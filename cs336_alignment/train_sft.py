"""SFT training script for CS336 Assignment 5.

Usage:
    python train_sft.py \
        --model_name Qwen/Qwen2.5-Math-1.5B \
        --data_path data/sft-reason/sft_gpt-oss-120b_filtered.jsonl \
        --output_dir output/sft \
        --epochs 3 \
        --lr 5e-5 \
        --microbatch_size 2 \
        --gradient_accumulation_steps 8 \
        --use_lora
"""

import argparse
import json
import math
import os
import sys

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cs336_alignment.plot_utils import MetricsLogger, plot_sft_curves
from tests.adapters import run_get_response_log_probs, run_sft_microbatch_train_step, run_tokenize_prompt_and_output

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <thinkPubMed> </thinkPubMed> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <thinkPubMed> reasoning process here </thinkPubMed> <answer> answer here </answer>.
User: {question}
Assistant: """


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(data_path):
    """Load SFT data. Supports JSON array or JSONL format."""
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]

    examples = []
    for item in data:
        problem = item["problem"]
        # reasoning_trace 包含完整的推理过程和 <answer>...</answer>
        reasoning_trace = item.get("reasoning_trace", "")
        expected_answer = item.get("expected_answer", "")
        extracted_answer = item.get("extracted_answer", "")

        # 构造 prompt 和 response
        prompt = R1_ZERO_PROMPT.format(question=problem)
        # response 以 <thinkPubMed>\n 开头
        if not reasoning_trace.startswith("<thinkPubMed>"):
            response = f"<thinkPubMed>\n{reasoning_trace}\n</thinkPubMed> <answer>{expected_answer or extracted_answer}</answer>"
        else:
            response = reasoning_trace

        examples.append({"prompt": prompt, "response": response})

    return examples


def setup_lora(model, lora_rank, lora_alpha):
    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.use_lora:
        model = setup_lora(model, args.lora_rank, args.lora_alpha)

    model.train()

    # 加载数据
    examples = load_data(args.data_path)
    print(f"Loaded {len(examples)} examples")

    # optimizer 和 scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(examples) // (args.microbatch_size * args.gradient_accumulation_steps)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    running_loss = 0.0
    logger = MetricsLogger(os.path.join(args.output_dir, "training_log.jsonl"))

    for epoch in range(args.epochs):
        # shuffle
        indices = torch.randperm(len(examples)).tolist()

        for i in tqdm(range(0, len(examples), args.microbatch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[i:i + args.microbatch_size]
            batch = [examples[idx] for idx in batch_indices]

            prompt_strs = [ex["prompt"] for ex in batch]
            response_strs = [ex["response"] for ex in batch]

            # tokenize
            tokenized = run_tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
            input_ids = tokenized["input_ids"].to(model.device)
            labels = tokenized["labels"].to(model.device)
            response_mask = tokenized["response_mask"].to(model.device)

            # forward pass
            result = run_get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            log_probs = result["log_probs"]

            # SFT train step (内含 backward)
            loss, _ = run_sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            running_loss += loss.item()
            global_step += 1

            # gradient accumulation: 每 N 步更新一次
            if global_step % args.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = None

            # logging
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                log_entry = {
                    "step": global_step,
                    "loss": avg_loss,
                    "perplexity": math.exp(min(avg_loss, 20)),
                    "lr": lr,
                    "epoch": epoch + 1,
                }
                if grad_norm is not None:
                    log_entry["grad_norm"] = grad_norm.item()
                logger.log(log_entry)
                running_loss = 0.0

            # save checkpoint
            if global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    # 保存最终模型
    save_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved final model to {save_path}")

    logger.close()
    plot_sft_curves(os.path.join(args.output_dir, "training_log.jsonl"), args.output_dir)


if __name__ == "__main__":
    main()

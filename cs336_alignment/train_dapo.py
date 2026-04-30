"""DAPO training script for CS336 Assignment 5.

DAPO 相比 GRPO 的三个改进：
1. Dynamic Sampling — 过滤全对/全错的 group（无梯度信号）
2. Token-level Loss — 按实际 response 长度归一化，而非固定常数
3. Over-long Filtering — 过滤超长回复

Usage:
    python train_dapo.py \
        --model_name output/sft/merged \
        --data_path data/sft-reason/train.jsonl \
        --output_dir output/dapo \
        --n_iterations 50 \
        --n_prompts_per_rollout 8 \
        --group_size 4 \
        --lr 5e-6 \
        --use_lora
"""

import argparse
import json
import os
import sys

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.adapters import (
    run_compute_group_normalized_rewards,
    run_get_response_log_probs,
    run_grpo_microbatch_train_step,
    run_tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.plot_utils import MetricsLogger, plot_grpo_curves

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <thinkPubMed> </thinkPubMed> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <thinkPubMed> reasoning process here </thinkPubMed> <answer> answer here </answer>.
User: {question}
Assistant: <thinkPubMed>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/dapo")
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--n_prompts_per_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--generation_temperature", type=float, default=0.7)
    parser.add_argument("--generation_top_p", type=float, default=0.9)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--loss_type", type=str, default="grpo_clip",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--normalize_by_std", action="store_true", default=True)
    parser.add_argument("--num_update_steps_per_rollout", type=int, default=1)
    parser.add_argument("--max_response_chars", type=int, default=4000,
                        help="DAPO: 过滤超过此长度的回复")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_problems(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]
    return data


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


def generate_rollouts(model, tokenizer, prompts, group_size, max_length, temperature, top_p, device):
    model.eval()
    all_prompts = []
    all_responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for j in range(group_size):
            generated_ids = outputs[j, prompt_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_prompts.append(prompt)
            all_responses.append(response)
    model.train()
    return all_prompts, all_responses


def dynamic_sampling_filter(prompts, responses, ground_truths, raw_rewards, group_size):
    """DAPO 改进1: 过滤全对或全错的 group（没有梯度信号）。"""
    n_groups = len(prompts) // group_size
    keep_indices = []

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_rewards = raw_rewards[start:end]

        # 如果全对或全错，跳过这个 group
        if (group_rewards == 1).all() or (group_rewards == 0).all():
            continue
        keep_indices.extend(range(start, end))

    if not keep_indices:
        return [], [], [], [], 0

    filtered_prompts = [prompts[i] for i in keep_indices]
    filtered_responses = [responses[i] for i in keep_indices]
    filtered_gts = [ground_truths[i] for i in keep_indices]
    filtered_rewards = raw_rewards[keep_indices]

    return filtered_prompts, filtered_responses, filtered_gts, filtered_rewards, len(keep_indices)


def overlong_filter(prompts, responses, ground_truths, max_chars):
    """DAPO 改进3: 过滤超长回复。"""
    keep = [i for i, r in enumerate(responses) if len(r) <= max_chars]
    if not keep:
        return [], [], []
    return [prompts[i] for i in keep], [responses[i] for i in keep], [ground_truths[i] for i in keep]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    problems = load_problems(args.data_path)
    print(f"Loaded {len(problems)} problems")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger = MetricsLogger(os.path.join(args.output_dir, "training_log.jsonl"))

    n_filtered_total = 0

    for iteration in range(args.n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.n_iterations} ===")

        # 1. 采样 prompt
        sample_indices = torch.randperm(len(problems))[:args.n_prompts_per_rollout].tolist()
        sampled_problems = [problems[i] for i in sample_indices]
        prompts_text = [R1_ZERO_PROMPT.format(question=p["problem"]) for p in sampled_problems]
        ground_truths = [str(p["expected_answer"]) for p in sampled_problems]

        # 2. 生成 rollout
        rollout_prompts, rollout_responses = generate_rollouts(
            model, tokenizer, prompts_text, args.group_size,
            args.max_response_length, args.generation_temperature, args.generation_top_p, device,
        )

        repeated_ground_truths = []
        for gt in ground_truths:
            repeated_ground_truths.extend([gt] * args.group_size)

        # DAPO 改进3: 过滤超长回复
        rollout_prompts, rollout_responses, repeated_ground_truths = overlong_filter(
            rollout_prompts, rollout_responses, repeated_ground_truths, args.max_response_chars,
        )
        if not rollout_prompts:
            print("All responses filtered (overlong), skipping iteration")
            continue

        # 3. 计算 reward 和 advantage
        # 先用原始 group_size 计算（过滤前的分组）
        # 需要重新构造分组：过滤后可能不是完整 group，用 raw_rewards 重新分组
        reward_details_all = [r1_zero_reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths)]
        raw_rewards_list = [d["reward"] for d in reward_details_all]
        raw_rewards = torch.tensor(raw_rewards_list)

        # DAPO 改进1: 动态采样过滤
        # 重新按 group_size 分组（过滤超长后可能打乱了原始分组）
        # 简化处理：对所有样本按当前顺序重新分组
        n_samples = len(rollout_prompts)
        n_groups = n_samples // args.group_size
        # 截断到整组
        valid_n = n_groups * args.group_size
        if valid_n == 0:
            print("Not enough samples for a complete group, skipping")
            continue

        rollout_prompts = rollout_prompts[:valid_n]
        rollout_responses = rollout_responses[:valid_n]
        repeated_ground_truths = repeated_ground_truths[:valid_n]
        raw_rewards = raw_rewards[:valid_n]

        # 动态过滤
        filtered_prompts, filtered_responses, filtered_gts, filtered_rewards, n_valid = \
            dynamic_sampling_filter(rollout_prompts, rollout_responses, repeated_ground_truths, raw_rewards, args.group_size)

        if n_valid == 0:
            n_filtered_total += 1
            print("All groups have uniform rewards (all correct/all wrong), skipping")
            continue

        # 对过滤后的样本重新计算 advantage
        advantages, _, _ = run_compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=filtered_responses,
            repeated_ground_truths=filtered_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.normalize_by_std,
        )

        # 用过滤前的 reward 做统计（包含所有样本）
        mean_reward = raw_rewards.mean().item()
        std_reward = raw_rewards.std().item()
        mean_advantage = advantages.mean().item()
        mean_format_reward = sum(d["format_reward"] for d in reward_details_all) / len(reward_details_all)
        mean_answer_reward = sum(d["answer_reward"] for d in reward_details_all) / len(reward_details_all)

        n_filtered = valid_n - n_valid
        print(f"Mean reward: {mean_reward:.4f} | Std: {std_reward:.4f} | "
              f"Format: {mean_format_reward:.2%} | Answer: {mean_answer_reward:.2%} | "
              f"Filtered: {n_filtered}/{valid_n}")

        # 4. 计算 old_log_probs（用过滤后的样本）
        model.eval()
        all_old_log_probs = []
        for i in range(0, len(filtered_prompts), args.microbatch_size):
            batch_prompts = filtered_prompts[i:i + args.microbatch_size]
            batch_responses = filtered_responses[i:i + args.microbatch_size]

            tokenized = run_tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)

            with torch.no_grad():
                result = run_get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            all_old_log_probs.append(result["log_probs"].cpu())

        max_len = max(t.size(1) for t in all_old_log_probs)
        all_old_log_probs = [torch.nn.functional.pad(t, (0, max_len - t.size(1))) for t in all_old_log_probs]
        old_log_probs = torch.cat(all_old_log_probs, dim=0)
        model.train()

        # 5. 策略更新
        for update_step in range(args.num_update_steps_per_rollout):
            perm = torch.randperm(len(filtered_prompts))
            optimizer.zero_grad()

            for i in range(0, len(filtered_prompts), args.microbatch_size):
                batch_idx = perm[i:i + args.microbatch_size]
                batch_prompts = [filtered_prompts[j] for j in batch_idx]
                batch_responses = [filtered_responses[j] for j in batch_idx]

                tokenized = run_tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
                input_ids = tokenized["input_ids"].to(device)
                labels = tokenized["labels"].to(device)
                response_mask = tokenized["response_mask"].to(device)

                result = run_get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
                policy_log_probs = result["log_probs"]
                batch_old_log_probs = old_log_probs[batch_idx].to(device)

                # 对齐序列长度
                cur_len = policy_log_probs.size(1)
                old_len = batch_old_log_probs.size(1)
                if cur_len > old_len:
                    batch_old_log_probs = torch.nn.functional.pad(batch_old_log_probs, (0, cur_len - old_len))
                elif old_len > cur_len:
                    batch_old_log_probs = batch_old_log_probs[:, :cur_len]

                batch_advantages = advantages[batch_idx].unsqueeze(-1).to(device)
                batch_raw_rewards = filtered_rewards[batch_idx].unsqueeze(-1).to(device)

                loss, metadata = run_grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=batch_raw_rewards,
                    advantages=batch_advantages,
                    old_log_probs=batch_old_log_probs,
                    cliprange=args.cliprange,
                )

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # logging
        if (iteration + 1) % args.log_every == 0:
            clip_frac = metadata.get("clip_fraction", 0)
            if isinstance(clip_frac, torch.Tensor):
                clip_frac = clip_frac.item()
            loss_val = loss.item()
            print(f"Loss: {loss_val:.4f} | Clip frac: {clip_frac:.4f}")
            logger.log({
                "iteration": iteration + 1,
                "loss": loss_val,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_advantage": mean_advantage,
                "mean_format_reward": mean_format_reward,
                "mean_answer_reward": mean_answer_reward,
                "clip_fraction": clip_frac,
                "grad_norm": grad_norm.item(),
                "n_filtered": n_filtered,
            })

        # save checkpoint
        if (iteration + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{iteration + 1}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")

    # 保存最终模型
    save_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved final model to {save_path}")

    logger.close()
    plot_grpo_curves(os.path.join(args.output_dir, "training_log.jsonl"), args.output_dir)


if __name__ == "__main__":
    main()

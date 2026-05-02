"""DAPO training script for CS336 Assignment 5.

DAPO improvements over GRPO:
1. Dynamic Sampling — filter groups with uniform rewards
2. Token-level Loss — normalize by actual response length per sample
3. Over-long Filtering — filter overly long responses
4. Asymmetric Clipping — cliprange_low=0.2, cliprange_high=0.28
5. Higher Temperature — temperature=1.0 for diversity

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
    run_tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, r1_zero_additive_reward_fn
from cs336_alignment.plot_utils import MetricsLogger, plot_dapo_curves

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <thinkPubMed> </thinkPubMed> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <thinkPubMed> reasoning process here </thinkPubMed> <answer> answer here </answer>.
User: {question}
Assistant: <thinkPubMed>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/dapo")
    parser.add_argument("--n_iterations", type=int, default=200)
    parser.add_argument("--n_prompts_per_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--generation_temperature", type=float, default=1.0,
                        help="DAPO uses higher temperature for diversity (default: 1.0)")
    parser.add_argument("--generation_top_p", type=float, default=0.9)
    parser.add_argument("--cliprange_low", type=float, default=0.2,
                        help="Lower clip ratio for DAPO asymmetric clipping")
    parser.add_argument("--cliprange_high", type=float, default=0.28,
                        help="Upper clip ratio for DAPO asymmetric clipping")
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--normalize_by_std", action="store_true", default=True)
    parser.add_argument("--num_update_steps_per_rollout", type=int, default=1)
    parser.add_argument("--reward_fn", type=str, default="multiplicative",
                        choices=["multiplicative", "additive"],
                        help="multiplicative: format*answer, additive: format+2*answer")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
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


def generate_rollouts(model, tokenizer, prompts, group_size, max_length,
                      temperature, top_p, device):
    """Generate rollouts using batched HF model.generate()."""
    model.eval()
    tokenizer.padding_side = "left"
    all_prompts = []
    all_responses = []
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_lens = (inputs["attention_mask"].sum(dim=1)).tolist()
        for i, prompt in enumerate(prompts):
            for j in range(group_size):
                idx = i * group_size + j
                generated_ids = outputs[idx, prompt_lens[i]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_prompts.append(prompt)
                all_responses.append(response)
    model.train()
    return all_prompts, all_responses


def dynamic_sampling_filter(prompts, responses, ground_truths, raw_rewards, group_size):
    """Filter groups with uniform rewards (no gradient signal)."""
    n_groups = len(prompts) // group_size
    keep_indices = []

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_rewards = raw_rewards[start:end]
        if (group_rewards == 1).all() or (group_rewards == 0).all():
            continue
        keep_indices.extend(range(start, end))

    if not keep_indices:
        return [], [], [], [], 0

    return (
        [prompts[i] for i in keep_indices],
        [responses[i] for i in keep_indices],
        [ground_truths[i] for i in keep_indices],
        raw_rewards[keep_indices],
        len(keep_indices),
    )


def overlong_filter(prompts, responses, ground_truths, max_chars):
    """Filter overly long responses."""
    keep = [i for i, r in enumerate(responses) if len(r) <= max_chars]
    if not keep:
        return [], [], []
    return [prompts[i] for i in keep], [responses[i] for i in keep], [ground_truths[i] for i in keep]


def dapo_loss(policy_log_probs, old_log_probs, advantages, response_mask,
              cliprange_low, cliprange_high, gradient_accumulation_steps):
    """DAPO asymmetric clip loss with token-level normalization."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Token-level: normalize per sample by actual response length
    response_lengths = response_mask.sum(dim=1, keepdim=True).clamp(min=1)
    per_sample_loss = (pg_losses * response_mask).sum(dim=1) / response_lengths.squeeze()
    loss = per_sample_loss.mean() / gradient_accumulation_steps

    with torch.no_grad():
        clip_mask = (ratio < 1 - cliprange_low) | (ratio > 1 + cliprange_high)
        clip_frac = (clip_mask.float() * response_mask).sum() / response_mask.sum()

    return loss, {"clip_fraction": clip_frac}


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

    reward_fn = r1_zero_additive_reward_fn if args.reward_fn == "additive" else r1_zero_reward_fn
    print(f"Using reward function: {args.reward_fn}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger = MetricsLogger(os.path.join(args.output_dir, "training_log.jsonl"))

    # Resume from checkpoint
    start_iteration = 0
    if args.resume_from_checkpoint:
        ckpt = args.resume_from_checkpoint
        print(f"Resuming from {ckpt}")
        if args.use_lora:
            base = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
            model = setup_lora(base, args.lora_rank, args.lora_alpha)
            adapter_file = os.path.join(ckpt, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                from safetensors.torch import load_file
                state_dict = load_file(adapter_file)
            else:
                state_dict = torch.load(
                    os.path.join(ckpt, "adapter_model.bin"),
                    map_location="cpu",
                )
            model.load_state_dict(state_dict, strict=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
        model.train()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_file = os.path.join(ckpt, "optimizer.pt")
        if os.path.exists(opt_file):
            optimizer.load_state_dict(torch.load(opt_file, map_location=device))
        iter_file = os.path.join(ckpt, "iteration.txt")
        if os.path.exists(iter_file):
            with open(iter_file) as f:
                start_iteration = int(f.read().strip())
        print(f"Resumed from iteration {start_iteration}")

    for iteration in range(start_iteration, args.n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.n_iterations} ===")

        # 1. Sample prompts
        sample_indices = torch.randperm(len(problems))[:args.n_prompts_per_rollout].tolist()
        sampled_problems = [problems[i] for i in sample_indices]
        prompts_text = [R1_ZERO_PROMPT.format(question=p["problem"]) for p in sampled_problems]
        ground_truths = [str(p["expected_answer"]) for p in sampled_problems]

        # 2. Generate rollouts
        rollout_prompts, rollout_responses = generate_rollouts(
            model, tokenizer, prompts_text, args.group_size,
            args.max_response_length, args.generation_temperature,
            args.generation_top_p, device,
        )

        repeated_ground_truths = []
        for gt in ground_truths:
            repeated_ground_truths.extend([gt] * args.group_size)

        # DAPO: overlong filter
        rollout_prompts, rollout_responses, repeated_ground_truths = overlong_filter(
            rollout_prompts, rollout_responses, repeated_ground_truths, args.max_response_chars,
        )
        if not rollout_prompts:
            print("All responses filtered (overlong), skipping iteration")
            continue

        # 3. Compute rewards
        reward_details_all = [
            reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths)
        ]
        raw_rewards = torch.tensor([d["reward"] for d in reward_details_all])

        # Truncate to complete groups
        n_groups = len(rollout_prompts) // args.group_size
        valid_n = n_groups * args.group_size
        if valid_n == 0:
            print("Not enough samples for a complete group, skipping")
            continue

        rollout_prompts = rollout_prompts[:valid_n]
        rollout_responses = rollout_responses[:valid_n]
        repeated_ground_truths = repeated_ground_truths[:valid_n]
        raw_rewards = raw_rewards[:valid_n]

        # DAPO: dynamic sampling filter
        filtered_prompts, filtered_responses, filtered_gts, filtered_rewards, n_valid = \
            dynamic_sampling_filter(
                rollout_prompts, rollout_responses, repeated_ground_truths,
                raw_rewards, args.group_size,
            )

        if n_valid == 0:
            print("All groups have uniform rewards, skipping")
            continue

        # Recompute advantages for filtered samples
        advantages, _, _ = run_compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=filtered_responses,
            repeated_ground_truths=filtered_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.normalize_by_std,
        )

        # Stats
        mean_reward = raw_rewards.mean().item()
        std_reward = raw_rewards.std().item()
        mean_advantage = advantages.mean().item()
        mean_format_reward = sum(d["format_reward"] for d in reward_details_all) / len(reward_details_all)
        mean_answer_reward = sum(d["answer_reward"] for d in reward_details_all) / len(reward_details_all)
        n_filtered = valid_n - n_valid
        print(f"Mean reward: {mean_reward:.4f} | Std: {std_reward:.4f} | "
              f"Format: {mean_format_reward:.2%} | Answer: {mean_answer_reward:.2%} | "
              f"Filtered: {n_filtered}/{valid_n}")

        # 4. Compute old_log_probs
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
        all_old_log_probs = [
            torch.nn.functional.pad(t, (0, max_len - t.size(1))) for t in all_old_log_probs
        ]
        old_log_probs = torch.cat(all_old_log_probs, dim=0)
        model.train()

        # 5. Policy update with DAPO asymmetric clip + token-level loss
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

                # Align sequence lengths
                cur_len = policy_log_probs.size(1)
                old_len = batch_old_log_probs.size(1)
                if cur_len > old_len:
                    batch_old_log_probs = torch.nn.functional.pad(
                        batch_old_log_probs, (0, cur_len - old_len)
                    )
                elif old_len > cur_len:
                    batch_old_log_probs = batch_old_log_probs[:, :cur_len]

                batch_advantages = advantages[batch_idx].unsqueeze(-1).to(device)

                loss, metadata = dapo_loss(
                    policy_log_probs, batch_old_log_probs, batch_advantages,
                    response_mask, args.cliprange_low, args.cliprange_high,
                    args.gradient_accumulation_steps,
                )
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Logging
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

        # Save checkpoint (model + optimizer + iteration)
        if (iteration + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{iteration + 1}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            with open(os.path.join(save_path, "iteration.txt"), "w") as f:
                f.write(str(iteration + 1))
            print(f"Saved checkpoint to {save_path}")

    # Save final model
    save_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved final model to {save_path}")

    logger.close()
    plot_dapo_curves(os.path.join(args.output_dir, "training_log.jsonl"), args.output_dir)


if __name__ == "__main__":
    main()

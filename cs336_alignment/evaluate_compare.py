"""
三阶段模型对比测评脚本 (Baseline / SFT / GRPO)。

复用 evaluate_vllm.py 的推理和打分逻辑，对多个模型分别测评后生成对比图表和表格。

用法:
    # 方式一：直接推理多个模型
    python cs336_alignment/evaluate_compare.py \
        --models baseline=Qwen/Qwen2.5-Math-1.5B sft=output/sft/final grpo=output/grpo/final \
        --data-path data/sft-reason/val.jsonl \
        --output-dir output/eval

    # 方式二：从已有结果文件加载（不重新推理）
    python cs336_alignment/evaluate_compare.py \
        --results baseline=output/eval/baseline.jsonl sft=output/eval/sft.jsonl grpo=output/eval/grpo.jsonl \
        --output-dir output/eval
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_vllm import (
    format_prompts,
    load_math_dataset,
    load_prompt_template,
)

logger = logging.getLogger(__name__)

R1_ZERO_PROMPT_PATH = "data/sft-reason/r1_zero.prompt"

STAGE_ORDER = ["baseline", "sft", "grpo", "dapo"]
STAGE_COLORS = {"baseline": "#4C72B0", "sft": "#55A868", "grpo": "#C44E52", "dapo": "#8172B2"}
STAGE_LABELS = {"baseline": "Baseline", "sft": "SFT", "grpo": "GRPO", "dapo": "DAPO"}


def evaluate_single_model(
    model_path: str,
    data_path: str,
    prompt_path: str,
    output_path: str,
    num_gpus: int = 1,
) -> list[dict]:
    """用 vLLM 对单个模型推理并保存 per-sample 结果。"""
    samples = load_math_dataset(data_path)
    template = load_prompt_template(prompt_path)
    prompts = format_prompts(samples, template)
    ground_truths = [s.get("expected_answer", s.get("answer", "")) for s in samples]

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=4096, top_p=0.9)
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for prompt, gt, output in tqdm(
        zip(prompts, ground_truths, outputs), total=len(prompts), desc=f"Eval {model_path}"
    ):
        generated = output.outputs[0].text
        reward_dict = r1_zero_reward_fn(generated, gt)
        results.append({
            "prompt": prompt,
            "generated": generated,
            "ground_truth": gt,
            "format_reward": reward_dict.get("format_reward", 0.0),
            "answer_reward": reward_dict.get("answer_reward", 0.0),
            "reward": reward_dict["reward"],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with xopen(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def load_results_file(path: str) -> list[dict]:
    """从 JSONL 结果文件加载。"""
    results = []
    with xopen(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def compute_metrics(results: list[dict]) -> dict:
    """从结果列表计算汇总指标。"""
    n = len(results)
    if n == 0:
        return {"accuracy": 0, "format_rate": 0, "answer_rate": 0,
                "correct": 0, "wrong_answer": 0, "wrong_format": 0, "total": 0}

    correct = sum(1 for r in results if r["reward"] == 1.0)
    wrong_answer = sum(1 for r in results if r.get("format_reward") == 1.0 and r["reward"] == 0.0)
    wrong_format = sum(1 for r in results if r.get("format_reward", 0.0) == 0.0)

    return {
        "accuracy": correct / n,
        "format_rate": (correct + wrong_answer) / n,
        "answer_rate": correct / n if (correct + wrong_answer) > 0 else 0.0,
        "correct": correct,
        "wrong_answer": wrong_answer,
        "wrong_format": wrong_format,
        "total": n,
    }


def parse_kv_args(items: list[str]) -> dict[str, str]:
    """解析 name=value 格式的参数列表。"""
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid format '{item}', expected name=value")
        name, value = item.split("=", 1)
        if name not in STAGE_ORDER:
            logger.warning(f"Unknown stage '{name}', will still be included")
        result[name] = value
    return result


def print_comparison_table(all_metrics: dict[str, dict]):
    """在终端打印对比表格。"""
    header = f"{'Stage':<12} {'Accuracy':>10} {'Format Rate':>12} {'Correct':>9} {'Wrong Ans':>10} {'Wrong Fmt':>10} {'Total':>7}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for stage in STAGE_ORDER:
        if stage not in all_metrics:
            continue
        m = all_metrics[stage]
        label = STAGE_LABELS.get(stage, stage)
        print(f"{label:<12} {m['accuracy']:>9.2%} {m['format_rate']:>11.2%} "
              f"{m['correct']:>9} {m['wrong_answer']:>10} {m['wrong_format']:>10} {m['total']:>7}")
    print(sep)
    print()


def save_comparison_csv(all_metrics: dict[str, dict], output_path: str):
    """保存对比结果到 CSV。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "accuracy", "format_rate", "answer_rate",
                         "correct", "wrong_answer", "wrong_format", "total"])
        for stage in STAGE_ORDER:
            if stage not in all_metrics:
                continue
            m = all_metrics[stage]
            writer.writerow([stage, f"{m['accuracy']:.4f}", f"{m['format_rate']:.4f}",
                             f"{m['answer_rate']:.4f}", m["correct"], m["wrong_answer"],
                             m["wrong_format"], m["total"]])
    logger.info(f"CSV saved to {output_path}")


def plot_comparison(all_metrics: dict[str, dict], output_dir: str):
    """生成对比图表并保存为 PNG。"""
    stages = [s for s in STAGE_ORDER if s in all_metrics]
    if len(stages) < 2:
        logger.warning("Need at least 2 stages to plot comparison, skipping.")
        return

    labels = [STAGE_LABELS.get(s, s) for s in stages]
    colors = [STAGE_COLORS.get(s, "#333333") for s in stages]
    metrics_data = {s: all_metrics[s] for s in stages}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Chart 1: Accuracy / Format Rate bar chart ---
    ax1 = axes[0]
    x = range(len(stages))
    width = 0.3
    accuracies = [metrics_data[s]["accuracy"] for s in stages]
    format_rates = [metrics_data[s]["format_rate"] for s in stages]

    bars1 = ax1.bar([i - width / 2 for i in x], accuracies, width, label="Accuracy", color=colors, alpha=0.9)
    bars2 = ax1.bar([i + width / 2 for i in x], format_rates, width, label="Format Rate",
                     color=colors, alpha=0.5, hatch="//")

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Rate")
    ax1.set_title("Accuracy & Format Rate")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # --- Chart 2: Stacked bar (category breakdown) ---
    ax2 = axes[1]
    correct = [metrics_data[s]["correct"] for s in stages]
    wrong_answer = [metrics_data[s]["wrong_answer"] for s in stages]
    wrong_format = [metrics_data[s]["wrong_format"] for s in stages]

    ax2.bar(labels, correct, label="Correct", color="#55A868")
    ax2.bar(labels, wrong_answer, bottom=correct, label="Format OK, Wrong Answer", color="#DDAA33")
    ax2.bar(labels, wrong_format, bottom=[c + w for c, w in zip(correct, wrong_answer)],
            label="Wrong Format", color="#C44E52")

    for i, s in enumerate(stages):
        total = metrics_data[s]["total"]
        ax2.text(i, total + max(total * 0.02, 20), str(total), ha="center", va="bottom", fontsize=9)

    ax2.set_ylabel("Count")
    ax2.set_title("Category Breakdown")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # --- Chart 3: Delta from baseline ---
    ax3 = axes[2]
    if "baseline" in metrics_data:
        baseline_acc = metrics_data["baseline"]["accuracy"]
        baseline_fmt = metrics_data["baseline"]["format_rate"]
        deltas_acc = [(metrics_data[s]["accuracy"] - baseline_acc) * 100 for s in stages]
        deltas_fmt = [(metrics_data[s]["format_rate"] - baseline_fmt) * 100 for s in stages]

        ax3.bar([i - width / 2 for i in x], deltas_acc, width, label="Accuracy Delta", color=colors, alpha=0.9)
        ax3.bar([i + width / 2 for i in x], deltas_fmt, width, label="Format Rate Delta",
                 color=colors, alpha=0.5, hatch="//")

        ax3.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("Delta (percentage points)")
        ax3.set_title("Improvement over Baseline")
        ax3.legend(loc="upper left")
        ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No baseline to compare", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Improvement over Baseline")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "eval_compare.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline / SFT / GRPO models")
    parser.add_argument("--models", nargs="+", help="Models as name=path pairs (e.g. baseline=Qwen/Qwen2.5-Math-1.5B)")
    parser.add_argument("--results", nargs="+", help="Existing result files as name=path pairs (e.g. baseline=results.jsonl)")
    parser.add_argument("--data-path", default="data/sft-reason/val.jsonl")
    parser.add_argument("--prompt-path", default=R1_ZERO_PROMPT_PATH)
    parser.add_argument("--output-dir", default="output/eval")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stage_results: dict[str, list[dict]] = {}

    # Load from --results files
    if args.results:
        for name, path in parse_kv_args(args.results).items():
            logger.info(f"Loading results for '{name}' from {path}")
            stage_results[name] = load_results_file(path)

    # Run inference for --models
    if args.models:
        for name, model_path in parse_kv_args(args.models).items():
            output_path = os.path.join(args.output_dir, f"{name}.jsonl")
            logger.info(f"Evaluating '{name}' model={model_path}")
            stage_results[name] = evaluate_single_model(
                model_path=model_path,
                data_path=args.data_path,
                prompt_path=args.prompt_path,
                output_path=output_path,
                num_gpus=args.num_gpus,
            )

    if not stage_results:
        parser.error("Provide at least --models or --results")

    # Compute metrics
    all_metrics = {name: compute_metrics(results) for name, results in stage_results.items()}

    # Output
    print_comparison_table(all_metrics)
    save_comparison_csv(all_metrics, os.path.join(args.output_dir, "eval_compare.csv"))
    plot_comparison(all_metrics, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()

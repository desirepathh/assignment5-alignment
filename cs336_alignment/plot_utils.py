"""训练指标日志和曲线绘制工具。

被 train_sft.py 和 train_grpo.py 共用。
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MetricsLogger:
    """将训练指标追加写入 JSONL 文件，每行一个 JSON 对象。"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")

    def log(self, metrics: dict):
        self._file.write(json.dumps(metrics) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def load_metrics(log_path: str) -> list[dict]:
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_sft_curves(log_path: str, output_dir: str):
    """绘制 SFT 训练曲线：loss + perplexity, learning rate, gradient norm。"""
    records = load_metrics(log_path)
    if not records:
        return

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    lrs = [r["lr"] for r in records]
    has_grad_norm = any("grad_norm" in r for r in records)

    nrows = 3 if has_grad_norm else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)

    # Loss + Perplexity
    ax1 = axes[0]
    ax1.plot(steps, losses, color="#4C72B0", linewidth=1, label="Train Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("SFT Training Loss & Perplexity")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    ax1b = ax1.twinx()
    ppls = [math.exp(min(r["loss"], 20)) for r in records]
    ax1b.plot(steps, ppls, color="#C44E52", linewidth=1, linestyle="--", label="Train PPL")
    ax1b.set_ylabel("Perplexity", color="#C44E52")
    ax1b.tick_params(axis="y", labelcolor="#C44E52")
    ax1b.legend(loc="upper left")

    # Learning Rate
    ax2 = axes[1]
    ax2.plot(steps, lrs, color="#55A868", linewidth=1)
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("SFT Learning Rate")
    ax2.grid(alpha=0.3)
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # Gradient Norm
    if has_grad_norm:
        ax3 = axes[2]
        grad_norms = [r.get("grad_norm", 0) for r in records]
        ax3.plot(steps, grad_norms, color="#DDAA33", linewidth=1)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Gradient Norm")
        ax3.set_title("SFT Gradient Norm")
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "sft_training_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {fig_path}")


def plot_grpo_curves(log_path: str, output_dir: str):
    """绘制 GRPO 训练曲线：loss, reward + reward breakdown, clip fraction, gradient norm。"""
    records = load_metrics(log_path)
    if not records:
        return

    iters = [r["iteration"] for r in records]
    has_grad_norm = any("grad_norm" in r for r in records)

    nrows = 4 if has_grad_norm else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)

    # 1. Loss
    ax1 = axes[0]
    losses = [r["loss"] for r in records]
    ax1.plot(iters, losses, color="#C44E52", linewidth=1)
    ax1.set_ylabel("Loss")
    ax1.set_title("GRPO Training Loss")
    ax1.grid(alpha=0.3)

    # 2. Reward + breakdown
    ax2 = axes[1]
    mean_rewards = [r["mean_reward"] for r in records]
    std_rewards = [r.get("std_reward", 0) for r in records]
    ax2.plot(iters, mean_rewards, color="#55A868", linewidth=1, label="Mean Reward")
    ax2.fill_between(iters,
                      [m - s for m, s in zip(mean_rewards, std_rewards)],
                      [m + s for m, s in zip(mean_rewards, std_rewards)],
                      color="#55A868", alpha=0.15)
    has_format = any("mean_format_reward" in r for r in records)
    if has_format:
        format_rewards = [r.get("mean_format_reward", 0) for r in records]
        answer_rewards = [r.get("mean_answer_reward", 0) for r in records]
        ax2.plot(iters, format_rewards, color="#4C72B0", linewidth=1, linestyle="--", label="Format Reward")
        ax2.plot(iters, answer_rewards, color="#DDAA33", linewidth=1, linestyle="--", label="Answer Reward")
    ax2.set_ylabel("Reward")
    ax2.set_title("GRPO Reward")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Clip Fraction
    ax3 = axes[2]
    clip_fracs = [r.get("clip_fraction", 0) for r in records]
    ax3.plot(iters, clip_fracs, color="#DDAA33", linewidth=1)
    ax3.set_ylabel("Clip Fraction")
    ax3.set_title("GRPO Clip Fraction")
    ax3.set_ylim(0, 1)
    ax3.axhspan(0.1, 0.3, alpha=0.08, color="green", label="Healthy range (0.1-0.3)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Gradient Norm
    if has_grad_norm:
        ax4 = axes[3]
        grad_norms = [r.get("grad_norm", 0) for r in records]
        ax4.plot(iters, grad_norms, color="#888888", linewidth=1)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Gradient Norm")
        ax4.set_title("GRPO Gradient Norm")
        ax4.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "grpo_training_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {fig_path}")

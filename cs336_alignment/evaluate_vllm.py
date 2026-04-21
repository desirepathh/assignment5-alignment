"""
评估 Qwen 2.5 Math 1.5B 模型在 MATH 数据集上的零样本性能

使用方法：
    uv run python cs336_alignment/evaluate_vllm.py \\
        --data-path data/sft-reason/val.jsonl \\
        --model-path Qwen/Qwen2.5-Math-1.5B \\
        --output-path math_baseline_results.jsonl
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Callable
from statistics import mean

from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

logger = logging.getLogger(__name__)


def load_prompt_template(template_path: str) -> str:
    """读取提示模板"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_math_dataset(data_path: str) -> List[dict]:
    """加载 MATH 数据集"""
    samples = []
    with xopen(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                # 支持 [ {...} ] 格式
                if line.startswith('['):
                    continue
                if line.endswith(']'):
                    continue
                # 去掉末尾逗号
                if line.endswith(','):
                    line = line[:-1]
                if line:
                    samples.append(json.loads(line))
    return samples


def format_prompts(samples: List[dict], template: str) -> List[str]:
    """使用 r1_zero 模板格式化提示"""
    prompts = []
    for sample in samples:
        problem = sample.get('problem', sample.get('question', ''))
        prompt = template.replace('{question}', problem)
        prompts.append(prompt)
    return prompts


def categorize_result(result: dict) -> str:
    """根据奖励值分类结果"""
    format_reward = result.get("format_reward", 0.0)
    answer_reward = result.get("answer_reward", 0.0)

    if format_reward == 0.0:
        return "wrong_format"
    elif format_reward == 1.0 and answer_reward == 1.0:
        return "correct_format_correct_answer"
    else:  # format_reward == 1.0 and answer_reward == 0.0
        return "correct_format_wrong_answer"


def generate_analysis_report(results: List[dict], samples_per_category: int = 10) -> dict:
    """
    生成分类统计和分析报告

    Args:
        results: 评估结果列表
        samples_per_category: 每个类别提取的样本数量

    Returns:
        分析报告字典
    """
    # 按类别分类
    categories = {
        "correct_format_correct_answer": [],
        "correct_format_wrong_answer": [],
        "wrong_format": [],
    }

    for result in results:
        category = categorize_result(result)
        categories[category].append(result)

    # 提取每类样本
    category_samples = {}
    for cat_name, cat_results in categories.items():
        category_samples[cat_name] = cat_results[:samples_per_category]

    # 统计
    total = len(results)
    stats = {
        "total": total,
        "correct_format_correct_answer": {
            "count": len(categories["correct_format_correct_answer"]),
            "percentage": len(categories["correct_format_correct_answer"]) / total if total > 0 else 0,
        },
        "correct_format_wrong_answer": {
            "count": len(categories["correct_format_wrong_answer"]),
            "percentage": len(categories["correct_format_wrong_answer"]) / total if total > 0 else 0,
        },
        "wrong_format": {
            "count": len(categories["wrong_format"]),
            "percentage": len(categories["wrong_format"]) / total if total > 0 else 0,
        },
    }

    return {
        "stats": stats,
        "samples": category_samples,
    }


def save_analysis_report(report: dict, output_path: str):
    """保存分析报告到文件"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# MATH Baseline Evaluation Report\n\n")

        # 写入统计
        f.write("## Category Statistics\n\n")
        stats = report["stats"]
        f.write(f"Total Samples: {stats['total']}\n\n")

        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")

        for cat_name in ["correct_format_correct_answer", "correct_format_wrong_answer", "wrong_format"]:
            cat_info = stats[cat_name]
            display_name = cat_name.replace("_", " ").title()
            f.write(f"| {display_name} | {cat_info['count']} | {cat_info['percentage']:.2%} |\n")

        f.write("\n")

        # 写入样本
        samples = report["samples"]

        for cat_name in ["correct_format_correct_answer", "correct_format_wrong_answer", "wrong_format"]:
            display_name = cat_name.replace("_", " ").title()
            f.write(f"## {display_name} Examples\n\n")

            cat_samples = samples[cat_name]
            if not cat_samples:
                f.write(f"No samples in this category.\n\n")
                continue

            for i, sample in enumerate(cat_samples[:10], 1):
                f.write(f"### Example {i}\n\n")
                f.write(f"**Format Reward:** {sample.get('format_reward', 0.0)} | ")
                f.write(f"**Answer Reward:** {sample.get('answer_reward', 0.0)} | ")
                f.write(f"**Total Reward:** {sample['reward']}\n\n")

                f.write(f"**Ground Truth:**\n```\n{sample['ground_truth']}\n```\n\n")

                f.write(f"**Generated Response:**\n```\n{sample['generated'][:1000]}\n```\n\n")

                if len(sample['generated']) > 1000:
                    f.write(f"... (truncated, total {len(sample['generated'])} chars)\n\n")

                f.write("---\n\n")

    logger.info(f"Analysis report saved to {output_path}")


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
    num_analysis_samples: int = 10,
) -> dict:
    """
    对一组提示评估语言模型，计算评估指标，并将结果序列化到磁盘。

    Args:
        vllm_model: vLLM 模型实例
        reward_fn: 奖励函数，接受 (response, ground_truth) 返回 reward 字典
        prompts: 格式化后的提示列表
        ground_truths: 对应的正确答案列表
        eval_sampling_params: 采样参数
        output_path: 结果保存路径
        num_analysis_samples: 每个类别提取的样本数量

    Returns:
        统计信息字典
    """
    # 1. 生成模型输出
    logger.info(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # 2. 计算奖励/评估指标
    results = []
    total_reward = 0.0
    total_format = 0.0
    total_answer = 0.0

    for prompt, ground_truth, output in tqdm(
        zip(prompts, ground_truths, outputs),
        total=len(prompts),
        desc="Evaluating"
    ):
        generated_text = output.outputs[0].text

        # 使用 reward_fn 评估
        reward_dict = reward_fn(generated_text, ground_truth)

        total_reward += reward_dict["reward"]
        total_format += reward_dict.get("format_reward", 0.0)
        total_answer += reward_dict.get("answer_reward", 0.0)

        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "ground_truth": ground_truth,
            "reward": reward_dict["reward"],
            "format_reward": reward_dict.get("format_reward", 0.0),
            "answer_reward": reward_dict.get("answer_reward", 0.0),
        })

    # 3. 保存详细结果
    logger.info(f"Saving results to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with xopen(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 4. 生成分析报告
    logger.info("Generating analysis report...")
    analysis_report = generate_analysis_report(results, num_analysis_samples)

    # 5. 保存分析报告
    report_path = output_path.replace('.jsonl', '_analysis.md')
    save_analysis_report(analysis_report, report_path)

    # 6. 返回统计信息
    num_samples = len(prompts)
    stats = {
        "num_samples": num_samples,
        "accuracy": total_reward / num_samples if num_samples > 0 else 0.0,
        "format_rate": total_format / num_samples if num_samples > 0 else 0.0,
        "answer_rate": total_answer / num_samples if num_samples > 0 else 0.0,
    }

    # 添加分类统计
    stats["categories"] = analysis_report["stats"]

    return stats


def main(
    data_path: str,
    model_path: str,
    prompt_path: str,
    output_path: str,
    num_gpus: int = 1,
    num_analysis_samples: int = 10,
):
    # 1. 加载数据
    logger.info(f"Loading dataset from {data_path}")
    samples = load_math_dataset(data_path)
    logger.info(f"Loaded {len(samples)} samples")

    # 2. 加载提示模板
    logger.info(f"Loading prompt template from {prompt_path}")
    template = load_prompt_template(prompt_path)

    # 3. 格式化提示
    prompts = format_prompts(samples, template)
    ground_truths = [sample.get('expected_answer', sample.get('answer', '')) for sample in samples]

    # 4. 初始化模型
    logger.info(f"Loading model from {model_path} with {num_gpus} GPU(s)")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=4096,
    )

    # 5. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,  # 零样本评估可以适当降低温度
        max_tokens=4096,
        top_p=0.9,
    )

    # 6. 导入 reward function
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    # 7. 评估
    stats = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=output_path,
        num_analysis_samples=num_analysis_samples,
    )

    # 8. 打印统计信息
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  Total Samples: {stats['num_samples']}")
    logger.info(f"  Accuracy: {stats['accuracy']:.2%}")
    logger.info(f"  Format Rate: {stats['format_rate']:.2%}")
    logger.info(f"  Answer Rate: {stats['answer_rate']:.2%}")
    logger.info("")

    # 打印分类统计
    logger.info("Category Breakdown:")
    if "categories" in stats:
        cat_stats = stats["categories"]
        logger.info(f"  Format Correct + Answer Correct: {cat_stats['correct_format_correct_answer']['count']} ({cat_stats['correct_format_correct_answer']['percentage']:.2%})")
        logger.info(f"  Format Correct + Answer Wrong:   {cat_stats['correct_format_wrong_answer']['count']} ({cat_stats['correct_format_wrong_answer']['percentage']:.2%})")
        logger.info(f"  Format Wrong:                    {cat_stats['wrong_format']['count']} ({cat_stats['wrong_format']['percentage']:.2%})")

    logger.info("=" * 50)
    logger.info(f"Detailed report saved to: {args.output_path.replace('.jsonl', '_analysis.md')}")
    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5 Math on MATH dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/sft-reason/val.jsonl",
        help="Path to MATH validation dataset (JSONL format)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Path to model (HF hub id or local path)",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="data/sft-reason/r1_zero.prompt",
        help="Path to r1_zero prompt template",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="math_baseline_results.jsonl",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for vLLM",
    )
    parser.add_argument(
        "--num-analysis-samples",
        type=int,
        default=10,
        help="Number of samples to extract per category for analysis",
    )
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_path=args.model_path,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        num_gpus=args.num_gpus,
        num_analysis_samples=args.num_analysis_samples,
    )

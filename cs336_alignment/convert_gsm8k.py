"""
将 GSM8K 数据集转换为 r1_zero 格式

GSM8K 原始格式：
    {"question": "...", "answer": "推理过程\\n#### 答案"}

转换后格式（符合 r1_zero prompt 要求）：
    {"problem": "...", "reasoning_trace": "</think> 推理过程

 <answer>答案</answer>", "expected_answer": "..."}

r1_zero 格式说明：
    -

 标签标记思考过程开始
    -

 标记思考过程结束
    - <answer> 标记答案开始
    - </answer> 标记答案结束

使用方法：
    uv run python cs336_alignment/convert_gsm8k.py \\
        --input data/gsm8k/train.jsonl \\
        --output data/gsm8k/train_r1zero.jsonl
"""
import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm
from xopen import xopen

logger = logging.getLogger(__name__)


def extract_answer_from_gsm8k(answer_text: str) -> tuple[str, str]:
    """
    从 GSM8K 答案中提取推理过程和最终答案

    Args:
        answer_text: GSM8K 格式的答案，如 "推理过程\\n#### 答案"

    Returns:
        (reasoning, answer): 推理过程和答案
    """
    if '####' in answer_text:
        parts = answer_text.split('####')
        reasoning = parts[0].strip()
        answer = parts[1].strip()
        return reasoning, answer
    else:
        # 如果没有 #### 标记，整段作为推理，尝试从末尾提取数字
        reasoning = answer_text.strip()
        # 尝试从末尾提取数字作为答案
        import re
        match = re.search(r'(\d+(?:\.\d+)?)$', reasoning)
        if match:
            answer = match.group(1)
        else:
            answer = ""
        return reasoning, answer


def convert_gsm8k_to_r1zero(input_path: str, output_path: str) -> dict:
    """
    将 GSM8K 数据集转换为 r1_zero 格式

    Args:
        input_path: GSM8K 原始数据路径
        output_path: 输出路径

    Returns:
        统计信息字典
    """
    results = []
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "no_answer_marker": 0,
    }

    logger.info(f"Reading from {input_path}")
    with xopen(input_path) as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Converting"):
        stats["total"] += 1

        try:
            data = json.loads(line.strip())

            # 提取问题
            problem = data.get("question", "")

            # 提取答案和推理过程
            answer_text = data.get("answer", "")
            reasoning, answer = extract_answer_from_gsm8k(answer_text)

            # 检查是否有 #### 标记
            if '####' not in answer_text:
                stats["no_answer_marker"] += 1

            # 构造 r1_zero 格式
            converted = {
                "problem": problem,
                "reasoning_trace": f"<think>\n\n{reasoning}\n\n</think> <answer>{answer}</answer>",
                "extracted_answer": answer,
                "expected_answer": answer,
            }
            results.append(converted)
            stats["success"] += 1

        except Exception as e:
            logger.warning(f"Failed to process line: {e}")
            stats["failed"] += 1

    # 保存结果
    logger.info(f"Saving to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with xopen(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return stats


def main():
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Convert GSM8K dataset to r1_zero format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to GSM8K dataset (JSONL format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted dataset (JSONL format)",
    )
    args = parser.parse_args()

    stats = convert_gsm8k_to_r1zero(args.input, args.output)

    # 打印统计
    logger.info("=" * 50)
    logger.info("Conversion Statistics:")
    logger.info(f"  Total samples: {stats['total']}")
    logger.info(f"  Successful: {stats['success']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  No #### marker: {stats['no_answer_marker']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

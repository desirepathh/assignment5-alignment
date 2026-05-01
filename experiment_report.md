# CS336 Assignment 5 实验报告：LLM Alignment

## 1. 实验概述

本实验基于 Qwen2.5-Math-1.5B 基座模型，依次进行 Baseline 评测、监督微调（SFT）、GRPO 强化学习训练和 DAPO 强化学习训练，对比四个阶段的数学推理能力。

### 实验结果总览

| 阶段 | Accuracy | Format Rate | 正确 | 格式正确但答案错 | 格式错误 | 总计 |
|------|----------|-------------|------|-----------------|---------|------|
| Baseline | 8.98% | 30.76% | 449 | 1,089 | 3,462 | 5,000 |
| SFT | 47.84% | 99.16% | 2,392 | 2,566 | 42 | 5,000 |
| GRPO | 49.94% | 99.18% | 2,497 | 2,462 | 41 | 5,000 |
| **DAPO** | **54.12%** | **98.88%** | **2,706** | **2,238** | **56** | 5,000 |

## 2. 模型与数据

### 2.1 基座模型

- **模型**: Qwen2.5-Math-1.5B
- **参数量**: 15 亿
- **精度**: BFloat16

### 2.2 数据集

| 数据集 | 用途 | 规模 |
|--------|------|------|
| `sft_gpt-oss-120b_filtered.jsonl` | SFT 训练 | ~70K 条 |
| `train.jsonl` | GRPO/DAPO 训练 | ~7.5K 条 |
| `val.jsonl` | 验证集 | 5,000 条 |

### 2.3 Prompt 模板

使用 R1-Zero 风格提示，要求模型在 `<thinkPubMed>` 标签内展示推理过程，在 `<answer>` 标签内给出最终答案。

### 2.4 评分函数

使用 `r1_zero_reward_fn`，返回三个指标：
- **format_reward**: 格式是否正确（包含 `<thinkPubMed>` 和 `<answer>` 标签）
- **answer_reward**: 答案是否正确（在格式正确的前提下）
- **reward**: 最终奖励 = format_reward × answer_reward

## 3. 实验方法

### 3.1 阶段一：Baseline 评测

对原始 Qwen2.5-Math-1.5B 模型进行零样本推理评测，使用 vLLM 加速推理。

### 3.2 阶段二：监督微调（SFT）

在 SFT 数据集上使用因果语言建模目标进行微调。

**训练配置：**

| 参数 | 值 |
|------|-----|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA dropout | 0.05 |
| 学习率 | 2e-5 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| 最大序列长度 | 1024 |

### 3.3 阶段三：GRPO 训练

在 SFT 模型基础上进行 GRPO（Group Relative Policy Optimization）强化学习训练。

**GRPO 核心公式：**

优势值计算：
```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

损失函数（对称裁剪）：
```
ratio = exp(log π_new - log π_old)
clipped_ratio = clamp(ratio, 1 - ε, 1 + ε)    # ε = 0.2
loss = -mean(min(ratio × advantage, clipped_ratio × advantage))
```

**训练配置：**

| 参数 | 值 |
|------|-----|
| 迭代次数 | 50 |
| 每轮采样 prompt 数 | 8 |
| Group size | 4 |
| 学习率 | 5e-6 |
| 裁剪范围 ε | 0.2（对称） |
| 生成温度 | 0.7 |
| Loss 归一化 | 全局 token 均值 |

### 3.4 阶段四：DAPO 训练

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）在 GRPO 基础上引入四项改进：

#### 改进 1：动态采样（Dynamic Sampling）

过滤掉 group 内所有回复全对或全错的 group。这类 group 的优势值全为 0，不提供梯度信号，只会浪费训练资源。

```python
if (group_rewards == 1).all() or (group_rewards == 0).all():
    skip this group
```

#### 改进 2：Token-level Loss

标准 GRPO 的损失在所有 token 上取均值，长回复会贡献更多梯度。DAPO 改为**先按每个样本的 response 长度归一化，再取均值**：

```python
# GRPO: 全局 token 均值
loss = mean(all_token_losses)

# DAPO: 先按样本归一化，再取均值
per_sample_loss = sum(token_losses) / response_length
loss = mean(per_sample_losses)
```

这确保每个样本对梯度的贡献相等，不受回复长度影响。

#### 改进 3：非对称裁剪（Asymmetric Clipping）

GRPO 使用对称裁剪 `clip(ratio, 1-ε, 1+ε)`，DAPO 使用非对称裁剪：

```python
clipped_ratio = clamp(ratio, 1 - 0.2, 1 + 0.28)
```

上界更宽松（1.28 vs 1.2），允许策略以更大幅度增加动作概率，鼓励探索新解法。

#### 改进 4：超长过滤（Overlong Filtering）

过滤掉超过 `max_response_chars`（默认 4000 字符）的回复，避免过长回复影响训练质量。

**DAPO 训练配置：**

| 参数 | 值 | 与 GRPO 的区别 |
|------|-----|---------------|
| 迭代次数 | 200 | 50 → 200 |
| Group size | 8 | 4 → 8 |
| 裁剪范围 | ε_low=0.2, ε_high=0.28 | 对称 0.2 → 非对称 |
| 生成温度 | 0.7 | 相同 |
| Loss 归一化 | Token-level（按样本） | 全局均值 → 按样本归一化 |
| 动态采样 | 开启 | GRPO 无 |
| 超长过滤 | 开启（4000 字符） | GRPO 无 |

## 4. 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4090 (24GB) |
| 框架 | PyTorch + HuggingFace Transformers + PEFT (LoRA) |
| 推理引擎 | vLLM (评测) |
| 训练方式 | LoRA 微调（仅训练 LoRA 参数） |

## 5. 结果分析

### 5.1 四阶段准确率对比

```
Baseline  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  8.98%
SFT       ████████████████████████████░░░░░░░░░░ 47.84%
GRPO      █████████████████████████████░░░░░░░░░ 49.94%
DAPO      ████████████████████████████████░░░░░░ 54.12%
```

### 5.2 各阶段提升幅度

| 阶段转换 | 提升幅度 | 说明 |
|----------|---------|------|
| Baseline → SFT | **+38.86%** | 监督微调带来最大提升，模型学会了格式和基本推理 |
| SFT → GRPO | **+2.10%** | RL 训练进一步提升，但受限于迭代次数和 group_size |
| GRPO → DAPO | **+4.18%** | DAPO 算法改进带来显著提升，验证了动态采样和 token-level loss 的有效性 |

### 5.3 格式正确率

- Baseline 格式正确率仅 30.76%，模型不知道如何格式化输出
- SFT 后格式正确率达到 99.16%，模型基本掌握了输出格式
- GRPO 和 DAPO 保持了接近 99% 的格式正确率

### 5.4 DAPO 改进分析

DAPO 相比 GRPO 的 4.18% 提升主要来自：

1. **动态采样**：过滤掉无梯度信号的 group，每个训练批次都有有效梯度
2. **更大 group_size**（8 vs 4）：优势值估计更准确，训练更稳定
3. **非对称裁剪**：允许更大的策略更新幅度，加速学习
4. **Token-level loss**：避免长回复主导梯度，训练更均衡
5. **更多训练迭代**（200 vs 50）：更充分的训练

### 5.5 训练过程观察

DAPO 训练过程中：
- 初始 mean_reward 约 18-25%，随着训练逐步提升
- 动态采样过滤比例从初期的 60-75% 逐步降低，说明模型在更多 prompt 上产生"边界"回答
- clip_fraction 维持在 0.1-0.3 的健康范围内
- format_reward 全程保持在 87-100%，未出现格式能力退化

## 6. 结论

1. **SFT 是最关键的一步**：Baseline → SFT 带来 38.86% 的提升，说明监督微调能有效教会模型输出格式和基本推理能力。

2. **DAPO 优于 GRPO**：在相同基座模型和数据集上，DAPO（54.12%）比 GRPO（49.94%）提升 4.18 个百分点，验证了动态采样、非对称裁剪和 token-level loss 的有效性。

3. **训练效率**：DAPO 的动态采样过滤机制确保每个训练批次都有有效梯度信号，避免了 GRPO 中常见的"全对/全错 group"浪费问题。

4. **模型容量限制**：1.5B 参数模型的推理能力有上限，SFT 后的 47.84% 已经接近该规模模型在数学推理任务上的瓶颈。GRPO 和 DAPO 的提升是在此基础上的增量优化。

## 7. 文件结构

```
assignment5-alignment/
├── cs336_alignment/
│   ├── train_sft.py          # SFT 训练脚本
│   ├── train_grpo.py         # GRPO 训练脚本
│   ├── train_dapo.py         # DAPO 训练脚本
│   ├── evaluate_vllm.py      # 单模型评测脚本（vLLM）
│   ├── evaluate_compare.py   # 多阶段对比评测脚本
│   ├── drgrpo_grader.py      # 评分函数
│   └── plot_utils.py         # 训练曲线绘制工具
├── tests/
│   └── adapters.py           # 核心函数实现
├── output/
│   ├── eval/
│   │   ├── eval_compare.csv  # 四阶段对比结果
│   │   ├── eval_compare.png  # 对比图表
│   │   ├── baseline.jsonl    # Baseline 评测结果
│   │   ├── sft.jsonl         # SFT 评测结果
│   │   ├── grpo.jsonl        # GRPO 评测结果
│   │   └── dapo.jsonl        # DAPO 评测结果
│   ├── sft/
│   │   ├── training_log.jsonl
│   │   └── sft_training_curves.png
│   ├── grpo/
│   │   ├── training_log.jsonl
│   │   └── grpo_training_curves.png
│   └── dapo/
│       ├── training_log.jsonl
│       └── grpo_training_curves.png
└── data/
    └── sft-reason/           # 训练和验证数据
```

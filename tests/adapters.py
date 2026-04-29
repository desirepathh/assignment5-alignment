from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    # ============================================================
    # 原理：SFT 的 tokenize + 构建 label + 构建 response mask
    # ============================================================
    # 语言模型训练的核心是"给定前面的 token，预测下一个 token"。
    # 所以 input_ids 和 labels 是错位（shift）的关系：
    #
    # 假设完整序列是 [p1, p2, p3, o1, o2, o3, EOS] （p=prompt, o=output）
    #
    #   input_ids = [p1, p2, p3, o1, o2, o3]      去掉最后一个 token
    #   labels    = [p2, p3, o1, o2, o3, EOS]      去掉第一个 token
    #
    # 这样 input_ids[i] 预测的就是 labels[i]，
    # 即 "p1 预测 p2", "p2 预测 p3", ..., "o3 预测 EOS"
    #
    # response_mask 标记 labels 中哪些位置属于 response（输出）部分：
    #   input_ids    = [p1, p2, p3, o1, o2, o3]
    #   labels       = [p2, p3, o1, o2, o3, EOS]
    #   response_mask= [ 0,  0,  1,  1,  1,  1 ]   只有 output 部分为 1
    #
    # 训练时只在 response_mask=1 的位置计算 loss（不训练 prompt 部分）
    #
    # 实现步骤：
    #   1. 用 tokenizer 分别 tokenize 每个 prompt 和 prompt+output，
    #      得到每个 prompt 的长度和完整序列的 token ids
    #   2. 把所有序列 padding 到同一长度（用 tokenizer 的 padding 功能）
    #   3. input_ids = tokens[:, :-1]（去掉最后一列）
    #   4. labels    = tokens[:, 1:] （去掉第一列）
    #   5. 构建 response_mask：对每个序列，根据 prompt 长度，
    #      在 labels 中标记哪些位置是 response
    #      注意：padding 位置也应该是 0
    #
    # 提示：
    #   - tokenizer(prompt, text=output, ...) 可以同时 tokenize 两者
    #   - tokenizer 返回的 input_ids 包含特殊 token（如 BOS），
    #     要仔细确认 prompt 部分到底占几个 token
    #   - padding_mask（attention_mask）可以帮你识别哪些位置是 padding
    # ============================================================
    tokenizer.pad_token=tokenizer.eos_token
    # 分别 tokenize prompt 和 output，避免 BPE 边界合并问题
    tokenized_prompts=tokenizer(prompt_strs)
    tokenized_outputs=tokenizer(output_strs)
    # 拼接 token ids: prompt_ids + output_ids + [EOS]，并记录 prompt 长度
    prompt_lens=[]
    all_ids=[]
    for p_ids, o_ids in zip(tokenized_prompts["input_ids"], tokenized_outputs["input_ids"]):
        prompt_lens.append(len(p_ids))
        all_ids.append(p_ids + o_ids)
    # padding 到同一长度
    max_len=max(len(ids) for ids in all_ids)
    padded_ids=[]
    attention_masks=[]
    for ids in all_ids:
        pad_len=max_len-len(ids)
        padded_ids.append(ids+[tokenizer.pad_token_id]*pad_len)
        attention_masks.append([1]*len(ids)+[0]*pad_len)
    tokens=torch.tensor(padded_ids)
    attention_mask=torch.tensor(attention_masks)
    # shift
    input_ids=tokens[:,:-1]
    labels=tokens[:,1:]
    # 构建 response_mask
    response_mask=torch.zeros_like(labels)
    for i in range(len(prompt_lens)):
        response_mask[i,prompt_lens[i]-1:]=1
    # 去掉 padding 部分
    response_mask=response_mask*attention_mask[:,1:]
    return {
        "input_ids":input_ids,
        "labels":labels,
        "response_mask":response_mask,
    }


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    # ============================================================
    # 原理：GRPO 的组归一化奖励（Group Relative Advantage）
    # ============================================================
    # GRPO 的核心思想：不用 value network（critic），
    # 而是用"同一个 prompt 生成的多个回复的相对表现"来算 advantage。
    #
    # 举个例子：group_size=4，同一个 prompt 生成 4 个回复
    #   回复1: reward=0.9  (好)
    #   回复2: reward=0.3  (差)
    #   回复3: reward=0.7  (中)
    #   回复4: reward=0.5  (中)
    #   mean=0.6, std=0.22
    #
    # 归一化后：
    #   回复1: (0.9-0.6)/0.22 = +1.36  → 正优势，增加概率
    #   回复2: (0.3-0.6)/0.22 = -1.36  → 负优势，降低概率
    #   回复3: (0.7-0.6)/0.22 = +0.45  → 略正
    #   回复4: (0.5-0.6)/0.22 = -0.45  → 略负
    #
    # 这样不需要 critic 也能知道"比平均水平好还是差"
    #
    # 实现步骤：
    #
    # 1. 对每个 rollout_response 调用 reward_fn 拿到 raw reward：
    #      rewards = [reward_fn(response, gt) for response, gt in zip(...)]
    #      rewards = torch.tensor([r["reward"] for r in rewards])
    #      shape: [rollout_batch_size]
    #
    # 2. 按 group 划分，计算每组的 mean 和 std：
    #      rewards → reshape 为 [n_groups, group_size]
    #      group_mean = rewards.reshape(n, group_size).mean(dim=1, keepdim=True)
    #      group_std  = rewards.reshape(n, group_size).std(dim=1, keepdim=True)
    #
    # 3. 归一化：
    #      if normalize_by_std:
    #          normalized = (rewards - group_mean) / (group_std + advantage_eps)
    #      else:
    #          normalized = (rewards - group_mean)
    #
    #   注意：group_mean/group_std 要广播回 rollout_batch_size 的 shape
    #
    # 4. 返回 (normalized_rewards, raw_rewards, metadata)
    #    metadata 里可以放 mean_reward、std_reward 等统计量
    # ============================================================
    rewards_list=[]
    for response,gt in zip(rollout_responses,repeated_ground_truths):
        rewards_list.append(reward_fn(response,gt))
    rewards=torch.tensor([r["reward"] for r in rewards_list])
    n_groups=len(rollout_responses)//group_size
    rewards=rewards.reshape(n_groups,group_size)
    group_mean=rewards.mean(dim=1,keepdim=True)
    group_std=rewards.std(dim=1,keepdim=True)
    if normalize_by_std:
        normalized=(rewards-group_mean)/(group_std+advantage_eps)
    else:
        normalized=(rewards-group_mean)
    return (
        normalized.flatten(),
        rewards.flatten(),
        {
            "mean_reward":rewards.mean().item(),
            "std_reward":rewards.std().item()
        }
    )


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # 步骤 1: 计算对数概率
    # logits 是模型输出的原始分数（未归一化），shape.logits=[Batch,Sequence,Prosibility]
    # log_softmax = log(softmax(logits))，数值更稳定
    log_probs=torch.log_softmax(logits,dim=-1)
    #                    ↑ 将 logits 转为概率分布再取 log
    #                                        ↑ 沿最后一个维度（vocab）计算
    # 步骤 2: 还原回概率
    # p = exp(log_p) 反向操作得到实际概率

    # 步骤 3: 计算熵
    # 熵公式：H = -Σ p(x) × log(p(x))
    # 对每个token位置的词汇分布计算熵
    probs=torch.exp(log_probs)
    entropy=-(probs*log_probs).sum(dim=-1)
    #          ↑ p(x) × log(p(x))  ↑ 求和得到熵值,按最后一列加，每个token计算一个熵值
    #      ↑ 负号（因为log_p是负数）
    #shape.entropy=[B,S]
    return entropy



def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # ============================================================
    # 原理：从模型获取每个 token 位置的 log 概率
    # ============================================================
    # 模型前向传播（model(input_ids)）输出 logits，shape=[B, S, V]
    #   logits[b, s, :] = 第 b 个序列、第 s 个位置、对所有 vocab 词的原始分数
    #
    # 但我们不需要整个 V 维度的分数，我们只需要知道：
    #   "模型对 labels[b,s] 这个 token 给了多少概率"
    #
    # 步骤：
    #   1. output = model(input_ids) → logits shape [B, S, V]
    #   2. log_probs = log_softmax(logits, dim=-1) → 每个 vocab 词的 log 概率
    #   3. 用 labels 做 gather，取出"实际 token"对应的 log 概率：
    #      log_probs_of_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    #      然后 squeeze 回 [B, S]
    #
    #   gather 的直觉：
    #     log_probs[b, s, :] 是一个 V 维向量（所有词的 log 概率）
    #     labels[b, s] 是实际下一个 token 的 id
    #     gather 就是把 log_probs[b, s, labels[b,s]] 这一个值挑出来
    #
    #   4. 如果 return_token_entropy=True，还要计算每个位置的熵
    #      可以直接复用你已经实现的 run_compute_entropy(logits)
    #
    # 提示：
    #   - labels 需要确保不是 -100 或 padding id，但这里先不管，
    #     masking 会在 train loop 里做
    #   - gather 前 labels 要 unsqueeze(-1) 变成 [B, S, 1] 才能做 gather
    # ============================================================
    output=model(input_ids)
    logits=output.logits
    log_probs=torch.log_softmax(logits,dim=-1)
    log_probs_of_labels=log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy is True:
        token_entropy=run_compute_entropy(logits)
    else:
        token_entropy=None
    return {
        "log_probs":log_probs_of_labels,
        "token_entropy":token_entropy,
    }


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    # ============================================================
    # 原理：最朴素的策略梯度 loss（REINFORCE）
    # ============================================================
    # 策略梯度的核心思想：
    #   如果一个动作（这里=生成的 token 序列）得到了好的奖励，
    #   就增加这个动作的概率；反之就降低。
    #
    # 数学公式：
    #   loss = - raw_rewards_or_advantages * policy_log_probs
    #
    #   shape 分析：
    #     raw_rewards_or_advantages: [B, 1]
    #     policy_log_probs:          [B, S]
    #     相乘时 [B,1] 自动广播到 [B,S]（每个 token 乘同一个 reward）
    #
    # 为什么加负号？
    #   策略梯度要做梯度**上升**来最大化 reward，
    #   但 PyTorch 的 optimizer 默认做梯度**下降**（minimize），
    #   所以加负号把"最大化"变成"最小化"。
    #
    # 直觉：
    #   reward=+5, log_prob=-2 → loss = -5 * (-2) = +10 → 梯度下降会降低 loss → log_prob 变大 → 好动作概率增加
    #   reward=-5, log_prob=-2 → loss = -(-5) * (-2) = -10 → 梯度下降会提高 loss → log_prob 变小 → 坏动作概率降低
    #
    # 实现就一行：
    #   return -raw_rewards_or_advantages * policy_log_probs
    # ============================================================
    naive_policy_gradient_loss=-raw_rewards_or_advantages * policy_log_probs
    return naive_policy_gradient_loss


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    # ============================================================
    # 原理：GRPO Clip Loss（来自 PPO 的核心思想）
    # ============================================================
    # 问题：朴素策略梯度（REINFORCE）每次更新步长不可控，
    #   可能一步就把策略改得面目全非，导致训练崩溃。
    #
    # PPO 的解决方案：限制"新旧策略的比值"在一个范围内。
    #
    # 步骤 1 — 计算概率比 ratio：
    #   ratio = exp(policy_log_probs - old_log_probs)
    #
    #   直觉：ratio > 1 表示新策略比旧策略更倾向这个 token
    #         ratio < 1 表示新策略比旧策略更不倾向这个 token
    #         ratio = 1 表示没变
    #
    # 步骤 2 — 计算 clipped ratio：
    #   clipped_ratio = clamp(ratio, 1-cliprange, 1+cliprange)
    #
    #   例如 cliprange=0.2，ratio 会被限制在 [0.8, 1.2] 之间
    #
    # 步骤 3 — 计算 loss：
    #   loss_unclipped = -advantages * ratio              （无限制版本）
    #   loss_clipped   = -advantages * clipped_ratio      （限制版本）
    #   loss = max(loss_unclipped, loss_clipped)
    #
    #   注意：这里 max 是取两个中更差（更大）的 loss，
    #   等价于 min(ratio, clipped_ratio) * advantages 的负数
    #
    # 为什么要取 max？
    #   - 当 advantage > 0（好动作）：ratio 被上界 clip，
    #     防止新策略对这个动作的概率增加太多
    #   - 当 advantage < 0（坏动作）：ratio 被下界 clip，
    #     防止新策略对这个动作的概率减少太多
    #   → 不管哪种情况，都防止策略变化太剧烈
    #
    # 步骤 4 — 返回元数据（用于监控训练）：
    #   clip_fraction = (ratio != clipped_ratio).float().mean()
    #   表示有多少比例的 token 被 clip 了
    #
    # shape 分析：
    #   advantages:     [B, 1]
    #   ratio:          [B, S]    （exp of [B,S] - [B,S]）
    #   clipped_ratio:  [B, S]
    #   loss:           [B, S]
    # ============================================================
    ratio=torch.exp(policy_log_probs-old_log_probs)
    clipped_ratio=torch.clamp(ratio,1-cliprange,1+cliprange)
    loss_unclipped=-advantages*ratio
    loss_clipped=-advantages*clipped_ratio
    loss=torch.maximum(loss_clipped,loss_unclipped)
    clip_fraction=(ratio!=clipped_ratio).float().mean()
    return (loss,{"clip_fraction":clip_fraction})


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    # ============================================================
    # 原理：策略梯度 loss 的路由器（wrapper）
    # ============================================================
    # 这个函数不做计算，只是根据 loss_type 选择调用哪个 loss 函数：
    #
    #   loss_type="no_baseline"              → 调用 naive_policy_gradient_loss(raw_rewards, ...)
    #   loss_type="reinforce_with_baseline"  → 调用 naive_policy_gradient_loss(advantages, ...)
    #   loss_type="grpo_clip"                → 调用 grpo_clip_loss(advantages, ...)
    #
    # 三种模式的区别：
    #
    # 1. no_baseline：直接用 raw_rewards 作为权重
    #    loss = -raw_rewards * log_probs
    #    问题：reward 总是正的，所有 token 概率都会增加（只是好的增加更多）
    #
    # 2. reinforce_with_baseline：用 advantages（= rewards - mean）作为权重
    #    loss = -advantages * log_probs
    #    好处：advantages 有正有负，好动作概率增加，差动作概率降低
    #
    # 3. grpo_clip：用 clip 限制更新幅度
    #    loss = clip_loss(advantages, ...)
    #    好处：防止策略一步变化太大
    #
    # 实现就是 if/elif 路由：
    #   if loss_type == "no_baseline":
    #       return naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    #   elif loss_type == "reinforce_with_baseline":
    #       return naive_policy_gradient_loss(advantages, policy_log_probs)
    #   elif loss_type == "grpo_clip":
    #       return grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    #   注意 grpo_clip 返回 (loss, metadata)，其他两个只返回 loss
    #   所以需要统一返回格式为 (loss, metadata)
    # ============================================================

    if loss_type == "no_baseline":
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    # ============================================================
    # 原理：带掩码的均值计算
    # ============================================================
    # 在 SFT/GRPO 中，一个 batch 的序列长度各不相同，短序列会被 padding。
    # 我们只想对"有效 token"（mask=1）计算均值，忽略 padding（mask=0）。
    #
    # 数学公式：
    #   masked_mean(x, mask, dim) = sum(x * mask, dim) / sum(mask, dim)
    #
    # 例如 dim=1，shape=[B, S]：
    #   对每行，只对 mask=1 的位置求和，再除以该行 mask=1 的个数。
    #   结果 shape=[B]
    #
    # 实现提示：
    #   1. tensor 和 mask 形状相同，先把 mask 转为 float
    #   2. 用 (tensor * mask_float).sum(dim=dim) 得到有效元素之和
    #   3. 用 mask_float.sum(dim=dim) 得到有效元素个数
    #   4. 相除得到均值（注意避免除以 0，可以加 clamp(min=1e-8)）
    #   5. dim=None 时，等价于对整个 tensor 做 sum / count
    # ============================================================
    mask=mask.float()
    
    masked_mean=((tensor*mask).sum(dim=dim))/(mask.sum(dim=dim))

    return masked_mean

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    # ============================================================
    # 原理：SFT 的 microbatch 训练步
    # ============================================================
    # SFT 的目标：让模型在给定 prompt 的情况下，
    # 最大化"正确输出"每个 token 的 log 概率。
    #
    # loss = -mean(log_probs * response_mask) / normalize_constant
    #
    # 拆解：
    #   1. log_probs 已经是"模型给每个 token 的 log 概率"，shape [B, S]
    #   2. response_mask 标记哪些位置是 response（有效 token），shape [B, S]
    #   3. 负号：因为要最大化 log_prob，但 PyTorch 做最小化
    #   4. mean：在 response token 上求平均
    #   5. 除以 normalize_constant：梯度累加时的归一化
    #
    # gradient_accumulation_steps 是什么？
    #   显存不够时，把一个 batch 拆成多个 microbatch，
    #   每个 microbatch 算梯度但不更新参数，累加几次后再更新。
    #   所以 loss 要除以 gradient_accumulation_steps，
    #   保证多个 microbatch 的梯度累加后量级和一整个 batch 一样。
    #
    # 实现步骤：
    #   1. loss = -run_masked_normalize(log_probs, response_mask, dim=-1, normalize_constant=...)
    #      用 masked_normalize 而不是 masked_mean，
    #      因为要除以固定的 normalize_constant
    #   2. 再除以 gradient_accumulation_steps
    #   3. loss.backward() 反向传播
    #   4. 返回 (loss, metadata)
    # ============================================================

    loss = -run_masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant).mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, {}

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    # ============================================================
    # 原理：GRPO 的 microbatch 训练步
    # ============================================================
    # 和 SFT 的 microbatch 步类似，但用策略梯度 loss 替代简单的负 log prob。
    #
    # 步骤：
    #   1. 调用 run_compute_policy_gradient_loss 得到 per-token loss [B, S]
    #      和 metadata
    #   2. 用 response_mask 屏蔽掉 prompt 和 padding 位置
    #   3. 对有效 token 求 masked_mean 得到标量 loss
    #   4. 除以 gradient_accumulation_steps
    #   5. backward() 反向传播
    #   6. 返回 (loss, metadata)
    #
    # 和 SFT 的区别：
    #   SFT: loss = -masked_normalize(log_probs)       → 直接用 log_probs
    #   GRPO: loss = masked_mean(pg_loss * mask)        → 先算策略梯度 loss 再求均值
    #   SFT 用 masked_normalize（除以固定常数），
    #   GRPO 用 masked_mean（除以实际有效 token 数）
    # ============================================================

    per_token_loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = run_masked_mean(per_token_loss, response_mask, dim=-1).mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    # ============================================================
    # 原理：带掩码的归一化求和
    # ============================================================
    # 和 masked_mean 很像，但区别是：
    #   masked_mean:  sum(x*mask) / count(mask)     — 除以有效元素个数
    #   masked_normalize: sum(x*mask) / normalize_constant — 除以一个固定常数
    #
    # 为什么需要除以常数而不是除以 count？
    #   在 Dr.GRPO 中，loss 要除以一个固定的"序列长度"来归一化，
    #   而不是除以实际有效 token 数，这样可以避免短序列的梯度被放大。
    #
    # 数学公式：
    #   result = sum(x * mask, dim) / normalize_constant
    #
    # 实现提示：
    #   1. mask 转 float
    #   2. (tensor * mask_float).sum(dim=dim) 得到有效元素之和
    #   3. 除以 normalize_constant
    # ============================================================
    mask_float=mask.float()
    up_num=(tensor*mask_float).sum(dim=dim)
    masked_mean=up_num/normalize_constant
    return masked_mean


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError

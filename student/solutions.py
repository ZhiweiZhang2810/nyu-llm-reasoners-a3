from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# Monkeypatch torch.library.register_fake if missing (older torch versions)
if not hasattr(torch.library, "register_fake"):
    def register_fake(name):
        def wrapper(fn):
            return fn
        return wrapper
    torch.library.register_fake = register_fake


def tokenize_prompt_and_output(
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
            "input_ids": torch.Tensor of shape (batch_size, max_seq_len - 1)
            "labels": torch.Tensor of shape (batch_size, max_seq_len - 1)
            "response_mask": torch.Tensor of shape (batch_size, max_seq_len - 1)
    """
    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    # The snapshot implies a max sequence length of 10
    max_seq_len = 10

    # Use eos_token_id as padding if pad_token_id is not set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    for prompt, output in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately to avoid boundary merging
        p_ids = tokenizer.encode(prompt, add_special_tokens=True)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        
        # Combine: prompt + output + EOS
        combined = p_ids + o_ids + [tokenizer.eos_token_id]
        
        # Truncate to max_seq_len
        combined = combined[:max_seq_len]
        
        # Padding to max_seq_len
        if len(combined) < max_seq_len:
            combined = combined + [pad_token_id] * (max_seq_len - len(combined))

        # input_ids: all tokens except the last one (length max_seq_len - 1)
        input_ids = combined[:-1]
        # labels: all tokens except the first one (length max_seq_len - 1)
        labels = combined[1:]

        # response_mask: 1 for tokens in labels that come from o_ids
        # In 'combined', o_ids are at indices [len(p_ids), len(p_ids) + len(o_ids) - 1]
        # In 'labels' (which is combined[1:]), o_ids are at [len(p_ids) - 1, len(p_ids) + len(o_ids) - 2]
        # We also need to account for truncation at max_seq_len.
        mask = [False] * len(labels)
        start_idx = len(p_ids) - 1
        end_idx = len(p_ids) + len(o_ids) - 1
        for i in range(start_idx, end_idx):
            if i < len(labels):
                mask[i] = True

        batch_input_ids.append(torch.tensor(input_ids))
        batch_labels.append(torch.tensor(labels))
        batch_response_mask.append(torch.tensor(mask, dtype=torch.bool))

    input_ids_tensor = torch.stack(batch_input_ids)
    labels_tensor = torch.stack(batch_labels)
    response_mask_tensor = torch.stack(batch_response_mask)

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "response_mask": response_mask_tensor,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.
    """
    masked_tensor = tensor * mask
    if dim is None:
        summed = torch.sum(masked_tensor)
    else:
        summed = torch.sum(masked_tensor, dim=dim)
    return summed / normalize_constant


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.
    Returns NaN for empty dimensions (where mask sum is zero).
    """
    # Use float for division to get NaN when count is 0
    mask_f = mask.float()
    masked_tensor = tensor * mask_f
    if dim is None:
        summed = torch.sum(masked_tensor)
        counts = torch.sum(mask_f)
    else:
        summed = torch.sum(masked_tensor, dim=dim)
        counts = torch.sum(mask_f, dim=dim)
    # 0.0 / 0.0 results in NaN for tensors
    return summed / counts


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
    and optionally the entropy of the next token predictions.
    """
    outputs = model(input_ids)
    logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather log_probs of actual labels
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": gathered_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    The loss is summed over response tokens per instance, then averaged over the batch.
    """
    # Negative log-likelihood over response tokens
    per_token_loss = -policy_log_probs * response_mask
    
    # Sum over response tokens for each instance
    per_instance_loss = torch.sum(per_token_loss, dim=-1)
    
    # Average over the batch
    batch_loss = torch.mean(per_instance_loss)
    
    # Normalize by constant and grad accumulation steps
    loss = batch_loss / (normalize_constant * gradient_accumulation_steps)

    loss.backward()

    return loss, {"loss": loss.detach()}


def compute_group_normalized_rewards(
    reward_fn: callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses,
    normalized by the group size.
    """
    raw_rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(resp, gt)
        raw_rewards.append(reward_dict["reward"])

    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)

    # Reshape to (n_groups, group_size)
    n_groups = len(raw_rewards) // group_size
    rewards_grouped = raw_rewards.view(n_groups, group_size)

    means = rewards_grouped.mean(dim=1, keepdim=True)
    advantages = rewards_grouped - means

    if normalize_by_std:
        stds = rewards_grouped.std(dim=1, keepdim=True)
        advantages = advantages / (stds + advantage_eps)

    # Flatten back
    advantages = advantages.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item() if len(raw_rewards) > 1 else 0.0,
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages."""
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss."""
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
        
    ratio = torch.exp(policy_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages

    # GRPO objective is to MAXIMIZE min(surr1, surr2)
    loss = -torch.min(surr1, surr2)

    # Metadata: clip fraction
    clip_mask = (surr2 < surr1).float()

    return loss, {"clip_mask": clip_mask}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapper that delegates to the appropriate policy gradient loss function."""
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch."""
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Aggregate over response tokens using masked_mean
    loss = masked_mean(per_token_loss, response_mask)

    # Scale by gradient accumulation
    loss = loss / gradient_accumulation_steps

    loss.backward()

    return loss, metadata

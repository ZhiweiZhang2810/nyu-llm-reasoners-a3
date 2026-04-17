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
    max_seq_len: int | None = None,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.
        max_seq_len: int | None, the maximum sequence length.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, seq_len - 1)
            "labels": torch.Tensor of shape (batch_size, seq_len - 1)
            "response_mask": torch.Tensor of shape (batch_size, seq_len - 1)
    """
    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    # Set max_seq_len based on the longest combined sequence in the batch
    # instead of hardcoding a small value like 10.
    encoded_pairs = []
    for prompt, output in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(prompt, add_special_tokens=True)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        combined = p_ids + o_ids + [tokenizer.eos_token_id]
        encoded_pairs.append((p_ids, o_ids, combined))
    
    if max_seq_len is None:
        max_seq_len = max(len(c) for _, _, c in encoded_pairs) if encoded_pairs else 0
    
    # Cap max_seq_len to a reasonable value for memory stability
    max_seq_len = min(max_seq_len, 2048)

    # Use eos_token_id as padding if pad_token_id is not set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    for p_ids, o_ids, combined in encoded_pairs:
        # Truncate to max_seq_len if necessary
        if len(combined) > max_seq_len:
            combined = combined[:max_seq_len]
            
        # Padding to max_seq_len
        if len(combined) < max_seq_len:
            combined = combined + [pad_token_id] * (max_seq_len - len(combined))

        # input_ids: all tokens except the last one (length max_seq_len - 1)
        input_ids = combined[:-1]
        # labels: all tokens except the first one (length max_seq_len - 1)
        labels = combined[1:]

        # response_mask: 1 for tokens in labels that come from o_ids
        mask = [False] * len(labels)
        
        # Start of output in labels is at index len(p_ids) - 1
        start_idx = len(p_ids) - 1
        actual_o_ids = o_ids
        if len(o_ids) > 0 and o_ids[0] == tokenizer.bos_token_id:
            actual_o_ids = o_ids[1:]
            start_idx += 1
            
        end_idx = start_idx + len(actual_o_ids)
        
        for i in range(start_idx, end_idx):
            if i < len(labels):
                mask[i] = True

        # Sanity check: if mask is still empty, something is wrong with tokenization
        if not any(mask) and len(output_strs[0]) > 0:
            # Fallback: mask the last part of the sequence
            for i in range(max(0, len(labels)-len(o_ids)-1), len(labels)):
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
    """Compute the SFT loss and backprop its gradients for a microbatch.
    The loss is averaged over response tokens.
    """
    # Negative log-likelihood over response tokens
    per_token_loss = -policy_log_probs
    
    # Use our masked_mean helper to average over the response mask
    loss = masked_mean(per_token_loss, response_mask)
    
    # Scale by grad accumulation steps
    loss = loss / gradient_accumulation_steps

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
    length_normalization: str = "masked_mean",
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

    # Aggregate over response tokens
    if length_normalization == "masked_mean":
        loss = masked_mean(per_token_loss, response_mask)
    elif length_normalization == "masked_normalize":
        loss = masked_normalize(per_token_loss, response_mask, normalize_constant=1.0)
    else:
        raise ValueError(f"Unknown length_normalization: {length_normalization}")

    # Scale by gradient accumulation
    loss = loss / gradient_accumulation_steps

    loss.backward()

    return loss, metadata


def grpo_train_loop(
    policy: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    reward_fn: callable,
    prompts: list[str],
    ground_truths: list[str],
    n_grpo_steps: int,
    learning_rate: float,
    group_size: int,
    rollout_batch_size: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    loss_type: str,
    cliprange: float = 0.2,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    length_normalization: str = "masked_mean",
    device: str = "cuda",
    vllm_instance = None, # vLLM instance for fast sampling
    vllm_sync_fn = None, # Function to sync policy weights to vLLM
) -> list[dict]:
    """Execute the full GRPO training loop."""
    from torch.optim import AdamW
    from tqdm import tqdm
    
    # Strictly follow Page 24: betas=(0.9, 0.95), weight_decay=0.0
    optimizer = AdamW(policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    stats_history = []
    
    # Validation helper
    def validate():
        # This would call evaluate.py logic
        pass

    for step in tqdm(range(n_grpo_steps), desc="GRPO Steps"):
        # Sync vLLM weights if using vLLM
        if vllm_instance and vllm_sync_fn:
            vllm_sync_fn(policy, vllm_instance)

        # 1. Sample a batch of prompts
        indices = torch.randint(0, len(prompts), (rollout_batch_size // group_size,))
        batch_prompts = [prompts[i] for i in indices]
        batch_gts = [ground_truths[i] for i in indices]
        
        # Repeat prompts for group sampling
        repeated_prompts = []
        for p in batch_prompts:
            repeated_prompts.extend([p] * group_size)
        repeated_gts = []
        for g in batch_gts:
            repeated_gts.extend([g] * group_size)

        # 2. Rollout (Generate responses)
        # In a real setup, we use vLLM here. For local CPU, we use policy.generate.
        policy.eval()
        responses = []
        with torch.no_grad():
            if vllm_instance:
                # vLLM path (high performance)
                from vllm import SamplingParams
                # Strictly follow Page 24: max_tokens=1024, temp=0.7
                sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
                outputs = vllm_instance.generate(repeated_prompts, sampling_params)
                responses = [output.outputs[0].text for output in outputs]
            else:
                # HF path (fallback)
                inputs = tokenizer(repeated_prompts, return_tensors="pt", padding=True).to(device)
                # Strictly follow Page 24: max_new_tokens=1024, temp=0.7
                gen_outputs = policy.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
                # Decode only the generated part
                responses = [tokenizer.decode(g[inputs.input_ids.shape[1]:], skip_special_tokens=True) for g in gen_outputs]

        # 3. Compute rewards and advantages
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn, responses, repeated_gts, group_size, advantage_eps, normalize_by_std
        )
        advantages = advantages.to(device)
        
        # 4. Get old log probs (for clipping)
        # Tokenize responses for training
        tokenized = tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)
        
        with torch.no_grad():
            old_output = get_response_log_probs(policy, input_ids, labels)
            old_log_probs = old_output["log_probs"]

        # 5. Policy Update (One epoch per rollout batch)
        # Clear cache to free up memory from rollouts/vLLM
        if device == "cuda":
            torch.cuda.empty_cache()
            
        policy.train()
        # Shuffle rollout batch for microbatches
        n_rollouts = len(responses)
        indices = torch.randperm(n_rollouts)
        
        step_loss = 0
        for i in range(0, n_rollouts, train_batch_size):
            mb_idx = indices[i : i + train_batch_size]
            
            # Forward pass
            mb_output = get_response_log_probs(policy, input_ids[mb_idx], labels[mb_idx])
            mb_log_probs = mb_output["log_probs"]
            
            # Train step
            loss, mb_metadata = grpo_microbatch_train_step(
                policy_log_probs=mb_log_probs,
                response_mask=response_mask[mb_idx],
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=raw_rewards[mb_idx].to(device) if raw_rewards is not None else None,
                advantages=advantages[mb_idx],
                old_log_probs=old_log_probs[mb_idx],
                cliprange=cliprange,
                length_normalization=length_normalization
            )
            step_loss += loss.item()
            
        optimizer.step()
        optimizer.zero_grad()
        
        # Log stats
        stats = {
            "step": step,
            "loss": step_loss,
            "reward_mean": reward_metadata["reward_mean"],
        }
        stats_history.append(stats)
        
    return stats_history


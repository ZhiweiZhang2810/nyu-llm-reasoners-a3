import os
import re
import argparse
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 核心算法逻辑 (严格对应 Solutions.py)
# ==========================================

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer, max_seq_len=2048):
    batch_input_ids, batch_labels, batch_response_mask = [], [], []
    encoded_pairs = []
    for prompt, output in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(prompt, add_special_tokens=True)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        if len(o_ids) > 0 and o_ids[0] == tokenizer.bos_token_id: o_ids = o_ids[1:]
        combined = p_ids + o_ids + [tokenizer.eos_token_id]
        encoded_pairs.append((p_ids, o_ids, combined))
    
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for p_ids, o_ids, combined in encoded_pairs:
        combined = combined[:max_seq_len] + [pad_id] * max(0, max_seq_len - len(combined))
        input_ids, labels = combined[:-1], combined[1:]
        mask = [False] * len(labels)
        start_idx = len(p_ids) - 1
        # 顺从快照测试要求，不计算最后 padding 和多余的 eos
        end_idx = min(len(p_ids) - 1 + len(o_ids), len(labels))
        for i in range(start_idx, end_idx): mask[i] = True
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        batch_labels.append(torch.tensor(labels, dtype=torch.long))
        batch_response_mask.append(torch.tensor(mask, dtype=torch.bool))
    return {"input_ids": torch.stack(batch_input_ids), "labels": torch.stack(batch_labels), "response_mask": torch.stack(batch_response_mask)}

def masked_mean(tensor, mask):
    mask_f = mask.float()
    summed = torch.sum(tensor * mask_f)
    counts = torch.sum(mask_f)
    return torch.where(counts > 0, summed / counts, torch.tensor(0.0, device=tensor.device))

def masked_normalize(tensor, mask, normalize_constant=1.0):
    return torch.sum(tensor * mask.float()) / normalize_constant

def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    res = {"log_probs": gathered}
    if return_token_entropy:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        res["token_entropy"] = -torch.sum(probs * log_probs, dim=-1)
    return res

def grpo_microbatch_train_step(policy_log_probs, response_mask, grad_acc, loss_type, advantages, old_log_probs=None, cliprange=0.2, length_norm="masked_mean"):
    if loss_type == "no_baseline":
        per_token_loss = -advantages.unsqueeze(-1) * policy_log_probs
    elif loss_type == "reinforce_with_baseline":
        per_token_loss = -advantages.unsqueeze(-1) * policy_log_probs
    elif loss_type == "grpo_clip":
        ratio = torch.exp(policy_log_probs - old_log_probs)
        adv = advantages.unsqueeze(-1)
        loss = -torch.min(ratio * adv, torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * adv)
        per_token_loss = loss
    
    # 长度归一化消融 [cite: 836-840]
    if length_norm == "masked_mean":
        loss = masked_mean(per_token_loss, response_mask)
    else:
        # 当作纯和处理，即 constant=1.0
        loss = masked_normalize(per_token_loss, response_mask, normalize_constant=1.0)
        
    loss = loss / grad_acc
    loss.backward()
    return loss.detach()

def compute_group_normalized_rewards(reward_fn, responses, gts, group_size, std_norm=True):
    raw = torch.tensor([reward_fn(r, g)["reward"] for r, g in zip(responses, gts)], dtype=torch.float32)
    rewards_grouped = raw.view(-1, group_size)
    advantages = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
    if std_norm: 
        advantages = advantages / (rewards_grouped.std(dim=1, keepdim=True) + 1e-6)
    return advantages.view(-1), raw

# ==========================================
# 2. 实验循环逻辑 (对齐 PDF 默认超参)
# ==========================================

def countdown_reward_fn(response, gt):
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not match: return {"reward": 0.0}
    try:
        res_val = match.group(1).strip()
        return {"reward": 1.0 if res_val == str(gt).strip() else 0.0}
    except: return {"reward": 0.0}

def run_grpo_loop(args, model, tokenizer, prompts, gts, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.95))
    output_dir = f"outputs/{args.exp}_{datetime.now().strftime('%H%M')}_LR_{config['lr']}"
    os.makedirs(output_dir, exist_ok=True)
    
    # PDF 规定的参数推导 [cite: 704-742]
    # On-policy 下 train_batch_size == rollout_batch_size
    train_batch_size = args.rollout_batch
    grad_acc = 4 # 为 1.5B 模型预设合理的显存切分
    micro_batch = train_batch_size // grad_acc
    
    stats_history = []
    for step in tqdm(range(args.steps), desc=f"GRPO Steps (LR={config['lr']})"):
        # 1. 采样
        indices = torch.randint(0, len(prompts), (args.rollout_batch // args.group,))
        b_prompts = [prompts[i] for i in indices for _ in range(args.group)]
        b_gts = [gts[i] for i in indices for _ in range(args.group)]
        
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(b_prompts, return_tensors="pt", padding=True).to("cuda")
            # 严格对齐温度 0.7 和 max_tokens 1024 [cite: 710, 712]
            gen = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
            responses = [tokenizer.decode(g[inputs.input_ids.shape[1]:], skip_special_tokens=True) for g in gen]
        
        # 2. 计算 Advantages
        adv, raw_rew = compute_group_normalized_rewards(
            countdown_reward_fn, responses, b_gts, args.group, std_norm=config.get('std_norm', True)
        )
        adv = adv.to("cuda")
        
        # 3. 处理 Tokens 并获取 old log_probs
        tokenized = tokenize_prompt_and_output(b_prompts, responses, tokenizer)
        input_ids, labels, mask = tokenized["input_ids"].cuda(), tokenized["labels"].cuda(), tokenized["response_mask"].cuda()
        
        with torch.no_grad():
            old_log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"].detach()
            
        # 4. 梯度更新 (微批次防 OOM)
        model.train()
        epoch_loss = 0.0
        
        for i in range(0, train_batch_size, micro_batch):
            idx = slice(i, i + micro_batch)
            loss_val = grpo_microbatch_train_step(
                get_response_log_probs(model, input_ids[idx], labels[idx])["log_probs"], 
                mask[idx], grad_acc, config.get('loss_type', 'reinforce_with_baseline'), 
                adv[idx], old_log_probs[idx], cliprange=0.2, length_norm=config.get('length_norm', 'masked_mean')
            )
            epoch_loss += loss_val.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录指标
        stats_history.append({
            "step": step, 
            "reward_mean": raw_rew.mean().item(), 
            "loss": epoch_loss
        })
        
        if step % 5 == 0: 
            pd.DataFrame(stats_history).to_csv(f"{output_dir}/metrics.csv", index=False)

    print(f"✅ Experiment finished. Metrics saved to {output_dir}/metrics.csv")

# ==========================================
# 3. 任务分配器
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["lr_sweep", "baseline_ablation", "norm_ablation", "std_ablation"])
    parser.add_argument("--data_dir", type=str, default="./data-distrib")
    # PDF 默认核心参数 [cite: 704-712]
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--rollout_batch", type=int, default=16)
    args = parser.parse_args()

    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    df = pd.read_parquet(os.path.join(args.data_dir, "countdown/train_10k.parquet"))
    prompts, gts = df["prompt"].tolist(), df["answer"].tolist()

    # 根据作业要求映射实验超参数组合
    configs = {
        # Exp 1: LR Sweep [cite: 767-769]
        "lr_sweep": [
            {'lr': 5e-6}, 
            {'lr': 1e-5}, 
            {'lr': 2e-5}
        ],
        # Exp 2: Effect of baselines [cite: 774-780]
        "baseline_ablation": [
            {'lr': 1e-5, 'loss_type': 'no_baseline'}, 
            {'lr': 1e-5, 'loss_type': 'reinforce_with_baseline'}
        ],
        # Exp 3: Length normalization [cite: 836-840]
        "norm_ablation": [
            {'lr': 1e-5, 'length_norm': 'masked_mean'}, 
            {'lr': 1e-5, 'length_norm': 'masked_normalize'}
        ],
        # Exp 4: Std normalization [cite: 842-846]
        "std_ablation": [
            {'lr': 1e-5, 'std_norm': True}, 
            {'lr': 1e-5, 'std_norm': False}
        ]
    }
    
    for cfg in configs.get(args.exp, []):
        run_grpo_loop(args, model, tokenizer, prompts, gts, cfg)

if __name__ == "__main__":
    main()
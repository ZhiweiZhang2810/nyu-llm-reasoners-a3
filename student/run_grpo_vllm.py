import os
import re
import torch
import wandb
import ast
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from unittest.mock import patch

from solutions import (
    tokenize_prompt_and_output, 
    grpo_microbatch_train_step, 
    get_response_log_probs,
    compute_group_normalized_rewards
)

# ==========================================
# 1. vLLM Starter Code (From HW)
# ==========================================
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id, device=device, dtype=torch.bfloat16,
            enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# ==========================================
# 2. 配置与实验调度
# ==========================================
class Config:
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct" # 注意：GRPO 用 Instruct 模型
    data_dir = "./data-distrib"
    
    # 硬件分配
    train_device = "cuda:0"
    vllm_device = "cuda:1"
    
    # GRPO 基础参数
    max_steps = 200
    rollout_batch_size = 16
    group_size = 8
    grad_acc = 16  # 防 OOM 神器
    eval_interval = 10

# 简单的 Countdown 奖励函数
import ast

def countdown_reward_fn(response_text, gt):
    """
    专门为 Countdown 游戏编写的奖励判定。
    要求模型生成的等式：1. 计算结果等于 target; 2. 使用的数字与给定的数字集完全一致。
    """
    match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    if not match: 
        return {"reward": 0.0}
    
    equation = match.group(1).strip()
    target = gt['target']
    # 将 numpy array 转换为排序后的列表，以便比对
    allowed_nums = sorted([int(x) for x in gt['numbers']])
    
    # 1. 验证模型使用的数字是否和题目要求的一模一样
    nums_in_eq = [int(n) for n in re.findall(r'\d+', equation)]
    if sorted(nums_in_eq) != allowed_nums:
        return {"reward": 0.0}
        
    # 2. 简单的防注入安全校验，只允许数字和基础运算符
    if not re.match(r'^[\d\+\-\*\/\(\)\s]+$', equation):
        return {"reward": 0.0}
        
    # 3. 计算算式结果是否等于 Target
    try:
        result = eval(equation)
        if abs(result - target) < 1e-5:
            return {"reward": 1.0}
    except:
        pass # 处理被 0 除等非法算式
        
    return {"reward": 0.0}

def evaluate_countdown_vllm(policy_model, llm, tokenizer, eval_df, step, limit=200):
    load_policy_into_vllm_instance(policy_model, llm)
    
    prompts = []
    gts = []
    for i in range(min(len(eval_df), limit)):
        row = eval_df.iloc[i]
        
        # 1. 精准提取 prompt 并应用 Qwen 的 Chat Template
        msgs = row['prompt']
        if isinstance(msgs, str): 
            msgs = ast.literal_eval(msgs) # 防御性转换
        prompt_str = tokenizer.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_str)
        
        # 2. 精准提取 ground_truth
        gts.append(row['reward_model']['ground_truth'])
        
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    correct = sum(1 for out, gt in zip(outputs, gts) if countdown_reward_fn(out.outputs[0].text, gt)["reward"] == 1.0)
    acc = correct / len(prompts)
    
    wandb.log({"eval/val_reward": acc, "eval_step": step})
    print(f"\n[Step {step}] Val Accuracy (Reward): {acc:.2%}")
    return acc

# ==========================================
# 3. GRPO 训练循环
# ==========================================
def run_grpo_experiment(exp_name, train_df, val_df, tokenizer, llm, 
                        lr=1e-5, loss_type="reinforce_with_baseline", 
                        length_norm="masked_mean", use_std_norm=True):
    print(f"\n🚀 启动实验: {exp_name} | LR: {lr} | Loss: {loss_type} | Norm: {length_norm} | StdNorm: {use_std_norm}")
    
    wandb.init(
        project="llm-reasoners-hw3-grpo", name=exp_name,
        config={"lr": lr, "loss_type": loss_type, "length_norm": length_norm, "use_std": use_std_norm}
    )
    wandb.define_metric("train_step"); wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    model = AutoModelForCausalLM.from_pretrained(Config.model_id, torch_dtype=torch.bfloat16, device_map={"": Config.train_device})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Rollout 采样参数
    rollout_params = SamplingParams(temperature=0.7, max_tokens=1024, n=Config.group_size)

    for step in tqdm(range(Config.max_steps), desc=exp_name):
        # 1. 采样 prompt 与数据清洗 (新版本，完美适配 Countdown)
        num_prompts = Config.rollout_batch_size // Config.group_size
        batch_df = train_df.sample(n=num_prompts, replace=True)
        
        prompts = []
        gts = []
        for _, row in batch_df.iterrows():
            msgs = row['prompt']
            if isinstance(msgs, str): 
                msgs = ast.literal_eval(msgs)
            # 把原始的字典列表转换成 Qwen 认识的纯文本 prompt
            prompt_str = tokenizer.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_str)
            # 从嵌套字典里精准提取 target 和 numbers
            gts.append(row['reward_model']['ground_truth'])
        
        # 2. vLLM 生成 Responses
        load_policy_into_vllm_instance(model, llm)
        outputs = llm.generate(prompts, rollout_params, use_tqdm=False)
        
        b_prompts, b_responses, b_gts = [], [], []
        for i, out in enumerate(outputs):
            for gen in out.outputs:
                b_prompts.append(prompts[i])
                b_responses.append(gen.text)
                b_gts.append(gts[i])
                
        # 3. 计算 Reward 和 Advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            countdown_reward_fn, 
            b_responses, 
            b_gts, 
            Config.group_size, 
            advantage_eps=1e-6,              # 补充你源码里需要的 eps
            normalize_by_std=use_std_norm    # 使用你源码里真正的参数名
        )
        advantages = advantages.to(Config.train_device)
        
        # 4. Tokenize
        tokenized = tokenize_prompt_and_output(b_prompts, b_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(Config.train_device)
        labels = tokenized["labels"].to(Config.train_device)
        mask = tokenized["response_mask"].to(Config.train_device)
        
        with torch.no_grad():
            old_log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"].detach()
            
        # 5. 微批次训练
        model.train()
        optimizer.zero_grad()
        
        mb_size = max(1, len(input_ids) // Config.grad_acc)
        total_loss = 0
        
        for i in range(0, len(input_ids), mb_size):
            mb_idx = slice(i, i + mb_size)
            log_probs = get_response_log_probs(model, input_ids[mb_idx], labels[mb_idx])["log_probs"]
            
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=mask[mb_idx],
                gradient_accumulation_steps=Config.grad_acc,
                loss_type=loss_type,
                raw_rewards=raw_rewards[mb_idx].to(Config.train_device),
                advantages=advantages[mb_idx],
                old_log_probs=old_log_probs[mb_idx],
                cliprange=0.2,
                length_normalization=length_norm
            )
            total_loss += loss.item()
            
        # 6. 计算 Gradient Norm (核心稳定性指标)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 记录日志
        wandb.log({
            "train/loss": total_loss, 
            "train/reward_mean": raw_rewards.mean().item(),
            "train/grad_norm": grad_norm.item(),
            "train_step": step
        })
        
        if step % Config.eval_interval == 0 or step == Config.max_steps - 1:
            evaluate_countdown_vllm(model, llm, tokenizer, val_df, step, limit=100)

    wandb.finish()
    del model
    torch.cuda.empty_cache()

# ==========================================
# 4. 主函数 (分发任务)
# ==========================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    print("加载 Parquet 数据集...")
    # 直接精准读取截图中的 parquet 文件
    train_df = pd.read_parquet(os.path.join(Config.data_dir, "countdown/train_10k.parquet"))
    val_df = pd.read_parquet(os.path.join(Config.data_dir, "countdown/dev.parquet"))
    
    # 打印一下数据量确保没读错
    print(f"✅ 成功加载: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条")
    
    print(f"初始化 vLLM 在 {Config.vllm_device} ...")
    llm = init_vllm(Config.model_id, device=Config.vllm_device, seed=42)
    
    # ========================================================
    # 实验执行区：你可以注释掉不需要的，每次只跑当前关注的实验
    # ========================================================
    
    # 实验 1: LR Sweep (跑完后找出达到 30% Acc 的最好 LR 填入后文)
    
    # run_grpo_experiment("LR_5e-6", train_df, val_df, tokenizer, llm, lr=5e-6)
    # run_grpo_experiment("LR_1e-5", train_df, val_df, tokenizer, llm, lr=1e-5)
    # run_grpo_experiment("LR_2e-5", train_df, val_df, tokenizer, llm, lr=2e-5)

    BEST_LR = 2e-5 # <--- 根据前 3 个的结果修改这个值
    
    # 实验 2: Baseline 消融 (使用 BEST_LR)
    # run_grpo_experiment("No_Baseline", train_df, val_df, tokenizer, llm, lr=BEST_LR, loss_type="no_baseline")
    # run_grpo_experiment("With_Baseline", train_df, val_df, tokenizer, llm, lr=BEST_LR, loss_type="reinforce_with_baseline")
    
    BEST_LOSS = "reinforce_with_baseline" # <--- 根据实验 2 修改
    
    # 实验 3: Length Normalization (密切关注 wandb 上的 grad_norm 曲线)
    run_grpo_experiment("Norm_Masked_Mean", train_df, val_df, tokenizer, llm, lr=BEST_LR, loss_type=BEST_LOSS, length_norm="masked_mean")
    run_grpo_experiment("Norm_Masked_Normalize", train_df, val_df, tokenizer, llm, lr=BEST_LR, loss_type=BEST_LOSS, length_norm="masked_normalize")

    BEST_NORM = "masked_normalize" # <--- 根据实验 3 修改
    
    # 实验 4: Standard Deviation Normalization
    # run_grpo_experiment("StdNorm_True", train_df, val_df, tokenizer, llm, lr=BEST_LR, loss_type=BEST_LOSS, length_norm=BEST_NORM, use_std_norm=True)
    # run_grpo_experiment("StdNorm_False", train_df, val_df, tokenizer, llm, lr=BEST_LR, loss_type=BEST_LOSS, length_norm=BEST_NORM, use_std_norm=False)

if __name__ == "__main__":
    main()
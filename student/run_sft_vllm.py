import os
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from unittest.mock import patch

from solutions import (
    tokenize_prompt_and_output, 
    sft_microbatch_train_step, 
    get_response_log_probs
)

# ==========================================
# 1. vLLM Starter Code (From HW PDF)
# ==========================================
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Start the inference process using vLLM on a separate GPU."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """Load updated policy weights into vLLM engine."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# ==========================================
# 2. 配置参数
# ==========================================
class Config:
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    dataset_sizes = [128, 256, 512, 1024, "full"]
    lr = 1e-5
    batch_size = 16 
    grad_acc = 4
    max_steps = 200 # 如果 loss 没有下降 40%，可以适当增加步数
    eval_interval = 20
    data_dir = "./data-distrib"
    
    # 严格分配 GPU：GPU 0 用于训练 (Policy)，GPU 1 用于评估 (vLLM)
    train_device = "cuda:0"
    vllm_device = "cuda:1"

# ==========================================
# 3. 评估函数 (使用 vLLM 提速)
# ==========================================
def evaluate_math_accuracy_vllm(policy_model, llm, tokenizer, eval_df, limit=200):
    """将最新权重同步到 vLLM 并进行快速评估"""
    load_policy_into_vllm_instance(policy_model, llm)
    
    prompt_tpl = "Solve the following math problem efficiently and clearly. Think carefully and step by step about your response and reason before providing a final response. Conclude your response with:\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct."
    
    prompts = []
    gts = []
    
    for i in range(min(len(eval_df), limit)):
        row = eval_df.iloc[i]
        # 兼容两种可能的数据结构
        question = row['prompt'] if 'prompt' in row else row['messages'][1]['content']
        gt = str(row['answer'] if 'answer' in row else row['ground_truth']).strip()
        
        full_prompt = f"{prompt_tpl}\n\nQuestion: {question}"
        prompts.append(full_prompt)
        gts.append(gt)
    
    # vLLM 贪婪解码参数
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    correct = 0
    for out, gt in zip(outputs, gts):
        response_text = out.outputs[0].text
        if gt in response_text:
            correct += 1
            
    return correct / len(prompts)

# ==========================================
# 4. SFT 训练循环
# ==========================================
def run_sft_experiment(size, train_df, val_df, test_df, tokenizer, llm):
    print(f"\n🚀 [Size: {size}] 正在启动 SFT 训练...")
    
    # 初始化 WandB 记录
    wandb.init(
        project="llm-reasoners-hw3-sft",
        name=f"sft_size_{size}",
        config={"size": size, "lr": Config.lr, "batch_size": Config.batch_size}
    )
    wandb.define_metric("train_step") 
    wandb.define_metric("eval_step") 
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    # 数据切分
    subset = train_df if size == "full" else train_df.sample(n=size, random_state=42)
    
    # 加载模型到 GPU 0
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_id, 
        torch_dtype=torch.bfloat16,
        device_map={"": Config.train_device} # 限制只在 GPU 0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    
    initial_loss = None
    
    for step in tqdm(range(Config.max_steps), desc=f"Training (Size {size})"):
        batch = subset.sample(n=Config.batch_size, replace=True)
        
        prompt_list = []
        answer_list = []
        
        for _, row in batch.iterrows():
            question = row['prompt'] if 'prompt' in row else row['messages'][1]['content']
            ans = row['answer'] if 'answer' in row else row['messages'][2]['content']
            full_prompt = f"Solve the following math problem efficiently and clearly. Think carefully and step by step about your response and reason before providing a final response. Conclude your response with:\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nQuestion: {question}"
            prompt_list.append(full_prompt)
            answer_list.append(ans)
        
        tokenized = tokenize_prompt_and_output(prompt_list, answer_list, tokenizer)
        input_ids = tokenized["input_ids"].to(Config.train_device)
        labels = tokenized["labels"].to(Config.train_device)
        mask = tokenized["response_mask"].to(Config.train_device)
        
        model.train()
        optimizer.zero_grad()
        
        mb_size = Config.batch_size // Config.grad_acc
        total_loss = 0
        for i in range(0, Config.batch_size, mb_size):
            mb_idx = slice(i, i + mb_size)
            output = get_response_log_probs(model, input_ids[mb_idx], labels[mb_idx])
            loss, _ = sft_microbatch_train_step(output["log_probs"], mask[mb_idx], Config.grad_acc, normalize_constant=1.0)
            total_loss += loss.item()
            
        if initial_loss is None: initial_loss = total_loss
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 记录 Train Loss
        wandb.log({"train/loss": total_loss, "train_step": step})
        
        # 定期评估
        if step % Config.eval_interval == 0 or step == Config.max_steps - 1:
            val_acc = evaluate_math_accuracy_vllm(model, llm, tokenizer, val_df, limit=200)
            print(f"\n[Step {step}] Loss: {total_loss:.4f} | Val Acc: {val_acc:.2%}")
            wandb.log({"eval/val_accuracy": val_acc, "eval_step": step})
            
            if initial_loss and total_loss < initial_loss * 0.6:
                print(f" ✨ Loss 已下降超过 40%！")

    # 训练结束后，在测试集上跑最终评估 (作业要求 Deliverable 2)
    print(f"\n🏃‍♂️ [Size: {size}] 正在运行 Test Set 最终评估...")
    test_acc = evaluate_math_accuracy_vllm(model, llm, tokenizer, test_df, limit=len(test_df))
    print(f"🏆 Size {size} 最终 Test Accuracy: {test_acc:.2%}")
    wandb.log({"eval/test_accuracy": test_acc, "eval_step": Config.max_steps})
    
    wandb.finish()
    
    # 释放显存供下一个消融实验使用
    del model
    torch.cuda.empty_cache()

# ==========================================
# 5. 主程序
# ==========================================
def main():
    print("加载数据集...")
    # 替换为你实际的本地缓存路径
    train_ds = load_from_disk(os.path.join(Config.data_dir, "intellect_math/train"))
    val_ds = load_from_disk(os.path.join(Config.data_dir, "intellect_math/dev"))
    # 注意：假设你有一个 test 集。如果没有单独的 test 集，可以 fallback 到使用另一个验证集
    test_ds = load_from_disk(os.path.join(Config.data_dir, "intellect_math/test")) if os.path.exists(os.path.join(Config.data_dir, "intellect_math/test")) else val_ds
    
    train_df = train_ds.to_pandas()
    val_df = val_ds.to_pandas()
    test_df = test_ds.to_pandas()
    
    tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"初始化 vLLM 在 {Config.vllm_device} ...")
    llm = init_vllm(Config.model_id, device=Config.vllm_device, seed=42)
    
    for size in Config.dataset_sizes:
        run_sft_experiment(size, train_df, val_df, test_df, tokenizer, llm)
        
    print("\n🎉 所有 SFT 消融实验完成！请登录 WandB 导出 Validation Accuracy 曲线用于 Writeup。")

if __name__ == "__main__":
    main()
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "pandas",
#   "tqdm",
#   "matplotlib",
#   "seaborn",
#   "datasets"
# ]
# ///

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from solutions import (
    tokenize_prompt_and_output, 
    sft_microbatch_train_step, 
    get_response_log_probs
)

# ==========================================
# 1. 配置与超参数 (对齐作业 PDF)
# ==========================================
class Config:
    model_id = "Qwen/Qwen2.5-Math-1.5B" # SFT 使用 Base 模型
    dataset_sizes = [128, 256, 512, 1024, "full"] # 消融实验规模
    lr = 1e-5 # 建议初始学习率，需观察 loss 下降是否达 40%
    batch_size = 16 
    grad_acc = 4
    max_steps = 200 # 根据 loss 下降情况调整，作业没定死步数，看 loss 表现
    eval_interval = 20
    output_base = "./outputs/sft_ablation"

# ==========================================
# 2. 评估函数 (MATH 验证集)
# ==========================================
def evaluate_math_accuracy(model, tokenizer, eval_data, limit=50):
    """
    使用 Prime Intellect 提示词在验证集上计算准确率
    """
    model.eval()
    correct = 0
    # 提示词模板
    prompt_tpl = "Solve the following math problem efficiently and clearly. Think carefully and step by step about your response and reason before providing a final response. Conclude your response with:\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct."
    
    with torch.no_grad():
        for i in range(min(len(eval_data), limit)):
            question = eval_data.iloc[i]['prompt'] if 'prompt' in eval_data.columns else eval_data.iloc[i]['question']
            gt = str(eval_data.iloc[i]['answer']).strip()
            
            full_prompt = f"{prompt_tpl}\n\nQuestion: {question}"
            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            
            # 使用贪婪解码评估
            gen = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=0.0)
            response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 简化判定：检查 GT 是否包含在模型的输出中
            if gt in response: 
                correct += 1
                
    return correct / min(len(eval_data), limit)

# ==========================================
# 3. 训练核心循环
# ==========================================
def run_sft_experiment(size, train_df, val_df, tokenizer):
    print(f"\n🚀 启动 SFT 消融实验 | 数据量: {size}")
    
    # 子集切分
    subset = train_df if size == "full" else train_df.sample(n=size, random_state=42)
    
    # 每次不同数据量实验都必须重新加载纯净的 Base 模型
    model = AutoModelForCausalLM.from_pretrained(Config.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    
    results = []
    initial_loss = None
    
    for step in tqdm(range(Config.max_steps), desc=f"Size {size}"):
        batch = subset.sample(n=Config.batch_size, replace=True)
        
        # 兼容列名可能是 prompt 或 question 的情况
        prompt_col = 'prompt' if 'prompt' in batch.columns else 'question'
        
        tokenized = tokenize_prompt_and_output(batch[prompt_col].tolist(), batch['answer'].tolist(), tokenizer)
        
        input_ids = tokenized["input_ids"].cuda()
        labels = tokenized["labels"].cuda()
        mask = tokenized["response_mask"].cuda()
        
        model.train()
        optimizer.zero_grad()
        
        mb_size = Config.batch_size // Config.grad_acc
        total_loss = 0
        for i in range(0, Config.batch_size, mb_size):
            mb_idx = slice(i, i + mb_size)
            output = get_response_log_probs(model, input_ids[mb_idx], labels[mb_idx])
            loss, _ = sft_microbatch_train_step(output["log_probs"], mask[mb_idx], Config.grad_acc)
            total_loss += loss.item()
            
        if initial_loss is None: initial_loss = total_loss
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 定期评估与记录
        if step % Config.eval_interval == 0 or step == Config.max_steps - 1:
            acc = evaluate_math_accuracy(model, tokenizer, val_df)
            results.append({"step": step, "acc": acc, "loss": total_loss})
            print(f" - Step {step}: Loss = {total_loss:.4f}, Val Acc = {acc:.2%}")
            
            if initial_loss and total_loss < initial_loss * 0.6:
                print(f" ✨ 达成 Loss 下降 40% 目标 (Initial: {initial_loss:.2f}, Current: {total_loss:.2f})")

    return results

# ==========================================
# 4. 数据加载、绘图与主程序
# ==========================================
def main():
    data_dir = "./data-distrib" 
    tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 修改点：使用 Hugging Face datasets 加载 Arrow 目录，然后转换为 Pandas DataFrame
    print("Loading Hugging Face datasets from disk...")
    train_ds = load_from_disk(os.path.join(data_dir, "intellect_math/train"))
    val_ds = load_from_disk(os.path.join(data_dir, "intellect_math/dev"))
    
    train_df = train_ds.to_pandas()
    val_df = val_ds.to_pandas()
    
    all_experiments = {}
    
    for size in Config.dataset_sizes:
        exp_results = run_sft_experiment(size, train_df, val_df, tokenizer)
        all_experiments[size] = exp_results
        
    # --- 绘制所有数据量规模的对比曲线 ---
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 6))
    for size, logs in all_experiments.items():
        df_logs = pd.DataFrame(logs)
        plt.plot(df_logs['step'], df_logs['acc'], label=f"Size: {size}", marker='o')
        
    plt.title("SFT Scaling: Validation Accuracy vs. Dataset Size")
    plt.xlabel("Training Steps")
    plt.ylabel("MATH Val Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(Config.output_base, exist_ok=True)
    out_path = f"{Config.output_base}/sft_scaling_curves.png"
    plt.savefig(out_path, dpi=300)
    print(f"\n📊 消融实验完成，曲线图已保存至: {out_path}")

if __name__ == "__main__":
    main()
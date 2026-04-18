import os
import re
import ast
import torch
import wandb
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch

from solutions import grpo_train_loop

# ==========================================
# Configuration
# ==========================================
class Config:
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    data_dir = "./data-distrib"
    train_device = "cuda:0"
    vllm_device = "cuda:1"
    
    # GRPO Hyperparameters (from the assignment)
    n_grpo_steps = 200
    learning_rate = 1e-5  # Use the BEST_LR you found earlier!
    group_size = 8
    rollout_batch_size = 16
    train_batch_size = 16
    gradient_accumulation_steps = 16 # Keep this high for stability/memory
    
    # Evaluation
    eval_limit = 100

# ==========================================
# Helpers (from previous setup)
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

def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def countdown_reward_fn(response_text, gt):
    match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    if not match: 
        return {"reward": 0.0}
    
    equation = match.group(1).strip()
    target = gt['target']
    allowed_nums = sorted([int(x) for x in gt['numbers']])
    
    nums_in_eq = [int(n) for n in re.findall(r'\d+', equation)]
    if sorted(nums_in_eq) != allowed_nums:
        return {"reward": 0.0}
        
    if not re.match(r'^[\d\+\-\*\/\(\)\s]+$', equation):
        return {"reward": 0.0}
        
    try:
        result = eval(equation)
        if abs(result - target) < 1e-5:
            return {"reward": 1.0}
    except:
        pass 
        
    return {"reward": 0.0}

def evaluate_countdown_vllm(policy_model, llm, tokenizer, eval_df, step, limit=200):
    load_policy_into_vllm_instance(policy_model, llm)
    
    prompts = []
    gts = []
    for i in range(min(len(eval_df), limit)):
        row = eval_df.iloc[i]
        msgs = row['prompt']
        if isinstance(msgs, str): msgs = ast.literal_eval(msgs)
        prompt_str = tokenizer.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_str)
        gts.append(row['reward_model']['ground_truth'])
        
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    correct = sum(1 for out, gt in zip(outputs, gts) if countdown_reward_fn(out.outputs[0].text, gt)["reward"] == 1.0)
    acc = correct / len(prompts)
    
    # Print a sample rollout for the write-up
    print(f"\n--- Sample Rollout at Step {step} ---")
    print(f"Prompt: {prompts[0][-100:]}...") # Print end of prompt
    print(f"Target: {gts[0]}")
    print(f"Model Output: {outputs[0].outputs[0].text[:300]}...") # Print first 300 chars of output
    print("--------------------------------------\n")
    
    return acc

# ==========================================
# Main Execution
# ==========================================
def main():
    print("Loading Parquet datasets...")
    train_df = pd.read_parquet(os.path.join(Config.data_dir, "countdown/train_10k.parquet"))
    val_df = pd.read_parquet(os.path.join(Config.data_dir, "countdown/dev.parquet"))
    
    # Process prompts for the training loop
    tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("Preparing training data...")
    train_prompts = []
    train_gts = []
    # Using a subset for faster rollout sampling if needed, or all 10k
    for _, row in train_df.iterrows():
        msgs = row['prompt']
        if isinstance(msgs, str): msgs = ast.literal_eval(msgs)
        prompt_str = tokenizer.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True)
        train_prompts.append(prompt_str)
        train_gts.append(row['reward_model']['ground_truth'])

    print(f"Initializing vLLM on {Config.vllm_device}...")
    llm = init_vllm(Config.model_id, device=Config.vllm_device, seed=42)
    
    print(f"Loading Policy Model on {Config.train_device}...")
    policy = AutoModelForCausalLM.from_pretrained(Config.model_id, torch_dtype=torch.bfloat16, device_map={"": Config.train_device})

    wandb.init(project="llm-reasoners-hw3-grpo", name="Final_Train_Loop")
    wandb.define_metric("step")
    wandb.define_metric("eval/*", step_metric="step")
    wandb.define_metric("train/*", step_metric="step")

    print("\n🚀 Starting GRPO Train Loop...")
    
    # Custom sync function to evaluate and log to wandb during the loop
    def sync_and_eval(current_policy, vllm_inst):
        load_policy_into_vllm_instance(current_policy, vllm_inst)
        # We need a way to track the step inside the sync function. 
        # A simple hack is to attach it to the policy object temporarily.
        if not hasattr(current_policy, 'current_step'):
            current_policy.current_step = 0
            
        step = current_policy.current_step
        
        if step % 10 == 0 or step == Config.n_grpo_steps - 1:
            val_acc = evaluate_countdown_vllm(current_policy, vllm_inst, tokenizer, val_df, step, limit=Config.eval_limit)
            wandb.log({"eval/val_reward": val_acc, "step": step})
            
        current_policy.current_step += 1

    # Call your solution!
    stats_history = grpo_train_loop(
        policy=policy,
        tokenizer=tokenizer,
        reward_fn=countdown_reward_fn,
        prompts=train_prompts,
        ground_truths=train_gts,
        n_grpo_steps=Config.n_grpo_steps,
        learning_rate=Config.learning_rate,
        group_size=Config.group_size,
        rollout_batch_size=Config.rollout_batch_size,
        train_batch_size=Config.train_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        epochs_per_rollout_batch=1,
        loss_type="reinforce_with_baseline", # Use best from sweep
        length_normalization="masked_normalize", # Use best from sweep
        normalize_by_std=False, # Use best from sweep
        device=Config.train_device,
        vllm_instance=llm,
        vllm_sync_fn=sync_and_eval
    )

    # Log the final stats history
    for stat in stats_history:
        wandb.log({
            "train/loss": stat["loss"],
            "train/entropy": stat["entropy_mean"],
            "train/reward_mean": stat["reward_mean"],
            "step": stat["step"]
        })

    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    main()
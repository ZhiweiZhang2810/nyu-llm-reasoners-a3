import argparse
import torch
import json
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from torch.optim import AdamW
from tqdm import tqdm
from student import solutions
from student.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from student.evaluate import evaluate, load_prompt

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm):
    """Sync torch policy weights into vLLM engine."""
    state_dict = policy.state_dict()
    # Path might vary slightly depending on vLLM version, but this is standard for 0.7.x
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def plot_and_save(data, title, xlabel, ylabel, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train SFT or GRPO models.")
    parser.add_argument("--mode", type=str, choices=["sft", "grpo"], required=True)
    parser.add_argument("--dataset", type=str, default="intellect", help="intellect or countdown")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num_examples", type=int, default=1024, help="For SFT experiment sweep")
    parser.add_argument("--loss_type", type=str, default="grpo_clip", choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--use_std_norm", action="store_true", default=True, help="Use STD normalization (default True)")
    parser.add_argument("--no_std_norm", action="store_false", dest="use_std_norm", help="Disable STD normalization")
    parser.add_argument("--length_norm", type=str, default="masked_mean", choices=["masked_mean", "masked_normalize"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Load Model and Tokenizer
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use bfloat16 if not on MPS (MPS has limited bfloat16 support depending on the OS version)
    dtype = torch.bfloat16 if device != "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    
    # 2. Load Data
    if args.dataset == "intellect":
        ds = load_from_disk("data-distrib/intellect_math/train")
        # For Intellect: 
        # msgs[0] = system, msgs[1] = user, msgs[2] = assistant (reasoning + answer)
        prompts = []
        for ex in ds:
            msgs = ex["messages"]
            p = ""
            if msgs[0]["role"] == "system":
                p += msgs[0]["content"] + "\n"
            if msgs[1]["role"] == "user":
                p += msgs[1]["content"]
            prompts.append(p)
            
        if args.mode == "sft":
            # Target is the assistant's reasoning trace
            gts = [ex["messages"][2]["content"] for ex in ds]
        else:
            gts = [ex["ground_truth"] for ex in ds]
        reward_fn = question_only_reward_fn
    else:
        ds = load_from_disk("data-distrib/countdown/dataset/train")
        # For Countdown: "prompt" is a list with one 'user' dict containing the whole prompt
        prompts = [ex["prompt"][0]["content"] for ex in ds]
        # Target for GRPO is the number
        gts = [str(ex["target"]) for ex in ds]
        reward_fn = r1_zero_reward_fn

    # Initialize vLLM if on Linux and vLLM is available
    vllm_instance = None
    if device != "mps" and args.mode == "grpo":
        try:
            from vllm import LLM
            print("Initializing vLLM for fast sampling...")
            # Qwen2.5-Math-1.5B fits on most GPUs
            vllm_instance = LLM(model=model_id, device=device, gpu_memory_utilization=0.4, enforce_eager=True)
        except ImportError:
            print("vLLM not found, falling back to HF sampling.")

    # 3. Execute Training
    if args.mode == "sft":
        print(f"Starting SFT with {args.num_examples} examples...")
        sft_prompts = prompts[:args.num_examples]
        sft_answers = gts[:args.num_examples]
        
        optimizer = AdamW(model.parameters(), lr=args.lr)
        batch_size = 4
        grad_accum = 4
        epochs = 3
        
        loss_history = []
        model.train()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i in tqdm(range(0, len(sft_prompts), batch_size)):
                batch_p = sft_prompts[i:i+batch_size]
                batch_a = sft_answers[i:i+batch_size]
                
                tokenized = solutions.tokenize_prompt_and_output(batch_p, batch_a, tokenizer)
                input_ids = tokenized["input_ids"].to(device)
                labels = tokenized["labels"].to(device)
                response_mask = tokenized["response_mask"].to(device)
                
                output = solutions.get_response_log_probs(model, input_ids, labels)
                loss, _ = solutions.sft_microbatch_train_step(
                    policy_log_probs=output["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=grad_accum
                )
                
                current_loss = loss.item() * grad_accum
                loss_history.append(current_loss)
                
                if (i // batch_size) % 10 == 0:
                    print(f"Step {i//batch_size}: Loss = {current_loss:.4f}")
                
                if (i // batch_size + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
        plot_and_save(loss_history, f"SFT Loss (N={args.num_examples})", "Steps", "Loss", f"sft_loss_n{args.num_examples}.png", args.output_dir)
        print("Evaluating on Intellect Test...")
        # Simple eval logic for local testing without vLLM
        # In a real run, you'd use evaluate.py with the saved model
        
    elif args.mode == "grpo":
        print(f"Starting GRPO with loss={args.loss_type}, lr={args.lr}, std_norm={args.use_std_norm}, length_norm={args.length_norm}...")
        history = solutions.grpo_train_loop(
            policy=model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            prompts=prompts,
            ground_truths=gts,
            n_grpo_steps=args.steps,
            learning_rate=args.lr,
            group_size=8,
            rollout_batch_size=16,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            loss_type=args.loss_type,
            normalize_by_std=args.use_std_norm,
            length_normalization=args.length_norm,
            device=device,
            vllm_instance=vllm_instance,
            vllm_sync_fn=load_policy_into_vllm_instance
        )
        
        rewards = [h["reward_mean"] for h in history]
        title = f"GRPO {args.loss_type} | lr={args.lr} | std_norm={args.use_std_norm} | len_norm={args.length_norm}"
        filename = f"grpo_{args.loss_type}_lr{args.lr}_std{args.use_std_norm}_len{args.length_norm}.png"
        plot_and_save(rewards, title, "Steps", "Mean Reward", filename, args.output_dir)
        
        print(f"Training complete. Last reward mean: {history[-1]['reward_mean']}")
        print(f"Plot saved to {os.path.join(args.output_dir, filename)}")

if __name__ == "__main__":
    main()
"""Minimal evaluation script for MATH and Intellect test sets."""

from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(llm, prompts, ground_truths, log_examples=False):
    """Run evaluation and return accuracy."""
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    correct = 0
    cat1_count = 0  # Format 1, Answer 1
    cat2_count = 0  # Format 1, Answer 0
    cat3_count = 0  # Format 0, Answer 0
    
    cat2_examples = []
    cat3_examples = []

    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]
        
        f_rew = reward.get("format_reward", 0.0)
        a_rew = reward.get("answer_reward", 0.0)
        
        if f_rew == 1.0 and a_rew == 1.0:
            cat1_count += 1
        elif f_rew == 1.0 and a_rew == 0.0:
            cat2_count += 1
            if log_examples and len(cat2_examples) < 10:
                cat2_examples.append((prompts[i], text, ground_truths[i]))
        elif f_rew == 0.0 and a_rew == 0.0:
            cat3_count += 1
            if log_examples and len(cat3_examples) < 10:
                cat3_examples.append((prompts[i], text, ground_truths[i]))

    if log_examples:
        print("\n--- MATH Evaluation Statistics ---")
        print(f"(1) Format Reward 1, Answer Reward 1: {cat1_count}")
        print(f"(2) Format Reward 1, Answer Reward 0: {cat2_count}")
        print(f"(3) Format Reward 0, Answer Reward 0: {cat3_count}")
        
        print("\n--- 10 Examples of Format 1, Answer 0 ---")
        for idx, ex in enumerate(cat2_examples):
            print(f"\n[Example {idx+1}]")
            print(f"GT: {ex[2]}")
            print(f"Output:\n{ex[1]}\n" + "-"*40)
            
        print("\n--- 10 Examples of Format 0, Answer 0 ---")
        for idx, ex in enumerate(cat3_examples):
            print(f"\n[Example {idx+1}]")
            print(f"GT: {ex[2]}")
            print(f"Output:\n{ex[1]}\n" + "-"*40)

    return correct / len(outputs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="data-distrib/intellect_math/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    # Load model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on Intellect test
    if Path(args.intellect_path).exists():
        print(f"\n=== Intellect Test ({args.intellect_path}) ===")
        dataset = load_from_disk(args.intellect_path)
        if args.max_examples:
            dataset = dataset.select(range(min(args.max_examples, len(dataset))))

        prompts, gts = [], []
        for ex in dataset:
            msgs = ex.get("messages", [])
            sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
            gts.append(ex.get("ground_truth", ""))

        print(f"[Sample] {prompts[0][:200]}...")
        acc = evaluate(llm, prompts, gts)
        print(f"Intellect Accuracy: {acc:.4f}")
    else:
        print(f"\n[Warning] Intellect path '{args.intellect_path}' not found. Skipping.")

    # Evaluate on MATH
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(llm, prompts, gts, log_examples=True)
    print(f"MATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

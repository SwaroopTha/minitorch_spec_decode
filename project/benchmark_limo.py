import argparse
import csv
import json
import time
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from sampling import autoregressive_generate, spec_gen
from utils.logits_proc import GreedyProcessor

# Import LIMO evaluation utilities.
from utils.utils.grader import check_is_correct
from utils.utils.parser import extract_answer, parse_ground_truth, parse_question

# LIMO’s evaluation function: rule-based for math problems.
def evaluate_math_answer(generated_answer, gold_answer):
    """
    Evaluate a model's prediction against a gold answer using LIMO's utilities.
    
    Args:
        generated_answer (str): The generated answer from the model.
        gold_answer (str): The correct answer to compare against.
    
    Returns:
        dict: A dictionary containing:
              - "correct": (bool) True if the generated answer is correct.
              - "quality": (int) 5 if correct, else 1.
              - "explanation": (str) A brief explanation.
    """
    extracted = extract_answer(generated_answer)
    correct = check_is_correct(extracted, gold_answer)

    print('Extracted Answer: '+extracted)
    print('Ground Truth: '+gold_answer)

    if correct:
        quality = 5
        explanation = "Answer is correct."
    else:
        quality = 1
        explanation = "Answer does not match the gold answer."
    return {"correct": correct, "quality": quality, "explanation": explanation}

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Define target and drafter models
    target_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    drafter_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    quant_config = QuantoConfig(weights="int8")

    print("Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True,
    )
    target.eval()

    print("Loading drafter model...")
    drafter = AutoModelForCausalLM.from_pretrained(
        drafter_model_name,
        quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True,
    )
    drafter.eval()

    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    eos_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")]

    # Load benchmark dataset:
    # If args.dataset is "limo", load LIMO dataset from Hugging Face
    if args.dataset.lower() == "limo":
        from datasets import load_dataset
        print("Loading LIMO dataset from Hugging Face...")
        benchmark_data = load_dataset("GAIR/LIMO", split="train")
    else:
        # Otherwise, load benchmark data from JSON file.
        with open(args.dataset, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    # Slice the dataset if a positive num_samples is specified for debugging.
    if args.num_samples > 0:
        print(f"Slicing the dataset to the first {args.num_samples} samples for debugging.")
        if hasattr(benchmark_data, "select"):
            benchmark_data = benchmark_data.select(list(range(args.num_samples)))
        else:
            benchmark_data = benchmark_data[:args.num_samples]

    # Determine CSV output file based on dataset if not explicitly provided.
    if args.output:
        output_file = args.output
    else:
        dataset_base = os.path.splitext(os.path.basename(args.dataset))[0]
        output_file = f"{dataset_base}_benchmark_results.csv"

    # Open CSV file in append mode and write header if new.
    write_header = not os.path.exists(output_file)
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    fieldnames = [
        "id",
        "question",
        "ground_truth",
        "speculative_answer",
        "baseline_answer",
        "speculative_latency",
        "baseline_latency",
        "speculative_throughput",
        "baseline_throughput",
        "acceptance_rate",
        "speculative_evaluation",
        "baseline_evaluation",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    total_speculative_time = 0.0
    total_baseline_time = 0.0
    total_speculative_tokens = 0
    total_baseline_tokens = 0

    # Determine the total number of samples.
    if hasattr(benchmark_data, "select"):
        num_samples = len(benchmark_data)
    else:
        num_samples = len(benchmark_data)

    # --- Update prompt prefix to use LIMO’s math problem prompt ---
    prompt_prefix = "Please reason step by step, and put your final answer within \\boxed{}.\n"

    # Iterate over the benchmark samples.
    for i, sample in enumerate(benchmark_data):
        # If the dataset does not have an "id" field, use the index.
        sample_id = sample.get("id", str(i))
        # Use LIMO's parser to extract the question and ground truth.
        question = parse_question(sample, args.data_name)
        _, ground_truth = parse_ground_truth(sample, args.data_name)

        input_text = prompt_prefix + question
        tokenized = tokenizer(input_text, return_tensors="pt").input_ids[0].tolist()

        # -------------------------------
        # Speculative decoding generation
        # -------------------------------
        set_seed(42)
        spec_start = time.time()
        spec_output_ids, accept_rate = spec_gen(
            tokenized,
            drafter,
            target,
            tokenizer=None,
            logits_processor=GreedyProcessor(temperature=1.0),
            gamma=args.gamma,
            max_gen_len=args.gen_len,
            eos_tokens_id=eos_tokens,
            use_cache=args.use_cache,
        )
        spec_end = time.time()
        speculative_latency = spec_end - spec_start
        spec_output = tokenizer.decode(spec_output_ids, skip_special_tokens=True)
        spec_tokens = len(spec_output)
        speculative_throughput = spec_tokens / speculative_latency if speculative_latency > 0 else 0

        # -------------------------------
        # Baseline autoregressive generation
        # -------------------------------
        set_seed(42)
        base_start = time.time()
        base_output_ids = autoregressive_generate(
            tokenized,
            target,
            use_cache=args.use_cache,
            max_gen_len=args.gen_len,
            eos_tokens_id=eos_tokens,
            logits_processor=GreedyProcessor(temperature=1.0),
            debug=False,
        )
        base_end = time.time()
        baseline_latency = base_end - base_start
        base_output = tokenizer.decode(base_output_ids, skip_special_tokens=True)
        base_tokens = len(base_output)
        baseline_throughput = base_tokens / baseline_latency if baseline_latency > 0 else 0

        # -------------------------------
        # Evaluate generated outputs using LIMO evaluation functions
        # -------------------------------
        spec_eval = evaluate_math_answer(spec_output, ground_truth)
        base_eval = evaluate_math_answer(base_output, ground_truth)

        total_speculative_time += speculative_latency
        total_baseline_time += baseline_latency
        total_speculative_tokens += spec_tokens
        total_baseline_tokens += base_tokens

        writer.writerow({
            "id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "speculative_answer": spec_output,
            "baseline_answer": base_output,
            "speculative_latency": speculative_latency,
            "baseline_latency": baseline_latency,
            "speculative_throughput": speculative_throughput,
            "baseline_throughput": baseline_throughput,
            "acceptance_rate": accept_rate,
            "speculative_evaluation": json.dumps(spec_eval),
            "baseline_evaluation": json.dumps(base_eval),
        })

        print(f"Sample {sample_id}:")
        print(f"  Speculative: {speculative_latency:.2f}s, {speculative_throughput:.1f} tokens/s")
        print(f"  Baseline:    {baseline_latency:.2f}s, {baseline_throughput:.1f} tokens/s")
        print("  Speculative Evaluation:", spec_eval)
        print("  Baseline Evaluation:", base_eval)
        print("-" * 60)

    csv_file.close()

    avg_spec_latency = total_speculative_time / num_samples
    avg_base_latency = total_baseline_time / num_samples
    avg_spec_throughput = total_speculative_tokens / total_speculative_time if total_speculative_time > 0 else 0
    avg_base_throughput = total_baseline_tokens / total_baseline_time if total_baseline_time > 0 else 0

    print("\n=== Overall Benchmark Summary ===")
    print(f"Number of samples: {num_samples}")
    print(f"Average Speculative Decoding Latency: {avg_spec_latency:.2f} seconds")
    print(f"Average Baseline Latency: {avg_base_latency:.2f} seconds")
    print(f"Average Speculative Throughput: {avg_spec_throughput:.1f} tokens/s")
    print(f"Average Baseline Throughput: {avg_base_throughput:.1f} tokens/s")
    print(f"Latency Improvement: {((avg_base_latency - avg_spec_latency) / avg_base_latency * 100):.1f}%")
    print(f"Inference Speed Improvement: {((avg_spec_throughput / avg_base_throughput) * 100):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Speculative Decoding vs Baseline using LIMO for Math Reasoning Evaluation"
    )
    parser.add_argument("--dataset", type=str, default='limo',
                        help="Path to benchmark dataset JSON file or use 'limo' to load the GAIR/LIMO dataset from Hugging Face")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV file to save benchmark results (if not provided, defaults to <dataset_basename>_benchmark_results.csv)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cuda or cpu)")
    parser.add_argument("--gamma", type=int, default=4,
                        help="Gamma parameter for speculative decoding")
    parser.add_argument("--gen_len", type=int, default=2048,
                        help="Maximum generation length (in tokens)")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use cache during generation")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples to use for benchmark (for debugging). Set to -1 to use the full dataset.")
    parser.add_argument("--data_name", type=str, default="math",
                        help="Name of the dataset for parsing answers (e.g., 'math')")
    args = parser.parse_args()

    main(args)
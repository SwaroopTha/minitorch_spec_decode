import argparse
import csv
import json
import time
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from minitorch.spec_decoding.spec_decoding import spec_gen
from minitorch.spec_decoding.auto_reg import autoregressive_generate
from utils.logits_proc import GreedyProcessor
import openai

# Set your OpenAI API key here or via environment variable


logits_processor = GreedyProcessor(temperature=1.0)

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

# Define the evaluation prompt for the LLM rating model
def build_evaluation_prompt(question, generated_answer, ground_truth):
    prompt = (
        "You are an expert evaluator for reasoning tasks. "
        "Below you are provided with a question, a generated answer, and a ground truth answer. "
        "Your task is to assess the generated answer based on its correctness and overall quality, "
        "taking into account that the generated answer may include extended reasoning. "
        "\n\n"
        f"Question: {question}\n\n"
        f"Generated Answer: {generated_answer}\n\n"
        f"Ground Truth Answer: {ground_truth}\n\n"
        "Please provide a JSON object with the following keys:\n"
        " - \"correct\": a boolean indicating if the generated answer is correct (true or false).\n"
        " - \"quality\": an integer from 1 (poor) to 5 (excellent) rating the overall quality.\n"
        " - \"explanation\": a brief explanation for your assessment.\n"
        "\n"
        "Only output the JSON object."
    )
    return prompt

# Call the evaluation LLM (using OpenAI's GPT-3.5-turbo) to rate an answer
def evaluate_answer(question, generated_answer, ground_truth, evaluation_model="gpt-3.5-turbo"):
    prompt = build_evaluation_prompt(question, generated_answer, ground_truth)
    try:
        response = openai.ChatCompletion.create(
            model=evaluation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        evaluation_text = response["choices"][0]["message"]["content"].strip()
        evaluation = json.loads(evaluation_text)
    except Exception as e:
        print("Evaluation LLM call failed:", e)
        evaluation = {"correct": False, "quality": 1, "explanation": "Evaluation failed."}
    return evaluation

def main(args):
    set_seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Define target and draft (drafter) models
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

    # Load benchmark dataset (JSON with keys: id, question, ground_truth)
    with open(args.dataset, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Determine CSV output file based on dataset if not explicitly provided
    if args.output:
        output_file = args.output
    else:
        dataset_base = os.path.splitext(os.path.basename(args.dataset))[0]
        output_file = f"{dataset_base}_benchmark_results.csv"

    # Open CSV file in append mode and write header if new
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
        "rejection_sampling_threshold",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    total_speculative_time = 0.0
    total_baseline_time = 0.0
    total_speculative_tokens = 0
    total_baseline_tokens = 0
    num_samples = len(benchmark_data)

    # Prompt prefix to enforce concise output
    prompt_prefix = "Answer concisely. "

    gammas = [8, 6, 2]

    for gamma in gammas:
        for sample in benchmark_data:
            sample_id = sample.get("id", "")
            question = sample["question"]
            ground_truth = sample["ground_truth"]

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
                logits_processor=logits_processor,
                gamma=gamma,
                max_gen_len=args.gen_len,
                eos_tokens_id=eos_tokens,
                use_cache=args.use_cache,
                rejection_sampling_threshold=1,
            )
            spec_end = time.time()
            speculative_latency = spec_end - spec_start
            spec_output = tokenizer.decode(spec_output_ids, skip_special_tokens=True)
            spec_tokens = len(spec_output)
            speculative_throughput = spec_tokens / speculative_latency if speculative_latency > 0 else 0

            # -------------------------------
            # Baseline autoregressive generation
            # -------------------------------
            # set_seed(42)
            # base_start = time.time()
            # base_output_ids = autoregressive_generate(
            #     tokenized,
            #     target,
            #     use_cache=args.use_cache,
            #     max_gen_len=args.gen_len,
            #     eos_tokens_id=eos_tokens,
            #     logits_processor=logits_processor,
            #     debug=False,
            # )
            # base_end = time.time()
            # baseline_latency = base_end - base_start
            # base_output = tokenizer.decode(base_output_ids, skip_special_tokens=True)
            # base_tokens = len(base_output)
            # baseline_throughput = base_tokens / baseline_latency if baseline_latency > 0 else 0

            # -------------------------------
            # Evaluate generated outputs using LLM rating
            # -------------------------------
            spec_eval = evaluate_answer(question, spec_output, ground_truth, evaluation_model=args.eval_model)
            # base_eval = evaluate_answer(question, base_output, ground_truth, evaluation_model=args.eval_model)

            total_speculative_time += speculative_latency
            # total_baseline_time += baseline_latency
            total_speculative_tokens += spec_tokens
            # total_baseline_tokens += base_tokens

            writer.writerow({
                "id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "speculative_answer": spec_output,
                # "baseline_answer": base_output,
                "speculative_latency": speculative_latency,
                # "baseline_latency": baseline_latency,
                "speculative_throughput": speculative_throughput,
                # "baseline_throughput": baseline_throughput,
                "acceptance_rate": accept_rate,
                "speculative_evaluation": json.dumps(spec_eval),
                # "baseline_evaluation": json.dumps(base_eval),
                # "rejection_sampling_threshold": threshold,
            })

            print(f"Sample {sample_id}:")
            print(f"  Speculative: {speculative_latency:.2f}s, {speculative_throughput:.1f} tokens/s")
            # print(f"  Baseline:    {baseline_latency:.2f}s, {baseline_throughput:.1f} tokens/s")
            print("  Speculative Evaluation:", spec_eval)
            # print("  Baseline Evaluation:", base_eval)
            print("-" * 60)

    csv_file.close()

    avg_spec_latency = total_speculative_time / num_samples
    # avg_base_latency = total_baseline_time / num_samples
    avg_spec_throughput = total_speculative_tokens / total_speculative_time if total_speculative_time > 0 else 0
    # avg_base_throughput = total_baseline_tokens / total_baseline_time if total_baseline_time > 0 else 0

    print("\n=== Overall Benchmark Summary ===")
    print(f"Number of samples: {num_samples}")
    print(f"Average Speculative Decoding Latency: {avg_spec_latency:.2f} seconds")
    # print(f"Average Baseline Latency: {avg_base_latency:.2f} seconds")
    print(f"Average Speculative Throughput: {avg_spec_throughput:.1f} tokens/s")
    # print(f"Average Baseline Throughput: {avg_base_throughput:.1f} tokens/s")
    # print(f"Latency Improvement: {((avg_base_latency - avg_spec_latency) / avg_base_latency * 100):.1f}%")
    # print(f"Inference Speed Improvement: {((avg_spec_throughput / avg_base_throughput) * 100):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Speculative Decoding vs Baseline for Reasoning Models")
    parser.add_argument("--dataset", type=str, required=True, help="Path to benchmark dataset JSON file")
    parser.add_argument("--output", type=str, default=None, help="CSV file to save benchmark results (if not provided, defaults to <dataset_basename>_benchmark_results.csv)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda or cpu)")
    parser.add_argument("--gamma", type=int, default=4, help="Gamma parameter for speculative decoding")
    parser.add_argument("--gen_len", type=int, default=1000, help="Maximum generation length (in tokens)")
    parser.add_argument("--use_cache", action="store_true", help="Use cache during generation")
    parser.add_argument("--eval_model", type=str, default="gpt-3.5-turbo", help="LLM model to use for answer evaluation")
    args = parser.parse_args()

    main(args)
import argparse
import random
import numpy as np
from minitorch.spec_decoding.spec_decoding import spec_gen
from minitorch.spec_decoding.auto_reg import autoregressive_generate

from utils.logits_proc import GreedyProcessor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import time
import os
from termcolor import colored

# torch for only seeding
import torch


class InferenceCLI:

    def __init__(self, device: str = "cuda"):
        print(
            colored("Speculative Decoding", "red"),
            colored("CLI", on_color="on_red", color="white"),
            "\n",
        )
        self.device = device

        self.gamma = 4
        self.gen_len = 100  # 350
        self.debug = False
        self.spec = True
        self.dr = False
        self.cache = False
        self.target_gen = True
        self.reset_in_between = True
        
        self.chat = True
        
        self.processors = {
            "greedy": {
                "processor": GreedyProcessor,
                "building_args": {"temperature": float},
            },
        }
        self.selected_processor = {
            "name": "greedy",
            "processor": GreedyProcessor,
            "args": {"temperature": 1.0},
        }
        self.processor = GreedyProcessor()

        self._load_models()
        self._run()

    def _load_models(self):
        # Target model
        target_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # "ibm-granite/granite-3.2-8b-instruct"
        target_quantize = QuantoConfig(weights="int8")  # QuantoConfig(weights="int8")  None
        
        # Drafter model
        drafter_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # "ibm-granite/granite-3.2-2b-instruct"
        drafter_quantize = QuantoConfig(weights="int8")  # QuantoConfig(weights="int8") None

        print(colored("Target model:", on_color="on_green"), target_model)
        print(colored("Drafter model:", on_color="on_blue"), drafter_model)
        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            quantization_config=target_quantize,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.target.eval()

        tokenizer_name = target_model
        if tokenizer_name != target_model:
            print(colored("Warning: Tokenizer is different from target model. Use with caution.", "red"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            quantization_config=drafter_quantize,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.drafter.eval()        
        self.end_tokens = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")]  # "<|end_of_text|>" for granite # "<|eot_id|>" is the end of turn token for Llama model.

    def _perform_command(self, command: str):
        args = command.split(" ")
        if args[0] == "/quit":
            print(colored("Goodbye!", on_color="on_red"))
            exit(0)
        if args[0] == "/debug":
            self.debug = not self.debug
            print(colored(f"Debug mode: {self.debug}", on_color="on_blue"))
            return
        if args[0] == "/speculative":
            self.spec = not self.spec
            print(colored(f"Speculative Decoding generation: {self.spec}", on_color="on_blue"))
            return
        if args[0] == "/drafter":
            self.dr = not self.dr
            print(colored(f"Drafter generation: {self.dr}", on_color="on_blue"))
            return
        if args[0] == "/cache":
            self.cache = not self.cache
            print(colored(f"Cache: {self.cache}", on_color="on_blue"))
            if self.cache:
                print(colored("Warning, cache feature is very unstable accross different models. It might generate errors or just perturb the generation. Use with caution.", "red"))
            return
        if args[0] == "/target":
            self.target_gen = not self.target_gen
            print(colored(f"Target generation: {self.target_gen}", on_color="on_blue"))
            return
        if args[0] == "/chat":
            self.chat = not self.chat
            print(colored(f"Chat mode: {self.chat}", on_color="on_blue"))
            return
        if args[0] == "/length":
            if len(args) < 2:
                print(colored("Usage: /length <value>", "red"))
                return
            self.gen_len = int(args[1])
            print(colored(f"Generation length: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/gamma":
            if len(args) < 2:
                print(colored("Usage: /gamma <value>", "red"))
                return
            self.gamma = int(args[1])
            print(colored(f"Gamma: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            return
        if args[0] == "/processor":
            # /processor <processor_name> <args0> <args1> ...
            if len(args) < 2:
                print(colored("Usage: /processor <processor_name> <args0> <args1> ...", "red"))
                return
            processor_name = args[1]
            if processor_name not in self.processors:
                print(colored("Invalid processor name", "red"))
                print(colored("Available processors:", "red"))
                for processor in self.processors.keys():
                    print(colored(f"\t{processor}", "red"))
                return
            processor = self.processors[processor_name]
            print(colored(f"Selected processor: {processor_name}", "blue"))
            building_args = processor["building_args"]
            args = args[2:]
            processor_args = {}
            for arg_name, arg_type in building_args.items():
                if len(args) == 0:
                    print(colored(f"Missing argument {arg_name}", "red"))
                    return
                try:
                    processor_args[arg_name] = arg_type(args[0])
                    print(colored(f"\t{arg_name}: {arg_type(args[0])}", "blue"))
                except ValueError:
                    print(colored(f"Invalid argument {arg_name} of type {arg_type}", "red"))
                    return
                args = args[1:]
            self.selected_processor = {
                "name": processor_name,
                "processor": processor["processor"],
                "args": processor_args,
            }
            self.processor = processor["processor"](**processor_args)
            return

        print(colored("Unknown command", "red"))
        self._help()

    def _help(self):
        print(colored("Commands:", on_color="on_blue"))
        print("/quit: quit the program")
        print("/debug: toggle speculative debug mode")
        print(colored(f"\t{self.debug}", "green" if self.debug else "red"))
        print("/clear: clear the screen")
        print("/speculative: toggle speculative decoding")
        print(colored(f"\t{self.spec}", "green" if self.spec else "red"))
        print("/target: toggle target generation")
        print(colored(f"\t{self.target_gen}", "green" if self.target_gen else "red"))
        print("/drafter: toggle drafter generation")
        print(colored(f"\t{self.dr}", "green" if self.dr else "red"))
        print("/cache: toggle cache")
        print(colored(f"\t{self.cache}", "green" if self.cache else "red"))
        print("/chat: toggle chat mode")
        print(colored(f"\t{self.chat}", "green" if self.chat else "red"))
        print("/length <value>: set generation length")
        print(colored(f"\t{self.gen_len}", "blue"))
        print("/gamma <value>: set gamma")
        print(colored(f"\t{self.gamma}", "blue"))
        print("/processor <processor_name> [args0] [args1] ...: set processor")
        print(colored(f"\t{self.selected_processor['name']}", "blue"))
        for arg_name, arg_value in self.selected_processor["args"].items():
            print(colored(f"\t\t{arg_name}: {arg_value}", "blue"))
        

    def _infer(self, prefix: str):
        if self.chat:
            prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": prefix}], add_generation_prompt=True, tokenize=False)
            
        tokenized = self.tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()
        
        spec_throughput = 0.0
        base_throughput = 0.0
        drafter_throughput = 0.0

        if self.spec:
            self._set_seed(42)
            spec_start_time = time.time()
            # print("type of tokenized:", type(tokenized))
            # print("tokenized:", tokenized)
            output_ids, accept_rate = spec_gen(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                use_cache=self.cache,
            )
            # print(output_ids)
            spec_end_time = time.time()
            spec_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Speculative ==========", "green"))
            print(colored("Out:", "green"), spec_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))
            spec_throughput = len(spec_output) / (spec_end_time - spec_start_time)
            print(colored(f"Throughput: {spec_throughput:.1f} tokens/s", "green"))
            print(colored("========== Speculative ==========", "green"))

        if self.target_gen:
            self._set_seed(42)
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
            )
            end_time = time.time()
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("=========== Target AR ===========", "orange"))
            print(colored("Out:", "orange"), output)
            base_throughput = len(output) / (end_time - start_time)
            print(colored(f"Throughput: {base_throughput:.1f} tokens/s", "orange"))
            print(colored("=========== Target AR ===========", "orange"))
            if self.spec and base_throughput > 0.0:
                print(colored(f"Throughput increase: {((spec_throughput - base_throughput) / base_throughput) * 100:.1f}%", "pink"))

        if self.dr:
            self._set_seed(42)
            output_ids = autoregressive_generate(
                tokenized,
                self.drafter,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
            )
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            print(colored("========== Drafter AR ==========", "cyan"))
            drafter_throughput = len(output) / (end_time - start_time)
            print(colored("Out:", "cyan"), output)
            print(colored(f"Throughput: {drafter_throughput:.1f} tokens/s", "cyan"))
            print(colored("========== Drafter AR ==========", "cyan"))

    def _run(self):
        while True:
            command = input("> ").replace('\\n', '\n').replace('\\t', '\t')
            if command.startswith("/"):
                self._perform_command(command)
                continue

            self._infer(command)
            
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding CLI")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    args = parser.parse_args()

    InferenceCLI(device=args.device)

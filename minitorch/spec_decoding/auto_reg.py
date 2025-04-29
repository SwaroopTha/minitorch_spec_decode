from math import inf
from utils.logits_proc import LogitsProcessor, GreedyProcessor
from typing import List
from minitorch.module import *
from minitorch.tensor import *
from minitorch.tensor_ops import *
from minitorch.nn import *
from minitorch.tensor_functions import *

# for model, we use nn.Module
import torch.nn as nn

def autoregressive_generate(
    inputs: List[int],
    model: nn.Module,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
) -> List[int]:
    backend = minitorch.TensorBackend(CudaKernelOps)
    cache = None
    prompt_len = len(inputs)
    # prepare input tensor
    max_seq_length = (
        getattr(model.config, "max_position_embeddings", None)
        or getattr(model.config, "max_context_length", None)
        or 1024
    )

    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = np.full(
        (1, total_len), pad_token_id, dtype=np.int64
    )
    input_ids[0, :prompt_len] = inputs
    input_ids = tensor_from_numpy(input_ids, backend=backend)

        
    list_tokens_id = (
        eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    )

    stop_tokens = np.array(list_tokens_id, dtype=np.int64)
    
    for curr in range(prompt_len, total_len):
        o = model(torch.tensor(input_ids.to_numpy(), dtype=torch.int64, device='cuda')[..., :curr], past_key_values=cache, use_cache=use_cache)
        logits = o.logits[..., -1, :]  # [1, vocab_size]
        # print("logits type", type(logits))
        logits = tensor(logits.detach().cpu().numpy().tolist(), backend, requires_grad=False)
        probs = logits_processor(logits)  # [1, vocab_size]
        x = logits_processor.sample(probs)  # [1, 1]
        input_ids[:, curr] = x
        cache = o.past_key_values

        # check for end token
        x_np = x.to_numpy() if hasattr(x, 'to_numpy') else np.array(x.item())
        if np.isin(x_np, stop_tokens).any():
            break

            


    # make sure elements are integers
    # input_ids = [int(x) for x in input_ids]
    output = list(map(int, input_ids[0, prompt_len : curr + 1].to_numpy().tolist()))
    return output

if __name__ == "__main__":
    import numpy as np
    from unittest.mock import MagicMock
    import torch

    class MockModel(Module):
        def __init__(self, vocab_size=10):
            super().__init__()
            self.config = MagicMock()
            self.config.max_position_embeddings = 1024
            self.config.vocab_size = vocab_size
            self.device = "cuda"
            self.backend = "cuda"

        def forward(self, input_ids, past_key_values=None, use_cache=False):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.config.vocab_size, device=self.device)
            return MagicMock(logits=logits, past_key_values=past_key_values)
        

    test_cases = [
        {
            "inputs": [1, 2, 3],
            "max_gen_len": 5,
            "eos_tokens_id": 9,
            "description": "Short input with early termination"
        },
        # {
        #     "inputs": [4, 5, 6, 7],
        #     "max_gen_len": 10,
        #     "eos_tokens_id": [8, 9],
        #     "description": "Multiple stop tokens"
        # },
        # {
        #     "inputs": [1] * 20,
        #     "max_gen_len": 100,
        #     "eos_tokens_id": 99,  # Unlikely to hit
        #     "description": "Long input with max length"
        # }
    ]

    for case in test_cases:
        model = MockModel()
        result = autoregressive_generate(
            inputs=case["inputs"],
            model=model,
            max_gen_len=case["max_gen_len"],
            eos_tokens_id=case["eos_tokens_id"],
            logits_processor=GreedyProcessor(),
        )

        print(f"Generated output: {result}")
        print(f"Output length: {len(result)}")

        assert isinstance(result, list), "Output should be a list"
        assert all(isinstance(x, (int, float)) for x in result), "All elements should be numbers"
        assert len(result) <= case["max_gen_len"], "Output exceeded max length"
        
        # Check stop token condition
        stop_tokens = case["eos_tokens_id"] if isinstance(case["eos_tokens_id"], list) else [case["eos_tokens_id"]]
        if len(result) < case["max_gen_len"]:
            assert result[-1] in stop_tokens, "Early termination should be due to stop token"
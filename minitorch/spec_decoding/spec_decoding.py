from typing import List, Tuple
from utils.cache import prune_cache
from transformers.cache_utils import DynamicCache
from minitorch.module import *
from minitorch.tensor import *
from minitorch.tensor_ops import *
from minitorch.nn import *
from minitorch.tensor_functions import *
from utils.logits_proc import LogitsProcessor, GreedyProcessor
import numpy as np
import time

# only for models
import torch
import torch.nn as nn


def max_fn(x: Tensor) -> Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    # print("x.shape: ", x.shape)
    x_max = x * (x > 0)  # + zeros(x) * (x <= 0).to(x.dtype)
    # print("x_max.shape: ", x_max.shape)
    x_sum = x_max.sum(dim=len(x_max.shape) - 1)
    # print("x_sum.shape: ", x_sum.shape)
    return x_max / x_sum

@torch.no_grad()
def spec_gen(
        inputs: List[int],
        drafter = nn.Module, # because huggingface transformers use nn.Module
        target = nn.Module, # because huggingface transformers use nn.Module
        tokenizer = None, # only for debugging (not yet implemented)
        gamma: int = 5,
        logits_processor: LogitsProcessor = GreedyProcessor(temperature=1.0),
        max_gen_len: int = 40,
        eos_tokens_id: int | List[int] = 1,
        pad_token_id: int = 0,
        use_cache: bool = False,
        skip_sample_adj: bool = False,
        first_target: bool = True,
) -> Tuple[List[int], float]:
    """Generate text seqence using speculative decoding."""

    # time1 = time.time()

    backend = minitorch.TensorBackend(CudaKernelOps)

    drafter_cache, target_cache = None, None
    stop_tokens = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    # stop_tokens = np.array(list_tokens_id)

    drafts_accepted, drafts_speculated = 0.0, 0.0
    vocab_size = min(drafter.config.vocab_size, target.config.vocab_size) 

    # prepare the input tensor
    prompt_len = len(inputs)
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)

    # input_ids = [[pad_token_id] * total_len]
    # input_ids[0][:prompt_len] = inputs  # replaced
    input_ids_np = np.full((1, total_len), pad_token_id, dtype=np.int64)
    input_ids_np[0, :prompt_len] = inputs


    curr_pos = prompt_len

    # time2 = time.time()
    # print("time to prepare input tensor: ", time2 - time1)

    # time3 = time.time()

    if first_target:
        # run target model before speculative decoding for prefilling

        inp_t = torch.from_numpy(input_ids_np).to(device='cuda')[:, :curr_pos]

        Mp = target(
            input_ids=inp_t,
            past_key_values=target_cache,
            use_cache=use_cache,
        )

        target_cache = Mp.past_key_values
        logits = Mp.logits[..., -1, :]
        # convert torch tensor to minitorch tensor
        # print("logits type: ", type(logits))
        logits = tensor(logits.cpu().numpy().tolist(), backend)

        p_p = logits_processor(logits)
        t = logits_processor.sample(p_p)[0]
        # print("input_ids0: ", input_ids)
        # input_ids[0][curr_pos] = int(t)  # replaced
        input_ids_np[0, curr_pos] = int(t)  
        # print("input_ids1: ", input_ids)
        curr_pos += 1

        if t in stop_tokens:
        #if np.isin(t, stop_tokens):
            # Handle the slice operation properly
            # output = list(map(int, input_ids[0, prompt_len:curr_pos].to_numpy().tolist()))
            output = input_ids_np[0, prompt_len:curr_pos].tolist()
            return output, 0.0

    # time4 = time.time()
    # print("time to prefill target model before speculative decoding: ", time4 - time3)
        
 
    while curr_pos < total_len:
        # print("curr_pos: ", curr_pos)
        # print("total_len: ", total_len)
        remaining_len = total_len - curr_pos
        corrected_gamma = min(gamma, remaining_len - 1)
        if corrected_gamma > 0:
            # time41 = time.time()
            q = zeros((1, corrected_gamma, vocab_size), backend=backend)
            # time42 = time.time()
            # print("zeros time: ", time42 - time41)

            # generate gamma drafts
            # print("input_ids2: ", input_ids)
            # time5 = time.time()
            for k in range(corrected_gamma):
                # time51 = time.time()
                # drafter_inputs = torch.tensor(input_ids, dtype=torch.int64, device='cuda')[:, :curr_pos + k]  replaced
                drafter_inputs = torch.from_numpy(input_ids_np).to(device='cuda')[:, : curr_pos + k]

                # time52 = time.time()
                Mq = drafter(
                    input_ids=drafter_inputs,  # torch.tensor(input_ids, dtype=torch.int64, device='cuda')[:, :curr_pos + k],
                    past_key_values=drafter_cache,
                    use_cache=use_cache
                )
                drafter_cache = Mq.past_key_values
                arr = Mq.logits[..., -1, :vocab_size].cpu().numpy()      # just once, no .tolist()
                # time53 = time.time()
                # convert torch tensor to minitorch tensor
                draft_logits = tensor_from_numpy(arr, backend)
                # draft_logits = tensor(draft_logits.cpu().numpy().tolist(), backend)
                # time54 = time.time()
                draft_probs = logits_processor(draft_logits)
                # time541 = time.time()
                q[:, k, :] = draft_probs  
                # time542 = time.time() 
                xi = logits_processor.sample(draft_probs)[0]
                # time55 = time.time()
                # input_ids[0][curr_pos + k] = int(xi)  replaced
                input_ids_np[0, curr_pos + k] = int(xi)
            
            drafts_speculated += corrected_gamma
            # time6 = time.time()
            # print("time to generate gamma drafts: ", time6 - time5)

            # time7 = time.time()
            drafter_inp = torch.from_numpy(input_ids_np).to(device='cuda')[:, : curr_pos + corrected_gamma]
            Mp = target(
                input_ids=drafter_inp,
                past_key_values=target_cache,
                use_cache=use_cache
            )
            # Mp = target(
            #     input_ids=torch.tensor(input_ids, dtype=torch.int64, device='cuda')[:, :curr_pos + corrected_gamma],  # .to_numpy()
            #     past_key_values=target_cache,
            #     use_cache=use_cache
            # )
            target_cache = Mp.past_key_values
            arr = Mq.logits[..., curr_pos - 1:curr_pos + corrected_gamma - 1, :vocab_size].cpu().numpy()  # just once, no .tolist()
            # draft_logits = Mp.logits[..., curr_pos - 1:curr_pos + corrected_gamma - 1, :vocab_size]
            # convert torch tensor to minitorch tensor
            # draft_logits = tensor(draft_logits, backend)
            draft_logits = tensor_from_numpy(arr, backend) 
            p = logits_processor(draft_logits)
            # time8 = time.time()
            # print("time to run target models: ", time8 - time7)

            # time9 = time.time()
            r = rand((corrected_gamma,), backend=backend)

            fractions_np = (p / q).to_numpy()
            r_np = r.to_numpy() 
            ids = input_ids_np[0, curr_pos : curr_pos + corrected_gamma]
            idxs = np.arange(corrected_gamma, dtype=int)
            sel = fractions_np[0, idxs, ids]
            mask = r_np > sel 
            if mask.any():
                n = int(np.argmax(mask))   # first position where r_np > sel
            else:
                n = corrected_gamma


            drafts_accepted += n
            # time10 = time.time()
            # print("time to compute fractions: ", time10 - time9)

            # time11 = time.time()
            # check if the end token is in the drafts
            new_ids = input_ids_np[0, curr_pos:curr_pos + n]  # .to_numpy().tolist()
            # stop_offset = next((i for i, tok in enumerate(new_ids) if tok in stop_tokens), None)  # can be optimized
            arr = input_ids_np[0, curr_pos: curr_pos + n]
            mask = np.isin(arr, stop_tokens)   # vectorized
            offs = np.nonzero(mask)[0]
            if offs.size:
                stop_offset = int(offs[0])
            else:
                stop_offset = None

            if stop_offset is not None:
                output = input_ids_np[0, prompt_len:curr_pos + stop_offset + 1].tolist()
                # output = list(map(int, input_ids[0, prompt_len:curr_pos+stop_offset+1].to_numpy().tolist()))
                return output, drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0.0
            # time12 = time.time()
            # print("time to check stop tokens: ", time12 - time11)

            # time13 = time.time()
            # adjust the distribution from Mp
            if n == corrected_gamma:
                p_p = Mp.logits[..., curr_pos + corrected_gamma - 1, :vocab_size]
                # convert torch tensor to minitorch tensor
                p_p = tensor(p_p.cpu().numpy().tolist(), backend)
                p_p = logits_processor(p_p)
            else:
                if use_cache:
                    drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                    target_cache = prune_cache(target_cache, corrected_gamma - n + 1)

                if not skip_sample_adj:
                    p_p = max_fn(p[:, n, :] - q[:, n, :])
                else:
                    p_p = p[..., n, :]
            
            x = logits_processor.sample(p_p)[0]
            # print("corrected_gamma: ", corrected_gamma)
            # print("n: ", n)
            if corrected_gamma - n > 0:
                # input_ids[0][curr_pos + n:curr_pos + corrected_gamma] = [pad_token_id] * (corrected_gamma - n)  # tensor([pad_token_id] * (corrected_gamma - n), backend=backend)
                input_ids_np[0, curr_pos + n:curr_pos + corrected_gamma] = pad_token_id
            input_ids_np[0, curr_pos + n] = int(x)

            curr_pos += n + 1
            # time14 = time.time()
            # print("time to adjust the distribution: ", time14 - time13)

            # print("x: ", x)
            # print("stop_tokens: ", stop_tokens)
            # if isinstance(x, Tensor):
            #     x = x.item()
            if x in stop_tokens:
                # print("stop_tokens found")
                # output = list(map(int, input_ids[0, prompt_len:curr_pos].to_numpy().tolist()))
                output = input_ids_np[0, prompt_len:curr_pos].tolist()
                return output, drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0.0
        else:
            # print("I'm here")
            inp_t = torch.from_numpy(input_ids_np).to(device='cuda')[:, : curr_pos]

            Mp = target(
                # input_ids=torch.tensor(input_ids, dtype=torch.int64, device='cuda')[:, :curr_pos],  # replaced
                input_ids=inp_t,
                past_key_values=target_cache,
                use_cache=use_cache,
            )
            target_cache = Mp.past_key_values

            final_logits = Mp.logits[..., curr_pos - 1, :].cpu().numpy().tolist()
            p_p = logits_processor(tensor(final_logits, backend))
            x = logits_processor.sample(p_p)[0]

            # input_ids[0][curr_pos] = int(x)
            input_ids_np[0, curr_pos] = int(x)
            curr_pos += 1

            if x in stop_tokens:
                # output = list(map(int, input_ids[0, prompt_len:curr_pos].to_numpy().tolist()))
                output = input_ids_np[0, prompt_len:curr_pos].tolist()
                # return input_ids[0][prompt_len:curr_pos], drafts_accepted / drafts_speculated  replaced
                return output, drafts_accepted / drafts_speculated
        # print("input_ids: ", input_ids)
        # print("prompt_len: ", prompt_len)
        # print("input_ids[0, prompt_len:]: ", input_ids[0, prompt_len:])

    # print("input_ids: ", input_ids)
    # print("input_ids dtype: input_ids.dtype")
    # output = list(map(int, input_ids[0, prompt_len:].to_numpy().tolist()))
    output = input_ids_np[0, prompt_len:].tolist()
    return output, drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0.0



if __name__ == "__main__":
    import numpy as np
    from unittest.mock import MagicMock

    class MockModel(nn.Module):
        def __init__(self, vocab_size=100):
            super().__init__()
            self.config = MagicMock()
            self.config.vocab_size = vocab_size
            self.config.max_position_embeddings = 1024
            self.device = "cuda"
            self.backend = "cuda"
        
        def forward(self, input_ids, past_key_values=None, use_cache=False):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.config.vocab_size, device=self.device)
            return MagicMock(logits=logits, past_key_values=past_key_values)
        
    drafter = MockModel()
    target = MockModel()

    # Test
    inputs = [1, 2, 3]
    gamma = 3
    max_gen_len = 10
    print("running spec_gen...")
    try:

        output, acceptance_ratio = spec_gen(
            inputs=inputs,
            drafter=drafter,
            target=target,
            gamma=gamma,
            max_gen_len=max_gen_len,
            eos_tokens_id=99,
            first_target=True,
        )
        print("\nTest Success!")
        print("Output:", output)
        print("Acceptance Ratio:", acceptance_ratio)

        assert isinstance(output, list), "Output should be a list"
        assert isinstance(acceptance_ratio, float), "Acceptance ratio should be a float"
        assert len(output) <= max_gen_len, "Output length exceeds max_gen_len"
    except Exception as e:
        print("\nTest Failed!")
        print("Error:", str(e))
        raise


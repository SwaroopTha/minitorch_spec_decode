from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # ASSIGN1.2
        ctx.save_for_backward(a, b)
        c = a * b
        return c
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # ASSIGN1.4
        a, b = ctx.saved_values
        return b * d_output, a * d_output
        # END ASSIGN1.4


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # ASSIGN1.2
        ctx.save_for_backward(a)
        return operators.inv(a)
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # ASSIGN1.4
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)
        # END ASSIGN1.4


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # ASSIGN1.2
        return -a
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # ASSIGN1.4
        return -d_output
        # END ASSIGN1.4


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # ASSIGN1.2
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # ASSIGN1.4
        sigma: float = ctx.saved_values[0]

        return sigma * (1.0 - sigma) * d_output
        # END ASSIGN1.4


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # ASSIGN1.2
        ctx.save_for_backward(a)
        return operators.relu(a)
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # ASSIGN1.4
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)
        # END ASSIGN1.4


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # ASSIGN1.2
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # ASSIGN1.4
        out: float = ctx.saved_values[0]
        return d_output * out
        # END ASSIGN1.4


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # ASSIGN1.2
        return 1.0 if a < b else 0.0
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # ASSIGN1.4
        return 0.0, 0.0
        # END ASSIGN1.4


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # ASSIGN1.2
        return 1.0 if a == b else 0.0
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # ASSIGN1.4
        return 0.0, 0.0
        # END ASSIGN1.4

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


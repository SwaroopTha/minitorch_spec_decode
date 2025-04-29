import abc
from minitorch.tensor import *
from minitorch.module import *
from minitorch.tensor_ops import *
from minitorch.nn import *
from minitorch.tensor_functions import *


class LogitsProcessor(abc.ABC):
    """Process Logits for decoding"""
    def __init__(self, temperature: float, backend: TensorBackend=None, use_fused_kernel: bool=False):
        self.temperature = temperature
        self.backend = backend

    def __call__(self, logits: Tensor) -> Tensor:
        """ Process the logits tensor. """
        # proc = self._process(logits)
        return softmax(logits / self.temperature, dim=len(logits.shape)-1)
        
    @abc.abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        """ Process the logits tensor. """
        pass

    @abc.abstractmethod
    def sample(self, logits: Tensor) -> Tensor:
        """Sample from the logits tensor."""
        pass

class GreedyProcessor(LogitsProcessor):
    """Most probable token."""
    def __init__(self, temperature: float=1.0, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__(temperature, backend, use_fused_kernel)

    def _process(self, logits: Tensor) -> Tensor:
        return logits
    
    def sample(self, probs: Tensor) -> Tensor:
        result = argmax(probs, dim=-1)
        # max_prob = max(probs, dim=1)
        # assert probs[0, int(result[0])] == max_prob[0, 0]
        return result  # .view(result.shape[0], 1)
from typing import Tuple, Union
from minitorch.tensor import Tensor
from transformers.cache_utils import DynamicCache

def prune_cache(cache: Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache], num_discard: int):
    """
        Prune the cache by removing num_discard from end of the cache.
        Args:
            cache: The cache to prune.
            num_discard: The number of items to discard from the end of the cache.
        Returns:
            The pruned cache. Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]
        
    """
    if cache is None:
        return cache
    if isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_discard)
    elif isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_discard)
    else:
        raise ValueError("Invalid cache type. Expected tuple or DynamicCache.")
    
def prune_tuple_cache(cache: Tuple[Tuple[Tensor, Tensor]], num_discard: int) -> Tuple[Tuple[Tensor, Tensor]]:
    """
        Prune the cache by removing num_discard from end of the cache.
        Args:
            cache: The cache to prune.
            num_discard: The number of items to discard from the end of the cache.
        Returns:
            The pruned cache. Tuple[Tuple[Tensor, Tensor]]
        
    """
    if cache is None:
        return None
    
    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue
        new_layer_cache = []
        for i in range(len(layer_cache)):
            tensor = layer_cache[i]
            new_tensor = tensor[:, :, :-num_discard, :]
            new_layer_cache.append(new_tensor)
        new_cache.append(tuple(new_layer_cache))

    return tuple(new_cache)

def prune_dynamic_cache(cache: DynamicCache, num_discard: int) -> DynamicCache:
    """
        Prune the cache by removing num_discard from end of the cache.
        Args:
            cache: The cache to prune.
            num_discard: The number of items to discard from the end of the cache.
        Returns:
            The pruned cache. DynamicCache
    """
    if cache is None:
        return None
    
    for layer in range(len(cache)):
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_discard, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_discard, :]
    
    cache._seen_tokens -= num_discard
    return cache

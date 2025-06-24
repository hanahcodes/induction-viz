# utils/hook_utils.py

def cache_attention_patterns(activation, hook, cache):
    """
    A hook function to cache the attention patterns.
    The shape of the attention pattern tensor is [batch, n_heads, query_pos, key_pos].
    """
    cache[hook.name] = activation.detach()

def attn_filter(name: str) -> bool:
    """
    A filter function to select only the attention pattern hooks.
    `TransformerLens` hook names are structured, e.g., 'blocks.0.attn.hook_pattern'.
    """
    return name.endswith("attn.hook_pattern")
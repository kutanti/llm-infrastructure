"""Storage arithmetic for conventional full-attention K/V tensors."""

from dataclasses import dataclass


MIB = 1024**2
GIB = 1024**3
# Elements per block, stored bytes per block (including the FP16 scale).
FORMATS = {"f16": (1, 2), "q8_0": (32, 34), "q4_0": (32, 18)}


@dataclass(frozen=True)
class AttentionShape:
    name: str
    full_attention_layers: int
    kv_heads: int
    head_dim: int
    scope: str = "conventional attention cache only"


SHAPES = (
    AttentionShape("Qwen3-4B / Qwen3-8B", 36, 8, 128),
    AttentionShape("Qwen3.5-4B", 8, 4, 256, "excludes 24 recurrent layers"),
    AttentionShape("GLM-4-9B-Chat (original)", 40, 2, 128),
    AttentionShape("Llama-2-7B", 32, 32, 128),
)


def cache_bytes(layers, kv_heads, head_dim, context, *,
                sequences=1, key_type="f16", value_type=None):
    """Assume equal K/V dimensions and round each head row to whole blocks.

    Excludes runtime padding, paging, recurrent state, and temporary buffers.
    A format being listed does not imply backend support for a given model.
    """
    for name, number in (
        ("layers", layers), ("kv_heads", kv_heads), ("head_dim", head_dim),
        ("context", context), ("sequences", sequences),
    ):
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError(f"{name} must be a positive integer")
    value_type = key_type if value_type is None else value_type
    row_bytes = 0
    for dtype in (key_type, value_type):
        if dtype not in FORMATS:
            raise ValueError(f"Unsupported cache format: {dtype}")
        block, size = FORMATS[dtype]
        row_bytes += ((head_dim + block - 1) // block) * size
    return layers * kv_heads * context * sequences * row_bytes

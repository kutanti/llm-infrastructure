"""
LESSON 5: FlashAttention and the online softmax.

Run:  python 05_flash_attention.py

THE SIMPLE ANALOGY
------------------
You must average 1,000,000 numbers, but your desk only fits 100 at a time.

  NAIVE:  demand a desk big enough for all 1,000,000. Fails.
  FLASH:  bring 100 at a time, keep a RUNNING total and a RUNNING count.
          Same exact average, desk never exceeds 100.

That illustrates streaming, one ingredient of FlashAttention. The catch is
that attention uses softmax, not a
plain average, and softmax needs the MAXIMUM of all values for numerical
stability - which you supposedly cannot know until you have seen them all.

The online softmax trick solves this: keep a running max too, and whenever
you meet a bigger one, RETROACTIVELY RESCALE everything accumulated so far
by exp(old_max - new_max). Mathematically equivalent, up to rounding.

WHY IT MATTERS ON A 6 GB CARD
-----------------------------
Fully materialized prefill attention builds an (seq x seq) score matrix
per head. FlashAttention avoids storing the full matrix. This NumPy demo
only implements a single-query streaming reduction, not GPU kernels or
tiled causal prefill. It makes no claim about actual GPU performance.
"""

import numpy as np

rng = np.random.default_rng(11)

D_HEAD = 128
BLOCK = 256          # keys processed per tile - the "desk size"


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def naive_attention(q, K, V):
    """Materializes the full score vector/matrix. Simple but memory-hungry."""
    scores = (K @ q) / np.sqrt(D_HEAD)     # <-- all SEQ scores exist at once
    return softmax(scores) @ V


def flash_attention(q, K, V, block=BLOCK):
    """
    Streaming attention. Never holds more than `block` scores.
    Running state is only: o (d_head,), l (scalar), m (scalar).
    """
    if block <= 0 or K.shape[0] == 0:
        raise ValueError("block and sequence length must be positive")
    d = q.shape[0]
    o = np.zeros(d)          # running weighted sum of V
    l = 0.0                  # running sum of exp() - the denominator
    m = -np.inf              # running max, for numerical stability

    for start in range(0, K.shape[0], block):
        Kb = K[start:start + block]
        Vb = V[start:start + block]

        s = (Kb @ q) / np.sqrt(d)        # only `block` scores exist now
        m_new = max(m, s.max())

        # Rescale everything accumulated so far onto the new maximum.
        correction = np.exp(m - m_new) if m > -np.inf else 0.0
        p = np.exp(s - m_new)

        o = o * correction + p @ Vb
        l = l * correction + p.sum()
        m = m_new

    return o / l


def peak_score_memory(seq, block, bytes_per=4):
    return seq * bytes_per, block * bytes_per


def main():
    print("=" * 70)
    print("PART 1: does the streaming version give the same answer?")
    print("=" * 70)
    for seq in (512, 4096, 16384):
        K = rng.normal(0, 1, (seq, D_HEAD))
        V = rng.normal(0, 1, (seq, D_HEAD))
        q = rng.normal(0, 1, D_HEAD)
        K[rng.integers(0, seq, 5)] *= 20.0    # nasty outliers to stress softmax

        ref = naive_attention(q, K, V)
        fla = flash_attention(q, K, V)
        np.testing.assert_allclose(ref, fla, rtol=1e-10, atol=1e-12)
        err = np.abs(ref - fla).max()
        cos = ref @ fla / (np.linalg.norm(ref) * np.linalg.norm(fla))
        print(f"seq={seq:6}  max abs diff = {err:.3e}   cosine = {cos:.12f}")

    print("\nExact to floating-point noise. FlashAttention is NOT an")
    print("approximation - it is the same math in a different order.\n")

    print("=" * 70)
    print("PART 2: the memory that disappears (one head, one layer, f32)")
    print("=" * 70)
    print(f"{'context':>10}{'naive scores':>16}{'flash tile':>14}{'saving':>12}")
    print("-" * 70)
    for seq in (1024, 4096, 8192, 16384, 32768):
        n_bytes, f_bytes = peak_score_memory(seq, BLOCK)
        print(f"{seq:>10}{n_bytes / 1024:>12.0f}KiB{f_bytes / 1024:>10.0f}KiB"
              f"{n_bytes / f_bytes:>11.0f}x")

    print()
    print("Hypothetical fully materialized PREFILL, not chunked prefill:")
    print("Scores are (seq x seq) per head. The following is storage math,")
    print("not allocations measured by the single-query demo above.")
    print()
    print(f"{'context':>10}{'naive seq x seq':>20}{'flash tile':>14}")
    print("-" * 70)
    for seq in (2048, 4096, 8192, 16384):
        naive = seq * seq * 4
        flash = BLOCK * BLOCK * 4
        print(f"{seq:>10}{naive / 1024**3:>16.2f}GiB{flash / 1024**2:>10.2f}MiB")

    print()
    print("An 8k full score matrix uses 0.25 GiB per head in FP32.")
    print("The tile column counts one score tile ONLY, not all workspace.")
    print("Q/K/V tiles, output accumulators and GPU occupancy add storage.")
    print("Layers need not retain every inference score matrix simultaneously.")
    print("Total memory still grows with context: K/V storage does not vanish.")

    print()
    print("=" * 70)
    print("PART 3: backend support is separate from the mathematical idea")
    print("=" * 70)
    print("""
FlashAttention already streams K and V in small tiles. That means it can
dequantize each tile on the fly, inside fast on-chip SRAM, and never
materialize a full f16 copy of the cache.

This is an implementation opportunity, not a universal rule that every
quantized cache requires FlashAttention. Support differs for K and V,
model architectures, runtimes, and versions.

Current llama.cpp can enable FlashAttention automatically when required
for quantized V cache, or explicitly fail an unsupported configuration.
The earlier claim that it always silently ignores the settings was wrong.

Ollama documents quantized KV cache with FlashAttention enabled. For a
hybrid model, check the installed runtime's logs and model support.
Do not assume these flags quantize recurrent states as well.
""")

    print("=" * 70)
    print("SUMMARY: four different things, often confused")
    print("=" * 70)
    print("""
  KV CACHE         Stop recomputing K/V for past tokens.
                   Saves TIME. Costs MEMORY. Exact. (Lesson 1)

  GQA              Fewer K/V heads shared across query heads.
                   Saves MEMORY. Architectural, baked in at training.
                   Learned architecture/quality tradeoff. (Lesson 4)

  FLASHATTENTION   Compute attention in tiles with online softmax.
                   Avoids full score matrices; hardware-dependent speed.
                   Mathematically equivalent, rounding may differ. (Lesson 5)

  KV QUANTIZATION  Store the cache in int8 instead of f16.
                   Saves MEMORY. Quality and speed must be evaluated. (Lesson 3)

None is distillation. GQA is baked into learned weight shapes; the other
three are execution/storage choices that do not retrain the model.
""")


if __name__ == "__main__":
    main()

"""
LESSON 1: Why the KV cache exists.

Run:  python 01_why_kv_cache.py

We feed predetermined random embeddings into ONE attention head, one at
a time, two ways. This is not a language model or token-generation benchmark:
    (a) NO CACHE  - recompute everything from scratch at every step
    (b) KV CACHE  - remember past Keys/Values, compute only the new token

The outputs are IDENTICAL to floating-point precision. The only difference
is how much work we did. That is the whole point: the KV cache is not an
approximation, it is a memo pad.
"""

import time
import numpy as np

rng = np.random.default_rng(0)

D_MODEL = 256   # embedding width
D_HEAD = 64     # width of this attention head
N_STEPS = 400   # number of predetermined embeddings

# Fixed "model weights". These never change - we are only doing inference.
Wq = rng.normal(0, 0.02, (D_MODEL, D_HEAD))
Wk = rng.normal(0, 0.02, (D_MODEL, D_HEAD))
Wv = rng.normal(0, 0.02, (D_MODEL, D_HEAD))

# The token embeddings that will "arrive" one at a time.
tokens = rng.normal(0, 1.0, (N_STEPS, D_MODEL))


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def attend(q, K, V):
    """Standard scaled dot-product attention for a single query vector."""
    scores = (K @ q) / np.sqrt(D_HEAD)
    return softmax(scores) @ V


def generate_no_cache():
    """At step t, rebuild K and V for the ENTIRE prefix from scratch."""
    outputs = []
    projections = 0
    for t in range(N_STEPS):
        prefix = tokens[: t + 1]          # every token seen so far
        K = prefix @ Wk                   # <-- recomputed every single step
        V = prefix @ Wv                   # <-- recomputed every single step
        q = tokens[t] @ Wq
        outputs.append(attend(q, K, V))
        projections += 2 * (t + 1)        # rows pushed through Wk and Wv
    return np.array(outputs), projections


def generate_with_cache():
    """At step t, project ONLY the new token and append to the cache."""
    outputs = []
    projections = 0
    K_cache = np.zeros((N_STEPS, D_HEAD))  # preallocated memo pad
    V_cache = np.zeros((N_STEPS, D_HEAD))
    for t in range(N_STEPS):
        K_cache[t] = tokens[t] @ Wk       # <-- one row of work
        V_cache[t] = tokens[t] @ Wv       # <-- one row of work
        q = tokens[t] @ Wq
        outputs.append(attend(q, K_cache[: t + 1], V_cache[: t + 1]))
        projections += 2
    return np.array(outputs), projections


def main():
    t0 = time.perf_counter()
    out_slow, proj_slow = generate_no_cache()
    slow_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_fast, proj_fast = generate_with_cache()
    fast_s = time.perf_counter() - t0

    max_diff = np.abs(out_slow - out_fast).max()
    np.testing.assert_allclose(out_slow, out_fast, rtol=1e-10, atol=1e-12)
    assert proj_slow == N_STEPS * (N_STEPS + 1)
    assert proj_fast == 2 * N_STEPS

    print("=" * 62)
    print(f"Processed {N_STEPS} synthetic embeddings, d_head={D_HEAD}")
    print("=" * 62)
    print(f"{'':22}{'NO CACHE':>16}{'KV CACHE':>16}")
    print(f"{'wall time (s)':22}{slow_s:>16.4f}{fast_s:>16.4f}")
    print(f"{'K/V row projections':22}{proj_slow:>16,}{proj_fast:>16,}")
    print("-" * 62)
    print(f"speedup                    {slow_s / fast_s:>5.1f}x")
    print(f"redundant work avoided     {proj_slow / proj_fast:>5.1f}x")
    print()
    print(f"max difference in output:  {max_diff:.3e}   <-- numerically identical")
    print()
    print("The no-cache version does O(n^2) projection work to produce")
    print("EXACTLY the same answer the cache gets in O(n). Nothing was")
    print("approximated, distilled, or thrown away. We just stopped")
    print("recomputing values we had already computed.")
    print("This counts K/V PROJECTIONS, not all attention arithmetic.")
    print("Each new query still reads its history. Cumulative full-attention")
    print("decode work remains O(n^2). The toy CPU timing is not LLM speed.")
    print()
    print("Cost of the trick: we now hold K and V for every past token")
    print("in memory. That memory is the KV cache, and on a 6 GB GPU it")
    print("competes with the model weights. See 02_kv_cache_size.py")


if __name__ == "__main__":
    main()

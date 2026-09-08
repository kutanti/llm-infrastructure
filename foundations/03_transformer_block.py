"""Trace an untrained causal decoder block and check that future tokens cannot leak."""

import numpy as np


D_MODEL = 8
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS
VOCAB_SIZE = 12


def layer_norm(x):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)


def position_vectors(length):
    positions = np.arange(length)[:, None]
    frequencies = np.exp(-np.arange(0, D_MODEL, 2) * np.log(10000) / D_MODEL)
    angles = positions * frequencies
    result = np.empty((length, D_MODEL))
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles)
    return result


def attention(x, weights):
    batch, length, _ = x.shape
    heads = []
    for name in ("q", "k", "v"):
        projected = x @ weights[name]
        heads.append(projected.reshape(batch, length, N_HEADS, D_HEAD).transpose(0, 2, 1, 3))
    q, k, v = heads
    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(D_HEAD)
    future = np.triu(np.ones((length, length), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    unnormalized = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities = unnormalized / unnormalized.sum(axis=-1, keepdims=True)
    combined = (probabilities @ v).transpose(0, 2, 1, 3).reshape(batch, length, D_MODEL)
    return combined @ weights["o"], probabilities


def model(ids, weights):
    x = weights["embedding"][ids] + position_vectors(ids.shape[1])
    mixed, probabilities = attention(layer_norm(x), weights)
    x = x + mixed
    hidden = np.maximum(0, layer_norm(x) @ weights["ff1"])
    x = x + hidden @ weights["ff2"]
    logits = layer_norm(x) @ weights["unembed"]
    return logits, probabilities


def main():
    rng = np.random.default_rng(11)
    shapes = {
        "embedding": (VOCAB_SIZE, D_MODEL),
        "q": (D_MODEL, D_MODEL),
        "k": (D_MODEL, D_MODEL),
        "v": (D_MODEL, D_MODEL),
        "o": (D_MODEL, D_MODEL),
        "ff1": (D_MODEL, 3 * D_MODEL),
        "ff2": (3 * D_MODEL, D_MODEL),
        "unembed": (D_MODEL, VOCAB_SIZE),
    }
    weights = {name: rng.normal(0, 0.2, shape) for name, shape in shapes.items()}
    ids = np.array([[1, 4, 2, 6]])
    logits, probabilities = model(ids, weights)
    assert logits.shape == (1, 4, VOCAB_SIZE)
    assert probabilities.shape == (1, N_HEADS, 4, 4)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1)
    future = np.triu(np.ones((4, 4), dtype=bool), k=1)
    np.testing.assert_array_equal(probabilities[:, :, future], 0)

    changed = ids.copy()
    changed[0, -1] = 9
    changed_logits, _ = model(changed, weights)
    np.testing.assert_allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-12)
    assert not np.allclose(logits[:, -1], changed_logits[:, -1])

    print("Token IDs shape:", ids.shape)
    print("Embedding/hidden shape:", (1, 4, D_MODEL))
    print("Per-head Q/K/V shape:", (1, N_HEADS, 4, D_HEAD))
    print("Attention weights shape:", probabilities.shape)
    print("Feed-forward intermediate shape:", (1, 4, 3 * D_MODEL))
    print("Vocabulary logits shape:", logits.shape)
    print("\nHead 0 attention weights (rows=query, columns=key):")
    print(probabilities[0, 0].round(3))
    print("\nChanging the final token leaves earlier logits unchanged.")
    print("All earlier predictions are causal; final-position logits do change.")
    print("\nThis block has random weights, sinusoidal positions, LayerNorm without")
    print("learned scale/bias, and a ReLU FFN. It is not a trained Qwen model.")


if __name__ == "__main__":
    main()

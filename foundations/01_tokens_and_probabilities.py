"""Trace a tiny word tokenizer, embedding lookup, and next-token distribution."""

import numpy as np


VOCAB = ("the", "cat", "sat", "on", "mat")
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


def encode(text):
    tokens = text.split()
    unknown = [token for token in tokens if token not in TOKEN_TO_ID]
    if unknown:
        raise ValueError(f"Words outside this toy vocabulary: {unknown}")
    return np.array([TOKEN_TO_ID[token] for token in tokens], dtype=np.int64)


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def main():
    ids = encode("the cat sat on the")
    embeddings = np.array([
        [0.1, 0.2, -0.1],
        [0.7, -0.3, 0.5],
        [0.0, 0.8, 0.4],
        [-0.4, 0.2, 0.1],
        [0.6, -0.2, 0.3],
    ])
    vectors = embeddings[ids]
    np.testing.assert_array_equal(ids, [0, 1, 2, 3, 0])
    np.testing.assert_array_equal(np.eye(len(VOCAB))[1] @ embeddings, embeddings[1])
    print("Toy word vocabulary:", TOKEN_TO_ID)
    print("Input IDs:", ids)
    print("Embedding shape:", embeddings.shape)
    print("Looked-up sequence shape:", vectors.shape)
    print("The two occurrences of 'the' have identical starting vectors.")
    print(vectors)

    projection = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    projected = vectors @ projection
    assert projected.shape == (5, 2)
    print("\nProject 3 features to 2 with a [3, 2] matrix:")
    print(projected)

    logits = np.array([2.0, 1.0, 0.0])
    print("\nSeparate sampling example: three candidates, logits [2, 1, 0].")
    print("These logits are hand-chosen, not predictions of a trained network.")
    for temperature in (0.5, 1.0, 2.0):
        probabilities = softmax(logits / temperature)
        np.testing.assert_allclose(probabilities.sum(), 1.0)
        print(f"T={temperature:.1f}: {probabilities.round(4)}")
    probabilities = softmax(logits)
    np.testing.assert_allclose(softmax(logits + 1000), probabilities)
    rng = np.random.default_rng(42)
    draws = rng.choice(3, size=10_000, p=probabilities)
    print("Greedy selection:", int(probabilities.argmax()))
    print("Sampling frequencies over 10,000 draws:",
          (np.bincount(draws, minlength=3) / draws.size).round(4))
    print("The sampler selects from the distribution; it does not judge truth.")


if __name__ == "__main__":
    main()

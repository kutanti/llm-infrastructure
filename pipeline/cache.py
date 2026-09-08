"""Inspect online softmax and a bounded, copy-on-write cache-page simulator."""

import json
import math

import numpy as np


def online_attention(query, keys, values, tile_size=16):
    """Attention over an already-allowed prefix, without a full score matrix."""
    q, k, v = (np.asarray(value, dtype=np.float64) for value in (query, keys, values))
    if any(value.ndim != 2 for value in (q, k, v)):
        raise ValueError("Query, keys and values must be matrices")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Attention dimensions do not match")
    if tile_size <= 0 or min(*q.shape, *k.shape, *v.shape) <= 0:
        raise ValueError("Attention requires positive dimensions and tile size")
    if not all(np.isfinite(value).all() for value in (q, k, v)):
        raise ValueError("Attention inputs must be finite")
    maximum = np.full(q.shape[0], -np.inf)
    denominator = np.zeros(q.shape[0])
    numerator = np.zeros((q.shape[0], v.shape[1]))
    for start in range(0, k.shape[0], tile_size):
        scores = q @ k[start:start + tile_size].T / math.sqrt(q.shape[1])
        new_maximum = np.maximum(maximum, scores.max(axis=1))
        old_scale = np.exp(maximum - new_maximum)
        probabilities = np.exp(scores - new_maximum[:, None])
        denominator = old_scale * denominator + probabilities.sum(axis=1)
        numerator = old_scale[:, None] * numerator + probabilities @ v[start:start + tile_size]
        maximum = new_maximum
    return numerator / denominator[:, None]


class PageCache:
    """Token IDs stand in for K/V payloads; this is not a GPU cache backend."""

    def __init__(self, page_size=4, max_pages=8):
        if type(page_size) is not int or type(max_pages) is not int or min(page_size, max_pages) <= 0:
            raise ValueError("Page size and capacity must be positive integers")
        self.page_size = page_size
        self.max_pages = max_pages
        self.pages = {}
        self.references = {}
        self.requests = {}

    def _allocate(self, tokens):
        free = set(range(self.max_pages)) - self.pages.keys()
        if not free:
            raise MemoryError("No free cache pages; admission or preemption is required")
        index = min(free)
        self.pages[index] = list(tokens)
        self.references[index] = 1
        return index

    def create(self, request, tokens):
        if request in self.requests:
            raise ValueError(f"Request already exists: {request}")
        tokens = list(tokens)
        needed = math.ceil(len(tokens) / self.page_size)
        if needed > self.max_pages - len(self.pages):
            raise MemoryError("Request does not fit; no state was allocated")
        self.requests[request] = [
            self._allocate(tokens[start:start + self.page_size])
            for start in range(0, len(tokens), self.page_size)
        ]

    def fork(self, source, target):
        if target in self.requests:
            raise ValueError(f"Request already exists: {target}")
        self.requests[target] = list(self.requests[source])
        for page in self.requests[target]:
            self.references[page] += 1

    def append(self, request, token):
        blocks = self.requests[request]
        if not blocks or len(self.pages[blocks[-1]]) == self.page_size:
            blocks.append(self._allocate([token]))
            return
        last = blocks[-1]
        if self.references[last] > 1:
            copied = self._allocate([*self.pages[last], token])
            self.references[last] -= 1
            blocks[-1] = copied
        else:
            self.pages[last].append(token)

    def release(self, request):
        for page in self.requests.pop(request):
            self.references[page] -= 1
            if self.references[page] == 0:
                del self.references[page]
                del self.pages[page]

    def tokens(self, request):
        return [token for page in self.requests[request] for token in self.pages[page]]

    def snapshot(self):
        return {"page_size": self.page_size, "max_pages": self.max_pages,
                "requests": {key: list(value) for key, value in self.requests.items()},
                "pages": {key: list(value) for key, value in self.pages.items()},
                "references": dict(self.references)}


def main():
    rng = np.random.default_rng(42)
    q, k, v = rng.normal(size=(1, 8)), rng.normal(size=(97, 8)), rng.normal(size=(97, 8))
    scores = q @ k.T / math.sqrt(8)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    expected = weights @ v / weights.sum(axis=1, keepdims=True)
    observed = online_attention(q, k, v, tile_size=16)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    print(f"Online attention max absolute error: {np.max(np.abs(expected - observed)):.3g}")
    cache = PageCache(page_size=2, max_pages=3)
    cache.create("A", [10, 20, 30])
    cache.fork("A", "B")
    print("Shared prefix:", json.dumps(cache.snapshot()))
    cache.append("A", 40)
    print("A diverges, copying the partial page:", json.dumps(cache.snapshot()))
    cache.release("A")
    print("A released, B remains valid:", json.dumps(cache.snapshot()))


if __name__ == "__main__":
    main()

import unittest

import numpy as np

from pipeline.cache import PageCache, online_attention


class CacheTests(unittest.TestCase):
    def test_online_softmax_matches_full_attention(self):
        rng = np.random.default_rng(8)
        q, k, v = rng.normal(size=(3, 8)), rng.normal(size=(19, 8)), rng.normal(size=(19, 5))
        scores = q @ k.T / np.sqrt(8)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        expected = weights @ v / weights.sum(axis=1, keepdims=True)
        for tile_size in (1, 4, 50):
            np.testing.assert_allclose(online_attention(q, k, v, tile_size), expected, atol=1e-12)

    def test_copy_on_write_and_release(self):
        cache = PageCache(page_size=2, max_pages=3)
        cache.create("a", [1, 2, 3])
        cache.fork("a", "b")
        self.assertEqual(len(cache.pages), 2)
        cache.append("a", 4)
        self.assertEqual(cache.tokens("b"), [1, 2, 3])
        self.assertEqual(cache.tokens("a"), [1, 2, 3, 4])
        cache.release("a")
        self.assertEqual(len(cache.pages), 2)
        cache.release("b")
        self.assertEqual(cache.snapshot()["pages"], {})

    def test_out_of_memory_does_not_mutate_shared_prefix(self):
        cache = PageCache(page_size=2, max_pages=1)
        cache.create("a", [1])
        cache.fork("a", "b")
        original = cache.snapshot()
        with self.assertRaises(MemoryError):
            cache.append("a", 2)
        self.assertEqual(cache.snapshot(), original)


if __name__ == "__main__":
    unittest.main()

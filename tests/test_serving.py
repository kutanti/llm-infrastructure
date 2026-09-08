import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pipeline.benchmark import loopback_url, request_one, run_load, summarize
from pipeline.serve import make_server


class FakeGenerator:
    def generate(self, prompt, max_new_tokens):
        return {"prediction": '{"category":"billing"}', "generated_tokens": 5,
                "finish_reason": "eos", "latency_s": 0.001}


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(FakeGenerator(), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_health_and_generation(self):
        with urlopen(self.url + "/health", timeout=3) as response:
            self.assertTrue(json.load(response)["ready"])
        result = request_one(self.url, "local", None, {"id": "a", "prompt": "hello"}, 8, 3,
                             time.perf_counter(), 0)
        self.assertTrue(result["ok"])
        self.assertEqual(result["generated_tokens"], 5)
        self.assertIsNone(result["first_output_s"])

    def test_bad_payload_is_rejected(self):
        request = Request(self.url + "/generate", json.dumps({"prompt": "hello", "max_new_tokens": True}).encode())
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=3)
        self.assertEqual(caught.exception.code, 400)

    def test_busy_slot_is_rejected_and_released(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingGenerator:
            def generate(self, prompt, max_new_tokens):
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Test generation timed out")
                return FakeGenerator().generate(prompt, max_new_tokens)

        server = make_server(BlockingGenerator(), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        first_result = []

        def first():
            first_result.append(request_one(url, "local", None, {"id": "a", "prompt": "hello"},
                                            8, 5, time.perf_counter(), 0))

        worker = threading.Thread(target=first)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=3))
            busy = request_one(url, "local", None, {"id": "b", "prompt": "hello"},
                               8, 3, time.perf_counter(), 1)
            self.assertEqual(busy["status"], 429)
            release.set()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertTrue(first_result[0]["ok"])
            following = request_one(url, "local", None, {"id": "c", "prompt": "hello"},
                                    8, 3, time.perf_counter(), 2)
            self.assertTrue(following["ok"])
        finally:
            release.set()
            worker.join(timeout=5)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_external_urls_are_rejected(self):
        for url in ("https://example.com", "http://127.0.0.1.evil.example",
                    "http://user@localhost", "http://localhost/path", "file:///tmp/data"):
            with self.assertRaises(ValueError):
                loopback_url(url)
        self.assertEqual(loopback_url(self.url + "/"), self.url)

    def test_open_loop_keeps_offered_and_dropped_counts(self):
        def slow_request(example, scheduled, index):
            time.sleep(0.08)
            return {"index": index, "id": example["id"], "ok": True, "latency_s": 0.08,
                    "generated_tokens": 1, "first_output_s": None}
        records, elapsed = run_load([{"id": "a"}], slow_request, 5, 1, "open", 1000)
        summary = summarize(records, elapsed)
        self.assertEqual(summary["offered_requests"], 5)
        self.assertGreater(summary["client_dropped"], 0)
        self.assertEqual(summary["failed_or_dropped"] + summary["completed_successfully"], 5)


if __name__ == "__main__":
    unittest.main()

"""Optional torch tests: python -m unittest tests.test_tiny -v."""

import importlib.util
import json
from pathlib import Path
import shutil
import unittest
import uuid


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from pipeline import tiny


@unittest.skipUnless(HAS_TORCH, "Install the optional pipeline CPU environment to test tiny")
class TinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1] / "results" / ("tiny-test-" + uuid.uuid4().hex)
        self.workspace.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.workspace)
        torch.manual_seed(123)
        self.model_config = tiny.ModelConfig(dim=16, heads=2, layers=1, max_seq_len=64, mlp_ratio=2)

    def make_data(self):
        data = self.workspace / "data"
        data.mkdir()
        for split in ("train", "validation", "test"):
            rows = [{
                "id": f"{split}-{index}", "prompt": f"bill {index}?",
                "response": '{"category":"billing"}', "category": "billing",
                "split": split, "template_id": f"{split}-bill",
            } for index in range(4)]
            (data / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return data

    def config(self, steps=30):
        return tiny.TrainConfig(steps=steps, batch_size=4, learning_rate=0.01,
                                warmup_steps=2, threads=1, eval_every=steps,
                                checkpoint_every=steps, validation_samples=4)

    def test_causal_invariant(self):
        model = tiny.TinyTransformer(self.model_config).eval()
        inputs = torch.randint(tiny.BYTE_OFFSET, tiny.VOCAB_SIZE, (2, 12))
        modified = inputs.clone()
        modified[:, 7:] = torch.randint(tiny.BYTE_OFFSET, tiny.VOCAB_SIZE, (2, 5))
        with torch.inference_mode():
            first, second = model(inputs), model(modified)
        torch.testing.assert_close(first[:, :7], second[:, :7], rtol=0, atol=0)
        self.assertGreater(float((first[:, 7:] - second[:, 7:]).abs().max()), 0)

    def test_assistant_target_mask_and_padding(self):
        x, y = tiny.encode_example("ticket", "{}", 32)
        prefix_length = 1 + len(tiny.encode("ticket")) + 1
        self.assertEqual(x[0], tiny.BOS)
        self.assertEqual(x[prefix_length - 1], tiny.ASSISTANT)
        self.assertTrue((y[:prefix_length - 1] == tiny.IGNORE_INDEX).all())
        self.assertEqual(y[prefix_length - 1:].tolist(), tiny.encode("{}") + [tiny.EOS])
        _, batch_targets = tiny.make_batch([(x, y), tiny.encode_example("a", "{}", 32)], [0, 1])
        self.assertTrue((batch_targets[1, -5:] == tiny.IGNORE_INDEX).all())
        with self.assertRaisesRegex(ValueError, "truncation"):
            tiny.encode_example("too long", "{}", 2)

    def test_byte_roundtrip_and_context(self):
        self.assertEqual(tiny.decode(tiny.encode("ASCII and café")), "ASCII and café")
        generator = tiny.Generator(tiny.TinyTransformer(self.model_config))
        with self.assertRaises(ValueError):
            generator.generate("x" * 64)
        with self.assertRaises(ValueError):
            generator.generate("ok", max_new_tokens=0)
        result = generator.generate("x" * 62, max_new_tokens=3)
        self.assertEqual(result["generated_tokens"], 1)
        self.assertIn(result["finish_reason"], ("eos", "context_length"))
        with torch.no_grad():
            generator.model.head.weight.zero_()
            generator.model.head.bias.zero_()
            generator.model.head.bias[tiny.EOS] = 10
        eos_result = generator.generate("bill?", max_new_tokens=8)
        self.assertEqual(eos_result["prediction"], "")
        self.assertEqual(eos_result["generated_tokens"], 1)
        self.assertEqual(eos_result["finish_reason"], "eos")

    def test_train_loss_save_load_and_int8(self):
        data = self.make_data()
        artifact = self.workspace / "fp32"
        metrics = tiny.train(data, artifact, self.model_config, self.config())
        self.assertLess(metrics["final_validation_loss"], metrics["initial_validation_loss"] * 0.5)
        checkpoint = torch.load(artifact / "checkpoint.pt", weights_only=True)
        model = tiny.TinyTransformer(self.model_config)
        model.load_state_dict(checkpoint["model_state"])
        expected = tiny.Generator(model).generate("bill 1?", 24)
        loaded = tiny.load_generator(artifact)
        actual = loaded.generate("bill 1?", 24)
        for key in ("prediction", "generated_tokens", "finish_reason"):
            self.assertEqual(actual[key], expected[key])
        self.assertGreater(actual["latency_s"], 0)
        self.assertTrue(actual["generated_tokens"] > 0)
        predictions = self.workspace / "train-predictions.jsonl"
        self.assertEqual(tiny.predict(artifact, data, "train", predictions, max_new_tokens=4), 4)
        rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in rows], [f"train-{index}" for index in range(4)])
        self.assertTrue(all(isinstance(row["prediction"], str) for row in rows))
        output = self.workspace / "int8"
        manifest = tiny.quantize(artifact, output)
        self.assertEqual(manifest["precision"], "dynamic-int8")
        self.assertFalse(manifest["quantization"]["gpu_low_bit_kernel"])
        quantized = tiny.load_generator(output)
        int8_modules = [module for module in quantized.model.modules()
                        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)]
        self.assertEqual(len(int8_modules), 5)
        self.assertTrue(all(module.weight().dtype == torch.qint8 for module in int8_modules))
        result = quantized.generate("bill 1?", 24)
        self.assertGreater(result["generated_tokens"], 0)
        reloaded = tiny.load_generator(output).generate("bill 1?", 24)
        self.assertEqual(reloaded["prediction"], result["prediction"])
        with self.assertRaisesRegex(ValueError, "CPU"):
            tiny.load_generator(output, device="cuda")
        with self.assertRaises(FileExistsError):
            tiny.quantize(artifact, output)

    def test_resume_is_exact_and_rejects_changed_data(self):
        data = self.make_data()
        config = self.config(steps=12)
        full = self.workspace / "full"
        partial = self.workspace / "partial"
        resumed = self.workspace / "resumed"
        tiny.train(data, full, self.model_config, config)
        tiny.train(data, partial, self.model_config, config, stop_after=5)
        tiny.train(data, resumed, self.model_config, config, resume=partial)
        uninterrupted = torch.load(full / "model.pt", weights_only=True)["state_dict"]
        restarted = torch.load(resumed / "model.pt", weights_only=True)["state_dict"]
        for key in uninterrupted:
            torch.testing.assert_close(uninterrupted[key], restarted[key], rtol=0, atol=0)
        full_metrics = json.loads((full / "metrics.json").read_text(encoding="utf-8"))
        resumed_metrics = json.loads((resumed / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(full_metrics["history"], resumed_metrics["history"])
        self.assertEqual(full_metrics["initial_validation_loss"], resumed_metrics["initial_validation_loss"])
        with self.assertRaisesRegex(ValueError, "config"):
            tiny.train(data, self.workspace / "wrong-config", self.model_config,
                       self.config(steps=13), resume=partial)
        path = data / "train.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("bill 0?", "bill 9?"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            tiny.train(data, self.workspace / "wrong-data", self.model_config, config, resume=partial)
        self.assertFalse((self.workspace / "wrong-data").exists())


if __name__ == "__main__":
    unittest.main()

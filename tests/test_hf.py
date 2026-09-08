"""Offline tests: random tiny GPT-2, locally constructed tokenizer, no Hub calls."""

import importlib.util
import json
from pathlib import Path
import shutil
import string
import unittest
from unittest.mock import patch
import uuid

from pipeline import hf


DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "transformers", "peft", "tokenizers", "accelerate", "safetensors",
))


class ContractTests(unittest.TestCase):
    def test_hub_requires_immutable_revision(self):
        for revision in (None, "main", "v1.0", "abc123"):
            with self.assertRaisesRegex(ValueError, "immutable"):
                hf.source_identity("example/model", revision)
        sha = "a" * 40
        self.assertEqual(hf.source_identity("example/model", sha)["revision"], sha)

    def test_offline_default_and_explicit_download(self):
        args = hf.parse_args(["inspect", "--revision", "a" * 40])
        self.assertEqual(hf.load_options(args), {
            "revision": "a" * 40, "cache_dir": str(Path("results") / "pipeline" / "hub"),
            "local_files_only": True, "trust_remote_code": False,
        })
        args.allow_download = True
        self.assertFalse(hf.load_options(args)["local_files_only"])

    def test_every_command_supports_explicit_project_cache(self):
        custom_cache = Path("results") / "custom-hub"
        for command in ("inspect", "train", "predict", "export"):
            values = [command, "--cache-dir", str(custom_cache), "--allow-download"]
            if command != "inspect":
                values += ["--device", "cpu", "--output", "unused"]
            if command == "export":
                values += ["--adapter", "unused-adapter"]
            with self.subTest(command=command):
                options = hf.load_options(hf.parse_args(values))
                self.assertEqual(options["cache_dir"], str(custom_cache))
                self.assertFalse(options["local_files_only"])
                self.assertFalse(options["trust_remote_code"])

    def test_predict_accepts_all_dataset_splits(self):
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                args = hf.parse_args([
                    "predict", "--device", "cpu", "--output", "unused.jsonl", "--split", split,
                ])
                self.assertEqual(args.split, split)

    def test_cli_validation(self):
        for extra in (
            ["--learning-rate", "nan"], ["--weight-decay", "-1"],
            ["--max-grad-norm", "0"], ["--lora-dropout", "1"],
            ["--max-steps", "2"], ["--target-modules", "q_proj,"],
        ):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                hf.parse_args(["train", "--device", "cpu", "--output", "unused", *extra])

    def test_full_conversation_boundary_not_prefix_token_count(self):
        class BoundaryTokenizer:
            eos_token_id = 9

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "USER: " if add_generation_prompt else "USER: answer"

            def __call__(self, text, **kwargs):
                self_text = text
                if self_text != "USER: answer":
                    raise AssertionError("Must tokenize the full conversation, not the isolated prefix")
                return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 4), (4, 8), (8, 12)]}

        row = hf.encode_record(BoundaryTokenizer(), {"id": "merged", "prompt": "x", "response": "answer"})
        self.assertEqual(row["labels"], [-100, 2, 3, 9])
        self.assertEqual(row["supervised_start"], 1)
        self.assertTrue(row["boundary_token_crosses_prefix"])
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            hf.encode_record(BoundaryTokenizer(), {"id": "merged", "prompt": "x", "response": "answer"}, 3)

    def test_incompatible_chat_template_rejected(self):
        class BadTemplate:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prefix" if add_generation_prompt else "different conversation"

        with self.assertRaisesRegex(ValueError, "not a prefix"):
            hf.encode_record(BadTemplate(), {"id": "x", "prompt": "x", "response": "x"})

    def test_training_rejects_different_generation_prefix_tokens(self):
        class PrefixTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prefix "

            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2]}

        row = {"id": "crossing", "prompt": "ticket", "response": "{}"}
        with self.assertRaisesRegex(ValueError, "changes prefix tokenization"):
            hf.validate_training_prefix(
                PrefixTokenizer(), row, {"input_ids": [1, 9, 3], "supervised_start": 1}
            )

    def test_template_cannot_drop_response(self):
        class DroppedResponse:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prefix" if add_generation_prompt else "prefix<eos>"

        with self.assertRaisesRegex(ValueError, "complete assistant response"):
            hf.encode_record(DroppedResponse(), {"id": "x", "prompt": "x", "response": "answer"})

    def test_scheduler_preserves_full_schedule(self):
        self.assertEqual([hf.lr_multiplier(i, 2, 4) for i in range(5)], [0, 0.5, 1, 0.5, 0])


@unittest.skipUnless(DEPENDENCIES, "Optional HF dependencies absent; install the documented environment")
class OfflineHFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.root = Path(__file__).resolve().parents[1] / "results" / f"hf-test-{uuid.uuid4().hex}"
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.base = cls.root / "base"
        cls.data = cls.root / "data"
        cls.data.mkdir()
        tokens = ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>"]
        tokens += list(dict.fromkeys(string.ascii_letters + string.digits + string.punctuation + " \n"))
        backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        backend.decoder = decoders.Fuse()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>",
            additional_special_tokens=["<user>", "<assistant>"],
            chat_template=(
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}<user>\n{{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}<assistant>\n{{ message['content'] }}<eos>"
                "{% endif %}{% endfor %}{% if add_generation_prompt %}<assistant>\n{% endif %}"
            ),
        )
        torch.manual_seed(42)
        config = GPT2Config(
            vocab_size=len(cls.tokenizer), n_positions=256, n_ctx=256,
            n_embd=16, n_layer=1, n_head=2, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
            eos_token_id=cls.tokenizer.eos_token_id, bos_token_id=None, pad_token_id=cls.tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(config)
        model.save_pretrained(cls.base, safe_serialization=True)
        cls.tokenizer.save_pretrained(cls.base)
        for split in ("train", "validation", "test"):
            rows = [
                {
                    "id": f"{split}-{i}", "prompt": f"Route ticket {i}: " + "fee " * (i + 1),
                    "response": '{"category":"billing"}', "category": "billing",
                    "split": split, "template_id": f"{split}-template-{i}",
                }
                for i in range(3 if split == "train" else 2)
            ]
            with (cls.data / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)
        shutil.rmtree(cls.root)

    def args(self, command, output=None, extra=()):
        values = [command, "--base", str(self.base)]
        if command != "export":
            values += ["--data-dir", str(self.data), "--max-seq-length", "256"]
        if command != "inspect":
            values += ["--device", "cpu", "--output", str(output)]
        if command == "train":
            values += [
                "--max-steps", "3", "--warmup-steps", "1",
                "--gradient-accumulation", "2", "--microbatch", "2", "--rank", "2",
                "--lora-alpha", "4", "--lora-dropout", "0.1", "--save-every", "3",
                "--eval-every", "3", "--learning-rate", "0.003",
            ]
        return hf.parse_args(values + list(extra))

    def new_output(self, prefix):
        return self.root / f"{prefix}-{uuid.uuid4().hex}"

    def test_assistant_eos_and_pad_masks(self):
        rows = [
            hf.encode_record(self.tokenizer, {"id": str(i), "prompt": "fee " * i, "response": '{"category":"billing"}'})
            for i in (1, 4)
        ]
        for row in rows:
            start = row["supervised_start"]
            self.assertTrue(all(label == -100 for label in row["labels"][:start]))
            self.assertEqual(row["labels"][start:], row["input_ids"][start:])
            self.assertEqual(row["labels"][-1], self.tokenizer.eos_token_id)
            self.assertEqual(
                self.tokenizer.decode(row["labels"][start:], skip_special_tokens=True),
                '{"category":"billing"}',
            )
        old_pad = self.tokenizer.pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            batch = hf.collate(rows, self.tokenizer, self.torch, "cpu")
        finally:
            self.tokenizer.pad_token = old_pad
        length = len(rows[0]["input_ids"])
        self.assertTrue((batch["labels"][0, length:] == -100).all())
        self.assertTrue((batch["attention_mask"][0, length:] == 0).all())
        self.assertEqual(batch["labels"][0, length - 1].item(), self.tokenizer.eos_token_id)

    def test_inspect_reports_all_supervised_spans(self):
        report = hf.inspect(self.args("inspect"))
        self.assertEqual(len(report["records"]), 2)
        self.assertIn('{"category":"billing"}', report["records"][0]["supervised_text"])
        self.assertFalse(report["records"][0]["exceeds_max_seq_length"])

    def test_sample_stream_resume(self):
        stream = hf.SampleStream(3, 17, self.torch)
        stream.take(7)
        resumed = hf.SampleStream(3, 17, self.torch, stream.epoch, stream.cursor)
        self.assertEqual(stream.take(20), resumed.take(20))

    def test_no_cuda_or_quantized_cpu_fallback(self):
        args = self.args("train", self.new_output("unused"), ["--method", "qlora"])
        with self.assertRaisesRegex(ValueError, "CUDA"):
            hf.runtime(args, self.torch)
        args.method, args.device = "lora", "cuda"
        with patch.object(self.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "no CPU fallback"):
                hf.runtime(args, self.torch)

    def test_train_resume_predict_and_merge(self):
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        full_dir, part_dir, resume_dir = [self.new_output(name) for name in ("full", "part", "resume")]
        full = hf.train(self.args("train", full_dir))
        self.assertEqual(full["step"], 3)
        self.assertTrue(full["completed_schedule"])
        self.assertLess(full["trainable_parameters"], full["total_parameters"])
        full_weights = load_file(str(full_dir / "adapter" / "adapter_model.safetensors"))
        self.assertTrue(any("lora_B" in name and weights.abs().sum() > 0 for name, weights in full_weights.items()))
        part = hf.train(self.args("train", part_dir, ["--stop-after-steps", "1"]))
        self.assertFalse(part["completed_schedule"])
        checkpoint = Path(part["checkpoint"])
        state = hf.load_training_state(checkpoint, self.torch)
        self.assertEqual(state["step"], 1)
        self.assertEqual((state["epoch"], state["cursor"]), (1, 1))
        self.assertIn("numpy", state["rng"])
        resumed = hf.train(self.args("train", resume_dir, ["--resume", str(checkpoint)]))
        self.assertTrue(resumed["completed_schedule"])
        resumed_weights = load_file(str(resume_dir / "adapter" / "adapter_model.safetensors"))
        for name in full_weights:
            self.assertTrue(self.torch.equal(full_weights[name], resumed_weights[name]), name)
        full_metrics = [json.loads(line) for line in (full_dir / "metrics.jsonl").read_text().splitlines()]
        resumed_metrics = [json.loads(line) for line in (resume_dir / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in resumed_metrics], [2, 3])
        for before, after in zip(full_metrics[1:], resumed_metrics):
            for key in ("loss", "learning_rate", "grad_norm_before_clip", "epoch", "cursor"):
                self.assertEqual(before[key], after[key], key)

        with self.assertRaises(FileExistsError):
            hf.train(self.args("train", full_dir))
        with self.assertRaisesRegex(ValueError, "identical"):
            hf.train(self.args(
                "train", self.new_output("bad-resume"),
                ["--resume", str(checkpoint), "--learning-rate", "0.004"],
            ))
        original_data = (self.data / "train.jsonl").read_text(encoding="utf-8")
        try:
            (self.data / "train.jsonl").write_text(original_data.replace("Route ticket", "Different ticket"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical"):
                hf.train(self.args("train", self.new_output("changed-data"), ["--resume", str(checkpoint)]))
        finally:
            (self.data / "train.jsonl").write_text(original_data, encoding="utf-8")

        predictions = self.new_output("prediction").with_suffix(".jsonl")
        predict_args = self.args("predict", predictions, [
            "--adapter", str(full_dir / "adapter"), "--max-new-tokens", "4",
        ])
        prediction_summary = hf.predict(predict_args)
        self.assertEqual(prediction_summary["records"], 2)
        rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in rows], ["test-0", "test-1"])
        self.assertTrue(all(1 <= row["generated_tokens"] <= 4 and row["latency_s"] > 0 for row in rows))
        self.assertTrue(all("Route ticket" not in row["prediction"] for row in rows))
        with self.assertRaises(FileExistsError):
            hf.predict(predict_args)

        merged_dir = self.new_output("merged")
        hf.export(self.args("export", merged_dir, ["--adapter", str(full_dir / "adapter")]))
        self.assertTrue((merged_dir / "model.safetensors").exists())
        self.assertFalse((merged_dir / "adapter_config.json").exists())
        merged = AutoModelForCausalLM.from_pretrained(merged_dir, local_files_only=True)
        merged_tokenizer = AutoTokenizer.from_pretrained(merged_dir, local_files_only=True)
        torch, transformers, peft = hf.libraries()
        attached, _ = hf.prediction_model(
            predict_args, torch, transformers, peft, torch.float32, False, hf.source_identity(str(self.base))
        )
        inputs = merged_tokenizer(hf.user_text(merged_tokenizer, "fee"), add_special_tokens=False, return_tensors="pt")
        with torch.inference_mode():
            expected = attached(**inputs).logits
            actual = merged(**inputs).logits
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-5))
        merged_predictions = self.new_output("merged-predictions").with_suffix(".jsonl")
        hf.predict(self.args("predict", merged_predictions, [
            "--base", str(merged_dir), "--max-new-tokens", "4",
        ]))
        merged_rows = [json.loads(line) for line in merged_predictions.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["prediction"] for row in rows], [row["prediction"] for row in merged_rows])
        with patch.object(hf, "load_options", wraps=hf.load_options) as options_spy:
            generator = hf.load_generator(merged_dir, device="cpu")
        self.assertEqual(options_spy.call_count, 2)
        for call in options_spy.call_args_list:
            self.assertFalse(call.args[0].allow_download)
            self.assertEqual(call.args[0].base, str(merged_dir.resolve()))
        served = generator.generate("Route ticket 0: fee ", max_new_tokens=4)
        self.assertEqual(served["prediction"], merged_rows[0]["prediction"])
        self.assertEqual(served["generated_tokens"], merged_rows[0]["generated_tokens"])
        self.assertIn(served["finish_reason"], ("eos", "length"))
        self.assertGreater(served["latency_s"], 0)
        with self.assertRaisesRegex(ValueError, "no prompt truncation"):
            generator.generate("fee " * 200, max_new_tokens=4)
        for invalid_budget in (0, -1, True, 1.5):
            with self.subTest(invalid_budget=invalid_budget):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    generator.generate("fee", invalid_budget)
        with self.assertRaisesRegex(ValueError, "nonempty string"):
            generator.generate("")
        with self.assertRaisesRegex(ValueError, "device must be"):
            hf.load_generator(merged_dir, device="automatic")
        with self.assertRaisesRegex(ValueError, "completed local"):
            hf.load_generator(full_dir / "adapter")

    def test_serving_requires_local_completed_export(self):
        with self.assertRaisesRegex(ValueError, "existing local directory"):
            hf.load_generator(self.root / "missing-model")
        with self.assertRaisesRegex(ValueError, "completed local"):
            hf.load_generator(self.base)

    def test_incomplete_checkpoint_is_rejected(self):
        checkpoint = self.new_output("incomplete")
        checkpoint.mkdir()
        with self.assertRaises(FileNotFoundError):
            hf.load_training_state(checkpoint, self.torch)

    def test_corrupt_safe_checkpoint_has_clear_error(self):
        checkpoint = self.new_output("corrupt")
        checkpoint.mkdir()
        hf.write_json(checkpoint / "completion.json", {"complete": True, "step": 1})
        (checkpoint / "training_state.pt").write_bytes(b"not a tensor checkpoint")
        with self.assertRaisesRegex(ValueError, "safe tensor training checkpoint"):
            hf.load_training_state(checkpoint, self.torch)


if __name__ == "__main__":
    unittest.main()

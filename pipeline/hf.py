"""Offline-by-default causal-LM LoRA/QLoRA lab. See pipeline/HF.md."""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import random
import re
import time


SCHEMA = 1
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MANIFEST = "hf_manifest.json"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def source_identity(base, revision=None):
    """Pin Hub repositories; fingerprint local snapshots including their weights."""
    path = Path(base)
    if path.is_dir():
        files = {}
        for file in sorted(path.rglob("*")):
            if not file.is_file() or any(part in {".git", ".cache"} for part in file.relative_to(path).parts):
                continue
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[file.relative_to(path).as_posix()] = digest.hexdigest()
        if not files:
            raise ValueError(f"Local base is empty: {path}")
        return {"kind": "local", "path": str(path.resolve()), "sha256": digest_json(files)}
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", base):
        raise ValueError("--base must be an existing local directory or an owner/repository Hub ID")
    if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Hub bases require an immutable 40-character --revision commit SHA, even offline")
    return {"kind": "hub", "repo": base, "revision": revision.lower()}


def load_options(args):
    return {
        "revision": args.revision,
        "cache_dir": str(args.cache_dir),
        "local_files_only": not args.allow_download,
        "trust_remote_code": False,
    }


def libraries():
    try:
        import torch
        import transformers
        import peft
    except ImportError as error:
        raise ImportError(
            "HF commands require PyTorch and pipeline\\requirements-hf.txt; "
            "install the documented CPU or CUDA environment explicitly."
        ) from error
    return torch, transformers, peft


def runtime(args, torch):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable; no CPU fallback is performed")
    quantized = getattr(args, "load_in_4bit", False) or getattr(args, "method", "lora") == "qlora"
    if quantized:
        if args.device != "cuda":
            raise ValueError("NF4/QLoRA requires an explicitly selected CUDA device")
        try:
            importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as error:
            raise ImportError("NF4 requires pipeline\\requirements-qlora.txt (bitsandbytes)") from error
    name = args.dtype or ("bfloat16" if args.device == "cuda" else "float32")
    if args.device == "cpu" and name != "float32":
        raise ValueError("This lab supports CPU execution in float32 only")
    if name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 requested but the selected CUDA device does not support it")
    return getattr(torch, name), name, quantized


def load_tokenizer(args, transformers, directory=None):
    options = {"local_files_only": True, "trust_remote_code": False} if directory else load_options(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(directory or args.base), use_fast=True, **options
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact assistant-boundary offset mapping")
    if not tokenizer.chat_template:
        raise ValueError("The tokenizer must provide a chat_template; no guessed prompt format is used")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def user_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def encode_record(tokenizer, record, max_seq_length=None):
    """Mask from offsets in the *full* conversation, not a separately tokenized prefix."""
    prefix = user_text(tokenizer, record["prompt"])
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["prompt"]},
         {"role": "assistant", "content": record["response"]}],
        tokenize=False, add_generation_prompt=False,
    )
    if not full.startswith(prefix):
        raise ValueError("Chat template's generation prompt is not a prefix of the full conversation")
    if not record["response"] or not full.startswith(prefix + record["response"]):
        raise ValueError(f"{record['id']}: chat template must preserve the complete assistant response")
    boundary = len(prefix)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    candidates = [i for i, (_, end) in enumerate(offsets) if end > boundary]
    if not candidates:
        raise ValueError(f"{record['id']}: chat template produced no assistant supervision")
    start = candidates[0]
    # A token can include both trailing role-prefix whitespace and response text.
    # Such a token belongs to the assistant span; prefix-token counts would miss it.
    if start < 1:
        raise ValueError(f"{record['id']}: assistant span needs at least one preceding context token")
    if tokenizer.eos_token_id not in ids[start:]:
        ids.append(tokenizer.eos_token_id)
    labels = [-100] * start + ids[start:]
    if max_seq_length is not None and len(ids) > max_seq_length:
        raise ValueError(
            f"{record['id']}: {len(ids)} tokens exceed --max-seq-length {max_seq_length}; "
            "refusing to truncate the prompt or assistant supervision"
        )
    return {
        "id": record["id"], "input_ids": ids, "labels": labels,
        "supervised_start": start, "supervised_end": len(ids),
        "boundary_token_crosses_prefix": offsets[start][0] < boundary,
    }


def collate(rows, tokenizer, torch, device):
    width = max(len(row["input_ids"]) for row in rows)
    ids, labels, attention = [], [], []
    for row in rows:
        length = len(row["input_ids"])
        ids.append(row["input_ids"] + [tokenizer.pad_token_id] * (width - length))
        labels.append(row["labels"] + [-100] * (width - length))
        attention.append([1] * length + [0] * (width - length))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


def validate_training_prefix(tokenizer, record, encoded):
    prefix_ids = tokenizer(user_text(tokenizer, record["prompt"]), add_special_tokens=False)["input_ids"]
    if prefix_ids != encoded["input_ids"][:encoded["supervised_start"]]:
        raise ValueError(
            f"{record['id']}: assistant boundary changes prefix tokenization; "
            "choose a compatible tokenizer/chat template rather than training on a different generation prefix"
        )


def load_base(args, torch, transformers, dtype, quantized):
    options = load_options(args)
    options.update(dtype=dtype, attn_implementation="eager", weights_only=True)
    if quantized:
        options["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        options["device_map"] = {"": torch.cuda.current_device()}
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base, **options)
    if not quantized:
        model.to(args.device)
    return model


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def memory_stats(torch, device):
    if device == "cpu":
        return {"cuda_max_allocated_bytes": None, "cuda_max_reserved_bytes": None}
    return {
        "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def autocast(torch, device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device == "cuda" and dtype == torch.bfloat16 else nullcontext()


def read_manifest(directory):
    with (Path(directory) / MANIFEST).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != SCHEMA or value.get("kind") not in {"adapter", "checkpoint"}:
        raise ValueError(f"{directory}: not a supported lab adapter/checkpoint")
    return value


def validate_adapter(directory, identity):
    value = read_manifest(directory)
    if value["identity"]["base"] != identity:
        raise ValueError("Adapter base identity differs from --base/--revision or local snapshot contents")
    return value


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers")
    }


def training_identity(args, base, splits, dtype_name):
    return {
        "base": base,
        "data_sha256": digest_json(splits),
        "versions": package_versions(),
        "hyperparameters": {
            key: getattr(args, key) for key in (
                "method", "device", "max_seq_length", "microbatch", "gradient_accumulation",
                "max_steps", "learning_rate", "weight_decay", "warmup_steps",
                "max_grad_norm", "rank", "lora_alpha", "lora_dropout", "target_modules",
                "gradient_checkpointing", "seed",
            )
        } | {"dtype": dtype_name},
    }


def capture_rng(torch, device):
    import numpy as np
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0], "keys": torch.tensor(numpy_state[1].astype("int64")),
            "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
    }


def restore_rng(torch, state):
    import numpy as np
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((
        ns["algorithm"], ns["keys"].numpy().astype("uint32"),
        ns["position"], ns["has_gauss"], ns["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG topology differs from the current environment")
        torch.cuda.set_rng_state_all(state["cuda"])


class SampleStream:
    """Deterministic shuffled epochs; cursor is saved only at optimizer boundaries."""

    def __init__(self, size, seed, torch, epoch=0, cursor=0):
        if size < 1 or epoch < 0 or not 0 <= cursor < size:
            raise ValueError("Invalid sample-stream checkpoint")
        self.size, self.seed, self.torch = size, seed, torch
        self.epoch, self.cursor = epoch, cursor
        self.order = self._order()

    def _order(self):
        generator = self.torch.Generator().manual_seed(self.seed + self.epoch)
        return self.torch.randperm(self.size, generator=generator).tolist()

    def take(self, count):
        result = []
        for _ in range(count):
            result.append(self.order[self.cursor])
            self.cursor += 1
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.order = self._order()
        return result


def lr_multiplier(step, warmup, total):
    if step < warmup:
        return float(step) / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def save_adapter(directory, model, tokenizer, manifest):
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    write_json(directory / MANIFEST, manifest)


def save_checkpoint(output, model, tokenizer, optimizer, scheduler, stream, step, identity, torch):
    directory = output / f"checkpoint-{step:08d}"
    save_adapter(directory, model, tokenizer, {
        "schema_version": SCHEMA, "kind": "checkpoint", "identity": identity, "step": step,
    })
    # completion.json is written last: an interrupted write is never resumable.
    torch.save({
        "schema_version": SCHEMA, "step": step, "epoch": stream.epoch, "cursor": stream.cursor,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng(torch, identity["hyperparameters"]["device"]),
    }, directory / "training_state.pt")
    write_json(directory / "completion.json", {"step": step, "complete": True})
    return directory


def load_training_state(directory, torch):
    directory = Path(directory)
    with (directory / "completion.json").open(encoding="utf-8") as handle:
        completion = json.load(handle)
    if completion.get("complete") is not True:
        raise ValueError("Checkpoint is incomplete")
    try:
        state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Cannot load safe tensor training checkpoint: {directory}") from error
    if state.get("schema_version") != SCHEMA or state.get("step") != completion.get("step"):
        raise ValueError("Checkpoint state/completion marker mismatch")
    return state


def validation_loss(model, rows, tokenizer, args, torch, dtype):
    model.eval()
    numerator, denominator = 0.0, 0
    with torch.inference_mode():
        for begin in range(0, len(rows), args.microbatch):
            batch = collate(rows[begin:begin + args.microbatch], tokenizer, torch, args.device)
            with autocast(torch, args.device, dtype):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            numerator += loss.item() * count
            denominator += count
    model.train()
    return numerator / denominator


def inspect(args):
    from pipeline.data import load_split
    _, transformers, _ = libraries()
    base = source_identity(args.base, args.revision)
    tokenizer = load_tokenizer(args, transformers)
    rows = load_split(args.data_dir, args.split)
    report = []
    for row in rows[:args.limit] if args.limit else rows:
        encoded = encode_record(tokenizer, row)
        start, end = encoded["supervised_start"], encoded["supervised_end"]
        report.append({
            "id": row["id"], "tokens": end, "prompt_tokens": start,
            "supervised_tokens": end - start, "supervised_span": [start, end],
            "supervised_token_ids": encoded["input_ids"][start:end],
            "supervised_text": tokenizer.decode(encoded["input_ids"][start:end], skip_special_tokens=False),
            "boundary_token_crosses_prefix": encoded["boundary_token_crosses_prefix"],
            "exceeds_max_seq_length": end > args.max_seq_length,
        })
    return {"base": base, "split": args.split, "records": report, "span_convention": "[start, end)"}


def train(args):
    from pipeline.data import load_split
    torch, transformers, peft = libraries()
    dtype, dtype_name, quantized = runtime(args, torch)
    base = source_identity(args.base, args.revision)
    splits = {split: load_split(args.data_dir, split) for split in ("train", "validation")}
    identity = training_identity(args, base, splits, dtype_name)
    state = None
    if args.resume:
        manifest = validate_adapter(args.resume, base)
        if manifest["kind"] != "checkpoint" or manifest["identity"] != identity:
            raise ValueError("Resume requires identical base, data, package versions, and training hyperparameters")
        state = load_training_state(args.resume, torch)
        if state["step"] != manifest["step"] or not 0 <= state["step"] < args.max_steps:
            raise ValueError("Checkpoint step is invalid or this training schedule is already complete")
    if args.stop_after_steps is not None and args.stop_after_steps <= (state["step"] if state else 0):
        raise ValueError("--stop-after-steps must exceed the resumed optimizer step")
    tokenizer = load_tokenizer(args, transformers, args.resume)
    encoded = {
        split: [encode_record(tokenizer, row, args.max_seq_length) for row in rows]
        for split, rows in splits.items()
    }
    for split, records in splits.items():
        for record, tokens in zip(records, encoded[split]):
            validate_training_prefix(tokenizer, record, tokens)
    # Never reuse a run directory, even on resume. Checkpoints remain immutable.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "run.json", {
        "schema_version": SCHEMA, "identity": identity,
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
    })
    transformers.set_seed(args.seed)
    model = load_base(args, torch, transformers, dtype, quantized)
    context_limit = getattr(model.config, "max_position_embeddings", args.max_seq_length)
    if any(len(row["input_ids"]) > context_limit for rows in encoded.values() for row in rows):
        raise ValueError(f"An encoded record exceeds the model's context limit ({context_limit})")
    model.config.use_cache = False
    if quantized:
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.requires_grad_(False)
    if args.resume:
        model = peft.PeftModel.from_pretrained(model, str(args.resume), is_trainable=True)
    else:
        config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            target_modules="all-linear" if args.target_modules == "all-linear" else args.target_modules.split(","),
        )
        model = peft.get_peft_model(model, config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("LoRA configuration has no trainable parameters")
    if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
        raise ValueError("Unexpected trainable base parameter: this lab trains LoRA adapters only")
    counts = {
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, args.warmup_steps, args.max_steps)
    )
    step = 0
    stream = SampleStream(len(encoded["train"]), args.seed, torch)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        stream = SampleStream(len(encoded["train"]), args.seed, torch, state["epoch"], state["cursor"])
        restore_rng(torch, state["rng"])
    model.train()
    end_step = min(args.max_steps, args.stop_after_steps or args.max_steps)
    synchronize(torch, args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    last_checkpoint = None
    with (args.output / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
        while step < end_step:
            synchronize(torch, args.device)
            update_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss, supervised_tokens = 0.0, 0
            used_lr = optimizer.param_groups[0]["lr"]
            for _ in range(args.gradient_accumulation):
                indices = stream.take(args.microbatch)
                batch = collate([encoded["train"][i] for i in indices], tokenizer, torch, args.device)
                with autocast(torch, args.device, dtype):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss at optimizer step {step + 1}")
                total_loss += loss.item() / args.gradient_accumulation
                supervised_tokens += int((batch["labels"][:, 1:] != -100).sum().item())
                (loss / args.gradient_accumulation).backward()
            if not any(p.grad is not None for p in parameters):
                raise ValueError("No adapter gradients were produced")
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError(f"Non-finite adapter gradient at optimizer step {step + 1}")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            step += 1
            synchronize(torch, args.device)
            entry = {
                "step": step, "loss": total_loss, "learning_rate": used_lr,
                "grad_norm_before_clip": float(grad_norm),
                "supervised_tokens": supervised_tokens,
                "optimizer_step_s": time.perf_counter() - update_start,
                "epoch": stream.epoch, "cursor": stream.cursor,
            }
            if step % args.eval_every == 0 or step == end_step:
                # Validation must not perturb the resumed dropout/RNG stream.
                rng = capture_rng(torch, args.device)
                entry["validation_loss"] = validation_loss(
                    model, encoded["validation"], tokenizer, args, torch, dtype
                )
                restore_rng(torch, rng)
            metrics.write(json.dumps(entry, allow_nan=False) + "\n")
            metrics.flush()
            if step % args.save_every == 0 or step == end_step:
                last_checkpoint = save_checkpoint(
                    args.output, model, tokenizer, optimizer, scheduler, stream, step, identity, torch
                )
    synchronize(torch, args.device)
    elapsed = time.perf_counter() - started
    save_adapter(args.output / "adapter", model, tokenizer, {
        "schema_version": SCHEMA, "kind": "adapter", "identity": identity, "step": step,
        "completed_schedule": step == args.max_steps,
    })
    summary = {
        **counts, "device": args.device, "dtype": dtype_name, "method": args.method,
        "step": step, "completed_schedule": step == args.max_steps,
        "training_wall_s": elapsed, **memory_stats(torch, args.device),
        "adapter": str(args.output / "adapter"), "checkpoint": str(last_checkpoint),
        "timing_scope": "optimizer loop including validation and checkpoints; excludes model loading and final adapter save",
    }
    write_json(args.output / "summary.json", summary)
    return summary


def prediction_model(args, torch, transformers, peft, dtype, quantized, base):
    if args.adapter:
        validate_adapter(args.adapter, base)
    tokenizer = load_tokenizer(args, transformers, args.adapter)
    model = load_base(args, torch, transformers, dtype, quantized)
    if args.adapter:
        model = peft.PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


class HFGenerator:
    """Shared greedy generation for batch prediction and local merged-model serving."""

    def __init__(self, model, tokenizer, torch, device, max_seq_length=512):
        self.model, self.tokenizer, self.torch = model, tokenizer, torch
        self.device = device
        self.context_limit = min(
            max_seq_length, getattr(model.config, "max_position_embeddings", max_seq_length)
        )
        eos = model.generation_config.eos_token_id
        self.eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [
            eos if eos is not None else tokenizer.eos_token_id
        ]

    def encode_prompt(self, prompt, max_new_tokens):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        ids = self.tokenizer(
            user_text(self.tokenizer, prompt), add_special_tokens=False
        )["input_ids"]
        if not ids or len(ids) + max_new_tokens > self.context_limit:
            raise ValueError(
                f"Prompt ({len(ids)}) + generation ({max_new_tokens}) exceeds context budget "
                f"{self.context_limit}; no prompt truncation is performed"
            )
        return ids

    def _generate_ids(self, ids, max_new_tokens):
        torch = self.torch
        with torch.inference_mode():
            inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
            synchronize(torch, self.device)
            start = time.perf_counter()
            generated = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                eos_token_id=self.eos_ids, pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
                num_return_sequences=1, return_dict_in_generate=False,
            )
            synchronize(torch, self.device)
            latency = time.perf_counter() - start
            new_ids = generated[0, inputs.shape[1]:].tolist()
        completed = bool(new_ids and new_ids[-1] in self.eos_ids)
        return {
            "prediction": self.tokenizer.decode(new_ids, skip_special_tokens=True),
            "latency_s": latency, "generated_tokens": len(new_ids),
            "completed": completed, "finish_reason": "eos" if completed else "length",
        }

    def generate(self, prompt, max_new_tokens=48):
        return self._generate_ids(self.encode_prompt(prompt, max_new_tokens), max_new_tokens)


def load_generator(artifact: Path, device="cpu"):
    """Load only a completed local merged export; never resolve or download a Hub base."""
    artifact = Path(artifact).resolve()
    if not artifact.is_dir():
        raise ValueError(f"Merged-model artifact must be an existing local directory: {artifact}")
    try:
        with (artifact / MANIFEST).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        with (artifact / "completion.json").open(encoding="utf-8") as stream:
            completion = json.load(stream)
    except FileNotFoundError as error:
        raise ValueError("Serving requires a completed local `pipeline.hf export` artifact") from error
    if (
        manifest.get("schema_version") != SCHEMA or manifest.get("kind") != "merged-model"
        or manifest.get("quantized") is not False or completion.get("complete") is not True
    ):
        raise ValueError("Serving requires a completed, nonquantized local merged-model export")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    args = argparse.Namespace(
        base=str(artifact), revision=None, allow_download=False,
        cache_dir=Path("results") / "pipeline" / "hub",
        device=device, dtype=None, load_in_4bit=False, adapter=None,
    )
    torch, transformers, peft = libraries()
    dtype, _, quantized = runtime(args, torch)
    model, tokenizer = prediction_model(args, torch, transformers, peft, dtype, quantized, None)
    return HFGenerator(model, tokenizer, torch, device)


def predict(args):
    from pipeline.data import dataset_digest, load_split
    torch, transformers, peft = libraries()
    dtype, dtype_name, quantized = runtime(args, torch)
    base = source_identity(args.base, args.revision)
    records = load_split(args.data_dir, args.split)
    if args.limit:
        records = records[:args.limit]
    metadata_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    for path in (args.output, metadata_path):
        if path.exists():
            raise FileExistsError(path)
    model, tokenizer = prediction_model(args, torch, transformers, peft, dtype, quantized, base)
    generator = HFGenerator(model, tokenizer, torch, args.device, args.max_seq_length)
    prompts = []
    for record in records:
        try:
            prompts.append(generator.encode_prompt(record["prompt"], args.max_new_tokens))
        except ValueError as error:
            raise ValueError(f"{record['id']}: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    synchronize(torch, args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    total_time, total_tokens = 0.0, 0
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        for record, ids in zip(records, prompts):
            row = {"id": record["id"], **generator._generate_ids(ids, args.max_new_tokens)}
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            total_time += row["latency_s"]
            total_tokens += row["generated_tokens"]
    summary = {
        "output": str(args.output), "records": len(records), "device": args.device, "dtype": dtype_name,
        "load_in_4bit": quantized, "generation_s": total_time, "generated_tokens": total_tokens,
        "generated_tokens_per_s": total_tokens / total_time if total_time else None,
        **memory_stats(torch, args.device),
        "timing_scope": "per-record generate only, CUDA synchronized; first request is cold; no warmup",
    }
    write_json(metadata_path, {
        "schema_version": SCHEMA, "backend": "hf",
        "base": base, "adapter": source_identity(str(args.adapter)) if args.adapter else None,
        "dataset_sha256": dataset_digest(args.data_dir), "split": args.split,
        "max_new_tokens": args.max_new_tokens, "max_seq_length": args.max_seq_length,
        "sampling": "greedy", "versions": package_versions(), "summary": summary,
    })
    return summary


def export(args):
    torch, transformers, peft = libraries()
    dtype, dtype_name, quantized = runtime(args, torch)
    if quantized:
        raise ValueError("Export requires a non-quantized base")
    base = source_identity(args.base, args.revision)
    manifest = validate_adapter(args.adapter, base)
    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = prediction_model(args, torch, transformers, peft, dtype, False, base)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    result = {
        "schema_version": SCHEMA, "kind": "merged-model", "base": base,
        "source_adapter": str(args.adapter.resolve()), "source_manifest": manifest,
        "dtype": dtype_name, "quantized": False, "versions": package_versions(),
    }
    write_json(args.output / MANIFEST, result)
    write_json(args.output / "completion.json", {"complete": True})
    return {"output": str(args.output), "dtype": dtype_name, "quantized": False}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "train", "predict", "export"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--base", default=DEFAULT_BASE)
        sub.add_argument("--revision", help="Immutable Hub commit SHA; required for Hub IDs")
        sub.add_argument("--allow-download", action="store_true", help="Opt in to Hub access")
        sub.add_argument(
            "--cache-dir", type=Path, default=Path("results") / "pipeline" / "hub",
            help="Model/tokenizer Hub cache (default: results\\pipeline\\hub)",
        )
        if command != "inspect":
            sub.add_argument("--device", choices=("cpu", "cuda"), required=True)
            sub.add_argument("--dtype", choices=("float32", "bfloat16"), default=None)
            sub.add_argument("--output", type=Path, required=True, help="New path; never overwrite")
        if command != "export":
            sub.add_argument("--data-dir", "--data", type=Path, default=Path("results") / "pipeline" / "data")
            sub.add_argument("--max-seq-length", type=positive_int, default=512)
        if command in {"inspect", "predict"}:
            sub.add_argument("--split", choices=("train", "validation", "test"), default="test")
            sub.add_argument("--limit", type=positive_int, help="Omit to process the entire split")
        if command == "train":
            sub.add_argument("--method", choices=("lora", "qlora"), default="lora")
            sub.add_argument("--microbatch", type=positive_int, default=1)
            sub.add_argument("--gradient-accumulation", type=positive_int, default=4)
            sub.add_argument("--max-steps", type=positive_int, default=100)
            sub.add_argument("--stop-after-steps", type=positive_int, help="Stop early, preserving the original LR schedule")
            sub.add_argument("--learning-rate", type=float, default=2e-4)
            sub.add_argument("--weight-decay", type=float, default=0.01)
            sub.add_argument("--warmup-steps", type=int, default=5)
            sub.add_argument("--max-grad-norm", type=float, default=1.0)
            sub.add_argument("--rank", type=positive_int, default=8)
            sub.add_argument("--lora-alpha", type=positive_int, default=16)
            sub.add_argument("--lora-dropout", type=float, default=0.05)
            sub.add_argument("--target-modules", default="all-linear", help="all-linear or comma-separated module suffixes")
            sub.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
            sub.add_argument("--seed", type=int, default=17)
            sub.add_argument("--save-every", type=positive_int, default=10)
            sub.add_argument("--eval-every", type=positive_int, default=10)
            sub.add_argument("--resume", type=Path, help="Completed checkpoint directory; --output must still be NEW")
        if command in {"predict", "export"}:
            sub.add_argument("--adapter", type=Path, required=command == "export")
        if command == "predict":
            sub.add_argument("--load-in-4bit", action="store_true")
            sub.add_argument("--max-new-tokens", type=positive_int, default=32)
        sub.set_defaults(handler=globals()[command])
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        for key in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
                parser.error(f"--{key.replace('_', '-')} must be finite and positive")
        if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            parser.error("--weight-decay must be finite and nonnegative")
        if not 0 <= args.lora_dropout < 1:
            parser.error("--lora-dropout must be in [0, 1)")
        if not 0 <= args.warmup_steps < args.max_steps:
            parser.error("--warmup-steps must be nonnegative and less than --max-steps")
        if args.seed < 0 or args.seed >= 2**32:
            parser.error("--seed must be in [0, 2**32)")
        if not all(part.strip() == part and part for part in args.target_modules.split(",")):
            parser.error("--target-modules must be all-linear or nonempty comma-separated suffixes")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

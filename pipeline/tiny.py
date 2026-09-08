"""A byte-tokenized causal transformer trained from random weights, without downloads."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


PAD, BOS, ASSISTANT, EOS = 0, 1, 2, 3
BYTE_OFFSET, VOCAB_SIZE, IGNORE_INDEX = 4, 260, -100
FORMAT_VERSION = 1
TOKENIZER = {
    "type": "utf8-bytes",
    "pad_id": PAD,
    "bos_id": BOS,
    "assistant_id": ASSISTANT,
    "eos_id": EOS,
    "byte_offset": BYTE_OFFSET,
    "vocab_size": VOCAB_SIZE,
}


def encode(text):
    return [value + BYTE_OFFSET for value in text.encode("utf-8")]


def decode(tokens):
    return bytes(token - BYTE_OFFSET for token in tokens
                 if BYTE_OFFSET <= token < VOCAB_SIZE).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 64
    heads: int = 4
    layers: int = 2
    max_seq_len: int = 256
    mlp_ratio: int = 4

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Model dimensions must be positive integers")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 400
    batch_size: int = 16
    learning_rate: float = 0.003
    warmup_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    threads: int = 2
    eval_every: int = 100
    checkpoint_every: int = 100
    validation_samples: int = 36

    def __post_init__(self):
        for field in ("steps", "batch_size", "threads", "eval_every",
                      "checkpoint_every", "validation_samples"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError("grad_clip must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (part.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for part in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(attended.transpose(1, 2).contiguous().reshape(batch, length, dim))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.dim)
        self.attention = CausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.dim * config.mlp_ratio, config.dim),
        )

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, VOCAB_SIZE)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.max_seq_len:
            raise ValueError("Expected [batch, length] tokens within the configured context")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def encode_example(prompt, response, max_seq_len):
    prefix = [BOS] + encode(prompt) + [ASSISTANT]
    tokens = prefix + encode(response) + [EOS]
    if len(tokens) - 1 > max_seq_len:
        raise ValueError(f"Example requires {len(tokens) - 1} input tokens; "
                         f"max_seq_len={max_seq_len}. Refusing silent truncation.")
    inputs, targets = tokens[:-1], tokens[1:]
    # The ASSISTANT input predicts the first response byte; prompt targets never contribute.
    targets[:len(prefix) - 1] = [IGNORE_INDEX] * (len(prefix) - 1)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def make_batch(examples, indices):
    selected = [examples[int(index)] for index in indices]
    length = max(len(inputs) for inputs, _ in selected)
    inputs = torch.full((len(selected), length), PAD, dtype=torch.long)
    targets = torch.full_like(inputs, IGNORE_INDEX)
    for row, (x, y) in enumerate(selected):
        inputs[row, :len(x)] = x
        targets[row, :len(y)] = y
    return inputs, targets


def supervised_loss(logits, targets):
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite supervised loss")
    return loss


def evaluate_loss(model, examples, batch_size):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            inputs, targets = make_batch(examples, range(start, min(start + batch_size, len(examples))))
            tokens = int((targets != IGNORE_INDEX).sum())
            total += float(supervised_loss(model(inputs), targets)) * tokens
            count += tokens
    model.train(was_training)
    return total / count


def learning_rate_at(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_atomic(value, path):
    path = Path(path)
    pending = path.with_name(path.name + ".part")
    torch.save(value, pending)
    os.replace(pending, path)


def _data_signature(train_rows, validation_rows):
    # Include IDs, ordering, template provenance, prompts, and targets in the resume identity.
    raw = json.dumps({"train": train_rows, "validation": validation_rows},
                     sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _manifest(model_config, precision, **extra):
    return {
        "format_version": FORMAT_VERSION,
        "backend": "tiny",
        "architecture": "tiny-byte-causal-transformer",
        "task": "synthetic-ticket-routing",
        "model_config": asdict(model_config),
        "tokenizer": TOKENIZER,
        "precision": precision,
        "torch_version": str(torch.__version__),
        "compatible_with_hf_base": False,
        **extra,
    }


def train(data_dir, output, model_config=None, train_config=None, resume=None, stop_after=None):
    """Train on CPU. Resume into a NEW output directory with the same total-step schedule."""
    from pipeline.data import load_split

    model_config = model_config or ModelConfig()
    config = train_config or TrainConfig()
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= config.steps):
        raise ValueError("stop_after must be between 1 and configured total steps")
    train_rows = load_split(data_dir, "train")
    validation_rows = load_split(data_dir, "validation")
    data_sha256 = _data_signature(train_rows, validation_rows)
    train_examples = [encode_example(row["prompt"], row["response"], model_config.max_seq_len)
                      for row in train_rows]
    # Evenly spaced selection avoids taking only the first category in sorted data.
    validation_indices = sorted(set(
        torch.linspace(0, len(validation_rows) - 1,
                       min(config.validation_samples, len(validation_rows))).long().tolist()
    ))
    validation_examples = [
        encode_example(validation_rows[index]["prompt"], validation_rows[index]["response"],
                       model_config.max_seq_len) for index in validation_indices
    ]
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    model = TinyTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    batch_rng = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    history, validation_history, start_step = [], [], 0
    initial_loss = None
    if resume is not None:
        source = Path(resume)
        checkpoint = torch.load(source / "checkpoint.pt" if source.is_dir() else source,
                                map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format")
        if checkpoint["model_config"] != asdict(model_config) or checkpoint["train_config"] != asdict(config):
            raise ValueError("Resume requires identical model/train config, including total steps")
        if checkpoint["data_sha256"] != data_sha256:
            raise ValueError("Resume dataset digest differs")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        batch_rng.set_state(checkpoint["batch_rng_state"])
        start_step = checkpoint["step"]
        history = checkpoint["history"]
        validation_history = checkpoint["validation_history"]
        initial_loss = checkpoint["initial_validation_loss"]
    end_step = config.steps if stop_after is None else stop_after
    if start_step >= end_step:
        raise ValueError("Resume checkpoint has already reached the requested stopping step")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if initial_loss is None:
        initial_loss = evaluate_loss(model, validation_examples, config.batch_size)
        validation_history.append({"step": 0, "loss": initial_loss})
    start_time = time.perf_counter()
    model.train()

    def save_checkpoint(step):
        _save_atomic({
            "format_version": FORMAT_VERSION,
            "step": step,
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "data_sha256": data_sha256,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "batch_rng_state": batch_rng.get_state(),
            "initial_validation_loss": initial_loss,
            "history": history,
            "validation_history": validation_history,
        }, output / "checkpoint.pt")

    for step in range(start_step, end_step):
        indices = torch.randint(len(train_examples), (config.batch_size,), generator=batch_rng)
        inputs, targets = make_batch(train_examples, indices)
        lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model(inputs), targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                                  error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise RuntimeError(f"Nonfinite model parameter after step {step + 1}")
        history.append({"step": step + 1, "loss": float(loss.detach()),
                        "learning_rate": lr, "grad_norm_before_clip": float(grad_norm)})
        if (step + 1) % config.eval_every == 0:
            value = evaluate_loss(model, validation_examples, config.batch_size)
            validation_history.append({"step": step + 1, "loss": value})
            print(f"step={step + 1} train_loss={float(loss.detach()):.4f} "
                  f"validation_loss={value:.4f}", flush=True)
        if (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(step + 1)
    final_loss = evaluate_loss(model, validation_examples, config.batch_size)
    if validation_history[-1]["step"] != end_step:
        validation_history.append({"step": end_step, "loss": final_loss})
    save_checkpoint(end_step)
    model.eval()
    _save_atomic({"format_version": FORMAT_VERSION, "state_dict": model.state_dict()}, output / "model.pt")
    metrics = {
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "validation_ids": [validation_rows[index]["id"] for index in validation_indices],
        "validation_loss_definition": "Mean cross entropy over response bytes and EOS only",
        "history": history,
        "validation_history": validation_history,
        "completed_steps": end_step,
        "resumed_from_step": start_step,
        "training_complete": end_step == config.steps,
        "elapsed_s_this_run": time.perf_counter() - start_time,
        "model_config": asdict(model_config),
        "train_config": asdict(config),
        "data_sha256": data_sha256,
        "data_dir": str(Path(data_dir).resolve()),
        "device": "cpu",
    }
    _json_write(output / "metrics.json", metrics)
    _json_write(output / "manifest.json", _manifest(
        model_config, "fp32", device="cpu", data_sha256=data_sha256,
        completed_steps=end_step, training_complete=end_step == config.steps,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    ))
    return metrics


def _quantization_engine(requested=None):
    supported = torch.backends.quantized.supported_engines
    if requested is not None:
        if requested not in supported or requested == "none":
            raise ValueError(f"INT8 engine {requested!r} unavailable; supported={supported}")
        engine = requested
    else:
        engine = next((name for name in ("x86", "fbgemm", "onednn", "qnnpack")
                       if name in supported), None)
        if engine is None:
            raise RuntimeError("No CPU dynamic quantization engine is available in this torch build")
    torch.backends.quantized.engine = engine
    return engine


def _quantize_model(model):
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear},
                                                 dtype=torch.qint8, inplace=False)


def _read_artifact(artifact, device="cpu"):
    artifact = Path(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version") != FORMAT_VERSION or manifest.get("backend") != "tiny"
            or manifest.get("tokenizer") != TOKENIZER):
        raise ValueError("Not a supported tiny transformer artifact")
    config = ModelConfig(**manifest["model_config"])
    precision = manifest["precision"]
    if precision not in ("fp32", "dynamic-int8"):
        raise ValueError(f"Unsupported artifact precision: {precision}")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Tiny generator supports cpu or cuda for FP32, cpu only for INT8")
    if precision == "dynamic-int8" and target.type != "cpu":
        raise ValueError("Dynamic INT8 artifacts require CPU; they are not GPU low-bit kernels")
    model = TinyTransformer(config)
    if precision == "dynamic-int8":
        _quantization_engine(manifest["quantization"]["engine"])
        model = _quantize_model(model)
    saved = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
    if saved.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported model state format")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(target).eval()
    return model, manifest


class Generator:
    def __init__(self, model, device="cpu"):
        self.model = model.eval()
        self.device = torch.device(device)

    def generate(self, prompt, max_new_tokens=48):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        tokens = [BOS] + encode(prompt) + [ASSISTANT]
        if len(tokens) > self.model.config.max_seq_len:
            raise ValueError("Prompt exceeds context length; no silent truncation is performed")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        generated, finish_reason = [], "length"
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if len(tokens) > self.model.config.max_seq_len:
                    finish_reason = "context_length"
                    break
                inputs = torch.tensor([tokens], dtype=torch.long, device=self.device)
                logits = self.model(inputs)[0, -1]
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError("Nonfinite generation logits")
                token = int(logits.argmax().item())
                generated.append(token)
                tokens.append(token)
                if token == EOS:
                    finish_reason = "eos"
                    break
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"prediction": decode(generated), "generated_tokens": len(generated),
                "finish_reason": finish_reason, "latency_s": time.perf_counter() - started}


def load_generator(artifact: Path, device="cpu", threads=2):
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if torch.device(device).type == "cpu":
        # Large CPU thread pools dominate the cost of this deliberately tiny model.
        torch.set_num_threads(threads)
    model, _ = _read_artifact(artifact, device)
    return Generator(model, device)


def quantize(artifact, output):
    model, source = _read_artifact(artifact, "cpu")
    if source["precision"] != "fp32":
        raise ValueError("Quantization input must be a trained FP32 tiny artifact")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    engine = _quantization_engine()
    quantized = _quantize_model(model)
    linears = []
    for name, module in quantized.named_modules():
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            weight = module.weight()
            if weight.dtype != torch.qint8 or not weight.is_quantized:
                raise RuntimeError(f"{name} was not quantized to INT8")
            linears.append({"name": name, "shape": list(weight.shape),
                            "weight_dtype": str(weight.dtype), "qscheme": str(weight.qscheme())})
    if not linears:
        raise RuntimeError("No linear layers were quantized")
    output.mkdir(parents=True, exist_ok=False)
    _save_atomic({"format_version": FORMAT_VERSION, "state_dict": quantized.state_dict()},
                 output / "model.pt")
    manifest = _manifest(
        model.config, "dynamic-int8", device="cpu",
        data_sha256=source.get("data_sha256"),
        completed_steps=source.get("completed_steps"),
        training_complete=source.get("training_complete"),
        quantization={
            "method": "torch.ao.quantization.quantize_dynamic",
            "engine": engine,
            "linear_weights": "qint8 tensors with scale/zero point, repacked on load",
            "activations": "dynamically quantized by CPU linear kernels; float outputs",
            "unquantized": ["token embeddings", "position embeddings", "layer norms", "biases"],
            "attention": "floating-point scaled dot product attention; INT8 projection linears",
            "gpu_low_bit_kernel": False,
            "linears": linears,
        },
    )
    manifest["model_bytes"] = (output / "model.pt").stat().st_size
    manifest["source_model_bytes"] = (Path(artifact) / "model.pt").stat().st_size
    _json_write(output / "manifest.json", manifest)
    # Exercise the weights-only reconstruction before reporting artifact creation as successful.
    reloaded = load_generator(output, threads=torch.get_num_threads())
    with torch.inference_mode():
        reloaded.model(torch.tensor([[BOS]], dtype=torch.long))
    return manifest


def predict(artifact, data_dir, split, output, max_new_tokens=48, device="cpu"):
    from pipeline.data import dataset_digest, load_split

    rows = load_split(data_dir, split)
    output = Path(output)
    metadata = output.with_suffix(output.suffix + ".manifest.json")
    for path in (output, metadata):
        if path.exists():
            raise FileExistsError(path)
    generator = load_generator(artifact, device, threads=torch.get_num_threads())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            result = {"id": row["id"], **generator.generate(row["prompt"], max_new_tokens)}
            stream.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n")
    prediction_manifest = {
        "backend": "tiny", "artifact": json.loads((Path(artifact) / "manifest.json").read_text(encoding="utf-8")),
        "model_sha256": hashlib.sha256((Path(artifact) / "model.pt").read_bytes()).hexdigest(),
        "dataset_sha256": dataset_digest(data_dir), "split": split,
        "max_new_tokens": max_new_tokens, "device": device, "sampling": "greedy",
        "timing_scope": "autoregressive loop; prompt byte encoding excluded; generated byte decoding included",
    }
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(prediction_manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train from random weights, or resume exactly")
    train_parser.add_argument("--data-dir", "--data", type=Path, default=Path("results") / "pipeline" / "data")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--stop-after", type=int)
    for name, default in asdict(ModelConfig()).items():
        train_parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in asdict(TrainConfig()).items():
        train_parser.add_argument("--" + name.replace("_", "-"),
                                  type=float if isinstance(default, float) else int, default=default)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--artifact", type=Path, required=True)
    predict_parser.add_argument("--data-dir", "--data", type=Path, default=Path("results") / "pipeline" / "data")
    predict_parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--max-new-tokens", type=int, default=48)
    predict_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    predict_parser.add_argument("--threads", type=int, default=2)
    quantize_parser = commands.add_parser("quantize")
    quantize_parser.add_argument("--artifact", type=Path, required=True)
    quantize_parser.add_argument("--output", type=Path, required=True)
    quantize_parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "train":
        model_config = ModelConfig(**{key: getattr(args, key) for key in asdict(ModelConfig())})
        config = TrainConfig(**{key: getattr(args, key) for key in asdict(TrainConfig())})
        metrics = train(args.data_dir, args.output, model_config, config, args.resume, args.stop_after)
        print(json.dumps({key: metrics[key] for key in (
            "initial_validation_loss", "final_validation_loss", "completed_steps",
            "training_complete", "elapsed_s_this_run")}, indent=2))
    else:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        if args.command == "predict":
            count = predict(args.artifact, args.data_dir, args.split, args.output,
                            args.max_new_tokens, args.device)
            print(f"Wrote {count} predictions to {args.output}")
        else:
            print(json.dumps(quantize(args.artifact, args.output), indent=2))


if __name__ == "__main__":
    main()

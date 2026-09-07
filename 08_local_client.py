"""Lesson 8: stream a local Ollama request and save real timing metrics."""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def get_json(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return json.load(response)


def generate(args):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
        "think": args.think,
        "keep_alive": "10m",
        "options": {
            "num_ctx": args.context, "num_predict": args.tokens,
            "temperature": 0, "seed": 42,
        },
    }
    request = urllib.request.Request(
        args.url + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_event_s = first_output_s = first_answer_s = None
    answer, thinking = [], []
    final = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            now = time.perf_counter() - start
            if "error" in event:
                raise RuntimeError(event["error"])
            if first_event_s is None:
                first_event_s = now
            message = event.get("message", {})
            text = message.get("content", "")
            thought = message.get("thinking", "")
            if first_output_s is None and (text or thought):
                first_output_s = now
            if first_answer_s is None and text:
                first_answer_s = now
            answer.append(text)
            thinking.append(thought)
            print(text, end="", flush=True)
            if event.get("done"):
                final = event
    wall_s = time.perf_counter() - start
    if final is None:
        raise RuntimeError("Stream ended without final completion metrics")
    if first_output_s is None:
        raise RuntimeError("Model produced neither answer nor thinking content")
    decode_s = final["eval_duration"] / 1e9
    prefill_s = final["prompt_eval_duration"] / 1e9
    result = {
        "request": payload,
        "answer": "".join(answer),
        "thinking": "".join(thinking),
        "first_event_s": first_event_s,
        "first_output_s": first_output_s,
        "first_answer_s": first_answer_s,
        "wall_s": wall_s,
        "load_s": final["load_duration"] / 1e9,
        "prompt_tokens": final["prompt_eval_count"],
        "prompt_eval_s": prefill_s,
        "generated_tokens": final["eval_count"],
        "eval_s": decode_s,
        "generation_tokens_per_s": final["eval_count"] / decode_s if decode_s else None,
        "done_reason": final.get("done_reason"),
    }
    print("\n" + json.dumps({k: result[k] for k in (
        "first_output_s", "first_answer_s", "wall_s",
        "generation_tokens_per_s", "done_reason",
    )}, indent=2), flush=True)
    if not result["answer"].strip():
        print("WARNING: no final answer; thinking may have exhausted the output budget.")
    if result["done_reason"] == "length":
        print("WARNING: output hit the token cap; inspect it for incompleteness.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--context", type=positive_int, default=4096)
    parser.add_argument("--tokens", type=positive_int, default=256)
    parser.add_argument("--repeat", type=positive_int, default=1)
    parser.add_argument("--think", action="store_true")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default="Explain KV caching in three concise sentences.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Read a UTF-8 prompt from a file")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            parser.error(f"Prompt file not found: {args.prompt_file}")
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    parsed = urllib.parse.urlsplit(args.url)
    if (parsed.scheme != "http" or
            parsed.hostname not in ("127.0.0.1", "localhost", "::1") or
            parsed.username or parsed.password or parsed.path not in ("", "/") or
            parsed.query or parsed.fragment):
        parser.error("--url must be a loopback HTTP server origin")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    args.url = args.url.rstrip("/")
    if args.save and args.save.exists():
        parser.error("--save already exists; choose a new file to preserve the old run")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": get_json(args.url, "/api/version"),
        "server": args.url,
        "notes": [
            "Sequential repeats may reuse a cached prefix.",
            "Does not unload models, change server settings, or execute generated code.",
            "first_output includes thinking; first_answer is visible answer content.",
            "No claim of peak GPU memory or reproducible cold-start conditions.",
        ],
        "cases": [],
    }
    for index in range(args.repeat):
        print(f"\nRequest {index + 1}/{args.repeat}", flush=True)
        report["cases"].append(generate(args))
    report["loaded_models_after"] = get_json(args.url, "/api/ps")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
        print(f"Saved: {args.save.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"Ollama HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Cannot reach the local Ollama endpoint: {error.reason}. "
            "Start the intended server and check --url; no reinstall is required."
        ) from error

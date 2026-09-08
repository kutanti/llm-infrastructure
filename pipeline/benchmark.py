"""Bounded local load generation with explicit failure and arrival accounting."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from pipeline.data import load_split
from pipeline.evaluate import percentile


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirects are disabled for local benchmarks", headers, fp)


def loopback_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1", "localhost")
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTP loopback origin with no path, credentials, query or fragment")
    # Access validates malformed/out-of-range port numbers.
    _ = parsed.port
    return value.rstrip("/")


def request_one(url, backend, model, example, tokens, timeout, scheduled, index):
    started = time.perf_counter()
    base = {"index": index, "id": example["id"], "dispatch_lag_s": started - scheduled}
    if backend == "local":
        route = "/generate"
        payload = {"prompt": example["prompt"], "max_new_tokens": tokens}
    else:
        route = "/api/chat"
        payload = {"model": model, "messages": [{"role": "user", "content": example["prompt"]}],
                   "stream": True, "think": False,
                   "options": {"num_predict": tokens, "temperature": 0, "seed": 42}}
    req = Request(url + route, json.dumps(payload).encode("utf-8"),
                  {"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=timeout) as response:
            if backend == "local":
                result = json.load(response)
                if not isinstance(result, dict) or not isinstance(result.get("prediction"), str):
                    raise ValueError("Server did not return a prediction string")
                generated = result.get("generated_tokens")
                first_output = None
                prediction = result["prediction"]
                reason = result.get("finish_reason")
            else:
                first_output = None
                chunks = []
                final = None
                for line in response:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("Ollama event must be a JSON object")
                    if "error" in event:
                        raise ValueError(f"Ollama error: {event['error']}")
                    message = event.get("message", {})
                    text = message.get("content", "")
                    if first_output is None and (text or message.get("thinking")):
                        first_output = time.perf_counter() - started
                    chunks.append(text)
                    if event.get("done"):
                        final = event
                if final is None:
                    raise ValueError("Ollama stream ended without completion metrics")
                generated = final.get("eval_count")
                prediction = "".join(chunks)
                reason = final.get("done_reason")
            if type(generated) is not int or generated < 0:
                raise ValueError("Server did not return a nonnegative token count")
            return {**base, "ok": True, "status": 200, "latency_s": time.perf_counter() - started,
                    "first_output_s": first_output, "generated_tokens": generated,
                    "prediction": prediction, "finish_reason": reason}
    except HTTPError as error:
        return {**base, "ok": False, "status": error.code, "error": str(error),
                "latency_s": time.perf_counter() - started}
    except (URLError, TimeoutError, ConnectionError, ValueError, OSError) as error:
        return {**base, "ok": False, "status": None,
                "error": f"{type(error).__name__}: {error}", "latency_s": time.perf_counter() - started}


def run_load(examples, request_fn, count, concurrency, mode, rate):
    if not examples or count <= 0 or concurrency <= 0:
        raise ValueError("Load requires examples, positive count, and positive concurrency")
    if mode not in ("open", "closed") or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Mode must be open/closed and rate must be finite and positive")
    records = []
    pending = set()
    start = time.perf_counter()

    def collect(done):
        for future in done:
            records.append(future.result())
            pending.remove(future)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index in range(count):
            if mode == "open":
                scheduled = start + index / rate
                remaining = scheduled - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                collect({future for future in pending if future.done()})
                if len(pending) >= concurrency:
                    records.append({"index": index, "id": examples[index % len(examples)]["id"],
                                    "ok": False, "status": None, "client_dropped": True,
                                    "error": "Client in-flight bound reached at scheduled arrival"})
                    continue
            else:
                if len(pending) >= concurrency:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done)
                scheduled = time.perf_counter()
            pending.add(pool.submit(request_fn, examples[index % len(examples)], scheduled, index))
        if pending:
            done, _ = wait(pending)
            collect(done)
    elapsed = time.perf_counter() - start
    return sorted(records, key=lambda row: row["index"]), elapsed


def summarize(records, elapsed):
    if not records or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Summary requires records and a positive finite elapsed time")
    successes = [row for row in records if row["ok"]]
    latencies = [row["latency_s"] for row in successes]
    first = [row["first_output_s"] for row in successes if row.get("first_output_s") is not None]
    tokens = sum(row["generated_tokens"] for row in successes)
    return {
        "offered_requests": len(records),
        "completed_successfully": len(successes),
        "failed_or_dropped": len(records) - len(successes),
        "client_dropped": sum(row.get("client_dropped", False) for row in records),
        "http_429": sum(row.get("status") == 429 for row in records),
        "success_fraction": len(successes) / len(records),
        "elapsed_s": elapsed,
        "successful_requests_per_s": len(successes) / elapsed,
        "generated_tokens_per_s": tokens / elapsed,
        "successful_request_latency_s": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "first_output_s": {"samples": len(first), "p50": percentile(first, 50), "p95": percentile(first, 95)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results") / "pipeline" / "data")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--backend", choices=("local", "ollama"), default="local")
    parser.add_argument("--url", default="http://127.0.0.1:11436")
    parser.add_argument("--model", help="Required for Ollama; model must already be installed")
    parser.add_argument("--mode", choices=("open", "closed"), default="closed")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rate", type=float, default=1.0, help="Scheduled requests/s in open mode")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.requests <= 10000 or not 1 <= args.concurrency <= 64:
        parser.error("Use 1..10000 requests and 1..64 client in-flight slots")
    if not 1 <= args.tokens <= 256 or not math.isfinite(args.timeout) or not 0 < args.timeout <= 600:
        parser.error("Use 1..256 tokens and a finite timeout in (0, 600]")
    if not math.isfinite(args.rate) or not 0 < args.rate <= 100:
        parser.error("Use a finite scheduled rate in (0, 100]")
    if args.backend == "ollama" and not args.model:
        parser.error("--model is required for Ollama")
    url = loopback_url(args.url)
    examples = load_split(args.data, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        def send(example, scheduled, index):
            return request_one(url, args.backend, args.model, example, args.tokens,
                               args.timeout, scheduled, index)
        records, elapsed = run_load(examples, send, args.requests, args.concurrency, args.mode, args.rate)
        result = {"configuration": {"backend": args.backend, "model": args.model, "mode": args.mode,
                                    "requests": args.requests, "concurrency": args.concurrency,
                                    "rate": args.rate if args.mode == "open" else None,
                                    "tokens": args.tokens, "split": args.split},
                  "summary": summarize(records, elapsed), "requests": records}
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))
    if result["summary"]["completed_successfully"] == 0:
        raise SystemExit("No requests succeeded; inspect the recorded errors.")


if __name__ == "__main__":
    main()

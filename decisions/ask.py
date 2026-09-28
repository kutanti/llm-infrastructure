"""Lessons 5-6: ask the course's questions of Jev, a Jev-compatible server, or Laya.

Nothing is sent and no model is loaded unless you add --live; without it the script
prints the request it would make.
"""

import argparse
import json
import os
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np

from .answers import options, read_answers
from .calibration import decide, expected_calibration_error
from .questions import DEFAULT_MODEL, TICKET, TRIAGE_QUESTIONS, check_request, request_body

JEV_URL = "https://api.typesafe.ai"
JEV_KEY = "TYPESAFE_API_KEY"
AUTH_SCHEME = "Bearer"
LOOPBACK = ("127.0.0.1", "::1", "localhost")
EXAMPLES = Path(__file__).with_name("tickets.jsonl")
LAYA_INSTALL = "python -m pip install laya==0.3.21"
MAX_RESPONSE_BYTES = 1 << 20
MIN_ECE = 100
HINTS = {
    401: "the API key is missing or wrong",
    403: "this key may not use the requested model",
    413: "the request is too large",
    422: "a question or the state failed validation; the detail names it",
    429: "rate limited: wait, then retry",
    503: "the server is busy or still loading a model: retry after a pause",
    529: "the service is overloaded: retry later",
}
NOTES = [
    "Records contain the ticket text that was sent; keep real customer data out of Git.",
    "wall_s is client wall time. Over HTTP it includes the network; the first in-process "
    "Laya call includes loading a checkpoint.",
    "top_probability is computed from the returned probabilities for every backend; "
    "the backend's own confidence fields are copied unchanged.",
]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def server_url(value):
    """Accept https://host[:port][/prefix], or plain http only on this machine."""
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("use http(s)://host[:port][/prefix] without credentials, query, or fragment")
    if parsed.scheme == "http" and parsed.hostname not in LOOPBACK:
        raise ValueError("plain http is allowed only for 127.0.0.1, ::1, or localhost; use https")
    _ = parsed.port  # rejects a malformed port
    return value.rstrip("/")


def post(url, body, api_key=None, timeout=30.0):
    """POST one body to {url}/v1/systemone; return the JSON response and timing metadata."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = " ".join((AUTH_SCHEME, api_key))
    request = Request(url + "/v1/systemone", json.dumps(body).encode("utf-8"), headers, method="POST")
    handlers = [NoRedirect()]
    if urlsplit(url).hostname in LOOPBACK:
        handlers.append(ProxyHandler({}))
    started = time.perf_counter()
    with build_opener(*handlers).open(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        received = response.headers
    wall_s = time.perf_counter() - started
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response is larger than 1 MiB")
    return json.loads(raw), {
        "wall_s": wall_s,
        "request_id": received.get("x-typesafe-request-id"),
        "server_inference_ms": received.get("x-inference-time-ms"),
    }


class LayaInProcess:
    """Laya's Router in this process. Without permission, only cached weights may load."""

    def __init__(self, allow_download, revision=None, model=None):
        if not allow_download:
            # huggingface_hub reads this when it is first imported, which happens below.
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import laya
        except ImportError as error:
            raise SystemExit(f"Laya is not installed. In a separate virtual environment: {LAYA_INSTALL}") from error
        self.version = getattr(laya, "__version__", None)
        self.router = laya.Router(revision=revision) if revision else laya.Router()
        self.allow_download, self.model = allow_download, model

    def __call__(self, body):
        started = time.perf_counter()
        try:
            result = self.router.predict(body["state"], body["questions"], model=self.model)
        except Exception as error:
            hint = "" if self.allow_download else " If the weights are not cached yet, add --allow-download."
            raise SystemExit(f"Laya could not answer: {type(error).__name__}: {error}.{hint}") from error
        return result, {"wall_s": time.perf_counter() - started}


def load_examples(path, questions):
    """Read labeled JSONL rows: {"id": ..., "state": ..., "labels": {question_id: label}}."""
    examples, seen = [], set()
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path}:{number}"
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in seen
                    or not isinstance(row.get("state"), (str, dict, list))
                    or not isinstance(row.get("labels", {}), dict)):
                raise ValueError(f"{where}: need a unique id, a state, and a labels object")
            seen.add(row["id"])
            row.setdefault("labels", {})
            for question_id, label in row["labels"].items():
                question = questions.get(question_id)
                if question is None:
                    raise ValueError(f"{where}: no question named {question_id!r}")
                valid = (isinstance(label, bool) if question["type"] == "noul"
                         else label in options(question) if question["type"] == "choice"
                         else isinstance(label, int) and not isinstance(label, bool)
                         and 0 <= label < len(question["criteria"]))
                if not valid:
                    raise ValueError(f"{where}: {label!r} is not a valid {question['type']} label for {question_id}")
            examples.append(row)
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples


def is_correct(question, summary, label):
    """choice: the chosen label; noul: yes at P >= 0.5; score: the most probable level."""
    if question["type"] == "choice":
        return summary["answer"] == label
    if question["type"] == "noul":
        return (summary["answer"] >= 0.5) == label
    probabilities = summary["probabilities"]
    return max(probabilities, key=probabilities.get) == str(label)


def summarize(records, questions, act_at=0.9, confirm_at=0.6):
    summary = {"examples": len(records), "questions": {}}
    for question_id, question in questions.items():
        rows = [(r["answers"][question_id], r["labels"][question_id])
                for r in records if question_id in r["labels"]]
        if not rows:
            continue
        top = np.array([answer["top_probability"] for answer, _ in rows])
        correct = np.array([is_correct(question, answer, label) for answer, label in rows])
        bands = {}
        for band in ("act", "confirm", "human"):
            members = np.array([decide(value, act_at, confirm_at) == band for value in top])
            bands[band] = {"count": int(members.sum()), "correct": int(correct[members].sum())}
        summary["questions"][question_id] = {
            "labeled": len(rows), "accuracy": float(correct.mean()),
            "mean_top_probability": float(top.mean()), "bands": bands,
            "ece": expected_calibration_error(top, correct) if len(rows) >= MIN_ECE else None,
        }
    walls = [record["wall_s"] for record in records]
    later = walls[1:]
    summary["latency_s"] = {
        "first": walls[0], "later_count": len(later),
        "later_p50": float(np.percentile(later, 50)) if later else None,
        "later_p95": float(np.percentile(later, 95)) if later else None,
    }
    tokens = [(record["response"].get("usage") or {}).get("input_tokens") for record in records]
    valid = all(isinstance(value, int) and not isinstance(value, bool) for value in tokens)
    summary["input_tokens"] = sum(tokens) if valid else None
    return summary


def print_answers(answers, act_at, confirm_at):
    print(f"{'question':18}{'type':8}{'answer':>12}{'top p':>8}  {'band':9}{'backend confidence':>18}")
    for question_id, summary in answers.items():
        value = summary["answer"]
        if summary["type"] == "choice":
            shown = str(value)
        elif summary["type"] == "noul":
            shown = f"P(yes)={value:.2f}"
        else:
            shown = f"{value:.3f}"
        reported = summary["reported_confidence"]
        reported = f"{reported:.3f}" if isinstance(reported, (int, float)) else "-"
        band = decide(summary["top_probability"], act_at, confirm_at)
        print(f"{question_id:18}{summary['type']:8}{shown:>12}{summary['top_probability']:8.3f}  "
              f"{band:9}{reported:>18}")


def print_summary(summary):
    print(f"\n{'question':18}{'labeled':>8}{'accuracy':>10}{'mean top p':>12}"
          "   correct/count in act, confirm, human")
    for question_id, row in summary["questions"].items():
        bands = ", ".join(f"{band['correct']}/{band['count']}" for band in row["bands"].values())
        print(f"{question_id:18}{row['labeled']:8}{row['accuracy']:10.1%}"
              f"{row['mean_top_probability']:12.3f}   {bands}")
        if row["ece"] is not None:
            print(f"{'':18}ECE {row['ece']:.3f}")
    if any(row["ece"] is None for row in summary["questions"].values()):
        print(f"ECE is not reported below {MIN_ECE} labeled answers per question; the bins would be too sparse.")
    latency = summary["latency_s"]
    line = f"Client wall time: first call {latency['first']:.3f} s"
    if latency["later_count"]:
        line += (f"; later calls p50 {latency['later_p50']:.3f} s, "
                 f"p95 {latency['later_p95']:.3f} s (n={latency['later_count']})")
    print(line)
    if summary["input_tokens"] is not None:
        print(f"Input tokens reported in usage: {summary['input_tokens']}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--jev", action="store_true", help=f"TypeSafe's hosted API at {JEV_URL}")
    target.add_argument("--url", help="a Jev-compatible server, such as laya-serve at http://127.0.0.1:8000")
    target.add_argument("--laya", action="store_true", help="Laya's Router in this process (needs pip install laya)")
    parser.add_argument("--live", action="store_true", help="send the request or run the model")
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--ticket", default=TICKET, help="ticket text to use as the state")
    state.add_argument("--examples", nargs="?", const=EXAMPLES, type=Path,
                       help="labeled JSONL file (default when given without a path: decisions/tickets.jsonl)")
    parser.add_argument("--model", help=f"model to request; HTTP default {DEFAULT_MODEL}, Laya default: routed")
    parser.add_argument("--api-key-env", help=f"environment variable with a bearer token (--jev default: {JEV_KEY})")
    parser.add_argument("--allow-download", action="store_true", help="--laya: let Hugging Face download weights")
    parser.add_argument("--revision", help="--laya: Hub commit SHA, or 'reviewed' for the laya package's pins")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--act-at", type=float, default=0.9)
    parser.add_argument("--confirm-at", type=float, default=0.6)
    parser.add_argument("--save", type=Path, help="write a new JSON report; an existing file is never replaced")
    args = parser.parse_args(argv)
    if args.url is not None:
        try:
            args.url = server_url(args.url)
        except ValueError as error:
            parser.error(f"--url: {error}")
    if args.live and not (args.jev or args.url or args.laya):
        parser.error("--live needs a target: --jev, --url, or --laya")
    if not args.laya and (args.allow_download or args.revision):
        parser.error("--allow-download and --revision apply only to --laya")
    if args.laya and args.api_key_env:
        parser.error("--api-key-env applies only to HTTP targets")
    if not 0 <= args.confirm_at <= args.act_at <= 1:
        parser.error("need 0 <= --confirm-at <= --act-at <= 1")
    if not args.timeout > 0:
        parser.error("--timeout must be positive")
    if args.save is not None and not args.live:
        parser.error("--save records a live run; add --live")
    if args.save is not None and args.save.exists():
        parser.error("--save already exists; choose a new file to keep the old run")
    return args


def http_error_message(error, target):
    try:
        detail = error.read(2000).decode("utf-8", errors="replace").strip()
    except Exception:
        detail = ""
    hint = "redirect refused; point --url at the final address" if 300 <= error.code < 400 else HINTS.get(error.code, "")
    return f"HTTP {error.code} from {target}" + (f" ({hint})" if hint else "") + (f": {detail}" if detail else "")


def main(argv=None):
    args = parse_args(argv)
    questions = TRIAGE_QUESTIONS
    try:
        examples = (load_examples(args.examples, questions) if args.examples is not None
                    else [{"id": "ticket", "state": args.ticket, "labels": {}}])
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cannot use the examples: {error}") from error
    bodies = [request_body(example["state"], questions, args.model or DEFAULT_MODEL) for example in examples]
    try:
        warnings = sorted({warning for body in bodies for warning in check_request(body)})
    except ValueError as error:
        raise SystemExit(f"Request refused before sending: {error}") from error
    for warning in warnings:
        print(f"warning: {warning}")

    if args.laya:
        target = "laya.Router().predict(state, questions) in this process"
    elif args.jev or args.url:
        target = (JEV_URL if args.jev else args.url) + "/v1/systemone"
    else:
        target = "none chosen yet (add --jev, --url URL, or --laya)"
    key_name = args.api_key_env or (JEV_KEY if args.jev else None)

    if not args.live:
        print("Dry run: nothing was sent and no model was loaded.")
        print(f"Target: {target}")
        if args.laya:
            print(f"Model: {args.model or 'chosen by the Router'}; downloads "
                  f"{'allowed' if args.allow_download else 'off (cached weights only)'}; "
                  f"revision {args.revision or 'Hub default (unpinned)'}")
            shown = {"state": bodies[0]["state"], "questions": bodies[0]["questions"]}
        else:
            print("Headers: Content-Type: application/json"
                  + (f"; Authorization: {AUTH_SCHEME} <value of ${key_name}>" if key_name else ""))
            shown = bodies[0]
        if len(bodies) > 1:
            print(f"Examples: {len(bodies)} from {args.examples}; the first request is shown.")
        print(json.dumps(shown, indent=2))
        print("Add --live to run it.")
        return None

    if args.laya:
        backend = LayaInProcess(args.allow_download, args.revision, args.model)
    else:
        api_key = None
        if key_name:
            api_key = os.environ.get(key_name, "").strip()
            if not api_key:
                raise SystemExit(f"Set {key_name} in the environment first; keys are never taken as arguments.")
        url = JEV_URL if args.jev else args.url

        def backend(body):
            return post(url, body, api_key, args.timeout)

    records = []
    for example, body in zip(examples, bodies):
        try:
            response, meta = backend(body)
            answers = read_answers(response, questions)
        except urllib.error.HTTPError as error:
            raise SystemExit(http_error_message(error, target)) from error
        except urllib.error.URLError as error:
            raise SystemExit(f"Cannot reach {target}: {error.reason}") from error
        except TimeoutError as error:
            raise SystemExit(f"No response from {target} within {args.timeout} s") from error
        except ValueError as error:
            raise SystemExit(f"Unexpected response from {target}: {error}") from error
        records.append({"id": example["id"], "state": example["state"], "labels": example["labels"],
                        "response": response, "answers": answers, **meta})

    first = records[0]
    routing = first["response"].get("routing") or {}
    print(f"Target: {target}")
    print(f"Answered by: {first['response'].get('model', 'not reported')}"
          + (f"; routed to {routing.get('model')} ({routing.get('reason')})" if routing else ""))
    summary = None
    if args.examples is None:
        print(f"Client wall time: {first['wall_s']:.3f} s\n")
        print_answers(first["answers"], args.act_at, args.confirm_at)
    else:
        summary = summarize(records, questions, args.act_at, args.confirm_at)
        print_summary(summary)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "requested_model": args.model or (None if args.laya else DEFAULT_MODEL),
        "laya": ({"version": backend.version, "revision": args.revision,
                  "downloads_allowed": args.allow_download} if args.laya else None),
        "thresholds": {"act_at": args.act_at, "confirm_at": args.confirm_at},
        "questions": questions, "records": records, "summary": summary, "notes": NOTES,
    }
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
        print(f"Saved: {args.save.resolve()}")
    return report


if __name__ == "__main__":
    main()

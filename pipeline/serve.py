"""A loopback-only, single-generation-slot teaching server."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading


MAX_BODY = 16_384


def make_server(generator, port=11436):
    slot = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            self.reply(200, {"ready": True, "generation_slots": 1, "streaming": False})

        def do_POST(self):
            if self.path != "/generate":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, {"error": "Chunked requests are not supported"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.reply(400, {"error": "Invalid Content-Length"})
                return
            if not 0 < length <= MAX_BODY:
                self.reply(413, {"error": "Body must contain 1 to 16384 bytes"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.reply(400, {"error": "Body must be UTF-8 JSON"})
                return
            if not isinstance(payload, dict):
                self.reply(400, {"error": "Body must be an object"})
                return
            prompt = payload.get("prompt")
            tokens = payload.get("max_new_tokens", 48)
            if not isinstance(prompt, str) or not prompt or len(prompt) > 2048:
                self.reply(400, {"error": "prompt must contain 1 to 2048 characters"})
                return
            if type(tokens) is not int or not 1 <= tokens <= 256:
                self.reply(400, {"error": "max_new_tokens must be an integer from 1 to 256"})
                return
            if not slot.acquire(blocking=False):
                self.reply(429, {"error": "The single generation slot is busy; no queue is retained"})
                return
            try:
                result = generator.generate(prompt, max_new_tokens=tokens)
                status = 200
            except ValueError as error:
                self.log_error("Rejected generation input: %s", error)
                status, result = 400, {"error": str(error)}
            except RuntimeError as error:
                self.log_error("Generation failed: %s", error)
                status, result = 500, {"error": "Generation failed; see local server diagnostics"}
            finally:
                slot.release()
            self.reply(status, result)

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--backend", choices=("tiny", "hf"), default="tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--port", type=int, default=11436)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.backend == "tiny":
        from pipeline.tiny import load_generator
    else:
        from pipeline.hf import load_generator
    generator = load_generator(args.artifact, device=args.device)
    server = make_server(generator, args.port)
    print(f"Ready: http://127.0.0.1:{args.port}; one slot, no batching, no streaming.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping the local server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

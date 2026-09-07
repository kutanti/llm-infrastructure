# Running a local LLM on a 6 GB GPU

The first Qwen3.5-4B request took **102 seconds** to start answering.
The same question, asked again, started in **0.19 seconds**.

These notes follow that deployment: what the runner did, where memory went,
and which changes are worth trying next. The machine has an RTX 4050 Laptop
GPU with 6,141 MiB VRAM, an i5-13420H, and 16 GB RAM. The model is
`qwen3.5:4b`, Q4_K_M, served by Ollama.

## Read

Start with [the investigation](GUIDE.md). It uses the recorded run to explain
loading, prefill, decode, KV caching, quantization, and request scheduling.
[The exercises](LABS.md) let you calculate the memory costs and make your own
requests. Answers are folded away so you can work through them first.

The initial run used a 4,096-token context, one request at a time, thinking
disabled, and FP16 KV cache:

| Request | First output | Decode speed |
| --- | ---: | ---: |
| Arithmetic, first load | 101.96 s | 1.16 tok/s |
| Same arithmetic, warm | 0.19 s | 17.59 tok/s |
| Python function, warm | 0.40 s | 10.25 tok/s |
| Explanation, warm | 0.56 s | 14.15 tok/s |

These are single observations. The four-token arithmetic answer is particularly
poor evidence of sustained speed. The selected answers and metrics are in
`data\qwen-baseline.json`; lesson 6 reads that file.

## Run

Use Python 3.10 or newer. Install NumPy if it is missing:

```powershell
python -m pip install -r requirements.txt
```

The offline examples need no model download:

```powershell
python 01_why_kv_cache.py
python 02_kv_cache_size.py --context 8192
python 03_kv_quant_error.py
python 04_gqa_explained.py
python 05_flash_attention.py
python 06_read_inference_results.py
python 07_performance_budget.py
```

For live requests, follow [SETUP.md](SETUP.md). It covers both an existing
Ollama server and starting one yourself. Then:

```powershell
python 08_local_client.py --prompt "Explain KV caching in three sentences." --save results\first.json
```

The default endpoint is `http://127.0.0.1:11435`, used by the original deployment.
Pass `--url http://127.0.0.1:11434` for a server on Ollama's usual port.

## Files

| File | Purpose |
| --- | --- |
| `01`-`05` | Small attention and memory examples |
| `06_read_inference_results.py` | Reconstruct the initial run's timings |
| `07_performance_budget.py` | Explore a weight-bandwidth bound |
| `08_local_client.py` | Stream a response and record timing data |
| `09_make_retrieval_prompts.py` | Prepare fixed documents for the input-length experiment |
| `SOURCES.md` | Architecture, API, and paper references |
| `ERRATA.md` | Corrections to the first draft |

Model files and runtimes are not part of this repository. New runs go in the
Git-ignored `results` directory because they may contain private prompts.
The client records generated code as text; it does not execute it.

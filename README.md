# Local LLM learning: inference, hosting, and optimization

A hands-on course built around a real **Qwen3.5-4B Q4_K_M** deployment on
an **RTX 4050 Laptop GPU (6,141 MiB), i5-13420H, and 16 GB installed RAM**.

The goal is to understand why an answer is slow or memory-hungry before
changing flags. The examples distinguish **mathematical models**, **synthetic
demos**, **runtime-specific behavior**, and **measured observations**.
There are no model weights, credentials, or bundled runtimes in this repository.

## Start here

Read [GUIDE.md](GUIDE.md) in order. It teaches the concepts with small examples
and connects each one to our measured deployment. Then work through
[LABS.md](LABS.md), which has experiments, questions, and answers.

| Stage | Read or run | What you will learn |
| --- | --- | --- |
| Fundamentals | Guide 1-3; `01_why_kv_cache.py` | Request lifecycle, prefill/decode, attention, caching |
| Memory | Guide 4; `02_kv_cache_size.py` | VRAM budgets, GQA, hybrid recurrent state, context |
| Precision | Guide 5; `03_kv_quant_error.py` | Weight vs cache quantization and quality tradeoffs |
| Architecture | `04_gqa_explained.py` | Different queries sharing the same K/V |
| Kernels | Guide 6; `05_flash_attention.py` | Online softmax and temporary workspace |
| Measurement | Guide 7-8; `06_read_inference_results.py` | Loading, first output, throughput, honest evidence |
| Optimization | Guide 9; `07_performance_budget.py` | Bottlenecks, rooflines, output budgets |
| Hosting | Guide 10-12; `08_local_client.py` | Local APIs, queues, concurrency, operations |

## Run the offline lessons

Python 3.10+ is sufficient. Lessons 1, 3, 4, and 5 require NumPy. It is
already installed on the laptop used for this course. On a fresh environment:

```powershell
python -m pip install -r requirements.txt
```

From this repository:

```powershell
python 01_why_kv_cache.py
python 02_kv_cache_size.py --context 8192 --sequences 1
python 03_kv_quant_error.py
python 04_gqa_explained.py
python 05_flash_attention.py
python 06_read_inference_results.py
python 07_performance_budget.py
```

These seven lessons do not download a model or contact a server. Cache arithmetic
is shared in `cache_math.py` so the examples use consistent units and block sizes.

## Talk to the existing local model

On the original laptop, the separate deployment is in `C:\Projects\Qwen-Local`.
Start its `Start-Server.ps1` in another terminal if the server is not running.
Do not install another Ollama copy just to run these lessons.

```powershell
python 08_local_client.py --prompt "Explain KV caching in three sentences." --save results\first.json
```

The default address is `http://127.0.0.1:11435`. If you use an existing
Ollama server on the usual port, pass `--url http://127.0.0.1:11434`.
That server must already have the requested model. The client uses only
Python's standard library and accepts only loopback HTTP addresses.

The client saves prompts, answers, timing data, and loaded-model metadata when
`--save` is supplied. It never executes generated code. `results` is ignored by
Git because future prompts may be private. Existing result files are not overwritten.

## What actually happened on this laptop

| Observation | Initial baseline |
| --- | --- |
| Runtime / model | Ollama 0.33.3 / `qwen3.5:4b`, Q4_K_M |
| Settings | 4,096 context, one request at a time, thinking off, FP16 cache requested |
| First-load first output | 101.96 seconds |
| Warm coding | 0.40 s first output; 10.25 generated tokens/s |
| Warm explanation | 0.56 s first output; 14.15 generated tokens/s |
| Placement | Ollama reported 100% GPU |
| NVIDIA memory snapshots | 3,829 MiB used out of 6,141 MiB |

Selected source observations and answers are in `data\qwen-baseline.json`.
These are short, single-run observations, not a GPU leaderboard. Memory snapshots
are not peak measurements. No optimized configuration has been established.

## Corrections to the original discussion

Earlier materials overstated several claims. The revised lessons remove:

- An invented 8B identity for GLM-4.7-Flash; it is a 30B-A3B MoE.
- Guaranteed tokens/s, VRAM-fit claims, or fixed runtime-overhead budgets.
- Claims that Q8 is always free or Q4 is always bad.
- The assertion that llama.cpp always silently ignores quantized KV flags.
- GQA quality conclusions based on random head correlations.
- The suggestion that FlashAttention makes total memory constant.

The original GLM-4-9B-Chat has 2 KV groups; similarly named later models can
have different architectures. Quantized cache estimates include block scales.
Distillation is a training method, not a mandatory quality loss or an inference flag.

Primary references and version caveats are in [SOURCES.md](SOURCES.md).

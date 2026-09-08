# Local LLM Learning: From Tokens to a Running Model

What happens between typing a prompt and receiving an answer?

This repository builds up the answer: tokenization, embeddings, training,
attention, and generation. Then it follows a real local deployment to understand
where time and memory go.

Start with basic Python. The fundamentals run on your CPU with NumPy; the local
inference exercises use Ollama. No GPU or model download is needed for the first
examples.

## Try a model you can read in one file

With Python 3.10 or newer:

```powershell
git clone https://github.com/kutanti/local-llm-learning.git
cd local-llm-learning
python -m pip install -r requirements.txt
python foundations\02_train_a_small_model.py
```

The script trains a 68-parameter model to predict the next token in simple
patterns such as `red blue red blue`. It prints the loss and learned
probabilities, then generates a sequence.

You can inspect every weight update. You can also see its limitation: it only
reads the most recent token. The later transformer example shows how attention
lets a prediction use more context.

For the explanation behind each line, start with
[Learning from mistakes](foundations/02-learning-from-mistakes.md).

## Start with the fundamentals

The [foundations path](foundations/README.md) assumes basic Python and introduces:

| Read | Work through |
| --- | --- |
| [Text and representations](foundations/01-text-and-representations.md) | Token IDs, embeddings, tensor shapes, logits, probabilities, and sampling |
| [Learning from mistakes](foundations/02-learning-from-mistakes.md) | Neural layers, loss, gradients, backpropagation, and generalization |
| [Inside a transformer](foundations/03-inside-a-transformer.md) | Positions, attention, causal masks, residuals, normalization, and feed-forward networks |
| [From training to answers](foundations/04-from-training-to-answers.md) | Pretraining, instruction tuning, LoRA, distillation, RAG, tools, and evaluation |

Three NumPy examples accompany the chapters. One trains a small next-token model;
another exposes the operations inside an untrained transformer block.
The [foundations exercises](foundations/EXERCISES.md) include worked answers.

## Then investigate a real deployment

The first Qwen3.5-4B request took **102 seconds** to start answering.
The same question, asked again, started in **0.19 seconds**.

These notes follow that deployment: what the runner did, where memory went,
and which changes are worth trying next. The machine has an RTX 4050 Laptop
GPU with 6,141 MiB VRAM, an i5-13420H, and 16 GB RAM. The model is
`qwen3.5:4b`, Q4_K_M, served by Ollama.

[The investigation](GUIDE.md) uses the recorded run to explain
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

## More examples

The offline examples need no model download:

```powershell
python foundations\01_tokens_and_probabilities.py
python foundations\02_train_a_small_model.py
python foundations\03_transformer_block.py
```

After the foundations, the inference examples explore individual operations
and read the recorded laptop run:

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
| `foundations` | Four foundational chapters, three runnable examples, and exercises |
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

## Help improve the explanations

If an exercise is unclear or a command fails, [open an issue](https://github.com/kutanti/local-llm-learning/issues)
with the chapter, command, and what you expected to happen. For performance
observations, include the model, runtime, and workload rather than a tokens/s
number alone. Remove private prompts and credentials before sharing output.

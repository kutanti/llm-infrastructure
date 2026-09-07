# Exercises

Work from the repository directory. Exercises 1-5 run offline.
For 6-8, follow [SETUP.md](SETUP.md) once.

Live commands below target port 11435 by default. Add
`--url http://127.0.0.1:11434` if that is where your server runs.
Choose a new result filename when repeating a command; the client will not
overwrite an earlier run.

## 1. Count the work the cache removes

For 400 steps, calculate how many token rows pass through the K and V projections:
first when every prefix is rebuilt, then when previous K/V rows are retained.

```powershell
python 01_why_kv_cache.py
```

Compare your counts with the output. Which work does the cache *not* remove?

<details>
<summary>Answer</summary>

Rebuilding prefixes requires `2 * (1 + ... + 400) = 160,400` row projections.
Caching requires `2 * 400 = 800`.

Each new query still compares against the retained keys. The reduction concerns
projection work, not all attention arithmetic. The script checks that its two
ways of computing the attention outputs agree within floating-point tolerance.

</details>

## 2. Decide whether Q8 cache is worth investigating first

Qwen3.5-4B has 8 full-attention layers, 4 KV heads, and head dimension 256.
Calculate their FP16 cache at 4,096 tokens for one sequence. Then calculate
Q8_0 storage using 34 bytes per 32 values.

```powershell
python 02_kv_cache_size.py --context 4096
python 02_kv_cache_size.py --context 8192 --sequences 2
```

How much would Q8 save at 4K? Would that explain our long first request?
What does this table leave out?

<details>
<summary>Answer</summary>

FP16: 128 MiB. Q8_0: 68 MiB. Saving: 60 MiB.
At 8K with two independent sequences, the FP16 portion is 512 MiB.

These calculations exclude the recurrent layers and runtime allocations.
The baseline has a large load/prefill delay, with no evidence tying it to
60 MiB of cache storage. Investigate those timings before choosing cache
precision as the remedy.

</details>

## 3. Follow one rounding error

```powershell
python 03_kv_quant_error.py
```

The script compares synthetic attention outputs against FP32. Notice how errors
change with query scale. Explain the difference between an error in K and an
error in V. Why can't this table tell you how well a model writes Python?

<details>
<summary>Answer</summary>

Changing K can change the attention weights; changing V changes the content
being blended. Which matters more depends on the inputs.

This script contains one attention operation, not the model's stack of layers
or its token-selection loop. It also uses a toy four-bit-like encoding, rather
than the runtime's packed Q4_0. A real quality comparison needs actual tasks
and a scoring rule.

</details>

## 4. Ask two questions of the same keys

```powershell
python 04_gqa_explained.py
```

Both query heads use the same K and V. Before inspecting the printed output,
predict whether their attention weights must match.

<details>
<summary>Answer</summary>

They need not match because the queries differ. One query favors the first
key; the other favors the second. GQA shares K/V projections without requiring
the query heads to do identical work.

</details>

## 5. Update a running softmax

The first tile contains score 2 with value 10. The second contains score 4
with value 20. Calculate the rescaling factor and final output using the
equations in the guide.

```powershell
python 05_flash_attention.py
```

The script uses larger random arrays. Inspect which arrays its memory table
counts, and which it does not.

<details>
<summary>Answer</summary>

The old accumulator is rescaled by `exp(2 - 4)`, about 0.1353.
The output is `(0.1353 * 10 + 20) / (0.1353 + 1)`, about 18.808.

The storage table counts scores or score tiles. It excludes other kernel
workspace and persistent K/V storage.

</details>

## 6. Explain the recorded first request

```powershell
python 06_read_inference_results.py
```

Account for the 102-second wait using the displayed durations. Then identify
one conclusion supported by the data and one explanation that still needs
another measurement.

<details>
<summary>Answer</summary>

The first request reports about 60.55 seconds loading and 41.37 seconds in
prefill. Later requests show almost no load time.

That supports separating initial and warm requests. It does not tell us whether
the initial prefill delay came from kernel initialization, memory pressure, or
another cause. The original report did not isolate those mechanisms.

</details>

Now make your own sequential requests:

```powershell
python 08_local_client.py --repeat 3 --tokens 192 --prompt "Explain prefill versus decode in six concise sentences." --save results\warm-sequence.json
```

Compare `load_s`, `prompt_eval_s`, and `eval_s`. The first request may load
the model; later requests may reuse both loaded weights and a processed prefix.
Treat those conditions separately rather than averaging them together.

## 7. Distinguish reserved context from actual input

First change only the configured capacity:

```powershell
python 08_local_client.py --context 4096 --repeat 3 --tokens 192 --save results\capacity4k.json
python 08_local_client.py --context 8192 --repeat 3 --tokens 192 --save results\capacity8k.json
```

The prompt is identical. Look at the saved `prompt_tokens` value: increasing
capacity did not make the prompt longer. A changed capacity may reload or
reallocate the runner, so compare the later requests separately.

Next, hold capacity fixed and use different input lengths:

```powershell
python 09_make_retrieval_prompts.py
python 08_local_client.py --context 8192 --tokens 16 --repeat 3 --prompt-file results\retrieval\short.txt --save results\short-input.json
python 08_local_client.py --context 8192 --tokens 16 --repeat 3 --prompt-file results\retrieval\long.txt --save results\long-input.json
```

The generator supplies two warehouse inventories. Each has a target record in
the middle, and both ask for its shelf number. The longer file has more
distracting records. Record `prompt_tokens`, `prompt_eval_s`, `first_answer_s`,
and whether the answer is correct.

Verify that the actual input plus output fits the context. If your tokenizer
or model differs and it does not, increase capacity for **both** conditions.

The first request for each document and its exact repeats answer different
questions. Compare the first encounters for input-length effects; inspect the
repeats for prefix-reuse effects. For a stronger result, repeat the comparison
in reverse order and investigate runner/prefix state rather than relying on
a single pair.

<details>
<summary>Expected answer and interpretation</summary>

Both documents identify shelf **42**. The target is supplied in the text;
this is retrieval, not a test of memorized knowledge.

The long document should produce a larger reported input token count. Timing
can depend on prefix reuse and runner state, so report the observations rather
than assuming a fixed ratio. Two documents are an exercise, not a long-context
quality evaluation.

</details>

## 8. Put a time budget on an answer

```powershell
python 07_performance_budget.py --decode-tps 12 --tokens 240
```

The example uses 0.5 seconds to first output. Estimate completion time for
240 tokens and 120 tokens. When would halving the output be a bad optimization?

<details>
<summary>Answer</summary>

About 20.42 and 10.42 seconds, respectively:
`0.5 + (tokens - 1) / 12`.

Shortening helps only if the answer still meets the task. A missing explanation
or truncated function is not an improvement. Check `done_reason` and read the
output before comparing speed.

</details>

For thinking mode, measure first generated output and first answer separately:

```powershell
python 08_local_client.py --think --tokens 512 --prompt "Explain why 17 times 23 is 391." --save results\thinking.json
```

Reasoning can consume the token budget before an answer appears. The client
records that condition. To compare thinking on/off, use this same prompt in
both conditions and score the answers as well as their times.

## Keep a short experiment record

For each change, save:

```text
Question and single changed variable:
Model digest / runtime / driver:
Server settings and evidence they took effect:
Prompt and actual input/output token counts:
Warm-up and prefix-reuse conditions:
Repeated timings and memory measurement method:
Answer correctness / formatting / truncation:
Decision and remaining uncertainty:
```

Keep the previous configuration available. If a change saves little memory,
adds latency, or harms the answers, returning to the baseline is a useful result.

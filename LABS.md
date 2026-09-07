# Labs: learn first, tune second

Run commands from this repository. Labs 1-5 are offline and do not need Ollama.
Labs 6-8 use an already-installed local server and model. No lab asks you to
install another runtime, expose a port publicly, or execute generated code.

## Lab 1: why caching works

```powershell
python 01_why_kv_cache.py
```

Predict the number of K/V row projections for 400 steps before running:
uncached `400 * 401 = 160,400`; cached `2 * 400 = 800`.

The output vectors should agree within floating-point tolerance.
The speedup depends on NumPy, CPU scheduling, and matrix shapes.

**Question:** does this make total attention work linear?

**Answer:** no. It removes repeated K/V projections. Every new query in ordinary
full attention still reads the available history. This is not a token-generation
benchmark and does not establish GPU tokens/s.

## Lab 2: budget the right layers

```powershell
python 02_kv_cache_size.py --context 8192 --sequences 1
python 02_kv_cache_size.py --context 8192 --sequences 2
```

Before running, predict Qwen3.5-4B's full-attention FP16 cache:
256 MiB for one sequence; 512 MiB for two independent sequences.

**Question:** why not use all 32 layers in this formula?

**Answer:** only 8 are full-attention layers. The other 24 maintain recurrent
state that must be accounted for separately. This is not proof of total memory
fit. A 32K context advertised by a model is not a hardware memory guarantee.

**Question:** what is the theoretical 4K FP16-to-Q8 saving for these layers?

**Answer:** `128 - 68 = 60 MiB`, including Q8 block scales. Actual support and
allocation behavior still need observation. A complicated tuning change for
60 MiB is unlikely to be the first priority when startup dominates.

## Lab 3: quantization error is not an accuracy score

```powershell
python 03_kv_quant_error.py
```

Compare the synthetic median relative errors across query scales. Identify
which reference is used: FP32 attention output, not generated text.

**Question:** can low error here prove the real model will solve a coding task?

**Answer:** no. Error propagates through layers and can change token selection.
The toy4 format is not the backend's actual packed `q4_0`. This lab illustrates
mechanics, not a ranking of real quantizations.

Design a real quality checklist before changing precision: valid syntax,
correct edge cases, factual correctness, constraints obeyed, and no truncation.

## Lab 4: shared shelves, different questions

```powershell
python 04_gqa_explained.py
```

The two queries share exactly the same K and V but produce different weights.

**Question:** is MQA forced to make every query head look at the same tokens?

**Answer:** no. The learned queries differ. Sharing K/V changes capacity and
weight shapes, but does not force identical attention patterns. Quality claims
require trained-model evaluations.

## Lab 5: stream a weighted sum

```powershell
python 05_flash_attention.py
python 07_performance_budget.py --weight-gib 3 --bandwidth-gbs 100
```

The streaming attention output should agree with the materialized result.
The performance-budget inputs are illustrative, not specifications of this GPU.

**Question:** if score workspace is tiled, is all inference memory constant?

**Answer:** no. Persistent K/V and other states remain. The score-tile column
also excludes other temporary storage and multiple simultaneously active tiles.

**Question:** why can measured decode fall below the bandwidth-only bound?

**Answer:** compute, recurrent updates, cache traffic, unpacking, kernel launches,
host work, and imperfect utilization were left out of the model.

## Lab 6: inspect the real deployment without changing it

On the original laptop, start `C:\Projects\Qwen-Local\Start-Server.ps1` in
another terminal only if its server is not already running.

```powershell
Invoke-RestMethod http://127.0.0.1:11435/api/version
Invoke-RestMethod http://127.0.0.1:11435/api/ps | ConvertTo-Json -Depth 6
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
python 06_read_inference_results.py
```

If `/api/ps` is empty, the model may have expired from memory while the server
remains healthy. If the endpoint is unreachable, check the server terminal and
port; do not immediately reinstall. Port 11434 may belong to another instance.

Distinguish reported model placement, GPU utilization, memory snapshots,
and sustained performance. They answer different questions.

**Question:** what explains the 102-second first output?

**Answer:** the recorded load time was about 60.55 seconds and reported prefill
about 41.37 seconds. The record does not isolate the underlying reasons for
those costs. Do not diagnose disk speed or compilation from the totals alone.

## Lab 7: capture your own warm sequence

```powershell
python 08_local_client.py --repeat 3 --tokens 192 --prompt "Explain prefill versus decode in six concise sentences." --save results\warm-sequence.json
```

This issues sequential requests, not concurrent ones. It does not unload the
model. The first request may load it; later requests may reuse both weights
and a prefix. The output file will not overwrite an existing run.

Compare first output, load, wall time, and generated-token rate. Read the
answers, not just the numbers. The script saves its requested settings and
post-run model metadata. Record the server-level settings separately.

With thinking enabled, time to first generated content may not equal time to
first answer. You can observe that explicitly:

```powershell
python 08_local_client.py --think --tokens 512 --prompt "Explain why 17 times 23 is 391." --save results\thinking.json
```

Do not compare this directly as a speed contest against a different prompt.
Reasoning can consume the output budget without yielding an answer; the client
reports that condition. A token cap is not a requirement to produce that many
tokens. `done_reason=length` warns that the response may be incomplete.

## Lab 8: plan a controlled comparison

Use the same prompt and output limit. Record a baseline, then change one
variable. The following context-capacity experiment is deliberately small:

```powershell
python 08_local_client.py --context 4096 --repeat 3 --tokens 192 --save results\ctx4k.json
python 08_local_client.py --context 8192 --repeat 3 --tokens 192 --save results\ctx8k.json
```

This changes **configured context capacity**, not actual input length. It may
reallocate or reload runner state. Treat the first request in each group
separately. Take memory observations while each configuration is loaded.
Do not claim it measures long-prompt quality or 8K prefill throughput.

For a later input-length experiment, prepare fixed documents at several token
lengths, hold `num_ctx` sufficient and constant, and ask the same retrieval
question. Record actual prompt tokens and check the retrieved answer.

### Experiments to postpone until the baseline is understood

| Change | Required evidence before adoption | Rollback |
| --- | --- | --- |
| Q8 cache | Backend supports this hybrid; effective cache changes; quality remains acceptable | Restore FP16 and restart intended server |
| FlashAttention | Effective state differs; compare matching workloads and memory | Restore previous setting and restart |
| Two request slots | No unexpected offload; per-user latency and aggregate throughput recorded | Restore one slot |
| Larger weight quantization | Record exact artifact/digest, placement and task scores | Request the previous model artifact |
| Different runtime | Verify model/template support and preserve the original deployment | Return to original endpoint |

The existing launcher explicitly sets FP16 cache and one slot. Changing an
environment variable in a different terminal will not override it.
Do not label a comparison "Q8 vs FP16" until logs confirm the effective setting.

## Experiment record template

Save your own records under the ignored `results` directory. Include:

```text
Question:
Hypothesis:
Model tag and immutable digest:
Runtime and driver:
Server settings and evidence they took effect:
Request settings:
Prompt suite and actual input/output token counts:
Power mode / battery / background workload:
Warm-up and prefix-reuse policy:
Measured repetitions:
First output / first answer / wall time / decode tokens per second:
GPU memory measurement method and sampling interval:
Quality rubric and failures:
Decision, uncertainty, rollback:
```

A good result can be "no useful improvement." Do not keep a setting merely
because its name sounds more advanced.

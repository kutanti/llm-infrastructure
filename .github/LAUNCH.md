# Launch drafts

Prepared for publication; not posted. The repository is currently private.
These links will not work for readers without access until its visibility changes.
Before posting, confirm public visibility and that the commit attribution is
acceptable. The earlier secret review covered the Git contents and history at
that time, not every future change.

## Main announcement

What happens between a prompt and an answer?

I've put together runnable notes that start with tokens, embeddings, and gradient
descent, then move through transformers to a local Qwen deployment.

The first training example has just 68 parameters. It learns a simple next-token
rule, and you can inspect every weight update. Another example shows a causal
transformer block, including why changing a future token cannot affect an earlier
prediction.

The deployment section follows a different question: why did one request take
102 seconds to start answering, while the next started in 0.19 seconds?
It breaks down the recorded load, prefill, and decode times rather than guessing
at an optimization.

Basic Python is enough to start. The fundamentals run on a CPU with NumPy;
the local inference exercises use Ollama.

https://github.com/kutanti/local-llm-learning

If you work through an exercise, I'd like to know where the explanation stops
making sense.

## Worked-example post: accuracy can hide uncertainty

A tiny next-token model can get every answer right and still have a lot left
to learn.

In this example, one update puts the correct token first for all four possible
inputs. Accuracy is already 100%. But each correct token only has about 26%
probability in a four-token vocabulary.

After more updates, the correct-token probabilities reach roughly 99.7%.
Accuracy hasn't changed; cross-entropy has dropped substantially.

That's why the example prints both metrics. Accuracy asks which candidate came
first. Cross-entropy also measures how much probability the target received.

This is a 68-parameter model trained on synthetic alternating patterns, not a
language benchmark. The complete training loop is short enough to read:

https://github.com/kutanti/local-llm-learning/blob/main/foundations/02_train_a_small_model.py

## Worked-example post: count the cache before changing it

Would an 8-bit KV cache help a model on a 6 GB GPU?

First count how much cache that model actually has.

Qwen3.5-4B has 32 text layers, but only 8 use full attention. At a 4,096-token
context, one sequence's conventional FP16 K/V arrays occupy:

```text
2 * 8 layers * 4 KV heads * 256 dimensions * 4096 tokens * 2 bytes
= 128 MiB
```

Q8_0 stores 32 values plus a two-byte scale in 34 bytes. That gives 68 MiB for
the same arrays: a saving of 60 MiB.

This excludes recurrent state and runtime allocations, and support still depends
on the backend. But it gives a concrete starting point: a 60 MiB saving is
unlikely to be the first thing to investigate when the visible problem is a
minute spent loading the model.

The calculation and runnable example:

https://github.com/kutanti/local-llm-learning/blob/main/02_kv_cache_size.py

## Posting order

Publish one worked example where you already participate, with a link to the
specific file. Use the main announcement for your profile or a project-introduction
post. Follow a community's promotion rules rather than posting the same message
everywhere.

Keep quoted results attached to their workload. The initial laptop run is a
case study, not a claim about expected RTX 4050 performance.

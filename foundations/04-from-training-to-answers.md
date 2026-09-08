# 4. From a pretrained model to a useful answer

The previous chapters explain how a network can predict and learn. A practical
assistant adds data preparation, several training choices, decoding rules,
application context, and evaluation.

Keep three questions separate:

```text
What did the weights learn?
What information did this request supply?
How did the application use the output?
```

Those questions distinguish pretraining, prompting, retrieval, fine-tuning,
and tool use.

## Pretraining learns from many next-token examples

A causal language model can turn a document into many training examples by
predicting each next token from the preceding prefix. During training, it uses
the actual preceding tokens from the document. This is often called
**teacher forcing**.

At generation time, it instead conditions on its own previously generated tokens.
An early error can therefore put later predictions on a prefix unlike the
intended answer.

Pretraining data may include prose, code, mathematics, dialogue, and other
material. Filtering, deduplication, source balance, licensing, and contamination
control all matter. More tokens are not automatically better if they are low
quality or repeatedly duplicate the same content.

The objective is predictive, but solving it across diverse data can produce
representations useful for translation, code generation, and many other tasks.
That does not mean every generated claim is grounded in a source.

A **base model** is trained for continuation. Given a question, it might answer,
continue with more questions, or imitate a document containing questions.
It has not necessarily learned the interaction contract expected of an assistant.

## Instruction tuning teaches the interaction format

Supervised fine-tuning, or **SFT**, trains on examples of desired behavior:
instructions, conversation context, and target responses.
For chat data, the tokenizer's chat template and role markers are part of the input.

Training may mask loss on user tokens so the optimizer focuses on assistant
responses. The exact masking policy is a training choice, not an inherent feature
of the transformer.

Preference-based methods use information about which responses are preferred.
Some train a reward model and use reinforcement learning; methods such as DPO
use preference pairs directly in an optimization objective.

These methods can improve helpfulness and instruction following. Their outcome
depends on the examples and preferences. They can also create tradeoffs such
as verbosity, overconfidence, or agreement with a user's mistaken premise.

Instruction tuning is still training: weights change. A system prompt is
inference-time input: weights usually do not change.

## Fine-tuning, LoRA, and distillation

**Full fine-tuning** updates the selected base-model weights directly.
**LoRA** learns a low-rank update while keeping the original weight matrix frozen.
For a matrix with input dimension d_in and output dimension d_out:

```text
delta_W = A @ B
A has shape [d_in, rank]
B has shape [rank, d_out]
```

Instead of training `d_in * d_out` update values, this factorization trains
`rank * (d_in + d_out)`. With d_in=d_out=4,096 and rank=8:

```text
full matrix: 16,777,216 values
low-rank update: 65,536 values
```

This reduces trainable-parameter and optimizer-state requirements. Training
still needs the base model's forward computation, relevant activations, and
backpropagation through the computation used by adapters.

QLoRA combines a quantized frozen base with adapter training and other memory
techniques. It does not mean every training value and gradient is four-bit.
The adapter must be used with a compatible base checkpoint.

**Distillation** uses a teacher to train a student. The teacher might supply
probabilities, text responses, or intermediate targets. A student can be smaller,
but it is not a compressed KV cache. Generating distillation data, updating a
student, and evaluating it are separate costs.

## Context is not persistent learning

Show a model a new rule in a prompt:

```text
For this exercise, a "dax" means a blue triangle.
What color is a dax?
```

It can answer `"blue"` by using the context. This is not evidence that the
weights permanently learned the word. A new request without that context may
have no reason to give the same answer.

A chat interface can appear to remember because the application resends
conversation history or retrieves stored notes. KV caching can reuse processing
of a retained prefix. Neither is necessarily a parameter update.

Keep prompt history within the actual context budget. Application-level memory
requires a storage and retrieval policy; it is not created by setting an
arbitrarily large output limit.

## RAG supplies information instead of changing weights

Retrieval-augmented generation typically has this path:

```text
documents -> split into chunks -> index
question -> retrieve relevant chunks -> build prompt -> generate answer
```

A dense index embeds documents and questions into vectors. A lexical index
matches words or terms. Hybrid retrieval combines approaches; a reranker can
score a small candidate set more carefully.

Suppose a manual says a device should be charged at 12 V. Rather than hoping
the model memorized the current manual, retrieve that passage and provide it
with the question. The model still has to interpret it correctly.

The pipeline can fail at several points:

| Failure | Consequence |
| --- | --- |
| Wrong document version | A fluent answer cites obsolete information |
| Relevant passage not retrieved | The model lacks the needed evidence |
| Chunk split removes a condition | The answer overgeneralizes a partial rule |
| Too much irrelevant context | Extra prefill cost and distraction |
| Model ignores the passage | Retrieval succeeds but the answer is wrong |

Evaluate retrieval separately from answer generation. A citation marker does
not prove the cited passage supports the claim. Retrieved text is also untrusted
input and may contain instructions an application should not follow.

Fine-tuning is often appropriate for a repeated behavior or format.
Retrieval is often appropriate for changing reference information. Some systems
need both; neither is universally the replacement for the other.

## Tools add actions outside the model

A model can emit a structured request to a calculator or search tool. The
application decides whether to execute it, supplies arguments, receives a result,
and may return that result to the model for another step.

The model is not performing an HTTP request merely because it describes one.
Actual tool use needs an application loop and permissions.

Separate reading a document from writing a file, and writing a draft from
sending an email. Tool output can also be wrong or malicious. Treat generated
arguments as untrusted input; validate them before execution.

An **agent** generally adds a loop that chooses actions, inspects results, and
continues until a stopping condition. More steps can solve harder tasks but also
increase latency, cost, and opportunities for mistakes.

## Why plausible answers can be wrong

Next-token probabilities measure how the model distributes probability over
continuations under its learned computation. They are not a calibrated
probability that a full claim is true.

A model can confidently continue a mistaken premise, invent a reference, or
reproduce a common error in its training data. Temperature zero does not provide
fact checking. Asking for reasoning can help on some tasks, but a long explanation
can rationalize a wrong answer.

Ground facts in suitable evidence, use tools for exact operations when appropriate,
and evaluate the response against the task rather than its tone.

The right quality measure depends on the task:

| Task | Useful evidence |
| --- | --- |
| Code | Correct outputs on representative and edge cases, not syntax alone |
| Extraction | Exact fields, schema validity, and source correspondence |
| Retrieval QA | Correct evidence retrieved and claims supported by it |
| Summarization | Important facts retained without invented additions |
| Conversation | Instruction compliance, usefulness, and suitable uncertainty |

Leakage and benchmark contamination matter. A model can score well on examples
it effectively saw during training without generalizing to your workload.
Compare fixed prompts, model artifacts, decoding settings, and scoring rules.

## Connect this back to local inference

The full request now has identifiable parts:

```text
text and optional retrieved/tool context
    -> tokenizer and chat template
    -> embeddings and model layers
    -> logits and sampling
    -> generated tokens and optional tool requests
    -> application response
```

During ordinary inference, the weights remain fixed. Longer input requires
more prompt processing; longer output requires more generation steps. Multiple
requests need scheduling and state. Quantization changes storage and arithmetic.
KV caching reuses intermediate computation.

That is the starting point for [the laptop investigation](../GUIDE.md).
When a request is slow, you can now ask which operation is slow instead of
treating "the model" as one indivisible box.

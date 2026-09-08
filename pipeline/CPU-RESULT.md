# The first tiny-model run did not solve the task

The 400-step CPU run reduced held-out supervised cross-entropy from **5.566 to
0.541**. Free generation still produced the correct category for only **6 of 36**
test tickets. Always returning the training-majority category scores **12 of 36**.
That is a failed task-quality result, despite the much lower loss.

| Artifact | Correct / 36 | Invalid JSON / 36 | Serialized model bytes | Mean generation time |
| --- | ---: | ---: | ---: | ---: |
| Training-majority response | 12 | 0 | Not a model | Not measured |
| Tiny FP32 transformer | 6 | 16 | 610,541 | 94 ms |
| Same transformer, INT8 linear layers | 6 | 16 | 272,329 | 145 ms |

The selected measurements and configuration are in
[the machine-readable record](../data/pipeline-cpu-example.json). Full raw outputs,
checkpoints, and profiler traces remain under the ignored `results` directory.
The recorded training loop took 62 seconds; this is one run on a Windows CPU,
not an expected runtime on every machine.

## What the loss did and did not measure

Validation loss uses teacher forcing: the model receives the correct earlier
response bytes while predicting each next byte. Free generation instead feeds
back its own choices. A wrong byte can take later predictions away from the
training sequence.

The JSON punctuation and repeated words also occupy many supervised positions.
Predicting them well can lower average byte-level loss without classifying
tickets correctly. The model has only 72 synthetic training examples with
limited wording patterns. These observations explain why loss and task score
can differ; they do not isolate the cause of every failure in this run.

Do not fix this result by giving the model test labels, extracting a convenient
category word from malformed output, or changing the scoring rule after seeing
the answers. Use the validation set to investigate data coverage, position
dependence, capacity, and the training budget, then evaluate a final choice on
fresh held-out material.

## What quantization changed

The saved model file shrank by about **55%**, not 75%, because embeddings,
normalization, biases, and metadata were not all reduced to one-quarter size.
This is artifact size, not resident-memory or peak-allocation measurement.

The reported accuracy and invalid-output counts did not change. That does not
make either model suitable for the task. INT8 generation was slower in this
single comparison. Conversion overhead, tiny matrix shapes, and CPU kernel
selection are hypotheses to investigate, not measured causal explanations.

Both generation timings include the full autoregressive loop and use different
model artifacts in separate processes. Repeat matched workloads, record spread,
and examine a profile before declaring a speed regression or improvement.
Tokens here are bytes and special IDs, so their throughput cannot be compared
directly with Qwen's subword-token throughput.

This example is retained because infrastructure work needs a way to discover
that a smaller artifact is slower, or that lower loss did not deliver a useful
model. A pipeline that only reports successful-looking numbers would conceal
both findings.

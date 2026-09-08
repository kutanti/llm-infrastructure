# Work through the fundamentals

Use the chapter notes when needed, but try each calculation before opening
the answer. Commands assume you are in the repository root.

## 1. An ID is not a feature

Our toy tokenizer maps `"the cat sat on the"` to `[0, 1, 2, 3, 0]`.
The embedding table has shape `[5, 3]`.

What shape results from looking up the whole sentence? Are the two occurrences
of `"the"` initially different? Could their final hidden states differ?

```powershell
python foundations\01_tokens_and_probabilities.py
```

<details>
<summary>Answer</summary>

The result has shape `[5, 3]`: one three-feature vector per position.
Both occurrences of `"the"` select row 0. Their contextual states can differ
after position information and context-dependent computation enter the model.

</details>

## 2. A learned matrix is shared across positions

Let `X` have shape `[2, 6, 8]` and projection `W` have shape `[8, 12]`.

What is the shape of `X @ W`? How many learned values are in W? Would processing
a longer sequence require a larger W?

<details>
<summary>Answer</summary>

The output is `[2, 6, 12]`. W contains `8 * 12 = 96` parameters.
It is applied at each position, so increasing sequence length does not change
this matrix's shape. Activation and state storage can still grow.

</details>

## 3. Probabilities from scores

Compute softmax for logits `[2, 1, 0]`. Then subtract 2 from every logit and
calculate again. Why is the answer unchanged?

Which candidate does greedy decoding choose? Could sampling choose another?

<details>
<summary>Answer</summary>

The probabilities are approximately `[0.665, 0.245, 0.090]`.
Subtracting 2 multiplies numerator and denominator by the same factor `exp(-2)`,
which cancels. Greedy chooses index 0. Sampling can choose any entry with
nonzero probability.

</details>

## 4. One gradient-descent update

For `prediction = w*x`, use `x=2`, `y=6`, `w=1`, and
`loss=0.5*(prediction-y)^2`.

Calculate the gradient and the new weight for learning rate 0.1.
Repeat with learning rate 1.0. Did both updates help?

<details>
<summary>Answer</summary>

The gradient is `(2-6)*2 = -8`.
At rate 0.1, `w_new=1.8`, prediction=3.6, loss=2.88, down from 8.
At rate 1.0, `w_new=9`, prediction=18, loss=72.
Following the negative gradient with too large a step can make the result worse.

</details>

## 5. The logit gradient points toward the target

The predicted distribution is `[0.7, 0.2, 0.1]` and the correct token is index 1.
Calculate its cross-entropy and the three logit gradients.

<details>
<summary>Answer</summary>

Loss is `-ln(0.2)`, approximately 1.609.
The gradient is `[0.7, -0.8, 0.1]`. Subtracting a small multiple raises the
target logit and lowers the others.

</details>

## 6. Train something small enough to understand

```powershell
python foundations\02_train_a_small_model.py
python foundations\02_train_a_small_model.py --steps 1
python foundations\02_train_a_small_model.py --learning-rate 0.05
```

Each invocation starts with the same initial seed. Compare final training loss
and next-token probabilities. A smaller rate may need more updates to reach the
same loss.

Locate the embedding, output matrix, and bias in the script. Account for its
68 parameters. What earlier information does it discard when generating?
If accuracy reaches 100% before loss becomes small, are the metrics inconsistent?

<details>
<summary>Answer</summary>

Four vocabulary entries times eight embedding features gives 32 parameters.
An `[8, 4]` output matrix adds 32. Four bias values give 68 total.
Only the most recent token enters the forward pass; earlier tokens are ignored.

With default settings, a representative run reduced training loss from about
1.37 to 0.0024 and learned all four deterministic transitions. The validation
sequences use that same rule, so this is not evidence of broad language ability.

With this seed, one update already puts the right token first in every case,
but its probability is only about 0.26. Accuracy counts the first-place choice;
cross-entropy also responds to how much probability the correct token receives.
The model can therefore have 100% greedy accuracy and still improve its loss
substantially.

</details>

## 7. A limitation more training cannot remove

Imagine the bigram model sees two contexts ending in the same token:

```text
river bank
savings bank
```

Can training make this implementation produce different distributions after
those contexts without changing its inputs or architecture?

<details>
<summary>Answer</summary>

No. The implementation supplies only the last token. Both contexts become the
same embedding lookup and therefore the same logits. More training cannot
recover information that the computation never receives.

It could learn a mixture of possible continuations, but not choose based on
the missing earlier word.

</details>

## 8. Read the causal attention matrix

```powershell
python foundations\03_transformer_block.py
```

Why is the upper-right triangle zero? Why must the first row be `[1, 0, 0, 0]`?
The script changes only the final token: which output positions may change?

<details>
<summary>Answer</summary>

The causal mask blocks keys after each query position. Position 0 has only one
allowed key: itself. Earlier positions cannot read the changed final token,
so their logits remain unchanged. The final position's output may change.

The code explicitly checks these properties. This is a computation check,
not a test of language quality.

</details>

## 9. Two kinds of information mixing

In the transformer example, attention and the feed-forward network both change
hidden states. Which directly combines information from different token positions?
Why have a nonlinearity in the feed-forward path?

<details>
<summary>Answer</summary>

Attention blends values from available positions. The standard FFN applies the
same network independently at each position, transforming its features.
Without a nonlinearity, successive linear maps could collapse into one linear map.
The input to the FFN is already contextualized by attention.

</details>

## 10. Choose the mechanism for the job

For each situation, name a first approach and explain its limitation:

- A manual changes every week and answers must cite the current version.
- Outputs consistently need a specialized format and style.
- The model must multiply two large integers exactly.
- An initial model load takes a minute but later requests are fast.

<details>
<summary>Answer</summary>

Current documents suggest retrieval with version control and citation evaluation.
Retrieval can still miss the needed passage or feed the wrong version.

A format/style requirement may be addressed first with prompting or structured
output constraints, then fine-tuning if a repeated behavior warrants it.
Syntactic constraints alone do not make the content correct.

Exact arithmetic suggests a validated calculator tool. The application must
check arguments and decide what execution is permitted.

An initial-load delay suggests investigating residency and preload policy.
Fine-tuning or a larger context limit does not address loading.

</details>

## 11. Trace one complete request

Write down the stages for:

> Using the attached inventory, which shelf holds item KESTREL?

Include how the text becomes numbers, where weights are used, how attention
handles the context, how logits become tokens, and why the answer does not
permanently update the model.

Then follow [the input-length experiment](../LABS.md#7-distinguish-reserved-context-from-actual-input).
It supplies inventories and records the local model's answers.

<details>
<summary>One possible trace</summary>

The application builds a prompt from instructions, inventory, and question.
The chat template and tokenizer turn it into IDs. Embedding lookup produces
vectors; layers process them using learned weights, positional information, and
causal context mixing. The last position projects to vocabulary logits.

The sampler selects a token, which becomes part of the next prefix. The runner
reuses retained state where supported and continues until stopping.
Ordinary inference does not compute optimizer updates, so supplying an inventory
does not permanently teach it to the weights.

</details>

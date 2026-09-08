# 1. From text to numbers

Suppose a model is shown:

```text
the cat sat on the
```

Its immediate job is not to return an entire sentence. It computes a distribution
over the next token. It might assign high probabilities to `"mat"` and `"floor"`,
lower probabilities to other continuations, then select one. The selected token
becomes part of the input for the next step.

What counts as a token, and where do those probabilities come from?

## A tokenizer assigns IDs

A tokenizer turns text into a sequence of integer IDs using a fixed vocabulary
and encoding rules. A tiny word vocabulary could be:

```text
"the" -> 0
"cat" -> 1
"sat" -> 2
"on"  -> 3
"mat" -> 4
```

Our prompt becomes `[0, 1, 2, 3, 0]`. ID 4 is not larger or more meaningful than
ID 1. These are addresses, not measured quantities.

Word-only vocabularies struggle with unfamiliar words and spelling variants.
Character tokenizers have small vocabularies but produce long sequences.
Modern language models commonly use subword or byte-based schemes that trade
vocabulary size against sequence length.

For an illustrative subword vocabulary:

```text
"playing" -> ["play", "ing"]
"played"  -> ["play", "ed"]
```

That is an example of segmentation, not the output of a particular Qwen tokenizer.
Real tokenizers often encode leading spaces and punctuation in ways that surprise
someone counting words. Some can fall back to bytes; unknown-word behavior
depends on the tokenizer.

Byte-pair encoding starts with small units and learns merges that compress
frequent adjacent pairs. It learns the **vocabulary construction**, not the
neural model's understanding of language. A tokenizer is usually fixed while
the language model trains.

Changing tokenizers is not like changing a text editor. The model's embedding
rows and output scores refer to the original vocabulary. A different ID mapping
would give existing weights the wrong meanings.

## An embedding is a learned lookup table

The next step looks up each ID in a matrix. If the vocabulary has 5 entries and
the embedding width is 3, the matrix has shape `[5, 3]`:

```text
E = [ 0.1,  0.2, -0.1 ]   row 0: "the"
    [ 0.7, -0.3,  0.5 ]   row 1: "cat"
    [ 0.0,  0.8,  0.4 ]   row 2: "sat"
    [-0.4,  0.2,  0.1 ]   row 3: "on"
    [ 0.6, -0.2,  0.3 ]   row 4: "mat"
```

Looking up `"cat"` returns `[0.7, -0.3, 0.5]`. Looking up five token IDs returns
a `[5, 3]` array. The first dimension counts positions; the second counts
features at each position.

The example numbers are hand-chosen. In a trained model, optimization changes
them to help prediction. Individual coordinates usually do not have labels
such as "animalness." Useful structure can be distributed across many coordinates.

An equivalent operation is multiplying a one-hot vector by `E`:

```text
[0, 1, 0, 0, 0] @ E = E[1]
```

An implementation uses the lookup because constructing a huge one-hot vector
would waste work.

The same token starts with the same embedding row, but its **contextual hidden
state** changes as it passes through layers. The token `"bank"` can end up with
different representations in `"river bank"` and `"bank account"`. A static lookup
and a contextual representation are not the same object.

## Enough linear algebra to read a model diagram

A **scalar** is one number. A **vector** is an ordered list of numbers. A **matrix**
is a rectangular array. A **tensor** generalizes this to more dimensions.

The dot product multiplies corresponding coordinates and adds:

```text
[1, 2, -1] dot [3, 0, 4] = 1*3 + 2*0 + (-1)*4 = -1
```

Matrix multiplication performs many dot products. For a token vector `x` with
3 features and a projection `W` with shape `[3, 2]`:

```text
x @ W has shape [2]
```

For 10 token positions, `X` has shape `[10, 3]`, and `X @ W` has shape `[10, 2]`.
Each position uses the same weights. This weight sharing lets a model process
different sequence lengths without learning a new matrix for every position.

With multiple sequences, a hidden-state tensor often has shape:

```text
[batch_size, sequence_length, hidden_width]
```

Reading shapes is more useful at first than memorizing model names.
Ask what each axis represents and which axes an operation combines.

### Similarity is useful, but not truth

Cosine similarity measures the angle between vectors:

```text
cosine(a, b) = (a dot b) / (length(a) * length(b))
```

This helps retrieve related items when the embedding model was trained for that
purpose. But two statements can be similar and contradict each other:
`"the door is open"` and `"the door is not open"`.

A language model's token embeddings are also not automatically a good
sentence-retrieval model. Pooling and training objective matter.

## Hidden states become logits

After the model processes the context, a final projection produces one score per
vocabulary entry. These scores are **logits**. They can be negative and need not
sum to anything.

For a toy vocabulary of three possible next tokens:

```text
logits = [2, 1, 0]
```

Softmax converts these scores to probabilities:

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
p   = approximately [0.665, 0.245, 0.090]
```

Implementations subtract the largest logit before exponentiating. Adding or
subtracting the same constant from every logit leaves softmax unchanged, but
avoids overflow.

A model with a vocabulary of 100,000 entries produces 100,000 next-token scores,
not a confidence score for the whole answer.

## Selecting a token is a separate step

**Greedy decoding** chooses the largest probability. **Sampling** draws a token
from the distribution. They can produce different text from identical logits.

Temperature rescales logits before softmax:

```text
p = softmax(logits / temperature)
```

For logits `[2, 1, 0]`:

| Temperature | Approximate probabilities |
| --- | --- |
| 0.5 | `[0.867, 0.117, 0.016]` |
| 1.0 | `[0.665, 0.245, 0.090]` |
| 2.0 | `[0.506, 0.307, 0.186]` |

Lower positive temperature concentrates probability on already-preferred tokens.
It does not look up which token is factually correct. A runtime's temperature-zero
mode usually means greedy selection; it should not literally divide by zero.

**Top-k** keeps the k highest-scored candidates and renormalizes.
**Top-p** keeps a high-probability prefix of candidates whose cumulative probability
reaches a threshold, then renormalizes. Both alter the sampling distribution.
Their interaction with temperature depends on the sampler's order of operations.

Once a token is chosen, append it to the sequence and recompute the next-token
distribution. Stop when an end token is emitted, a configured stop condition
matches, or a length limit is reached.

Run `foundations\01_tokens_and_probabilities.py` from the repository root.
It traces a word-token lookup, an embedding projection, and temperature sampling.
The next chapter explains how training changes the numbers in those tables.

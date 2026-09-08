# 3. How a token uses its context

The bigram model in chapter 2 only sees the preceding token. A transformer
builds a representation of each position using other available positions.

Consider:

```text
Maya gave Ana her book.
```

To predict a continuation, information about names, the action, and possible
references may matter. The architecture does not contain a hand-written pronoun
resolver. It provides operations that training can use to combine contextual
information.

## Follow the shapes

Use a deliberately small decoder with:

```text
batch size B = 1
sequence length T = 4
hidden width D = 8
attention heads H = 2
head dimension Dh = 4
vocabulary size V = 12
```

Token IDs have shape `[B, T]`. Embedding lookup gives `[B, T, D]`.
Each block takes a tensor of that shape and returns one of the same shape.
The final vocabulary projection gives logits with shape `[B, T, V]`.

During training, many positions supply predictions at once. If the input is
`[a, b, c, d]`, the shifted targets might be `[b, c, d, e]`. The prediction at
position 0 must not see `b`, even though `b` appears later in the training input.

At generation time, the last position's logits predict the next token.

## Position must enter the computation

Without position information, ordinary self-attention cannot distinguish order
in the way language needs. Compare:

```text
dog bites person
person bites dog
```

They contain the same token types but do not mean the same thing.

Some transformers add learned or sinusoidal position vectors to embeddings.
Many modern decoders use rotary position embeddings, or **RoPE**, which rotate
query/key coordinate pairs according to position. This lets attention scores
depend on positional relationships.

The runnable block uses fixed sinusoidal positions because the numbers are easy
to inspect. It is an example transformer, not a reproduction of Qwen's position
scheme. Position handling is also one reason cached prefixes must correspond
to compatible positions.

## One decoder block

A common pre-normalization layout is:

```text
X1 = X  + attention(norm(X))
X2 = X1 + feed_forward(norm(X1))
```

The additions are **residual connections**. They provide a direct path for the
input representation and gradients, while each sublayer learns an adjustment.
They help optimization; they do not make every learned change useful.

Normalization keeps the scale of features manageable. LayerNorm subtracts a mean
and divides by a standard deviation across features, usually followed by learned
scale and bias. RMSNorm divides by the root mean square without subtracting the
mean, commonly with a learned scale.

Different model families use different normalization placement and details.
Understanding the data flow is more important than memorizing one block diagram.

## Project queries, keys, and values

For ordinary multi-head attention:

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

The weights in our example have shape `[D, D]`. The results start as `[B, T, D]`,
then are reshaped into `[B, H, T, Dh]`.

Each head forms:

```text
scores = Q @ K_transposed / sqrt(Dh)
```

The scores have shape `[B, H, T, T]`: for each query position, one score for
every key position. The square root scaling helps keep dot-product magnitudes
from growing excessively with head dimension.

The head is a learned subspace, not a hard-coded job title. It is tempting to
say "head 1 handles grammar, head 2 handles facts," but that is not guaranteed
by the architecture.

### Causal masking

Before softmax, block future positions with negative infinity:

```text
query position 0 can read keys 0
query position 1 can read keys 0, 1
query position 2 can read keys 0, 1, 2
query position 3 can read keys 0, 1, 2, 3
```

Softmax gives masked entries zero weight. Each row of the remaining weights
sums to one. Multiplying by V produces weighted contextual features.

The heads are concatenated back into `[B, T, D]` and projected through an output
matrix. This mixes information from the different heads before the residual addition.

The runnable block changes only the last input token and checks that earlier
outputs remain unchanged. If those outputs changed, the implementation would
be leaking future information.

## The feed-forward network does a different job

Attention moves information between positions. A standard feed-forward sublayer
applies the same nonlinear network independently at each position:

```text
FFN(x) = activation(x @ W1 + b1) @ W2 + b2
```

It often expands the feature dimension and then projects back. For D=8 and
intermediate width 24:

```text
[B, T, 8] -> [B, T, 24] -> [B, T, 8]
```

Modern models commonly use gated variants such as SwiGLU, which combine a
nonlinear branch with another projected branch by elementwise multiplication.
Large feed-forward matrices account for substantial parameter storage and work.
Attention is not the entire model.

A **mixture-of-experts** architecture replaces some dense feed-forward computation
with multiple expert networks and a router that selects a subset for each token.
It can increase total parameter capacity without activating every expert on every
token. The inactive experts still have to be stored somewhere.

## Stack blocks and predict

Several blocks successively transform the hidden states. The last state is
normalized and mapped to vocabulary logits:

```text
logits = final_norm(hidden) @ output_weights
```

Some models tie the output weights to the transpose of the input embedding
matrix. With vocabulary V and hidden width D, an embedding matrix alone has
`V * D` parameters. For V=100,000 and D=4,096, that is 409.6 million values.
Tying can avoid a second separate matrix of that size.

A parameter count describes learned values, not their byte precision.
Four billion parameters at FP16 is roughly eight billion bytes of values;
quantization changes the representation, not the nominal parameter count.

Run:

```powershell
python foundations\03_transformer_block.py
```

The output shows the shapes, one attention matrix, and the result of the causal
invariance check. Its random weights have not learned grammar or facts. The
point is to expose the operations that chapter 2's training process would update.

## Decoder, encoder, and encoder-decoder

The block above is causal, suitable for a **decoder-only** language model.
An **encoder** often uses bidirectional attention so each position can read
both earlier and later positions in the input. This is useful for representation
learning and tasks such as classification.

An **encoder-decoder** model processes an input with an encoder and generates
an output with a decoder. The decoder can use cross-attention to the encoder's
representations. Input and output sequence lengths can differ.

"Decoder" does not mean text decoding from bytes; it names a role in the neural
architecture. Likewise, a tokenizer's decoding function simply converts token
IDs back into text.

## Not every current model is a pure transformer

Recurrent and state-space layers maintain evolving state instead of retaining
the same per-token attention arrays. Hybrid models combine such layers with
full attention. Our Qwen3.5-4B deployment has 24 Gated DeltaNet layers and
8 full-attention layers.

That difference becomes important for memory budgeting. It does not change
the basic learning loop: forward prediction, loss, gradients, parameter updates.
The inference guide calculates the cache for the actual model only after
identifying which layers retain which kind of state.

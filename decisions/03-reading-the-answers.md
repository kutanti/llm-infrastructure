# 3. Reading the answers

Every answer is computed from one probability per allowed option. This lesson reads the
three answer types. It then compares three different numbers that are all called
"confidence".

## The three answer shapes

| Type | Jev returns (typesafe-sdk 0.7.2 schema) | Laya 0.3.21 also returns |
| --- | --- | --- |
| choice | `choice`, `confidence`, and `probabilities` keyed by label | `answer_confidence` |
| score | `score`, `confidence`, `legend`, and `probabilities` keyed `"0"`, `"1"`, ... | `answer_confidence` |
| noul | `noul`, the probability of yes | `confidence` and `answer_confidence` |

Jev's noul answer has no `confidence` field, because `noul` is already a probability.
Laya also adds an `action` block, which this course does not use.

## One choice answer, three "confidences"

Run the example:

```powershell
python -m decisions.answers
```

The first block uses hand-chosen probabilities for `department`, not model output:

```text
department  {'billing': 0.7, 'account': 0.1, 'technical': 0.1, 'other': 0.1}
  choice = 'billing'
  top probability     0.700   (Laya: answer_confidence)
  above uniform       0.600   (third-party fit to Jev's confidence)
  1 - entropy/log(4)  0.322   (Laya: confidence)
```

With `n` options and top probability `max(p)`:

- **Top probability** is `max(p) = 0.70`. Laya returns it as `answer_confidence`.
- **Above uniform** is `(n × max(p) − 1) / (n − 1) = (4 × 0.70 − 1) / 3 = 0.60`. It is 0
  when every option is equally likely and 1 when one option has all the probability.
  TypeSafe does not publish how Jev computes `confidence`. An independent analysis found
  that this formula matches Jev's choice confidence and is a lower bound for its score
  confidence. Treat it as a description of the field, not its definition.
- **One minus normalized entropy** is `1 − H(p) / ln(n)`, with `H(p) = −Σ p ln p`. Here
  `H = −(0.7 ln 0.7 + 3 × 0.1 ln 0.1) = 0.940` and `ln 4 = 1.386`, so the value is
  `1 − 0.940 / 1.386 = 0.322`. Laya returns this as `confidence` for choice and score
  answers.

So `0.70`, `0.60`, and `0.32` all describe the same answer. Laya's HTTP API documentation
states the rule for its two fields: "Never compare the two against one threshold." The
same applies to a threshold carried over from Jev.

## A score is an average

```text
urgency: the score is an expected level, so check where the probability is
  probabilities [0.0, 1.0, 0.0]  ->  score 1.00, top probability 1.00
  probabilities [0.5, 0.0, 0.5]  ->  score 1.00, top probability 0.50
  Both average to level 1; the second puts no probability on level 1 itself.
```

Jev's schema defines `score` as the "probability-weighted average of the rubric levels"
and notes it "may fall between integer levels". A score of 1.00 can mean "certainly
Soon" or "either Can wait or Now". Before acting on a score, read `probabilities`. To
page the on-call team for urgent tickets, test `P(Now) ≥ 0.5` rather than `score ≥ 1.5`.

## A noul is a probability

`noul = 0.92` means the probability of yes is 0.92. Its top probability is
`max(p, 1 − p) = 0.92`. A value near 0.5 is the uncertain one: `noul = 0.08` is as
confident a "no" as 0.92 is a "yes".

## More options raise two of the numbers

The last block keeps the top probability at 0.6 and spreads the rest evenly over more
options:

```text
Same top probability (0.6), more options (rest spread evenly):
 options   top p  above uniform  1 - H/log n
       2    0.60          0.200        0.029
       4    0.60          0.467        0.198
      10    0.60          0.556        0.326
      20    0.60          0.579        0.382
A threshold on one definition does not transfer to another, or to another option count.
```

The model gives its answer the same 0.6 in every row, yet both formula-based numbers rise
as options are added. A gate at 0.5 on "above uniform" rejects the four-option answer and
accepts the ten-option one. The independent analysis calls this option padding. The table
holds `max(p)` fixed to isolate the formulas; a real model's probabilities also shift when
its options change.

## A rule for gating

Compute one statistic from `probabilities`, the same way for every backend, and set its
threshold separately for each question, from data (lesson 4). A 0.9 top probability means
something different for a two-option noul, where a coin flip scores 0.5, than for a
four-way choice, where a guess scores 0.25.

`read_answers` in [`answers.py`](answers.py) follows this rule. It checks each answer
against the question that was asked. It rejects a wrong type, a label you did not offer,
probabilities for the wrong keys or not summing to about 1, and a score outside the
levels. It then records `top_probability` beside the backend's own `confidence` fields.
`decisions.ask` bands answers on `top_probability`.

References: [TypeSafe Python SDK wire schema, v0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python/blob/v0.7.2/src/typesafe_sdk/_schemas/models.py),
[Laya HTTP API: confidence](https://github.com/NandhaKishorM/laya/blob/v0.3.21/docs/http-api.md#confidence-two-numbers-not-interchangeable),
[independent analysis of Jev's confidence field (third-party)](https://github.com/slashdaemon/jev-expert/blob/5e7504d564bb7f590e65c0b5c949e0f9ae535129/references/confidence.md).

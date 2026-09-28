# 4. Thresholds and calibration

A threshold turns a probability into an action. Choose it from the cost of a mistake
and from accuracy measured on labeled tickets. First check whether the probabilities
mean what they say.

## Three bands

`decide` in [`calibration.py`](calibration.py) sorts each answer by its top probability:

| Band | Default rule | What happens |
| --- | --- | --- |
| act | top probability ≥ 0.9 | The program acts on the answer |
| confirm | ≥ 0.6 | The program proposes the answer, and a person accepts or corrects it |
| human | below 0.6 | A person decides without a proposal |

The defaults are examples, not recommendations. Set the thresholds for each question
from what a wrong answer costs:

- A ticket routed to the wrong team costs a reassignment. Act on `department`
  automatically when the probability is high.
- A refund sent by mistake costs money and may not be recoverable. Use
  `refund_requested` to prioritize a ticket or prepare a form, and let a person approve
  the payment.

If acting wrongly costs `C_wrong` and asking a person costs `C_review`, acting is cheaper
on average when `(1 − p) × C_wrong < C_review`. That is, when
`p > 1 − C_review / C_wrong`. If a wrong route wastes 5 minutes and a review takes 1
minute, act above `1 − 1/5 = 0.8`. This arithmetic assumes `p` is calibrated: among
answers given 0.8, about 80% are right.

## Measuring calibration

Measure calibration on labeled tickets the model has not been tuned on:

- **ECE** (expected calibration error) sorts answers into 15 equal-width bins by top
  probability. In each bin it takes the gap between accuracy and mean top probability,
  and it averages the gaps weighted by bin size. Zero is perfect. With few examples
  most bins are empty or tiny, so `decisions.ask` reports ECE only from 100 labeled
  answers per question.
- **Brier score** is the mean squared distance between the probability vector and the
  one-hot true label.
- **NLL** (negative log-likelihood) is the mean of `−ln p(true label)`. It punishes a
  confident mistake hardest.

Brier and NLL are *proper scoring rules*: a forecaster gets the best expected score by
reporting its true probabilities. Training on them does not guarantee calibration on your
tickets, though. Laya's README says it trains against strictly proper scoring rules. It
also says "both checkpoints are over-confident as shipped", and that `laya-multilingual`
ships with no fitted temperatures.

## Temperature scaling, simulated

```powershell
python -m decisions.calibration
```

The script simulates a three-department router whose correct logits are known. It then
reports them three times too sharp, a deliberately over-confident model. It fits one
temperature `T` on 2,000 validation tickets and reports on 2,000 separate test tickets:

```text
Temperature fitted on validation only: T = 3.000

test split               accuracy mean top p     ECE   Brier     NLL
as reported                 0.778      0.922   0.144   0.354   0.875
divided by T=3.00           0.778      0.775   0.014   0.312   0.543
```

- **Accuracy does not change.** Dividing every logit by the same positive `T` keeps the
  order of the options, so the answer stays the same. Only the probabilities change.
- **The probabilities start to mean what they say.** Mean top probability falls from
  0.922 to 0.775, next to the accuracy of 0.778, and ECE falls from 0.144 to 0.014.
- **Fit on one split, report on another.** A temperature chosen on the test tickets
  would make the test report look better than new traffic will be.

`fit_temperature` searches `T` from 0.5 to 5.0, the range Laya clamps its stored
temperatures to. Laya fits one temperature per question type and option count, and you
should fit per question too. Jev returns probabilities, not logits. Use `ln p` as the
logits, which gives the same rescaling, after clipping exact zeros. The third-party
analysis of Jev reports that its probabilities are rounded to 0.01 and are often exactly
0 or 1, which limits what one temperature can repair. The same analysis reports a large
ECE reduction from isotonic regression fitted on about 1,000 labels.

## Coverage and accuracy

The calibrated test split, acting only at or above a threshold:

```text
 threshold  coverage  accuracy of those
       0.0    100.0%              77.8%
       0.6     79.3%              84.6%
       0.8     51.9%              91.5%
       0.9     33.0%              94.5%

Bands at act >= 0.9, confirm >= 0.6:
  act       33.0% of tickets,  94.5% correct
  confirm   46.4% of tickets,  77.6% correct
  human     20.6% of tickets,  51.8% correct
```

A higher threshold automates fewer tickets, and more of those are right. Report both
numbers with their denominators. "94.5% accurate" hides that two-thirds of the tickets
still went to people. Raising a threshold moves work to people; it does not make the
model better.

## Confidence is not correctness

Laya's README states: "Confidence orders decisions; it does not establish that a
decision is correct." In its negation example, one wrong answer had probability 0.9998.
The third-party analysis of Jev reports 44.7% accuracy with an average probability of
0.74 on a question whose answer needed a policy that was absent from the state. Keep hard
rules, such as permissions and refund limits, in code.

Check which field an example gates on before copying its threshold. Laya's README gating
example reads `confidence`, the entropy value. Its abstention section says to use
`answer_confidence` instead.

## A threshold belongs to one configuration

A threshold fitted today is tied to the model version, the question wording, and the
serving setup. The alias `jev-latest` can move to a new model. A Laya update can change
weights or temperatures. Laya's README reports that on its GPU parity test set, bf16
autocast moved probabilities by up to 0.073 against fp32 and flipped 3 of 864 answers.
Record the version, and measure again after any change. Lesson 5 shows how to pin
versions.

References: [On Calibration of Modern Neural Networks (temperature scaling, ECE)](https://arxiv.org/abs/1706.04599),
[Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437),
[Laya README: calibration](https://github.com/NandhaKishorM/laya/blob/v0.3.21/README.md#calibration),
[Laya README: confidence gating](https://github.com/NandhaKishorM/laya/blob/v0.3.21/README.md#automated-confidence-gating),
[independent analysis of Jev's confidence field (third-party)](https://github.com/slashdaemon/jev-expert/blob/5e7504d564bb7f590e65c0b5c949e0f9ae535129/references/confidence.md).

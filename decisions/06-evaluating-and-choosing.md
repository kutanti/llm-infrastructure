# 6. Evaluating and choosing

Lessons 1 to 5 showed what the two models return and how to run them. This lesson is
about choosing: Jev, Laya, or neither. Base the choice on your own questions and your
own tickets. A vendor's table can suggest where to look, but it cannot make the choice.

## What decides the choice

Several of these rows are settled before any accuracy test.

| Factor | Jev | Laya |
| --- | --- | --- |
| Where the ticket text goes | To TypeSafe's API | Stays on the machine that runs Laya |
| Who operates it | TypeSafe | You: install, update, monitor, and secure it |
| Cost | Billed per input token; `usage.input_tokens` counts them | Hardware, power, and your time; no per-call fee |
| Changing the model | Only through instructions and criteria. SDK 0.7.2 has two calls: `system_one` and `models.list` | Open weights; Laya's README describes fine-tuning on your own decisions |
| Pinning a version | Request a model name from `models.list()` instead of `jev-latest` | Pin `laya==0.3.21` and a Hub commit with `--revision` |
| Many options | Laya's README reports Jev ahead above about 20 options | Options share a 192-token budget on the English checkpoint (lesson 2) |
| Accuracy on your tickets | Measure it | Measure it; the base checkpoints are weak zero-shot (lesson 1) |

If customer text may not leave your network, only a local model such as Laya remains. If
nobody can run a model service, a hosted API is simpler. Check your organization's rules
before you send customer text to any hosted API.

## Reading a vendor comparison

Laya's README compares Laya with Jev 1.13.0. It says of its own Jev column: "Jev figures
are third-party published, never measured here (no TypeSafe API access), so sample sizes
and prompts differ." The README states most of its limits openly, which makes the table
good practice for reading any vendor's numbers, TypeSafe's included. These are the vendor's
figures; this course measured none of them.

| Row | Jev 1.13.0 | Laya (routed) | What to check |
| --- | --- | --- | --- |
| typed-decisions accuracy | 0.727 | 0.766 | Laya's figure comes from `laya-typed-decisions`, a checkpoint fine-tuned on that benchmark's four synthetic workflows. The base English checkpoint scores 0.362 on the same decisions |
| Banking77 accuracy | 0.870, 72 labels | 0.425, 77 labels | The two sides used different label counts. The README attributes Laya's result to the option budget |
| ECE | 0.246 | 0.081 | Laya's figure is after temperature fitting. The same README's benchmark table lists Jev at 0.144 and the fine-tuned Laya checkpoint at 0.213 before fitting |
| p50 latency, 1 question | 236–276 ms | 32.8 ms | Jev's range comes from third-party benchmarks of the hosted API, so it includes a network. Laya's figure is one question on a T4 GPU |
| Cost | $0.042 per 1M tokens | $0 self-hosted | Self-hosting still costs hardware, power, and operations |

Ask these questions of any comparison table:

- Which exact model produced each number, and was it adapted to this benchmark?
- Did both sides answer the same questions with the same wording?
- Is the statistic the same on both sides, for example raw or calibrated?
- Where was time measured: inside the process, on the server, or at the client?
- How many examples were there, and is the difference larger than the uncertainty?
- What does the cost row leave out?

None of these make a table wrong. They show which question the table answers, and it is
seldom the question you have.

## A fair comparison on your tickets

1. **Label first.** Write the expected answers before you run either model. Labels written
   after seeing an answer drift toward it. Include the hard cases you know about, such as
   negation (t08), other languages (t13), and tickets that fit no team.
2. **Freeze the configuration.** `--save` records the target, the requested model, the
   Laya version and revision, the thresholds, the questions, and every answer. Note the
   device yourself.
3. **Ask the same questions and compare the same statistic.** `decisions.ask` sends one
   question set to every backend and bands on `top_probability`, never on either
   backend's own `confidence` (lesson 3).
4. **Tune on one set and report on another.** Thresholds and temperatures chosen on the
   test tickets make the test look better than new tickets will.
5. **Report accuracy together with coverage.** For each question, give the number of
   answers in each band and how many of them were right.
6. **Report latency with its boundary.** `wall_s` is measured at the client. It includes
   the network for HTTP targets and the model load on the first in-process Laya call.
   `server_inference_ms` comes from the `X-Inference-Time-Ms` header, which `laya-serve`
   sends, and covers inference only. Keep the first call apart and report p50 and p95 of
   the rest.
7. **Price from the usage field.** For Jev, multiply the total `usage.input_tokens` by the
   current price. For Laya, count the hardware hours you would run.

## Run the comparison

```powershell
python -m decisions.ask --jev --examples --live --save results\decisions\jev-examples.json
.\decisions\.venv\Scripts\python.exe -m decisions.ask --laya --examples --live --revision reviewed --allow-download --save results\decisions\laya-examples.json
python -m decisions.ask --url http://127.0.0.1:8000 --examples --live --save results\decisions\laya-serve-examples.json
```

`--allow-download` is needed only when the pinned revision is not cached yet. Pass your own
tickets with `--examples path\to\tickets.jsonl`, one JSON object per line:

```json
{"id": "t01", "state": "Ticket text", "labels": {"department": "billing", "refund_requested": true}}
```

A `choice` label must be one of the options, a `noul` label is `true` or `false`, and a
`score` label is a level's position, starting at 0.

The layout below comes from the test suite's fake server, which gives every ticket the same
fixed answer. Its numbers describe that fake, not a model:

```text
question           labeled  accuracy  mean top p   correct/count in act, confirm, human
department              15     33.3%       0.900   5/15, 0/0, 0/0
refund_requested        15     80.0%       0.800   0/0, 12/15, 0/0
ECE is not reported below 100 labeled answers per question; the bins would be too sparse.
Client wall time: first call 0.025 s; later calls p50 0.015 s, p95 0.019 s (n=14)
Input tokens reported in usage: 150
```

- `labeled` counts tickets with a label for that question. `tickets.jsonl` has no
  `urgency` labels, so urgency is asked but not scored.
- `accuracy` is correct answers over labeled tickets. A `noul` answer counts as yes when
  P(yes) is at least 0.5; a `score` answer counts as its most probable level.
- `mean top p` is the average `top_probability`. Compare it with accuracy: 0.900 against
  33.3% is badly over-confident.
- `correct/count` covers the act, confirm, and human bands at `--act-at 0.9` and
  `--confirm-at 0.6`. `5/15, 0/0, 0/0` means all 15 answers landed in act, and 5 were right.

The fake also shows why a baseline matters. It always answers `billing` and scores 5 of 15,
because 5 tickets are billing tickets. It always answers no refund and scores 12 of 15,
because only 3 tickets ask for one. A model must beat these majority-class baselines to
be useful: 80% on `refund_requested` would be no better than never saying yes.

## How much 15 tickets can tell you

Fifteen tickets are enough to practice the procedure, but not to choose a vendor. A rough
95% range for an accuracy p measured on n tickets is p ± 1.96·√(p(1 − p)/n). For 12 of
15 correct (0.80), that is 0.80 ± 0.20. The Wilson interval, which behaves better on small
samples, gives 0.55 to 0.93. At the same accuracy on 400 tickets, the range narrows to
about ± 0.04.

When two models answer the same tickets, compare them ticket by ticket. Only the tickets
where exactly one model is right carry information. Suppose model A alone is right on 8
tickets and model B alone on 2. If the models were equally accurate, a split at least that
lopsided would still happen about 11% of the time (a two-sided sign test, the exact form
of McNemar's test). At 15 against 5, it would happen about 4% of the time.

## Deciding

- **Laya:** the text must stay in your network, or you want to adapt the model to your own
  labels. Plan to fine-tune and fit temperatures, as Laya's README recommends.
- **Jev:** you want a hosted API and cannot operate a model service, or your questions
  have many options. Pin a model name and set your own thresholds.
- **Neither:** neither model beats the majority-class baseline, or a simple rule, by more
  than the uncertainty.

Whichever you choose, keep the labeled tickets, the saved reports, and the thresholds
together. Run the set again whenever the model, the questions, or the serving setup
change (lesson 4).

References: [Laya README: Laya (with routing) vs Jev](https://github.com/NandhaKishorM/laya/blob/v0.3.21/README.md#laya-with-routing-vs-jev),
[Laya README: typed-decisions on all three checkpoints](https://github.com/NandhaKishorM/laya/blob/v0.3.21/README.md#typed-decisions-measured-on-all-three-checkpoints),
[Laya README: fine-tuning](https://github.com/NandhaKishorM/laya/blob/v0.3.21/README.md#fine-tuning),
[Laya HTTP API (timing headers)](https://github.com/NandhaKishorM/laya/blob/v0.3.21/docs/http-api.md),
[TypeSafe Python SDK 0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python/tree/v0.7.2),
[Wilson (1927), Probable Inference, the Law of Succession, and Statistical Inference](https://doi.org/10.1080/01621459.1927.10502953),
[McNemar (1947), Note on the sampling error of the difference between correlated proportions or percentages](https://doi.org/10.1007/BF02295996).

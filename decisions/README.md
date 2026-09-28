# Typed decisions with Jev and Laya

A generative model writes text. A **typed decision model** answers a fixed question
with one of the answers you allowed, and gives a probability for each of them.
This short course teaches that kind of model through two examples:

- **Jev**, from TypeSafe AI, is a hosted model. You call it over HTTPS with an API
  key. Its weights and architecture are not published.
- **Laya**, from Convai Innovations, is an open-weights model (Apache-2.0). It runs
  on your own CPU or GPU and can serve the same HTTP API as Jev.

The lessons and the code use one invented support ticket:

```text
I was charged twice for my March invoice. Please refund the duplicate payment.
```

They ask three questions about it: which team should handle it (a *choice*),
whether the customer asks for money back (a yes/no *noul*), and how urgent it is
(a *score*).

## Reading order

| Lesson | Question it answers | Runnable example |
| --- | --- | --- |
| [1. Decisions, not text](01-decisions-not-text.md) | What does a decision model return, and when is it the wrong tool? | None; reading only |
| [2. Asking typed questions](02-asking-typed-questions.md) | How do I write questions that both Jev and Laya accept? | `python -m decisions.questions` |
| [3. Reading the answers](03-reading-the-answers.md) | What do the answer, the probabilities, and "confidence" mean? | `python -m decisions.answers` |
| [4. Thresholds and calibration](04-thresholds-and-calibration.md) | When should a program act, ask for confirmation, or pass to a person? | `python -m decisions.calibration` |
| [5. Running Jev and Laya](05-running-jev-and-laya.md) | How do I call the hosted API, a local server, or Laya in-process? | `python -m decisions.ask` |
| [6. Evaluating and choosing](06-evaluating-and-choosing.md) | How do I compare the two on my own labeled examples? | `python -m decisions.ask --examples` |

Then do the [exercises](EXERCISES.md). Each has a worked answer.

## Prerequisites

You need Python 3.10 or newer and NumPy, as for the rest of this repository.
Lessons 3 and 4 use softmax and temperature, which
[Text and representations](../foundations/01-text-and-representations.md)
introduces. Lessons 1 to 4 and the tests need no account, API key, GPU, or model
download.

## Run the examples

From the repository root:

```powershell
python -m decisions.questions
python -m decisions.answers
python -m decisions.calibration
python -m decisions.ask
python -m unittest tests.test_decisions -v
```

None of these commands contacts an outside service or loads a model. The tests
start a small fake server on `127.0.0.1`. Every number the scripts print comes
from hand-chosen probabilities, arithmetic, or a seeded simulation. **None of them
is a measurement of Jev or Laya.**

## What leaves your machine

`decisions.ask` only prints the request it would make until you add `--live`:

| With `--live` | Where the ticket text goes | Credentials |
| --- | --- | --- |
| `--jev` | TypeSafe's API at `https://api.typesafe.ai` | `TYPESAFE_API_KEY` from the environment |
| `--url http://127.0.0.1:8000` | A server you run, such as `laya-serve` | Optional, from the variable named by `--api-key-env` |
| `--laya` | Nowhere; the model runs inside the Python process | None; only cached weights load unless you add `--allow-download` |

The course tickets are invented. Read your provider's data terms before sending
real customer text to any hosted API.

## What you should be able to do afterward

- Write a choice, noul, and score question that both models accept, and explain
  why `yes`/`no` labels belong in a noul question rather than a choice.
- Compute the top probability, Jev's confidence, and Laya's confidence from one
  probability list, and explain why a threshold on one does not transfer to another.
- Choose act, confirm, and human-review thresholds from measured coverage and
  accuracy, and fit a temperature on validation data only.
- Run the same labeled tickets against Jev and Laya. Report accuracy, coverage,
  latency, and input tokens with the model version, and without claiming more
  than 15 tickets can show.

## Versions and sources

Checked on 2026-09-28 against `typesafe-sdk` 0.7.2, the official Python SDK for Jev,
and `laya` 0.3.21. Both projects change quickly, so check current documentation
before relying on a default or limit quoted here. The sources are listed under
[typed decision models](../SOURCES.md#typed-decision-models-jev-and-laya).

# Exercises: typed decisions with Jev and Laya

Try each one before you open the answer. Commands assume you are in the repository root.
None of them need a network connection or an API key.

## 1. One answer, three numbers

A `department` answer has probabilities `billing 0.55, account 0.25, technical 0.15,
other 0.05`. Compute the top probability, the "above uniform" value, and
`1 − H/ln 4`. Which one does Laya return as `answer_confidence`?

<details>
<summary>Answer</summary>

- The top probability is 0.55. Laya returns it as `answer_confidence`.
- Above uniform is `(4 × 0.55 − 1) / 3 = 0.40`.
- `H = 1.110` and `ln 4 = 1.386`, so `1 − H/ln 4 = 0.200`. Laya returns this as `confidence`.

These are three different numbers for one answer, so a threshold set on one of them
does not carry over to another (lesson 3).

</details>

## 2. Fix a question

What is wrong with this question? Run the checker, then read the options yourself.

```powershell
python -c "from decisions.questions import question_problems; print(question_problems({'type': 'choice', 'instructions': '', 'criteria': {'billing': 'Payments, invoices, or login problems', 'account': 'Login, password, or profile changes', 'technical': 'The app or website fails'}}))"
```

<details>
<summary>Answer</summary>

The checker reports `needs non-empty instructions (Jev allows none; Laya rejects the question)`.
It cannot see two other problems. Login appears under both `billing` and `account`, so
the options overlap. There is also no `other` option, so a ticket that fits no team is
forced into one. Keep login under `account` only, and add
`"other": "Anything that fits none of the above"` (lesson 2).

</details>

## 3. An average that hides the answer

The `urgency` levels are Can wait (0), Soon (1), and Now (2). An answer has
probabilities `[0.40, 0.15, 0.45]`. What is the score? Should this ticket page the
on-call team under the rule from lesson 3?

<details>
<summary>Answer</summary>

The score is `0 × 0.40 + 1 × 0.15 + 2 × 0.45 = 1.05`, which reads as "Soon". Yet Soon
is the least likely level at 0.15. P(Now) is 0.45, below 0.5, so the rule does not page
anyone automatically. The answer is split between two levels, so a person should look
at it.

</details>

## 4. Adding options changes the gate

An answer over 4 options has top probability 0.6. After six rarely used options are
added, a similar ticket again gets top probability 0.6, now over 10 options. A gate
accepts answers with "above uniform" at 0.5 or more. What does it do with each answer?

<details>
<summary>Answer</summary>

With 4 options, `(4 × 0.6 − 1) / 3 = 0.467`, so the gate rejects the answer. With 10
options, `(10 × 0.6 − 1) / 9 = 0.556`, so it accepts it. The top probability is the same
in both cases, but the decision changed. Gate on `top_probability`, and fit the
threshold again whenever the options change (`python -m decisions.answers` prints this
table).

</details>

## 5. A threshold from costs

Undoing a wrong automatic refund takes 30 minutes. A human check takes 3 minutes. Above
what probability should the system act without a check? What if the check takes
6 minutes?

<details>
<summary>Answer</summary>

Act when the expected cost of acting, `(1 − p) × 30`, is below the cost of a check, 3.
That gives `p > 1 − 3/30 = 0.9`. With a 6-minute check, `p > 1 − 6/30 = 0.8`. The
threshold comes from costs and calibrated probabilities, not from the model (lesson 4).

</details>

## 6. Temperature keeps the answer

Compute softmax of the logits `[2, 1, 0]` at temperature 1 and at temperature 3. Why does
`python -m decisions.calibration` report the same accuracy, 0.778, before and after
scaling?

<details>
<summary>Answer</summary>

At T = 1 the probabilities are `[0.665, 0.245, 0.090]`. At T = 3 they are
`[0.448, 0.321, 0.230]`. Dividing by a positive T keeps the order of the logits, so the
same option wins and accuracy cannot change. Only the probabilities move: the mean top
probability falls from 0.922 to 0.775, closer to the 0.778 accuracy.

</details>

## 7. Pick the act threshold

Run `python -m decisions.calibration`. Automatic actions must be right at least 90% of
the time. Which threshold in its table meets that, and what share of tickets goes to
people?

<details>
<summary>Answer</summary>

A threshold of 0.8 acts on 51.9% of the tickets with 91.5% accuracy. At 0.6 the accuracy
is 84.6%, which is too low. The other 48.1% go to confirmation or to a person. These
numbers come from simulated data. Repeat the procedure on your own labeled tickets.

</details>

## 8. What leaves your machine

Which of these send ticket text over the network?

1. `python -m decisions.ask --jev`
2. `python -m decisions.ask --jev --live`
3. `python -m decisions.ask --laya --live`
4. `python -m decisions.ask --url http://127.0.0.1:8000 --live`

<details>
<summary>Answer</summary>

Only 2. Command 1 is a dry run that prints the request and contacts nothing. Command 3
runs Laya in this process, and it reads only cached weights unless you add
`--allow-download`. Downloading weights does not send the ticket. Command 4 sends the
ticket to a server on the same machine through the loopback address. `ask.py` accepts
plain `http` only for loopback addresses, and anything else must use `https` (lesson 5).

</details>

## 9. Beat the baseline

In `tickets.jsonl`, what accuracy does "always billing" get on `department`, and "never a
refund" on `refund_requested`? A model gets 13 of 15 refund labels right. It is right on
all three refund tickets and wrong on two others. Is it better than the baseline?

<details>
<summary>Answer</summary>

"Always billing" gets 5 of 15 (33.3%) and "never a refund" gets 12 of 15 (80.0%). The
model's 13 of 15 (0.867) has a rough 95% range of ± 0.172, which includes 0.80. Compare
the two on the same tickets. The model alone is right on 3 tickets and the baseline
alone on 2. If the two were equally good, a split at least that lopsided would happen
every time, so a sign test gives p = 1.0. Fifteen tickets cannot tell them apart
(lesson 6).

</details>

## 10. Read a vendor row

Laya's README lists ECE as Jev 0.246 and Laya 0.081. What stops this from being a
like-for-like comparison?

<details>
<summary>Answer</summary>

- Laya's figure is after temperature fitting. Jev's comes from third-party results,
  which the README says it never measured.
- The two sides used different samples and prompts.
- The same README's typed-decisions table lists raw ECE as Jev 0.144 and the fine-tuned
  Laya checkpoint 0.213, so on that benchmark Jev was better calibrated before fitting.

A fair comparison needs the same tickets, the same statistic, and the same stage for
both models (lesson 6).

</details>

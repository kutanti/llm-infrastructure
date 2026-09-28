# 2. Asking typed questions

A request has three parts: the `model` to use, the `state` the questions refer to, and
named `questions`. This lesson writes one question of each type about the course ticket
and checks them before anything is sent.

## One question of each type

[`questions.py`](questions.py) builds the questions with three small functions:

```python
TRIAGE_QUESTIONS = {
    "department": choice("Which team should handle this support ticket?", {
        "billing": "Charges, invoices, payments, or refunds.",
        "account": "Signing in, passwords, profile details, or account access.",
        "technical": "Errors, crashes, or features that do not work.",
        "other": "Anything else, such as pricing questions before buying or general feedback.",
    }),
    "refund_requested": noul(
        "Does the customer ask for money back?",
        true="The customer asks for a refund or for a charge to be reversed.",
        false="The customer does not ask for money back.",
    ),
    "urgency": score("How urgent is this ticket?", [
        "Can wait: a question or request with no current harm.",
        "Soon: something is wrong, but the customer can still work.",
        "Now: the customer is blocked or is losing money.",
    ]),
}
```

- **choice**: `criteria` maps each label to a description of when it applies. The label
  comes back as the answer, so pick labels your code can branch on.
- **noul**: `instructions` is the yes/no question, or a statement to test. `criteria`
  may describe what counts as `true` and as `false`. The answer is the probability of yes.
- **score**: `criteria` is an ordered list. A description's position is its level: 0, 1,
  and 2 here. The answer is the expected level.

The question names (`department`, `refund_requested`, `urgency`) are yours. The response
uses the same names.

Run the example:

```powershell
python -m decisions.questions
```

It prints the full request body and sends nothing. It then checks three broken questions:

```text
Three questions with problems:
  is_spam  warning  labels ['no', 'yes'] read as booleans; use a noul question or labels that describe outcomes
  priority error    score criteria must list at least two level descriptions, level 0 first
  refund   error    needs non-empty instructions (Jev allows none; Laya rejects the question)
```

An error stops the request before it is sent. A warning is advice.

## A portable subset

Jev and Laya accept slightly different inputs. A question inside the overlap works with
both, so you can switch backends without rewriting it. `check_request` in
[`questions.py`](questions.py) enforces these rules:

| Rule | Jev (typesafe-sdk 0.7.2 schema) | Laya 0.3.21 |
| --- | --- | --- |
| Give `instructions` as text | Optional; text, an object, or a list | Required |
| Choice `criteria` is an object from label to description | Must be an object. A description may be `null`: "interpreted by its name alone" | An object, or a list of labels. Null or duplicate labels are refused |
| Score `criteria` lists at least two levels, level 0 first | At least one entry | Each entry becomes `level i: ...` |
| Noul `criteria` has only `true` and `false` keys | Both optional | Other keys are refused |

## Writing criteria that work

1. **Make the options exclusive, and give an exit.** The `other` label catches tickets
   that fit no team. Without it, a pricing question must still be split across three
   teams that do not fit, and one of them comes back as the answer.
2. **Name outcomes, not booleans.** Laya's README warns that its checkpoints "can follow
   labels such as `true`/`false` or `yes`/`no` instead of the option descriptions". A
   yes/no question belongs in a `noul`. For two named outcomes, use labels that describe
   them, such as `refund` and `no_refund`.
3. **Put the meaning in short descriptions.** Laya renders each option as
   `label: description` and keeps at most 48 tokens of it. The options of one question
   share a budget of 192 tokens on the English checkpoint. Many long options get cut
   until they may no longer be told apart. Laya reports that in `usage["options"]`, and
   `laya-serve` can refuse such a question with status 422. `questions.py` warns above
   20 options. Check your provider's documented limits for larger lists.
4. **Send only the text the questions need.** The English checkpoint has 512 tokens in
   total, which leaves about 320 for the state. Laya's README says `predict` truncates a
   longer state, "silently dropping the rest", and offers `predict_long` for long
   documents. Jev bills by input tokens, so extra text also costs money.
5. **Ask one thing per question.** "Is this an urgent billing issue?" mixes two
   decisions. Ask `department` and `urgency` separately, and combine them in code.
6. **Test negation and other languages.** "I do not want a refund" contains the word
   "refund". Laya's README reports a run in which a cancellation question chose
   `cancel_account` for all four negated requests on the English checkpoint. Ticket `t08`
   in [`tickets.jsonl`](tickets.jsonl) is a negation case, and `t13` is in Spanish.
   Lesson 6 measures such cases instead of assuming them.

## What the state can be

The course sends plain text. Both models also accept JSON, such as
`{"subject": "Duplicate charge", "body": "..."}`. Laya serializes a JSON state to text
before tokenizing it. Use the same form for evaluation as for production, because the
text the model reads changes with it.

References: [TypeSafe Python SDK wire schema, v0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python/blob/v0.7.2/src/typesafe_sdk/_schemas/models.py),
[Laya option rendering and budgets (`common.py`)](https://github.com/NandhaKishorM/laya/blob/v0.3.21/laya/common.py),
[Laya README: honest limits](https://github.com/NandhaKishorM/laya/blob/v0.3.21/README.md#honest-limits).

"""Lesson 2: write typed questions and the request body that Jev and Laya both accept."""

import json

QUESTION_TYPES = ("noul", "choice", "score")
BOOLEAN_WORDS = {"yes", "no", "true", "false"}
OPTION_WARNING = 20
DEFAULT_MODEL = "jev-latest"

TICKET = "I was charged twice for my March invoice. Please refund the duplicate payment."


def noul(instructions, true=None, false=None):
    """A yes/no question. The answer is the probability of yes."""
    question = {"type": "noul", "instructions": instructions}
    criteria = {key: text for key, text in (("true", true), ("false", false)) if text is not None}
    if criteria:
        question["criteria"] = criteria
    return question


def choice(instructions, options):
    """Pick one label. `options` maps each label to a description (or None)."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions, levels):
    """Rate on ordered levels. The first description is level 0."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


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


def question_problems(question):
    """Return (errors, warnings) for one question, using rules both Jev and Laya accept."""
    if not isinstance(question, dict):
        return ["must be an object"], []
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        return [f"type must be one of {', '.join(QUESTION_TYPES)}"], []
    errors, warnings = [], []
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        errors.append("needs non-empty instructions (Jev allows none; Laya rejects the question)")
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, dict)
                                     or not criteria or not set(criteria) <= {"true", "false"}):
            errors.append("noul criteria may only describe 'true' and 'false'")
    elif kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            errors.append("choice criteria must map at least two labels to descriptions")
        elif not all(isinstance(label, str) and label.strip() and (text is None or isinstance(text, str))
                     for label, text in criteria.items()):
            errors.append("choice labels must be non-empty text; descriptions must be text or null")
        else:
            boolean = sorted(label for label in criteria if label.strip().lower() in BOOLEAN_WORDS)
            if boolean:
                warnings.append(f"labels {boolean} read as booleans; use a noul question "
                                "or labels that describe outcomes")
            if len(criteria) > OPTION_WARNING:
                warnings.append(f"{len(criteria)} options; Laya fits every option into one "
                                "token budget, so long lists get truncated")
    elif (not isinstance(criteria, list) or len(criteria) < 2
          or not all(isinstance(level, str) and level.strip() for level in criteria)):
        errors.append("score criteria must list at least two level descriptions, level 0 first")
    return errors, warnings


def request_body(state, questions=None, model=DEFAULT_MODEL):
    """Assemble the POST /v1/systemone body. The key order matches the wire schema."""
    return {"model": model, "state": state,
            "questions": TRIAGE_QUESTIONS if questions is None else questions}


def check_request(body):
    """Raise ValueError if the body cannot be sent; otherwise return a list of warnings."""
    if not isinstance(body, dict):
        raise ValueError("the request must be a JSON object")
    errors, warnings = [], []
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        errors.append("model: name a model, for example jev-latest")
    state = body.get("state")
    if not isinstance(state, (str, dict, list)) or not state or (isinstance(state, str) and not state.strip()):
        errors.append("state: give non-empty text, a JSON object, or a JSON array")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        errors.append("questions: ask at least one question")
    else:
        for question_id, question in questions.items():
            if not isinstance(question_id, str) or not question_id.strip():
                errors.append("questions: every question needs a non-empty id")
                continue
            found, advice = question_problems(question)
            errors += [f"{question_id}: {message}" for message in found]
            warnings += [f"{question_id}: {message}" for message in advice]
    if errors:
        raise ValueError("; ".join(errors))
    return warnings


def main():
    body = request_body(TICKET)
    assert check_request(body) == []
    print("POST /v1/systemone with this body works for Jev and for Laya's server.")
    print("Nothing is sent by this script.\n")
    print(json.dumps(body, indent=2))

    broken = {
        "is_spam": choice("Is this message spam?", {"yes": None, "no": None}),
        "priority": score("How urgent is this ticket?", ["Urgent."]),
        "refund": {"type": "noul", "criteria": {"true": "The customer wants money back."}},
    }
    print("\nThree questions with problems:")
    for question_id, question in broken.items():
        errors, warnings = question_problems(question)
        for label, messages in (("error", errors), ("warning", warnings)):
            for message in messages:
                print(f"  {question_id:9}{label:9}{message}")
    assert question_problems(broken["is_spam"]) == ([], [
        "labels ['no', 'yes'] read as booleans; use a noul question or labels that describe outcomes"])
    assert question_problems(broken["priority"])[0] and question_problems(broken["refund"])[0]
    try:
        check_request(request_body(TICKET, broken))
    except ValueError as error:
        assert "priority" in str(error) and "refund" in str(error)
    else:
        raise AssertionError("the broken request should be refused")
    print("\nErrors stop the request before it is sent. Warnings are advice; the request can still go.")


if __name__ == "__main__":
    main()

"""Lesson 3: turn probabilities into typed answers and compare three confidence numbers."""

import math

import numpy as np

from .questions import TRIAGE_QUESTIONS


def options(question):
    """Answer keys in model order: choice labels, score levels "0", "1", ..., or noul's false/true."""
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(level) for level in range(len(question["criteria"]))]
    return ["false", "true"]


def distribution(probabilities, count):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (count,) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError(f"expected {count} non-negative probabilities")
    # Services round each value (Laya to 4 decimals), so allow half a unit of rounding per value.
    if abs(p.sum() - 1) > 0.005 * count + 1e-6:
        raise ValueError(f"probabilities sum to {p.sum():.4f}, not 1")
    return p


def answer(question, probabilities):
    """Build the answer that the wire schema describes from one probability per option."""
    keys = options(question)
    p = distribution(probabilities, len(keys))
    if question["type"] == "noul":
        return {"type": "noul", "noul": float(p[1])}
    result = {"type": question["type"], "probabilities": dict(zip(keys, p.tolist()))}
    if question["type"] == "choice":
        result["choice"] = keys[int(p.argmax())]
    else:
        result["score"] = float(np.arange(len(p)) @ p)
        result["legend"] = dict(zip(keys, question["criteria"]))
    return result


def top_probability(p):
    """The probability of the chosen option. Laya returns it as `answer_confidence`."""
    return float(np.max(p))


def above_uniform(p):
    """How far the top probability sits between uniform (0) and certain (1).

    Third parties fitted this closed form to Jev's choice `confidence`; TypeSafe does not
    publish a formula, so treat it as a model of the field, not its definition.
    """
    n = len(p)
    return float(np.clip((n * np.max(p) - 1) / (n - 1), 0, 1))


def entropy_confidence(p):
    """One minus normalized entropy: Laya's `confidence` for choice and score answers."""
    p = np.asarray(p, dtype=float)
    used = p[p > 0]
    entropy = float(-(used * np.log(used)).sum())
    return float(np.clip(1 - entropy / math.log(len(p)), 0, 1))


def probabilities_of(reply):
    """Probabilities in option order from any answer, Jev-shaped or Laya-shaped."""
    if reply["type"] == "noul":
        return np.array([1 - reply["noul"], reply["noul"]])
    return np.array(list(reply["probabilities"].values()), dtype=float)


def read_answers(response, questions):
    """Check a /v1/systemone response against the questions that were asked.

    Returns one summary per question with the answer, its probabilities, a top
    probability computed the same way for every backend, and the backend's own
    confidence fields when present.
    """
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise ValueError("response has no answers object")
    results = {}
    for question_id, question in questions.items():
        reply = response["answers"].get(question_id)
        if not isinstance(reply, dict) or reply.get("type") != question["type"]:
            raise ValueError(f"{question_id}: missing answer or wrong type")
        keys = options(question)
        try:
            if question["type"] == "noul":
                value = float(reply["noul"])
                if not 0 <= value <= 1:
                    raise ValueError
                p = np.array([1 - value, value])
            else:
                if set(reply["probabilities"]) != set(keys):
                    raise ValueError
                p = distribution([reply["probabilities"][key] for key in keys], len(keys))
                value = reply[question["type"]]
                if question["type"] == "choice" and value not in keys:
                    raise ValueError
                if question["type"] == "score":
                    value = float(value)
                    if not 0 <= value <= len(keys) - 1:
                        raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{question_id}: answer does not match the question") from error
        results[question_id] = {
            "type": question["type"], "answer": value,
            "probabilities": dict(zip(keys, p.tolist())), "top_probability": top_probability(p),
            "reported_confidence": reply.get("confidence"),
            "answer_confidence": reply.get("answer_confidence"),
        }
    return results


def main():
    department = TRIAGE_QUESTIONS["department"]
    urgency = TRIAGE_QUESTIONS["urgency"]
    refund = TRIAGE_QUESTIONS["refund_requested"]
    print("Hand-chosen probabilities, not model output.\n")

    p = [0.70, 0.10, 0.10, 0.10]
    reply = answer(department, p)
    print(f"department  {dict(zip(options(department), p))}")
    print(f"  choice = {reply['choice']!r}")
    print(f"  top probability     {top_probability(p):.3f}   (Laya: answer_confidence)")
    print(f"  above uniform       {above_uniform(p):.3f}   (third-party fit to Jev's confidence)")
    print(f"  1 - entropy/log(4)  {entropy_confidence(p):.3f}   (Laya: confidence)")
    np.testing.assert_allclose([top_probability(p), above_uniform(p), entropy_confidence(p)],
                               [0.70, 0.60, 0.3216], atol=5e-5)
    assert reply["choice"] == "billing"

    print("\nurgency: the score is an expected level, so check where the probability is")
    for p in ([0.0, 1.0, 0.0], [0.5, 0.0, 0.5]):
        reply = answer(urgency, p)
        print(f"  probabilities {p}  ->  score {reply['score']:.2f}, top probability {top_probability(p):.2f}")
        assert reply["score"] == 1.0
    print("  Both average to level 1; the second puts no probability on level 1 itself.")

    reply = answer(refund, [0.08, 0.92])
    top = top_probability(probabilities_of(reply))
    print(f"\nrefund_requested: noul = {reply['noul']:.2f}; top probability max(p, 1 - p) = {top:.2f}")
    np.testing.assert_allclose(top, 0.92)

    print("\nSame top probability (0.6), more options (rest spread evenly):")
    print(f"{'options':>8}{'top p':>8}{'above uniform':>15}{'1 - H/log n':>13}")
    previous = -1.0
    for n in (2, 4, 10, 20):
        p = np.array([0.6] + [0.4 / (n - 1)] * (n - 1))
        print(f"{n:8}{top_probability(p):8.2f}{above_uniform(p):15.3f}{entropy_confidence(p):13.3f}")
        assert above_uniform(p) > previous
        previous = above_uniform(p)
    print("A threshold on one definition does not transfer to another, or to another option count.")


if __name__ == "__main__":
    main()

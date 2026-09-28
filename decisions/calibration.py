"""Lesson 4: measure calibration, fit one temperature, and pick act/confirm/human bands."""

import numpy as np

DEPARTMENTS = ("billing", "account", "technical")


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(confidence, correct, bins=15):
    """Average |accuracy - confidence| over equal-width bins, weighted by bin size."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            total += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    return float(total)


def brier_score(probabilities, labels):
    """Mean squared distance between the probability vector and the one-hot label."""
    probabilities = np.asarray(probabilities, dtype=float)
    target = np.eye(probabilities.shape[1])[labels]
    return float(((probabilities - target) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probabilities, labels):
    picked = np.asarray(probabilities, dtype=float)[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1)).mean())


def fit_temperature(logits, labels, grid=None):
    """Pick the temperature that minimizes validation NLL. Argmax, and so accuracy, stays the same.

    The grid matches the range Laya clamps its fitted temperatures to. With only
    probabilities, as from Jev, pass log(p) as the logits: softmax(log p / T) is the
    same rescaling. Clip exact zeros first; the clip value then influences the fit.
    """
    grid = np.linspace(0.5, 5.0, 181) if grid is None else grid
    losses = [negative_log_likelihood(softmax(logits, t), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def coverage_and_accuracy(confidence, correct, threshold):
    """Share of answers at or above the threshold, and how often those answers are right."""
    covered = np.asarray(confidence) >= threshold
    accuracy = float(np.asarray(correct)[covered].mean()) if covered.any() else None
    return float(covered.mean()), accuracy


def decide(confidence, act_at=0.9, confirm_at=0.6):
    """Three bands: act automatically, act after a person confirms, or hand the ticket over."""
    if not 0 <= confirm_at <= act_at <= 1:
        raise ValueError("need 0 <= confirm_at <= act_at <= 1")
    if confidence >= act_at:
        return "act"
    return "confirm" if confidence >= confirm_at else "human"


def synthetic_router(rng, count, signal=1.5, overconfidence=3.0):
    """Tickets whose calibrated logits are known, reported `overconfidence` times too sharp."""
    labels = rng.integers(0, len(DEPARTMENTS), count)
    features = signal * np.eye(len(DEPARTMENTS))[labels] + rng.normal(size=(count, len(DEPARTMENTS)))
    # With unit Gaussian noise and equal priors, signal * features are the true log-odds.
    return overconfidence * signal * features, labels


def report(name, probabilities, labels):
    top = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    row = {"accuracy": float(correct.mean()), "mean_top": float(top.mean()),
           "ece": expected_calibration_error(top, correct),
           "brier": brier_score(probabilities, labels), "nll": negative_log_likelihood(probabilities, labels)}
    print(f"{name:24}{row['accuracy']:9.3f}{row['mean_top']:11.3f}{row['ece']:8.3f}"
          f"{row['brier']:8.3f}{row['nll']:8.3f}")
    return row


def main():
    rng = np.random.default_rng(42)
    validation_logits, validation_labels = synthetic_router(rng, 2000)
    test_logits, test_labels = synthetic_router(rng, 2000)
    print("Synthetic 3-department router: 2000 validation and 2000 test tickets.")
    print("Its logits are 3x the calibrated values: a deliberately over-confident model.")
    print("This is a simulation of the procedure, not a measurement of Jev or Laya.\n")

    temperature = fit_temperature(validation_logits, validation_labels)
    print(f"Temperature fitted on validation only: T = {temperature:.3f}\n")
    print(f"{'test split':24}{'accuracy':>9}{'mean top p':>11}{'ECE':>8}{'Brier':>8}{'NLL':>8}")
    before = report("as reported", softmax(test_logits), test_labels)
    calibrated = softmax(test_logits, temperature)
    after = report(f"divided by T={temperature:.2f}", calibrated, test_labels)
    assert 2.5 < temperature < 3.5
    assert before["accuracy"] == after["accuracy"]
    assert after["ece"] < before["ece"] / 3 and after["nll"] < before["nll"]

    top = calibrated.max(axis=1)
    correct = calibrated.argmax(axis=1) == test_labels
    print("\nAct only at or above a threshold (calibrated test split):")
    print(f"{'threshold':>10}{'coverage':>10}{'accuracy of those':>19}")
    rows = [coverage_and_accuracy(top, correct, t) for t in (0.0, 0.6, 0.8, 0.9)]
    for threshold, (coverage, accuracy) in zip((0.0, 0.6, 0.8, 0.9), rows):
        print(f"{threshold:10.1f}{coverage:10.1%}{accuracy:19.1%}")
    assert all(a[0] >= b[0] and a[1] <= b[1] for a, b in zip(rows, rows[1:]))

    bands = [decide(value) for value in top]
    print("\nBands at act >= 0.9, confirm >= 0.6:")
    for band in ("act", "confirm", "human"):
        members = np.array([b == band for b in bands])
        print(f"  {band:8}{members.mean():7.1%} of tickets, {correct[members].mean():6.1%} correct")
    print("Raising a threshold moves tickets toward people; it does not make the model better.")


if __name__ == "__main__":
    main()

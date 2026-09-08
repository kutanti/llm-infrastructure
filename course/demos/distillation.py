"""Compare hard and soft teacher supervision with a feature-limited student."""

import numpy as np

from .objectives import log_softmax, soft_cross_entropy


def train(features, teacher_logits, use_soft):
    classes = teacher_logits.shape[1]
    hard_targets = np.eye(classes)[teacher_logits.argmax(axis=1)]
    temperature = 2.0
    soft_targets = np.exp(log_softmax(teacher_logits / temperature))
    weights = np.zeros((features.shape[1], classes))
    mixture = 0.7 if use_soft else 0.0
    initial = None
    for _ in range(700):
        logits = features @ weights
        hard_loss, hard_gradient = soft_cross_entropy(logits, hard_targets)
        soft_loss, soft_gradient = soft_cross_entropy(logits, soft_targets, temperature)
        loss = (1 - mixture) * hard_loss + mixture * soft_loss
        if initial is None:
            initial = loss
        gradient = (1 - mixture) * hard_gradient + mixture * soft_gradient
        weights -= 0.2 * features.T @ gradient
    final_logits = features @ weights
    final = ((1 - mixture) * soft_cross_entropy(final_logits, hard_targets)[0]
             + mixture * soft_cross_entropy(final_logits, soft_targets, temperature)[0])
    assert final < initial
    return weights


def evaluate(features, teacher_logits, weights):
    student_logits = features @ weights
    log_teacher = log_softmax(teacher_logits)
    log_student = log_softmax(student_logits)
    kl = (np.exp(log_teacher) * (log_teacher - log_student)).sum(axis=1).mean()
    agreement = (teacher_logits.argmax(axis=1) == student_logits.argmax(axis=1)).mean()
    return float(kl), float(agreement)


def main():
    rng = np.random.default_rng(42)
    teacher_weights = rng.normal(0, 0.8, (6, 3))
    teacher_weights[4:] *= 0.4
    train_x = rng.normal(size=(256, 6))
    heldout_x = rng.normal(size=(256, 6))
    teacher_train = train_x @ teacher_weights
    teacher_heldout = heldout_x @ teacher_weights

    logits = rng.normal(size=(5, 3))
    targets = np.exp(log_softmax(rng.normal(size=(5, 3)) / 2))
    _, gradient = soft_cross_entropy(logits, targets, temperature=2)
    epsilon = 1e-5
    above, below = logits.copy(), logits.copy()
    above[1, 2] += epsilon
    below[1, 2] -= epsilon
    numeric = (soft_cross_entropy(above, targets, 2)[0]
               - soft_cross_entropy(below, targets, 2)[0]) / (2 * epsilon)
    np.testing.assert_allclose(numeric, gradient[1, 2], rtol=1e-4, atol=1e-8)

    print("Teacher uses 6 input features; student uses only the first 4.")
    print("Soft objective: 0.3 * hard CE + 0.7 * T^2 * soft CE at T=2.")
    print("Held-out KL is measured at T=1 for both students.\n")
    print(f"{'training':18}{'teacher agreement':>20}{'KL(teacher||student)':>24}")
    for label, use_soft in (("hard labels", False), ("hard + soft", True)):
        weights = train(train_x[:, :4], teacher_train, use_soft)
        kl, agreement = evaluate(heldout_x[:, :4], teacher_heldout, weights)
        assert np.isfinite(kl) and kl >= -1e-12
        print(f"{label:18}{agreement:19.1%}{kl:24.5f}")
    print("\nAgreement is not independent task correctness.")
    print("The teacher is a fixed numerical function, not ground truth about the world.")
    print("Missing student features impose a limit that optimization cannot remove.")


if __name__ == "__main__":
    main()

"""Stable probability calculations shared by the course's training examples."""

import numpy as np


def log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite [batch, classes] matrix")
    if min(logits.shape) < 1:
        raise ValueError("logits must have nonempty dimensions")
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def soft_cross_entropy(logits, targets, temperature=1.0):
    """Return T^2-scaled mean CE and its derivative with respect to raw logits."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    targets = np.asarray(targets, dtype=np.float64)
    log_probs = log_softmax(np.asarray(logits) / temperature)
    if targets.shape != log_probs.shape or not np.isfinite(targets).all():
        raise ValueError("targets must be a finite matrix matching logits")
    if np.any(targets < 0) or not np.allclose(targets.sum(axis=1), 1):
        raise ValueError("each target row must be a probability distribution")
    loss = -(targets * log_probs).sum(axis=1).mean() * temperature**2
    gradient = temperature * (np.exp(log_probs) - targets) / targets.shape[0]
    return float(loss), gradient

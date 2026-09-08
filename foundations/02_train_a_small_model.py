"""Train a neural bigram model with explicit NumPy gradients on synthetic text."""

import argparse
import math

import numpy as np


WORDS = ("red", "blue", "sun", "moon")
IDS = {word: index for index, word in enumerate(WORDS)}


def pairs(sentences):
    inputs, targets = [], []
    for sentence in sentences:
        ids = [IDS[word] for word in sentence.split()]
        inputs.extend(ids[:-1])
        targets.extend(ids[1:])
    return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)


def forward(inputs, embedding, output, bias):
    hidden = np.tanh(embedding[inputs])
    logits = hidden @ output + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return hidden, log_probs


def loss_and_gradients(inputs, targets, embedding, output, bias):
    hidden, log_probs = forward(inputs, embedding, output, bias)
    loss = -log_probs[np.arange(inputs.size), targets].mean()
    d_logits = np.exp(log_probs)
    d_logits[np.arange(inputs.size), targets] -= 1
    d_logits /= inputs.size
    d_output = hidden.T @ d_logits
    d_bias = d_logits.sum(axis=0)
    d_hidden = (d_logits @ output.T) * (1 - hidden**2)
    d_embedding = np.zeros_like(embedding)
    np.add.at(d_embedding, inputs, d_hidden)
    return loss, (d_embedding, d_output, d_bias)


def check_gradients(inputs, targets, parameters):
    _, analytic = loss_and_gradients(inputs, targets, *parameters)
    epsilon = 1e-5
    for parameter, gradient, index in zip(
        parameters, analytic, ((0, 0), (0, 1), (1,))
    ):
        old = parameter[index]
        parameter[index] = old + epsilon
        above = loss_and_gradients(inputs, targets, *parameters)[0]
        parameter[index] = old - epsilon
        below = loss_and_gradients(inputs, targets, *parameters)[0]
        parameter[index] = old
        numeric = (above - below) / (2 * epsilon)
        np.testing.assert_allclose(numeric, gradient[index], rtol=1e-4, atol=1e-8)


def positive_rate(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("learning rate must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=positive_rate, default=0.8)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    training = [
        "red blue red blue red blue",
        "blue red blue red blue red",
        "sun moon sun moon sun moon",
        "moon sun moon sun moon sun",
    ]
    validation = [
        "red blue red blue",
        "blue red blue red",
        "sun moon sun moon",
        "moon sun moon sun",
    ]
    inputs, targets = pairs(training)
    val_inputs, val_targets = pairs(validation)
    rng = np.random.default_rng(7)
    parameters = (
        rng.normal(0, 0.1, (len(WORDS), 8)),
        rng.normal(0, 0.1, (8, len(WORDS))),
        np.zeros(len(WORDS)),
    )
    check_gradients(inputs, targets, parameters)
    initial = loss_and_gradients(inputs, targets, *parameters)[0]
    print("Vocabulary:", WORDS)
    print("Model: current token -> embedding -> tanh -> next-token logits")
    print("Parameter count:", sum(parameter.size for parameter in parameters))
    print(f"Initial training loss: {initial:.4f}")
    checkpoints = {1, args.steps, max(1, args.steps // 2)}
    for step in range(1, args.steps + 1):
        _, gradients = loss_and_gradients(inputs, targets, *parameters)
        for parameter, gradient in zip(parameters, gradients):
            parameter -= args.learning_rate * gradient
        if step in checkpoints:
            loss = loss_and_gradients(inputs, targets, *parameters)[0]
            if not np.isfinite(loss):
                raise RuntimeError("Training diverged; try a smaller learning rate")
            print(f"After {step:4} updates: training loss = {loss:.4f}")
    _, val_log_probs = forward(val_inputs, *parameters)
    val_loss = -val_log_probs[np.arange(val_inputs.size), val_targets].mean()
    accuracy = (val_log_probs.argmax(axis=1) == val_targets).mean()
    print(f"Validation loss: {val_loss:.4f}; accuracy: {accuracy:.1%}")
    print("Validation uses new sequences with the SAME transition rule.")

    print("\nLearned next-token choices:")
    for index, word in enumerate(WORDS):
        _, log_probs = forward(np.array([index]), *parameters)
        prediction = int(log_probs[0].argmax())
        print(f"  {word:4} -> {WORDS[prediction]:4}"
              f"  probability={np.exp(log_probs[0, prediction]):.4f}")
    generated = [IDS["red"]]
    for _ in range(11):
        _, log_probs = forward(np.array([generated[-1]]), *parameters)
        generated.append(int(log_probs[0].argmax()))
    print("\nGreedy generation:", " ".join(WORDS[index] for index in generated))
    print("Only the last token enters each prediction; earlier context is discarded.")
    print("Weights live in this process only. Rerunning starts from the same seed.")


if __name__ == "__main__":
    main()

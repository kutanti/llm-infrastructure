"""Learn a rank-two update while keeping a synthetic base projection frozen."""

import numpy as np


def predict(x, base, a, b, scale):
    return x @ base + scale * (x @ a) @ b


def objective(x, target, base, a, b, scale):
    error = predict(x, base, a, b, scale) - target
    loss = np.mean(error**2)
    upstream = 2 * error / error.size
    da = scale * x.T @ upstream @ b.T
    db = scale * (x @ a).T @ upstream
    return float(loss), da, db


def check_gradients(x, target, base, a, b, scale):
    _, da, db = objective(x, target, base, a, b, scale)
    for parameter, gradient, index in ((a, da, (0, 0)), (b, db, (0, 0))):
        original = parameter[index]
        epsilon = 1e-5
        parameter[index] = original + epsilon
        above = objective(x, target, base, a, b, scale)[0]
        parameter[index] = original - epsilon
        below = objective(x, target, base, a, b, scale)[0]
        parameter[index] = original
        np.testing.assert_allclose(
            (above - below) / (2 * epsilon), gradient[index], rtol=1e-4, atol=1e-8
        )


def main():
    rng = np.random.default_rng(31)
    d_in, d_out, rank = 12, 8, 2
    scale = 1.0
    base = rng.normal(0, 0.2, (d_in, d_out))
    original_base = base.copy()
    true_update = rng.normal(0, 0.4, (d_in, rank)) @ rng.normal(0, 0.4, (rank, d_out))
    x_train = rng.normal(size=(128, d_in))
    x_heldout = rng.normal(size=(64, d_in))
    target_train = x_train @ (base + true_update)
    target_heldout = x_heldout @ (base + true_update)
    a = rng.normal(0, 0.1, (d_in, rank))
    b = np.zeros((rank, d_out))

    probe_b = rng.normal(0, 0.05, b.shape)
    check_gradients(x_train, target_train, base, a, probe_b, scale)
    np.testing.assert_allclose(predict(x_train, base, a, b, scale), x_train @ base)
    initial, da, db = objective(x_train, target_train, base, a, b, scale)
    np.testing.assert_array_equal(da, 0)
    assert np.linalg.norm(db) > 0
    print(f"Base parameters: {base.size}; trainable adapter parameters: {a.size + b.size}")
    print("B starts at zero: initial adapter output and A gradient are zero.")
    print(f"Initial train MSE: {initial:.6f}")
    for _ in range(1200):
        loss, da, db = objective(x_train, target_train, base, a, b, scale)
        if not np.isfinite(loss):
            raise RuntimeError("Adapter training diverged")
        a -= 0.2 * da
        b -= 0.2 * db
    final = objective(x_train, target_train, base, a, b, scale)[0]
    heldout = objective(x_heldout, target_heldout, base, a, b, scale)[0]
    merged = base + scale * a @ b
    np.testing.assert_array_equal(base, original_base)
    np.testing.assert_allclose(
        x_heldout @ merged, predict(x_heldout, base, a, b, scale), atol=1e-12
    )
    assert final < initial * 0.01
    print(f"Final train MSE: {final:.8f}; held-out MSE: {heldout:.8f}")
    print("The base stayed unchanged. Merged and unmerged projections agree.")
    print("The target update was rank two by construction; real tasks may need")
    print("different ranks and targets. This trains no language-model checkpoint.")


if __name__ == "__main__":
    main()

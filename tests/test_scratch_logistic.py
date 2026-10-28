"""Unit tests for the first-principles ScratchLogisticRegression implementation."""
import numpy as np
import pytest
from src.models.scratch_logistic import ScratchLogisticRegression


def test_scratch_logistic_separable_data():
    """Scratch logistic regression should easily separate well-separated 2D clusters."""
    rng = np.random.default_rng(42)
    # Cluster 0
    X0 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(50, 2))
    y0 = np.zeros(50)
    # Cluster 1
    X1 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(50, 2))
    y1 = np.ones(50)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    model = ScratchLogisticRegression(l2_reg=0.01, solver="lbfgs", random_state=42)
    model.fit(X, y)

    acc = model.score(X, y)
    assert acc == 1.0  # 100% accuracy on linearly separable data

    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_scratch_logistic_gradient_descent_solver():
    """Gradient descent solver should converge and achieve high accuracy."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 2))
    # True hyperplane: 1.5 * x0 - 2.0 * x1 + 0.5 > 0
    y = ((1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5) > 0).astype(int)

    model = ScratchLogisticRegression(l2_reg=0.1, solver="gd", max_iter=500, learning_rate=0.2)
    model.fit(X, y)

    assert len(model.cost_history_) > 0
    # Loss should decrease from start to finish
    assert model.cost_history_[-1] < model.cost_history_[0]
    assert model.score(X, y) > 0.90


def test_scratch_logistic_l2_shrinkage():
    """Higher L2 regularization parameter should shrink parameter magnitudes."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 3))
    y = rng.choice([0, 1], size=60)

    model_low_reg = ScratchLogisticRegression(l2_reg=0.001, solver="lbfgs", random_state=42).fit(X, y)
    model_high_reg = ScratchLogisticRegression(l2_reg=50.0, solver="lbfgs", random_state=42).fit(X, y)

    # Exclude bias term when comparing weight norms
    norm_low = np.linalg.norm(model_low_reg.theta_[1:])
    norm_high = np.linalg.norm(model_high_reg.theta_[1:])

    assert norm_high < norm_low


def test_scratch_logistic_decision_boundary_2d():
    """Verify calculation of 2D boundary line."""
    X = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    y = np.array([1, 1, 0, 0])

    model = ScratchLogisticRegression(l2_reg=0.1, solver="lbfgs", random_state=42).fit(X, y)
    x1_vals = np.linspace(-2, 2, 10)
    boundary = model.get_decision_boundary_2d(x1_vals)

    assert len(boundary) == 10
    assert not np.any(np.isnan(boundary))

"""
Scratch Logistic Regression Classifier with Scipy BFGS and Gradient Descent Solvers.
Elevates the reference implementation into a robust, regularized, API-compliant estimator.
"""

from typing import Optional, Tuple, Literal
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


class ScratchLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Numerically stable Logistic Regression implemented from first mathematical principles.
    Supports L2 (Ridge) regularization and dual optimization solvers: Scipy BFGS and Gradient Descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-6,
        l2_reg: float = 0.01,
        fit_intercept: bool = True,
        solver: Literal["bfgs", "gd"] = "bfgs",
        random_state: Optional[int] = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.random_state = random_state

        self.theta_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.n_iter_: int = 0
        self.loss_history_: list = []

    @property
    def cost_history_(self) -> list:
        return self.loss_history_

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function with exponent clipping."""
        clipped_z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped_z))

    def _cost_and_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculates regularized binary cross-entropy loss and its analytical gradient."""
        m = len(y)
        h = np.clip(self._sigmoid(X @ theta), 1e-15, 1.0 - 1e-15)

        # Cross-entropy loss
        bce = -np.mean(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))

        # L2 Regularization (exclude bias intercept from penalty)
        reg_theta = theta.copy()
        if self.fit_intercept:
            reg_theta[0] = 0.0
        reg_penalty = (self.l2_reg / (2.0 * m)) * np.sum(reg_theta ** 2)
        total_cost = float(bce + reg_penalty)

        # Gradient
        grad = (1.0 / m) * (X.T @ (h - y)) + (self.l2_reg / m) * reg_theta
        return total_cost, grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLogisticRegression":
        """
        Fits the logistic regression parameters to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary target vector of shape (n_samples,)
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()

        if len(X_arr) == 0:
            raise ValueError("Input feature matrix X is empty.")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X_arr)} rows, y has {len(y_arr)} labels.")

        n_samples, n_features = X_arr.shape

        # Prepend column of ones for intercept
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n_samples), X_arr])
        else:
            X_design = X_arr

        n_params = X_design.shape[1]

        # Parameter initialization
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            init_theta = rng.normal(0, 0.01, size=n_params)
        else:
            init_theta = np.zeros(n_params)

        if self.solver == "bfgs":
            def cost_fun(t):
                c, _ = self._cost_and_grad(t, X_design, y_arr)
                return c

            def grad_fun(t):
                _, g = self._cost_and_grad(t, X_design, y_arr)
                return g

            res = minimize(
                fun=cost_fun,
                x0=init_theta,
                jac=grad_fun,
                method="BFGS",
                options={"maxiter": self.max_iter, "gtol": self.tol},
            )
            self.theta_ = res.x
            self.n_iter_ = res.nit
        else:
            # Gradient descent solver
            theta = init_theta.copy()
            prev_cost = float("inf")
            self.loss_history_ = []

            for i in range(self.max_iter):
                cost, grad = self._cost_and_grad(theta, X_design, y_arr)
                self.loss_history_.append(cost)

                if abs(prev_cost - cost) < self.tol:
                    self.n_iter_ = i + 1
                    break

                prev_cost = cost
                theta -= self.learning_rate * grad
            else:
                self.n_iter_ = self.max_iter

            self.theta_ = theta

        # Separate intercept and coefficients
        if self.fit_intercept:
            self.intercept_ = float(self.theta_[0])
            self.coef_ = self.theta_[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta_.copy()

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Computes linear score z = X @ coef + intercept."""
        if self.theta_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X_arr = np.asarray(X, dtype=float)
        if self.fit_intercept:
            return X_arr @ self.coef_ + self.intercept_
        return X_arr @ self.coef_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns binary class probability estimates of shape (n_samples, 2)."""
        z = self.decision_function(X)
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary class labels {0, 1}."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_decision_boundary_2d(self, x_values: np.ndarray) -> np.ndarray:
        """
        Calculates analytical decision boundary y_values where w1*x + w2*y + b = 0.
        Only valid for 2D feature input!
        """
        if self.coef_ is None or len(self.coef_) != 2:
            raise ValueError("Decision boundary calculation requires exactly 2 features.")
        w1, w2 = self.coef_[0], self.coef_[1]
        b = self.intercept_
        if abs(w2) < 1e-9:
            raise ZeroDivisionError("w2 is near zero; boundary is vertical.")
        return -(w1 * x_values + b) / w2

import numpy as np
from sklearn.metrics import mean_squared_error
from project1_d import ols_gradient, ridge_gradient, momentum_gd, ADAgrad, RMSProp, ADAM

def lasso_gradient(X, y, theta, lam):
    n_samples = X.shape[0]
    grad = (2 / n_samples) * X.T @ (X @ theta - y)
    subgrad = lam * np.sign(theta)
    subgrad[0] = 0.0  # do not penalize intercept
    return grad + subgrad

def run_optimizer_lasso(method, X, y, lam=0.01):
    iterations = 400
    n_steps = 0.01
    momentum = 0.9

    if method == "GD":
        return momentum_gd(X, y, iterations, 0, n_steps, "Lasso", lam)
    elif method == "Momentum":
        return momentum_gd(X, y, iterations, momentum, n_steps, "Lasso", lam)
    elif method == "Adagrad":
        return ADAgrad(X, y, iterations, n_steps, "Lasso", lam)
    elif method == "RMSProp":
        return RMSProp(X, y, iterations, n_steps, "Lasso", lam)
    elif method == "Adam":
        return ADAM(X, y, iterations, n_steps, "Lasso", lam)
    else:
        raise ValueError("Unknown method")




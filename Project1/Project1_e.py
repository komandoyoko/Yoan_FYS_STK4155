import numpy as np
from sklearn.metrics import mean_squared_error
from project1_d import ADAgrad, RMSProp, ADAM

def lasso_gradient(X, y, theta, lam):
    n_samples = X.shape[0]
    grad = (2 / n_samples) * X.T @ (X @ theta - y)
    subgrad = lam * np.sign(theta)
    subgrad[0] = 0.0
    return grad + subgrad

def run_optimizer_lasso(method, X, y, lam=0.01):
    iterations = 400
    n_steps = 0.001
    momentum = 0.9

    if method in ["GD", "Momentum"]:
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)
        mse_val = np.zeros(iterations)
        mom = 0 if method == "GD" else momentum

        for i in range(iterations):
            grad = lasso_gradient(X, y, theta, lam)
            change = mom * change - n_steps * grad
            theta += change
            mse_val[i] = mean_squared_error(y, X @ theta)

        return theta, mse_val

    elif method == "Adagrad":
        return ADAgrad(X, y, iterations, n_steps, "Lasso", lam)
    elif method == "RMSProp":
        return RMSProp(X, y, iterations, n_steps, "Lasso", lam)
    elif method == "Adam":
        return ADAM(X, y, iterations, n_steps, "Lasso", lam)
    else:
        raise ValueError("Unknown method")






def lasso_gradient(X, y, theta, lam):
    grad = (2 / n_samples) * X.T @ (X @ theta - y)
    subgrad = lam * np.sign(theta)
    subgrad[0] = 0.0  # do not penalize intercept
    return grad + subgrad


def momentum_gd(X, y, iterations, momentum, n_steps, func, lam=0.01):
    theta = np.zeros(X.shape[1])
    change = np.zeros_like(theta)
    mse_val = np.zeros(iterations)
    
    for i in range(iterations):
        if func == "OLS":
            grad = ols_gradient(X, y, theta)
        elif func == "Ridge":
            grad = ridge_gradient(X, y, theta, lam)
        elif func == "Lasso":
            grad = lasso_gradient(X, y, theta, lam)
        else:
            raise ValueError("Unknown func type")

        change = momentum * change + n_steps * grad
        theta -= change
        mse_val[i] = mean_squared_error(y, X @ theta)
    
    return theta, mse_val

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


theta_lasso, mse_lasso = run_optimizer_lasso("GD", X_train, y_train, lam=0.01)
y_pred_lasso = X_test @ theta_lasso

print("MSE (Lasso, GD):", mean_squared_error(y_test, y_pred_lasso))
print("R2 (Lasso, GD):", R2_score(y_test, y_pred_lasso))



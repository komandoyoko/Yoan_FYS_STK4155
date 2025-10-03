import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures , StandardScaler

from project1_d import  ADAgrad , RMSProp , ADAM , momentum_gd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def Stochastic_gradient(X, y, lambd, method, optimizer, size_batch, epochs):
    datapoint = len(y)
    iterations = 1  # one update per batch
    n_steps = 0.0001 #we set a fixes learn rate
    momentum = 0.3 #this is the best momentum when testing
    lam = lambd
    theta = np.zeros(X.shape[1])
    mse_vals = []

    for epoch in range(epochs):
        indices = np.random.permutation(datapoint)
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, datapoint, size_batch):
            X_batch = X_shuffled[i:i+size_batch]
            y_batch = y_shuffled[i:i+size_batch]

            if method == "OLS" and optimizer == "Adagrad":
                theta, mse_val = ADAgrad(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)
            elif method == "OLS" and optimizer == "RMSProp":
                theta, mse_val = RMSProp(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)
            elif method == "OLS" and optimizer == "Adam":
                theta, mse_val = ADAM(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)
            elif method == "OLS" and optimizer == "Momentum":
                theta, mse_val = momentum_gd(X_batch, y_batch, iterations=iterations, momentum=momentum, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)
            elif method == "Ridge" and optimizer == "Momentum":
                theta, mse_val = momentum_gd(X_batch, y_batch, iterations=iterations, momentum=momentum, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)
            elif method == "Ridge" and optimizer == "Adagrad":
                theta, mse_val = ADAgrad(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)
            elif method == "Ridge" and optimizer == "RMSProp":
                theta, mse_val = RMSProp(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)
            elif method == "Ridge" and optimizer == "Adam":
                theta, mse_val = ADAM(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)

            mse_vals.extend(mse_val if isinstance(mse_val, (list, np.ndarray)) else [mse_val])

    return theta, np.array(mse_vals)
'''
def Stochastic_gradient(X , y  , lambd , method , optimizer , size_batch , epochs):
    
    datapoint = len(x)
    iterations = 1000          # one update per batch
    n_steps = 0.0001
    momentum = 0.3
    lam = lambd
    theta  = np.zeros(X.shape[1])
    mse_val = np.array([])
    for epoch in range(epochs):
        indices = np.random.permutation(datapoint)
        X_shuffled , y_shuffled = X[indices] , y[indices]

        for i in range(0, datapoint , size_batch):
            X_batch = X_shuffled[i:i+size_batch]
            y_batch = y_shuffled[i:i+size_batch]

            if method == "OLS" and optimizer == "Adagrad":
                theta, mse_val = ADAgrad(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="OLS", lam=lam)

            elif method == "OLS" and optimizer == "RMSProp":
                theta, mse_val = RMSProp(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="OLS", lam=lam)

            elif method == "OLS" and optimizer == "Adam":
                theta, mse_val = ADAM(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="OLS", lam=lam)
            
            elif method == "OLS" and optimizer == "Momentum":
                theta, mse_val = momentum_gd(X_batch, y_batch, iterations=iterations, momentum=momentum, n_steps=n_steps, func="OLS", lam=lam)

            elif method == "Ridge" and optimizer == "Momentum":
                theta, mse_val = momentum_gd(X_batch, y_batch, iterations=iterations, momentum=momentum, n_steps=n_steps, func="Ridge", lam=lam)

            elif method == "Ridge" and optimizer == "Adagrad":
                theta, mse_val = ADAgrad(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam)

            elif method == "Ridge" and optimizer == "RMSProp":
                theta, mse_val = RMSProp(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam)

            elif method == "Ridge" and optimizer == "Adam":
                theta, mse_val = ADAM(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam)
    return theta , mse_val
'''





x = np.linspace(-3 , 3 , 50)
y = 1/(1+25*x**2)

lam = 0.005
degree = 10

X = PolynomialFeatures(degree).fit_transform(x.reshape(-1 , 1))
X = StandardScaler().fit_transform(X)






optimizers = [ "Adagrad", "RMSProp", "Adam", "Momentum"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- OLS ---
for opt in optimizers:
    theta, mse_val = Stochastic_gradient(
        X, y,
        lambd=0,
        method="OLS",
        optimizer=opt,
        size_batch=50,
        epochs=50
    )
    axes[0].plot(mse_val, label=f"{opt}")
axes[0].set_title(f"SGD Convergence (OLS, degree={degree})")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

# --- Ridge ---
for opt in optimizers:
    theta, mse_val = Stochastic_gradient(
        X, y,
        lambd=lam,
        method="Ridge",
        optimizer=opt,
        size_batch=50,
        epochs=50
    )
    axes[1].plot(mse_val, label=f"{opt}")
axes[1].set_title(f"SGD Convergence (Ridge, degree={degree}, λ={lam})")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("MSE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
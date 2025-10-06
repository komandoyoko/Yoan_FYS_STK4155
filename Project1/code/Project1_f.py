import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures , StandardScaler
from project1_d import  ADAgrad , RMSProp , ADAM , momentum_gd



def Stochastic_gradient(X, y, lambd, method, optimizer, size_batch, epochs, total_updates=400):
    datapoint = len(y)
    n_steps = 0.03   # learning rate
    momentum = 0.3   # momentum factor
    lam = lambd
    theta = np.zeros(X.shape[1])

    mse_vals = []
    updates = 0   # keep track of total updates

    while updates < total_updates:   # run until we have the same number of updates as vanilla GD
        indices = np.random.permutation(datapoint) #running a random on the datapoints to shuffle
        X_shuffled, y_shuffled = X[indices], y[indices] #createing the new shuffled versions

        for i in range(0, datapoint, size_batch):
            if updates >= total_updates:
                break  # stop exactly at total_updates

            X_batch = X_shuffled[i:i+size_batch] #creating the minibatches
            y_batch = y_shuffled[i:i+size_batch]

            if method == "OLS" and optimizer == "Adagrad":
                theta, mse_val = ADAgrad(X_batch, y_batch, iterations=1, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)
            elif method == "OLS" and optimizer == "RMSProp":
                theta, mse_val = RMSProp(X_batch, y_batch, iterations=1, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)
            elif method == "OLS" and optimizer == "Adam":
                theta, mse_val = ADAM(X_batch, y_batch, iterations=1, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)
            elif method == "OLS" and optimizer == "Momentum":
                theta, mse_val = momentum_gd(X_batch, y_batch, iterations=1, momentum=momentum, n_steps=n_steps, func="OLS", lam=lam, theta_init=theta)

            elif method == "Ridge" and optimizer == "Adagrad":
                theta, mse_val = ADAgrad(X_batch, y_batch, iterations=1, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)
            elif method == "Ridge" and optimizer == "RMSProp":
                theta, mse_val = RMSProp(X_batch, y_batch, iterations=1, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)
            elif method == "Ridge" and optimizer == "Adam":
                theta, mse_val = ADAM(X_batch, y_batch, iterations=1, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)
            elif method == "Ridge" and optimizer == "Momentum":
                theta, mse_val = momentum_gd(X_batch, y_batch, iterations=1, momentum=momentum, n_steps=n_steps, func="Ridge", lam=lam, theta_init=theta)

            # store MSE
            mse_vals.append(mse_val[-1] if isinstance(mse_val, (list, np.ndarray)) else mse_val) #Here we append the MSE values, and create a robust append method

            updates += 1

    return theta, np.array(mse_vals)




x = np.linspace(-3 , 3 , 80)
y = 1/(1+25*x**2)

lam = 0.005
degree = 10

X = PolynomialFeatures(degree).fit_transform(x.reshape(-1 , 1)) #here we create the polynomial features
X = StandardScaler().fit_transform(X) #scale the data




from project1_d import run_optimizer_OLS , run_optimizer_ridge #import the optimizers ran in previous assignment



optimizers = [ "Adagrad", "RMSProp", "Adam", "Momentum"] #here we call our optimizers for the adaptive learning step

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- OLS ---
theta_gd, mse_gd = run_optimizer_OLS("GD", X, y, lambd=0)
axes[0].plot(mse_gd, label="Vanilla GD", color='black', linewidth=2)

for opt in optimizers:
    theta, mse_val = Stochastic_gradient(
        X, y,
        lambd=0,
        method="OLS",
        optimizer=opt,
        size_batch=80,
        epochs=400
    )
    axes[0].plot(mse_val, label=f"SGD {opt}", linestyle='--')

axes[0].set_title(f"OLS Convergence: Vanilla GD vs SGD (degree={degree})")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

# --- Ridge ---
theta_gd, mse_gd = run_optimizer_ridge("GD", X, y, lamb=lam)
axes[1].plot(mse_gd, label="Vanilla GD", color='black', linewidth=2)

for opt in optimizers:
    theta, mse_val = Stochastic_gradient(
        X, y,
        lambd=lam,
        method="Ridge",
        optimizer=opt,
        size_batch=80,
        epochs=400
    )
    axes[1].plot(mse_val, label=f"SGD {opt}", linestyle='--')

axes[1].set_title(f"Ridge Convergence: Vanilla GD vs SGD (λ={lam})")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("MSE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
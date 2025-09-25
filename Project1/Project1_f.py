import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from project1_d import  ADAgrad , RMSProp , ADAM , momentum_gd




def Stochastic_gradient(x , y  , degree  , lambd , method , optimizer , size_batch , epochs):
    X = PolynomialFeatures(degree).fit_transform(x.reshape(-1 , 1))
    
    datapoint = len(x)
    iterations = 400          # one update per batch
    n_steps = 0.0001
    momentum = 0.9
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


x = np.linspace(-3 , 3 , 50)
y = 1 / (1 + 25 * x**2)

lam = 0.005
degree = 3
'''
for e in lamb:
    theta , mse_val = Stochastic_gradient(x , y , method ="OLS", optimizer= "Adagrad", size_batch= = 30 , epochs = 50)
'''

optimizers = ["Momentum", "Adagrad", "RMSProp", "Adam"]

plt.figure(figsize=(12, 8))

for opt in optimizers:
    theta, mse_val = Stochastic_gradient(
        x, y,
        degree=degree,
        lambd=0,           # doesn’t matter for OLS
        method="OLS",
        optimizer=opt,
        size_batch=30,
        epochs=50
    )
    plt.plot(mse_val, label=f"{opt}")

plt.title(f"SGD Convergence (OLS, degree={degree})")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()




plt.figure(figsize=(12, 8))

for opt in optimizers:
    theta, mse_val = Stochastic_gradient(
        x, y,
        degree=degree,
        lambd=lam,
        method="Ridge",
        optimizer=opt,
        size_batch=30,
        epochs=50
    )
    plt.plot(mse_val, label=f"{opt}")

plt.title(f"SGD Convergence (Ridge, degree={degree}, λ={lam})")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()
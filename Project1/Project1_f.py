import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from project1_d import  ADAgrad , RMSProp , ADAM , momentum_gd




def Stochastic_gradient(x , y  , degree , theta , lambd , method , optimizer , nr_batches , size_batch , epochs):
    X = PolynomialFeatures(degree).fit_transform(x.reshape(-1 , 1))
    
    datapoint = len(x)
    iterations = 1          # one update per batch
    n_steps = 0.01
    momentum = 0.9
    lam = lambd
    
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

            elif method == "Ridge" and optimizer == "Momentum":
                theta, mse_val = momentum_gd(X_batch, y_batch, iterations=iterations, momentum=momentum, n_steps=n_steps, func="Ridge", lam=lam)

            elif method == "Ridge" and optimizer == "Adagrad":
                theta, mse_val = ADAgrad(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam)

            elif method == "Ridge" and optimizer == "RMSProp":
                theta, mse_val = RMSProp(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam)

            elif method == "Ridge" and optimizer == "Adam":
                theta, mse_val = ADAM(X_batch, y_batch, iterations=iterations, n_steps=n_steps, func="Ridge", lam=lam)
    return theta , mse_val


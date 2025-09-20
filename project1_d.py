import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


n_samples = 100


def ols_gradient(X, y, theta):
    return (2/n_samples) * X.T @ (X @ theta - y)

def ridge_gradient(X, y, theta, lam):
    return (2/n_samples) * X.T @ (X @ theta - y) + 2 * lam  * theta





def momentum_gd(X , y , iterations, momentum , n_steps , func , lam):
    theta = np.zeros(X.shape[1]) #we start by implementing the starting point of the theta
    change = np.zeros_like(theta) #we initialize the change
    for i in range(iterations): #here we want to iterate until we reach our point
        if func == "OLS": #we split between OLS and ridge gradients
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)

        change = momentum * change + n_steps * grad #the change is based on the momentum * change + direction we are headed
        theta -= change #the final result is theta is changed based on the previous change + the new one.
    return theta


def ADAgrad(X , y , iterations , n_steps , func , lam = 0.01 , eps = 1e-6 ):
    theta = np.zeros(X.shape[1])
    r = np.zeros_like(theta)

    for i in range(iterations):
        if func == "OLS":
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)
        
        gtgt = grad ** 2
        r= r+ gtgt

        theta = theta - (n_steps / (np.sqrt(r) + eps)) * grad

    return theta 


def RMSProp(X , y , iterations , n_steps , func , lam = 0.01 , p = 0.9 , eps = 1e-5):
    theta = np.zeros(X.shape[1])
    v = np.zeros_like(theta)

    for i in range(iterations):
        if func == "OLS":
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)

        v = p*v + (1-p) * grad**2

        theta = theta - (n_steps/np.sqrt(v + eps)) * grad

    return theta



def ADAM(X , y , iterations , n_steps , func , lam = 0.01 , b1 = 0.9 , b2 = 0.999 , eps = 1e-5):
    theta = np.zeros(X.shape[1])
    v = np.zeros_like(theta)
    m = np.zeros_like(theta)
    for i in range(1 , iterations + 1):
        if func == "OLS":
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)
        
        m = b1*m + (1-b1) * grad
        v = b2*v + (1-b2)* grad ** 2

        m_correct = m / (1-b1**i)
        v_correct = v / (1-b2**i)

        theta = theta - ( n_steps / (np.sqrt(v_correct) + eps) ) * m_correct

    return theta





x = np.linspace(-1, 1, 200)
y_true = 1 / (1 + 25 * x**2)
degree = 4
poly = PolynomialFeatures(degree)
X = poly.fit_transform(x.reshape(-1, 1))
y = y_true

# Split data for training (optional, but you can use all data for fitting here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def run_optimizer(method, X, y):
    iterations = 1000
    n_steps = 0.01
    lam = 0.01
    momentum = 0.9
    if method == "GD":
        # Standard gradient descent (no momentum)
        return momentum_gd(X, y, iterations, 0, n_steps, "OLS", lam)
    elif method == "Momentum":
        return momentum_gd(X, y, iterations, momentum, n_steps, "OLS", lam)
    elif method == "Adagrad":
        return ADAgrad(X, y, iterations, n_steps, "OLS", lam)
    elif method == "RMSProp":
        return RMSProp(X, y, iterations, n_steps, "OLS", lam)
    elif method == "Adam":
        return ADAM(X, y, iterations, n_steps, "OLS", lam)
    else:
        raise ValueError("Unknown method")

plt.plot(x, y_true, 'k', label="Runge function")
for method in ["GD", "Momentum", "Adagrad", "RMSProp", "Adam"]:
    theta = run_optimizer(method, X_train, y_train)
    y_pred = poly.transform(x.reshape(-1, 1)) @ theta
    plt.plot(x, y_pred, label=method)
plt.legend()
plt.title(f"Runge function approximation, degree={degree}")
plt.show()




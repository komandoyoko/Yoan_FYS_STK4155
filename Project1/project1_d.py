import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt


n_samples = 200


def ols_gradient(X, y, theta):
    return (2/n_samples) * X.T @ (X @ theta - y)

def ridge_gradient(X, y, theta, lam):
    return (2/n_samples) * X.T @ (X @ theta - y) + 2 * lam  * theta





def momentum_gd(X , y , iterations, momentum , n_steps , func , lam , theta_init = None):
    if theta_init is None:
        theta = np.zeros(X.shape[1])
    else:
        theta = theta_init.copy() #we start by implementing the starting point of the theta
    
    change = np.zeros_like(theta) #we initialize the change

    mse_val = np.zeros(iterations) # We also initialize mse_val as the measure of how far away from our goal we are
    for i in range(iterations): #here we want to iterate until we reach our point
        if func == "OLS": #we split between OLS and ridge gradients
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)

        change = momentum * change + n_steps * grad #the change is based on the momentum * change + direction we are headed
        theta -= change #the final result is theta is changed based on the previous change + the new one.

        mse_val[i] =  mean_squared_error(y , X @ theta) #we want to calculate the error away from the true values of the function
    return theta , mse_val





def ADAgrad(X , y , iterations , n_steps , func , lam  , eps = 1e-6 , theta_init = None): 
    if theta_init is None:
        theta = np.zeros(X.shape[1])
    else:
        theta = theta_init.copy() #initialize the parameters we need
    r = np.zeros_like(theta)

    mse_val = np.zeros(iterations)
    for i in range(iterations):
        if func == "OLS":
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)
        
        gtgt = grad ** 2 #here we are taking the square of the gradient
        r= r+ gtgt #we apply the adaptive step

        theta = theta - (n_steps / (np.sqrt(r) + eps)) * grad #we calculate the new theta here based on the adaptive step

        mse_val[i] = mean_squared_error(y , X @ theta)  #and we also calculate the MSE
    return theta , mse_val




def RMSProp(X , y , iterations , n_steps , func , lam  , p = 0.9 , eps = 1e-5, theta_init = None):
    if theta_init is None:
        theta = np.zeros(X.shape[1])
    else:
        theta = theta_init.copy() #initalize the first steps
    v = np.zeros_like(theta)

    mse_val = np.zeros(iterations)
    for i in range(iterations):
        if func == "OLS":
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)

        v = p*v + (1-p) * grad**2 # Here is our step

        theta = theta - (n_steps/np.sqrt(v + eps)) * grad #the new theta calculated from the RMSprop algorythm

        mse_val[i] = mean_squared_error(y , X @ theta) #as always we calculate the MSE
    return theta , mse_val



def ADAM(X , y , iterations , n_steps , func , lam  , b1 = 0.9 , b2 = 0.999 , eps = 1e-5, theta_init = None):
    if theta_init is None:
        theta = np.zeros(X.shape[1])
    else:
        theta = theta_init.copy() # we initialize the different things we need
    v = np.zeros_like(theta)
    m = np.zeros_like(theta)

    mse_val = np.zeros(iterations)
    for i in range(1 , iterations + 1): #we cannot start with 0 in this loop because of the bias corrections need to raised to the power of the iteration.
        if func == "OLS":
            grad = ols_gradient(X , y , theta)
        elif func == "Ridge":
            grad = ridge_gradient(X , y , theta , lam)
        
        m = b1*m + (1-b1) * grad #we define our unbiased terms
        v = b2*v + (1-b2)* grad ** 2

        m_correct = m / (1-b1**i) #here we apply the bias correction terms
        v_correct = v / (1-b2**i)

        theta = theta - ( n_steps / (np.sqrt(v_correct) + eps) ) * m_correct #calculate the theta according to the algorythm

        mse_val[i-1] = mean_squared_error(y , X @ theta)  #calculkate the MSE as always
    return theta , mse_val












'''
In this block of the code we define the Runge function, create a design matrix and try to extract the optimal theta values using the 
gradient descents we have implemented. This will be plotted against the actual runge function to visually see how close we are, and we will
also implement a plot of the MSE vs iterations of the different methods
'''
x = np.linspace(-1, 1, 200)  #we define x
y_true = 1 / (1 + 25 * x**2) #the true runge function
lam = 1e-4
degree = 6 #we choose a polynomial of 4th degree


# setup
poly = PolynomialFeatures(degree)
X = poly.fit_transform(x.reshape(-1, 1))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # <--- mean/std learned here

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_true, test_size=0.2)







'''
Here we define a function called run_optimizer that will take all the different functions and go thourgh them creating different gradient descent methods
and extracting the mse_val and theta values
'''


def run_optimizer_OLS(method, X, y , lambd):
    iterations = 400
    n_steps = 0.03
    momentum = 0.4
    lam = lambd
    if method == "GD":
        # Standard gradient descent (no momentum)
        return momentum_gd(X, y, iterations, momentum, n_steps, "OLS" , lam)
    elif method == "Momentum":
        return momentum_gd(X, y, iterations, momentum, n_steps, "OLS" , lam)
    elif method == "Adagrad":
        return ADAgrad(X, y, iterations, n_steps, "OLS" , lam)
    elif method == "RMSProp":
        return RMSProp(X, y, iterations, n_steps, "OLS" , lam)
    elif method == "Adam":
        return ADAM(X, y, iterations, n_steps, "OLS" , lam)
    else:
        raise ValueError("Unknown method")
    
def run_optimizer_ridge(method, X, y, lamb):
    iterations = 400
    n_steps = 0.03
    momentum = 0.4
    lam = lamb
    if method == "GD":
        return momentum_gd(X, y, iterations, momentum, n_steps, "Ridge", lam)
    elif method == "Momentum":
        return momentum_gd(X, y, iterations, momentum, n_steps, "Ridge", lam)
    elif method == "Adagrad":
        return ADAgrad(X, y, iterations, n_steps, "Ridge", lam)
    elif method == "RMSProp":
        return RMSProp(X, y, iterations, n_steps, "Ridge", lam)
    elif method == "Adam":
        return ADAM(X, y, iterations, n_steps, "Ridge", lam)
    else:
        raise ValueError("Unknown method")













if __name__ == "__main__" :

    plt.figure(figsize=(12, 10))

    # --- OLS fit ---
    plt.subplot(2, 2, 1)
    plt.plot(x, y_true, 'k', label="Runge function")
    for method in ["GD", "Momentum", "Adagrad", "RMSProp", "Adam"]:
        theta, mse_val = run_optimizer_OLS(method, X_train, y_train, lam)

        # Use the same poly + scaler from above
        X_pred = poly.transform(x.reshape(-1, 1))
        X_pred = scaler.transform(X_pred)   # <--- FIX: use transform, not fit_transform
        y_pred = X_pred @ theta

        plt.plot(x, y_pred, label=method)
    plt.legend()
    plt.title(f"OLS Runge approximation (degree={degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)




    # OLS convergence
    plt.subplot(2, 2, 2)
    for method in ["GD", "Momentum", "Adagrad", "RMSProp", "Adam"]:
        _, mse_val = run_optimizer_OLS(method, X_train, y_train, lam)
        plt.plot(mse_val, label=method)
    plt.legend()
    plt.title("OLS Convergence (MSE vs iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.grid(True)












    # Ridge fit
    plt.subplot(2, 2, 3)
    plt.plot(x, y_true, 'k', label="Runge function")
    for method in ["GD", "Momentum", "Adagrad", "RMSProp", "Adam"]:
        theta, mse_val = run_optimizer_ridge(method, X_train, y_train, lam)

        X_pred = poly.transform(x.reshape(-1, 1))
        X_pred = scaler.transform(X_pred)   # <--- same scaler here too
        y_pred = X_pred @ theta

        plt.plot(x, y_pred, label=method)
    plt.legend()
    plt.title(f"Ridge Runge approximation (degree={degree}, λ={lam})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # Ridge convergence 
    plt.subplot(2, 2, 4)
    for method in ["GD", "Momentum", "Adagrad", "RMSProp", "Adam"]:
        _, mse_val = run_optimizer_ridge(method, X_train, y_train, lam)
        plt.plot(mse_val, label=method)
    plt.legend()
    plt.title("Ridge Convergence (MSE vs iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
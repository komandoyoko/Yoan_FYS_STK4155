import numpy as np
from matplotlib import pyplot as plt

from data import generate_data
from models.ols import OLS
from models.ridge import Ridge
from models.lasso import Lasso
from models.optimization import (
    Optimizer, 
    FixedLearningRate,
    AdaGrad, 
    Adam,
    RMSProp
)

def visualize_data(x: np.ndarray, y: np.ndarray,
                   model: OLS,
                   optimizer: Optimizer | None = None,
                   batch_size: int | None = None) -> None:
    """
    Visualize the input data and a simple OLS fit.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        y (np.ndarray): Output data of shape (n_samples,).
        model (OLS): The OLS model to fit and visualize.
        optimizer (Optimizer | None): The optimization method to use. If None, use analytical solution.
        batch_size (int | None): The batch size for stochastic gradient descent. If None,
    Returns:
        None
    """
    name = model.name()
    if optimizer is None:
        name += "(Analytical Solution)"
    else:
        name += f"(Gradient Descent with {optimizer.name()})"
        if batch_size is not None:
            name += f", batch_size={batch_size}"

    # fit model and predict
    model.fit(optimizer=optimizer, batch_size=batch_size)
    y_pred = model.predict()

    # Create figure or use existing one
    plt.figure(figsize=(10, 6))  
    # Plot data and prediction points
    plt.scatter(x, y, alpha=0.6, color='lightgray', s=20, label='Data', zorder=1)
    plt.scatter(model.x_test, y_pred, label='Prediction', color='blue', zorder=2)

    # Create smooth prediction line
    x_smooth = np.linspace(x.min(), x.max(), 300)
    X_smooth = np.vander(x_smooth, N=model.degree + 1, increasing=True)
    y_smooth = X_smooth @ model.theta
    plt.plot(x_smooth, y_smooth, color='red', linewidth=2, label='Prediction', zorder=2)

    # Final plot settings
    plt.title(f"{name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def analyze_bias_variance(x: np.ndarray, y: np.ndarray, 
                          max_degree: int = 15, 
                          n_bootstraps: int = 100) -> None:
    """
    Analyze bias-variance trade-off using bootstrap resampling.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        y (np.ndarray): Output data of shape (n_samples,).
        degree (int): Polynomial degree for the model.
        n_bootstraps (int): Number of bootstrap samples.
    """
    from models.ols import OLS
    model: OLS = OLS(x, y)

    degrees:  np.ndarray = np.arange(1, max_degree + 1)
    error:    np.ndarray = np.zeros(max_degree)
    bias:     np.ndarray = np.zeros(max_degree)
    variance: np.ndarray = np.zeros(max_degree)

    for degree in degrees:
        model.set_degree(degree)
        error[degree - 1], bias[degree - 1], variance[degree - 1] = \
            model.bootstrap(n_bootstraps=n_bootstraps)
    
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, error, marker='o', label='Total Error')
    plt.plot(degrees, bias, marker='o', label='Bias²')
    plt.plot(degrees, variance, marker='o', label='Variance')
    plt.title("Bias-Variance Trade-off")
    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_cross_validation(x: np.ndarray, y: np.ndarray, 
                             models: list[OLS],
                             max_degree: int = 15, 
                             k_folds: int = 5,
                             optimizer: Optimizer | None = None) -> None:
    """
    Analyze model performance using k-fold cross-validation.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        y (np.ndarray): Output data of shape (n_samples,).
        models (list[OLS]): List of models to evaluate.
        max_degree (int): Maximum polynomial degree to evaluate.
        k_folds (int): Number of folds for cross-validation.
    """

    degrees: np.ndarray = np.arange(1, max_degree + 1)
    mse_cv:  np.ndarray = np.zeros((len(models), max_degree))

    for i, model in enumerate(models):
        for degree in degrees:
            model.set_degree(degree)
            mse_cv[i, degree - 1] = model.kfold_cross_validation(optimizer=optimizer, k=k_folds)

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.plot(degrees, mse_cv[i], marker='o', label=f"{model.name()} λ={getattr(model, 'reg_lambda', 'N/A')}")
    plt.title(f"Mean Squared Error - Cross Validation ({k_folds}-fold), Optimizer: {optimizer.name() if optimizer else 'Analytical'}")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Generate data
    n = 20
    x, y = generate_data(n_samples=n, noise=True, random_state=42)
    optimizer = FixedLearningRate()

    # Visualize data with OLS fit
    ols = OLS(x, y, degree=8)
    visualize_data(x, y, ols, optimizer=optimizer)

    # Visualize data with Ridge fit
    ridge = Ridge(x, y, degree=8, reg_lambda=0.001)
    visualize_data(x, y, ridge, optimizer=optimizer)

    # Visualize data with Lasso fit
    lasso = Lasso(x, y, degree=8, reg_lambda=0.01)
    visualize_data(x, y, lasso, optimizer=optimizer)

    # Cross-Validation
    models: list[OLS] = [ols, ridge, lasso]
    analyze_cross_validation(x, y, models=models, max_degree=15, k_folds=5, optimizer=optimizer)

    # Analyze bias-variance trade-off
    analyze_bias_variance(x, y, max_degree=20, n_bootstraps=100)

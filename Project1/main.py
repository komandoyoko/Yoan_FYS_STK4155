import numpy as np
from matplotlib import pyplot as plt
from models.optimization import (
    Optimizer, 
    FixedLearningRate,
    AdaGrad, 
    Adam,
    RMSProp
)

def visualize_data(x: np.ndarray, y: np.ndarray, 
                   degree: int, 
                   optimizer: Optimizer | None = None,
                   batch_size: int | None = None) -> None:
    """
    Visualize the input data and a simple OLS fit.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        y (np.ndarray): Output data of shape (n_samples,).
    """
    from models.ols import OLS

    if optimizer is None:
        name = "OLS (Analytical Solution)"
    else:
        name = f"OLS (Gradient Descent with {optimizer.name()})"
        if batch_size is not None:
            name += f", batch_size={batch_size}"

    # Create and fit OLS model
    model: OLS = OLS(x, y, degree=degree)
    model.gradient_descent(optimizer=optimizer, batch_size=batch_size) if optimizer else model.fit()
    y_pred = model.predict()

    # Plot data and prediction points
    plt.scatter(x, y, alpha=0.6, color='lightgray', s=20, label='Data', zorder=1)
    plt.scatter(model.x_test, y_pred, label='OLS Prediction', color='blue', zorder=2)
    
    # Create smooth prediction line
    x_smooth = np.linspace(x.min(), x.max(), 300)
    X_smooth = np.vander(x_smooth, N=degree + 1, increasing=True)
    y_smooth = X_smooth @ model.theta
    plt.plot(x_smooth, y_smooth, color='red', linewidth=2, label='OLS Prediction', zorder=2)

    # Final plot settings
    plt.title(f"{name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_ols(x: np.ndarray, y: np.ndarray, 
                degrees: int = 15, 
                optimizer: Optimizer | None = None,
                batch_size: int | None = None) -> None:
    """
    Analyze OLS model performance using Gradient Descent.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        y (np.ndarray): Output data of shape (n_samples,).
        degrees (int): Maximum polynomial degree to evaluate.
    """
    from models.ols import OLS
    model: OLS = OLS(x, y)

    if optimizer is None:
        name = "OLS"
    else:
        name = f"OLS (GD: {optimizer.name()})"
        if batch_size is not None:
            name += f", batch_size={batch_size}"

    degrees: np.ndarray = np.arange(1, degrees + 1)
    mse: list[float] = []
    r2: list[float] = []

    for degree in degrees:
        model.set_degree(degree)
        model.fit() 
        if optimizer is None:
            model.fit()
        else:
            model.gradient_descent(optimizer=optimizer, batch_size=batch_size)
        mse.append(model.mse())
        r2.append(model.r2_score())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(degrees, mse, marker='o')
    plt.title(f"Mean Squared Error - {name}")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(degrees, r2, marker='o')
    plt.title(f"R² Score - {name}")
    plt.xlabel("Degree")
    plt.ylabel("R²")
    plt.grid()

    plt.tight_layout()
    plt.show()

def analyze_ridge(x: np.ndarray, y: np.ndarray, 
                  degrees: int = 15, 
                  lambdas: list[float] = [0.001, 0.01, 0.1, 1, 10],
                  optimizer: Optimizer | None = None,
                  batch_size: int | None = None) -> None:
    """
    Analyze Ridge regression model performance using Gradient Descent.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        y (np.ndarray): Output data of shape (n_samples,).
        degrees (int): Maximum polynomial degree to evaluate.
    """
    from models.ridge import Ridge
    model: Ridge = Ridge(x, y)

    degrees: np.ndarray = np.arange(1, degrees + 1)

    plt.figure(figsize=(12, 5))

    for reg_lambda in lambdas:
        model.set_lambda(reg_lambda)
        
        if optimizer is None:
            name = "Ridge"
        else:
            name = f"Ridge (GD: {optimizer.name()})"
            if batch_size is not None:
                name += f", batch_size={batch_size}"

        mse: list[float] = []
        r2: list[float] = []

        for degree in degrees:
            model.set_degree(degree)
            if optimizer is None:
                model.fit()
            else:
                model.gradient_descent(optimizer=optimizer, batch_size=batch_size)
            mse.append(model.mse())
            r2.append(model.r2_score())

        plt.subplot(1, 2, 1)
        plt.plot(degrees, mse, marker='o', label=f'λ={reg_lambda}')

        plt.subplot(1, 2, 2)
        plt.plot(degrees, r2, marker='o', label=f'λ={reg_lambda}')

    plt.subplot(1, 2, 1)
    plt.title(f"Mean Squared Error - {name}")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f"R² Score - {name}")
    plt.xlabel("Degree")
    plt.ylabel("R²")
    plt.grid()
    plt.legend()

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
                         max_degree: int = 15, 
                         k_folds: int = 5) -> None:
    """
    Analyze model performance using k-fold cross-validation.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        y (np.ndarray): Output data of shape (n_samples,).
        max_degree (int): Maximum polynomial degree to evaluate.
        k_folds (int): Number of folds for cross-validation.
    """
    from models.ols import OLS
    from sklearn.model_selection import KFold

    model: OLS = OLS(x, y)
    degrees: np.ndarray = np.arange(1, max_degree + 1)
    mse_cv:  np.ndarray = np.zeros(max_degree)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for degree in degrees:
        model.set_degree(degree)
        mse_fold = []

        for train_index, test_index in kf.split(x):
            model.x_train, model.y_train = x[train_index], y[train_index]
            model.x_test, model.y_test = x[test_index], y[test_index]
            model.fit()
            mse_fold.append(model.mse())

        mse_cv[degree - 1] = np.mean(mse_fold)

    plt.figure(figsize=(8, 6))
    plt.plot(degrees, mse_cv, marker='o')
    plt.title(f"{k_folds}-Fold Cross-Validation MSE")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    from data import generate_data
    """
    n = 1000
    x, y = generate_data(n_samples=n, noise=True, random_state=42)
    batch_size = 50
    optimizers = [ FixedLearningRate(), AdaGrad(), Adam(), RMSProp() ]

    # Analytical solutions
    visualize_data(x, y, degree=8)
    analyze_ols(x, y)
    analyze_ridge(x, y)

    # Gradient Descent solutions
    for optimizer in optimizers:
        visualize_data(x, y, degree=8, optimizer=optimizer)
        analyze_ols(x, y, optimizer=optimizer)
        analyze_ridge(x, y, optimizer=optimizer)

    # Mini-batch Gradient Descent solutions
    for optimizer in optimizers:
        visualize_data(x, y, degree=8, optimizer=optimizer, batch_size=batch_size)
        analyze_ols(x, y, optimizer=optimizer, batch_size=batch_size)
        analyze_ridge(x, y, optimizer=optimizer, batch_size=batch_size)

    # Cross-Validation
    analyze_cross_validation(x, y, max_degree=10, k_folds=5)
    """
    # Analyze bias-variance trade-off
    x, y = generate_data(n_samples=100, noise=True, random_state=42)
    analyze_bias_variance(x, y, max_degree=16, n_bootstraps=100)


from typing import override
import numpy as np
from .ols import OLS
from .optimization import Optimizer, FixedLearningRate # default optimizer if none provided

class Lasso(OLS):
    """
    LASSO (Least Absolute Shrinkage and Selection Operator) Regression Model.
    """
    
    def __init__(self, 
                 x: np.ndarray, 
                 y: np.ndarray, 
                 degree: int = 1, 
                 reg_lambda: float = 0.01,
                 test_size: float = 0.2) -> None:
        """
        Initialize the LASSO regression model with data.
        Args:
            x (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Output data of shape (n_samples,).
            degree (int): The polynomial degree for the model.
            reg_lambda (float): L1 regularization strength (lambda).
            test_size (float): Proportion of the dataset to include in the test split.
        """
        super().__init__(x, y, degree, test_size)
        self.reg_lambda = reg_lambda
    
    @override
    def name(self):
        return "LASSO"
    
    def set_lambda(self, reg_lambda: float) -> None:
        """
        Set the regularization strength for the model.
        Args:
            reg_lambda (float): L1 regularization strength.
        """
        self.reg_lambda = reg_lambda

    @override
    def fit(self, optimizer: Optimizer | None = None, batch_size: int | None = None) -> None:
        """
        Fit the LASSO model using gradient descent.
        LASSO cannot be solved analytically, so it requires an optimizer.
        Args:
            optimizer (Optimizer | None): The optimization method to use. Required for LASSO.
            batch_size (int | None): The batch size for stochastic gradient descent.
        """
        if optimizer is None:
            optimizer = FixedLearningRate()
        
        self.gradient_descent(optimizer=optimizer, batch_size=batch_size)

    @override
    def update_parameters(self, theta: np.ndarray, eta: float) -> None:
        """
        Update the model parameters. 
        Args:
            theta (np.ndarray): The new parameter values.
        """
        self.theta = np.sign(theta) * np.maximum(0, np.abs(theta) - self.reg_lambda * eta)

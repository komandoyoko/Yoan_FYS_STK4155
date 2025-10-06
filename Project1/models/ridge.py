from typing import override
import numpy as np

from .optimization import Optimizer
from .ols import OLS

class Ridge(OLS):
    def __init__(self, 
                 x: np.ndarray, 
                 y: np.ndarray, 
                 degree: int = 1, 
                 reg_lambda: float = 1.0, 
                 test_size: float = 0.2) -> None:
        """
        Initialize the Ridge regression model with data.
        Args:
            x (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Output data of shape (n_samples,).
            degree (int): The polynomial degree for the model.
            reg_lambda (float): Regularization strength.
            test_size (float): Proportion of the dataset to include in the test split.
        """
        super().__init__(x, y, degree, test_size)
        self.reg_lambda = reg_lambda
    
    @override
    def name(self):
        return "Ridge"
    
    def set_lambda(self, reg_lambda: float) -> None:
        """
        Set the regularization strength for the model.
        Args:
            reg_lambda (float): Regularization strength.
        """
        self.reg_lambda = reg_lambda

    @override
    def fit(self, optimizer: Optimizer | None = None, batch_size: int | None = None) -> None:
        """
        Fit the model to the data (analytical).
        """
        if optimizer is not None:
            self.gradient_descent(optimizer=optimizer, batch_size=batch_size)
        else:
            X = self.design_matrix()
            self.theta = np.linalg.inv(X.T @ X + self.reg_lambda * np.eye(X.shape[1])) @ X.T @ self.y_train

    @override
    def gradient(self, X):
        return super().gradient(X) + 2 * self.reg_lambda * self.theta

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from .optimization import Optimizer

class OLS:
    """ 
    Ordinary Least Squares Regression Model.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, 
                 degree: int = 1, 
                 test_size: float = 0.2) -> None:
        """
        Initialize the OLS model with data.
        Args:
            x (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Output data of shape (n_samples,).
            degree (int): The polynomial degree for the model.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=test_size)
        self.degree = degree
        self.theta: np.ndarray = None
    
    def set_degree(self, degree: int) -> None:
        """
        Set the polynomial degree for the model.
        Args:
            degree (int): The polynomial degree for the model.
        """
        self.degree = degree
    
    def design_matrix(self) -> np.ndarray:
        """
        Create the design matrix for polynomial features.
        Returns:
            np.ndarray: The design matrix.
        """
        return np.vander(self.x_train, N=self.degree + 1, increasing=True)

    def fit(self) -> None:
        """
        Fit the model to the data (analytical).
        """
        X = self.design_matrix()
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ self.y_train

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function.
        Args:
            X (np.ndarray): The design matrix.
        Returns:
            np.ndarray: The gradient vector.
        """
        return -2/len(self.y_train) * X.T @ (self.y_train - X @ self.theta)

    def hessian(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian matrix of the loss function.
        Args:
            X (np.ndarray): The design matrix.
        Returns:
            np.ndarray: The Hessian matrix.
        """
        return 2/len(self.y_train) * X.T @ X

    def gradient_descent(self, optimizer: Optimizer, batch_size: int | None = None) -> None:
        """
        Fit the model using Gradient Descent.
        Args:
            optimizer (Optimizer): The optimization method to use.
            batch_size (int | None): The batch size for stochastic gradient descent. If None, use full batch gradient descent.
        """
        X = self.design_matrix()
        self.theta = np.random.randn(X.shape[1])

        H = self.hessian(X)        # Hessian matrix
        eig = np.linalg.eigvals(H) # Eigenvalues of the Hessian
        eta = 1.0 / np.max(eig)    # Learning rate based on Hessian's largest eigenvalue

        optimizer.reset()          # Reset optimizer state for new parameter size
        while optimizer.iterate():
            # Stochastic Gradient Descent with mini-batches
            if batch_size is not None:
                indices = np.random.choice(len(self.y_train), batch_size, replace=False)
                X_batch = X[indices]
                y_batch = self.y_train[indices]
                grad = -2/batch_size * X_batch.T @ (y_batch - X_batch @ self.theta)
                self.theta = optimizer.step(self.theta, grad, eta)
            # Full Batch Gradient Descent
            else:
                grad = self.gradient(X)
                self.theta = optimizer.step(self.theta, grad, eta)

    def predict(self) -> np.ndarray:
        """
        Predict using the fitted model.
        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        if self.theta is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        X = np.vander(self.x_test, N=self.degree + 1, increasing=True)
        return X @ self.theta

    def mse(self) -> float:
        """
        Calculate the Mean Squared Error of the model.
        Returns:
            float: Mean Squared Error.
        """
        return np.mean((self.y_test - self.predict()) ** 2)

    def r2_score(self) -> float:
        """
        Calculate the R² score of the model.
        Returns:
            float: R² score.
        """
        return 1 - np.sum((self.y_test - self.predict()) ** 2) \
                 / np.sum((self.y_test - np.mean(self.y_test)) ** 2)
    
    def bootstrap(self, n_bootstraps: int = 100) -> tuple[float, float, float]:
        """
        Perform bootstrap resampling to estimate bias and variance.
        Args:
            n_bootstraps (int): Number of bootstrap samples.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing
                - error (np.ndarray): Total error of the model.
                - bias (np.ndarray): Bias of the model.
                - variance (np.ndarray): Variance of the model.
        """
        x, y = self.x_train, self.y_train # Save original data
        y_test = self.y_test.reshape(-1, 1)
        y_pred = np.zeros((self.y_test.shape[0], n_bootstraps))

        for i in range(n_bootstraps):
            self.x_train, self.y_train = resample(x, y) # Bootstrap resample
            self.fit()
            y_pred[:, i] = self.predict().ravel()

        error: float = np.mean( (y_test - y_pred)**2 )
        bias: float = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance: float = np.mean( np.var(y_pred, axis=1, keepdims=True))

        self.x_train, self.y_train = x, y  # Restore original data

        return error, bias, variance

    def kfold_cross_validation(self, k: int = 5) -> float:
        """
        Perform k-fold cross-validation to estimate the model's performance.
        Args:
            k (int): Number of folds.
        Returns:
            float: Average Mean Squared Error across all folds.
        """

        kf = KFold(n_splits=k, shuffle=True)
        mse_list = []

        x, y = self.x_train, self.y_train  # Use training data for cross-validation

        for train_index, val_index in kf.split(x):
            x_train_fold, x_val_fold = x[train_index], x[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            self.x_train, self.y_train = x_train_fold, y_train_fold
            self.fit()

            X_val_design = np.vander(x_val_fold, N=self.degree + 1, increasing=True)
            y_val_pred = X_val_design @ self.theta
            mse_fold = np.mean((y_val_fold - y_val_pred) ** 2)
            mse_list.append(mse_fold)

        self.x_train, self.y_train = x, y  # Restore original training data

        return np.mean(mse_list)

from typing import override
import numpy as np

MAX_ITER_DEFAULT: int = 1000

class Optimizer:
    """ 
    Base class for optimization methods.
    """

    def __init__(self, max_iter: int = MAX_ITER_DEFAULT):
        self.max_iter = max_iter
        self.iterations = max_iter
    
    def iterate(self) -> bool:
        """
        Check if more iterations are allowed.
        Returns:
            bool: True if more iterations are allowed, False otherwise.
        """
        self.iterations -= 1
        return self.iterations > 0

    def name(self) -> str:
        return self.__class__.__name__

    def step(self, theta: np.ndarray, gradient: np.ndarray, eta: float = 1.0) -> np.ndarray:
        """
        Perform a single optimization step.
        Args:
            theta (np.ndarray): Current parameters.
            gradient (np.ndarray): Current gradient.
            eta (float): Learning rate.
        Returns:
            np.ndarray: Updated parameters.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def reset(self) -> None:
        """Reset optimizer state. Override in stateful optimizers."""
        self.iterations = self.max_iter

class FixedLearningRate(Optimizer):
    """
    Fixed Learning Rate optimization method.
    """

    @override
    def name(self) -> str:
        return "Fixed Learning Rate"

    @override
    def step(self, theta: np.ndarray, gradient: np.ndarray, eta: float = 1.0) -> np.ndarray:
        return theta - eta * gradient

class AdaGrad(Optimizer):
    """
    AdaGrad optimization method.
    """

    def __init__(self, max_iter: int = MAX_ITER_DEFAULT, epsilon: float = 1e-8) -> None:
        super().__init__(max_iter)
        self.epsilon = epsilon
        self.G = None

    @override
    def name(self) -> str:
        return "AdaGrad"

    @override
    def reset(self) -> None:
        super().reset()
        self.G = None

    @override
    def step(self, theta: np.ndarray, gradient: np.ndarray, eta: float = 1.0) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros_like(gradient)
        self.G += gradient**2
        lr = eta / (np.sqrt(self.G) + self.epsilon)
        return theta - lr * gradient

class Adam(Optimizer):
    """
    Adam optimization method.
    """

    def __init__(self, max_iter: int = MAX_ITER_DEFAULT, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__(max_iter)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    @override
    def name(self) -> str:
        return "Adam"

    @override
    def reset(self) -> None:
        super().reset()
        self.m = None
        self.v = None
        self.t = 0

    @override
    def step(self, theta: np.ndarray, gradient: np.ndarray, eta: float = 1.0) -> np.ndarray: 
        if self.m is None:
            self.m = np.zeros_like(gradient)
        if self.v is None:
            self.v = np.zeros_like(gradient)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return theta - (eta * m_hat) / (np.sqrt(v_hat) + self.epsilon)

class Momentum(Optimizer):
    """
    Momentum optimization method.
    """

    def __init__(self, max_iter: int = MAX_ITER_DEFAULT, momentum: float = 0.9) -> None:
        super().__init__(max_iter)
        self.momentum = momentum
        self.velocity = None

    @override
    def name(self) -> str:
        return "Momentum"

    @override   
    def reset(self) -> None:
        super().reset()
        self.velocity = None

    @override
    def step(self, theta: np.ndarray, gradient: np.ndarray, eta: float = 1.0) -> np.ndarray:     
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        self.velocity = self.momentum * self.velocity - eta * gradient
        return theta + self.velocity

class RMSProp(Optimizer):
    """
    RMSProp optimization method.
    """
    
    def __init__(self, max_iter: int = MAX_ITER_DEFAULT, decay_rate: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__(max_iter)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    @override
    def name(self) -> str:
        return "RMSProp"

    @override
    def reset(self) -> None:
        super().reset()
        self.cache = None

    @override
    def step(self, theta: np.ndarray, gradient: np.ndarray, eta: float = 1.0) -> np.ndarray:
        if self.cache is None:
            self.cache = np.zeros_like(gradient)
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * (gradient ** 2)
        lr = eta / (np.sqrt(self.cache) + self.epsilon)
        return theta - lr * gradient

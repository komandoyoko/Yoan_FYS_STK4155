import numpy as np

START = -1.0
STOP = 1.0

def runge(x: np.ndarray, noise: bool = False) -> np.ndarray:
    """
    Compute the Runge function with optional Gaussian noise.
    Args:
        x (np.ndarray): Input data of shape (n_samples,).
        noise (bool): If True, add Gaussian noise to the output.

    Returns:
        np.ndarray: Output data of shape (n_samples,).
    """
    y = 1 / (1 + 25 * x**2)
    if noise:     
        y += np.random.normal(0, 0.1, x.shape)   
    return y

def x(n_samples: int = 100, random_state: int | None = None) -> np.ndarray:
    """
    Generate uniformly spaced input data in the range [start, stop].
    Args:
        start (float): Start of the range.
        stop (float): End of the range.
        n_samples (int): Number of samples to generate.
        random_state (int | None): Seed for the random number generator.

    Returns:
        np.ndarray: Input data of shape (n_samples,).
    """
    if random_state is not None:
        np.random.seed(random_state)
    return np.linspace(START, STOP, n_samples)

def generate_data(n_samples: int = 1000, noise: bool = True, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate input and output data using the Runge function.
    Args:
        n_samples (int): Number of samples to generate.
        noise (bool): If True, add Gaussian noise to the output.
        random_state (int | None): Seed for the random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing input data (x) and output data (y).
    """
    x_data = x(n_samples=n_samples, random_state=random_state)
    y_data = runge(x_data, noise=noise)
    return x_data, y_data

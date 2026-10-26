import numpy as np

def to_01(y):
    """Convert labels from {-1, 1} to {0, 1}."""
    return (y + 1) // 2

def to_pm1(y):
    """Convert labels from {0, 1} back to {-1, 1} for submission."""
    return np.where(y == 0, -1, 1)

def standardize(x, mean=None, std=None):
    """
    Standardize features: (x - mean) / std
    If mean and std are not provided, calculate them (fit on train).
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
        
    # Avoid division by zero by replacing 0 std with 1
    std = np.where(std == 0, 1.0, std)
    
    x_normalized = (x - mean) / std
    return x_normalized, mean, std

def add_bias(x):
    """Add a column of 1s to the feature matrix."""
    return np.hstack([np.ones((x.shape[0], 1)), x])
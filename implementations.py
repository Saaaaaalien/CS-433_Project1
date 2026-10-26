import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    # e = y - Xw
    # gradient = -1/N*(X.T @ e)
    # w(t+1) = w(t) - gamma*gradient
    # mean squared error  = 1/(2N)*sum(y-tx@w)^2 = 1/2*mean(y-tx@w)^2
    """
    w = initial_w.copy()
    n = len(y)
    for i in range(max_iters):
        e = y - tx @ w
        gradient = -1 / n * (tx.T @ e)
        w = w - gamma * gradient

    loss = 0.5 * np.mean((y - tx @ w) ** 2)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    # gradient = -xi*(yi - xi @ w)
    # w(t+1) = w(t) - gamma*sgd
    # mean_squared_error = 1/2 * np.mean(e^2)
    """

    w = initial_w.copy()
    n = len(y)
    randg = np.random.default_rng(0)
    for i in range(max_iters):
        j = randg.integers(n)
        xj, yj = tx[j], y[j]
        gradient = -xj * (yj - xj @ w)
        w = w - gamma * gradient
    loss = 0.5 * np.mean((y - tx @ w) ** 2)
    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations

    # w = (x.T@X)^-1 (X.T@y)
    # e = y - X @ w
    # loss = 1/2 * np.mean(e^2)
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = 0.5 * np.mean(e**2)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    # w = (X.T@X + 2 * N * lambda * I)^-1 (X.T @ y)
    # e = y - X @ w
    # loss = 1/2 * np.mean(e**2)
    """

    a = tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)

    e = y - tx @ w
    loss = 0.5 * np.mean(e**2)

    return w, loss


def _sigmoid(eta):
    # sigmoid = 1/(1 + e^-eta) for eta >= 0
    # sigmoid = e^eta/ (1 + e^eta) for eta < 0
    sig = np.empty_like(eta, dtype=float)

    pos = eta >= 0

    sig[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    sig[~pos] = np.exp(eta[~pos]) / (1.0 + np.exp(eta[~pos]))

    return sig


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y € {0,1})

    # gradient = 1/ N * (X.T @ (sigmoid(X @ w) - y))
    # w = w - gamma * gradient
    # loss = 1/N * (-y * X @ w + log(1 + e^(X @ w)))
    """

    w = initial_w.copy()

    for i in range(max_iters):
        eta = tx @ w
        sig = _sigmoid(eta)
        gradient = (tx.T @ (sig - y)) / len(y)
        w = w - gamma * gradient

    eta = tx @ w
    loss = np.mean(-y * eta + np.logaddexp(0, eta))

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    (y E {0,1}, with regularization term Lambda||w||^2)

    # gradient = 1/ N * (X.T @ (sigmoid(X @ w) - y)) + lambda_ * w
    # w = w - gamma * gradient
    # loss = 1/N * (-y * X @ w + log(1 + e^(X @ w)))
    """

    w = initial_w.copy()

    for i in range(max_iters):
        eta = tx @ w
        sig = _sigmoid(eta)
        gradient = (tx.T @ (sig - y)) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient
    eta = tx @ w
    loss = np.mean(-y * eta + np.logaddexp(0, eta))

    return w, loss

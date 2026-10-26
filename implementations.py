import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
   
    # e = y - Xw
    # gradient = -1/N*(X.T @ e)
    # w(t+1) = w(t) - gamma*gradient
     # mean squared error  = 1/(2N)*sum(y-tx@w)^2 = 1/2*mean(y-tx@w)^2

    w = initial_w.copy()
    n = len(y)
    for i in range(max_iters):
        e = y - tx @ w
        gradient = -1/n * (tx.T @ e)
        w = w - gamma*gradient
    
    loss = 1/2 * np.mean((y - tx @ w)** 2)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):

    # gradient = -xi*(yi - xi @ w)
    # w(t+1) = w(t) - gamma*sgd
    # mean_squared_error = 1/2 * np.mean(e^2)

    w = initial_w.copy()
    n = len(y)
    randg = np.random.default_rng(0)
    for i in range(max_iters):
        j = randg.integers(n)
        xj, yj = tx[j], y[j]
        gradient = -xj * (yj - xj @ w)
        w = w - gamma * gradient
    loss = 1/2 * np.mean((y - tx @ w)**2)
    return w, loss

def least_squraes(y, tx):
    # w = (x.T@X)^-1 (X.T@y)
    # e = y - X @ w
    # loss = 1/2 * np.mean(e^2)

    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = 1/2 * np.mean(e**2)

    return w, loss

def ridge_regression(y, tx, lambda_):
    return

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return
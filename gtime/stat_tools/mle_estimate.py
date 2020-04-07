import pandas as pd
import numpy as np
from numpy.linalg import multi_dot
from .kalman_filter import KalmanFilter
from scipy.optimize import minimize

log_yhat = []



def _likelihood(X, mu, sigma, phi, theta):

    loglikelihood = 0.0
    m = phi.size
    kalman = KalmanFilter(mu, sigma, phi, theta)
    p = np.identity(m)
    a = np.zeros((m, 1))
    for x in X.flatten():
        a_hat, p_hat, y_hat, F, nu = kalman.predict(a, p, x)
        LL_last = -0.5 * (np.log(np.abs(F)) + multi_dot([nu, np.linalg.inv(F), np.transpose(nu)]) + np.log(2 * np.pi))
        a, p = kalman.update(a_hat, p_hat, F, nu)
        loglikelihood += LL_last

    return loglikelihood


def _run_mle(params, X):
    m = (len(params) - 2) / 2
    mu = params[0]
    sigma = params[1]
    phi = params[2:m + 2]
    theta = params[-m:]
    return _likelihood(X, mu, sigma, phi, theta)


class MLEModel:

    def __init__(self, order):

        self.length = max(order[0], order[1] + 1)
        self.order = order

    def fit(self, X):

        p0 = np.random.random(self.order[0])
        q0 = np.random.random(self.order[2])
        p = np.pad(p0, (0, self.length - len(p0)), mode='constant')
        q = np.pad(q0, (0, self.length - len(q0) + 1), mode='constant')
        mu = np.random.random()
        sigma = abs(np.random.random())
        initial_params = np.concatenate([mu, sigma, p, q])
        Xmin = minimize(lambda phi: _run_mle(phi, X), x0=initial_params)
        self.mu = Xmin[0][0]
        self.sigma = Xmin[0][1]
        self.p = Xmin[0][2:self.length+2]
        self.q = Xmin[0][-self.length+2:]

        return self


def gen_y(size, coefs):

    mu, sigma, p, q = coefs
    m = max(len(p), len(q))
    y = np.zeros(size + m)
    epses = np.zeros(len(q))
    y[:m] = 0.0
    for i in range(m, size + m):
        eps = np.random.standard_normal(1) * sigma**2
        y[i] = mu + eps + np.dot(p, y[i-m:i]) + np.dot(q, epses)
        epses[1:] = epses[:-1]
        epses[0] = eps
    return y[m:]


if __name__ == '__main__':

    np.random.seed(0)
    p = np.array([0.5])
    q = np.array([0.0])
    mu = 0.0
    sigma = 0.7
    y = gen_y(1000, (mu, sigma, p, q))



    pd.DataFrame(np.array([np.array(log_yhat).flatten(), y])).T.to_clipboard()

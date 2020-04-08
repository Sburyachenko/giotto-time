import numpy as np
from numpy.linalg import multi_dot
from gtime.stat_tools.kalman_filter import KalmanFilter
from scipy.optimize import minimize


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
    print(F, loglikelihood)
    return float(loglikelihood)


def _run_mle(params, X):
    m = (len(params) - 2) // 2
    if len(params.shape) > 1:
        print(params.shape)
    mu = params[0]
    sigma = params[1]
    phi = params[2:m + 2]
    theta = params[-m:]
    return -_likelihood(X, mu, sigma, phi, theta)


class MLEModel:

    def __init__(self, order):

        self.length = max(order[0], order[1] + 1)
        self.order = order

    def fit(self, X):

        p0 = np.random.random(self.order[0])
        q0 = np.random.random(self.order[1])
        p = np.pad(p0, (0, self.length - len(p0)), mode='constant')
        q = np.pad(q0, (0, self.length - len(q0)), mode='constant')
        mu = np.random.random(1)
        sigma = abs(np.random.random(1))
        initial_params = np.concatenate([mu, sigma, p, q])
        Xmin = minimize(lambda phi: _run_mle(phi, X), x0=initial_params, tol=0.1)
        fitted_params = Xmin['x']
        self.mu = fitted_params[0]
        self.sigma = fitted_params[1]
        self.p = fitted_params[2:self.length+2]
        self.q = fitted_params[-self.length:]

        return self

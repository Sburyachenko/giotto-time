import numpy as np
from numpy.linalg import multi_dot
from gtime.stat_tools.kalman_filter import KalmanFilter
from scipy.optimize import minimize
from scipy.signal import lfilter
import time


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
    return float(loglikelihood)


def _run_mle(params, X, len_p):
    if len(params.shape) > 1:
        print(params.shape)
    mu = params[0]
    sigma = params[1]
    len_q = len(params) - len_p - 2
    max_lag = max(len_p, len_q + 1)
    phi = params[2:len_p + 2]
    theta = params[len_p + 2:]
    phi = np.pad(phi, (0, max_lag - len_p), mode='constant', constant_values=(0, 0))
    theta = np.pad(theta, (0, max_lag - len_q), mode='constant', constant_values=(0, 0))
    return -_likelihood(X, mu, sigma, phi, theta)


def _run_css(params, X, len_p):

    if len(params.shape) > 1:
        print(params.shape)

    mu = params[0]
    nobs = len(X) - len_p
    phi = np.r_[1, params[2:len_p + 2]]
    theta = np.r_[1, params[len_p + 2:]]
    y = X - mu
    eps = lfilter(phi, theta, y)
    ssr = np.dot(eps, eps)
    sigma2 = ssr / nobs
    loglikelihood = -nobs / 2. * (np.log(2 * np.pi) + np.log(sigma2)) - ssr / (2. * sigma2)
    return -loglikelihood




class MLEModel:

    def __init__(self, order, method='mle'):

        self.length = max(order[0], order[1] + 1)
        self.order = order
        self.method = method

    def fit(self, X):

        start = time.time()

        p0 = np.random.random(self.order[0])
        q0 = np.random.random(self.order[1])
        mu = X.mean(keepdims=True)
        sigma = X.std(keepdims=True)
        initial_params = np.concatenate([mu, sigma, p0, q0])

        if self.method == 'mle':
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]), x0=initial_params, tol=0.001, method='L-BFGS-B')
        elif self.method == 'css':
            Xmin = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=initial_params,
                            method='L-BFGS-B')
        else:
            x_css = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=initial_params)['x']
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]), x0=x_css, tol=0.001,
                            method='L-BFGS-B')

        print(f'Time: {time.time() - start:.2f} s')
        print(Xmin['fun'])
        fitted_params = Xmin['x']
        self.mu = fitted_params[0]
        self.sigma = fitted_params[1]
        self.p = fitted_params[2:len(p0) + 2]
        self.q = fitted_params[-len(q0):]

        return self

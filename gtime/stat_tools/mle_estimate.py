import numpy as np
from numpy.linalg import multi_dot
from gtime.stat_tools.kalman_filter import KalmanFilter
from gtime.stat_tools.tools import mat_square
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.signal import lfilter
import time


def loglikelihood_ns(nu, F):
    return -0.5 * (np.log(2 * np.pi * np.abs(F)) + mat_square(np.linalg.inv(F), nu))

def _likelihood(X, mu, sigma, phi, theta, errors=False):

    m = phi.size
    kalman = KalmanFilter(mu, sigma, phi, theta)
    eps = np.zeros(len(X))
    a_hat, p_hat, x_hat, F, nu = kalman.initial_predict(X[0])
    LL_last = loglikelihood_ns(nu, F)
    loglikelihood = LL_last
    eps[0] = nu
    for i, x in enumerate(X[1:]):
        a, p = kalman.update(a_hat, p_hat, F, nu)
        a_hat, p_hat, x_hat, F, nu = kalman.predict(a, p, x)
        eps[i] = nu
        LL_last = loglikelihood_ns(nu, F)
        loglikelihood += LL_last
    if errors:
        return eps
    else:
        return -float(loglikelihood)


def _run_mle(params, X, len_p, errors=False):
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
    return _likelihood(X, mu, sigma, phi, theta, errors)


def _run_css(params, X, len_p, errors=False, transform=False):


    mu = params[0]
    nobs = len(X) - len_p

    phi = np.r_[1, -params[2:len_p + 2]]
    theta = np.r_[1, params[len_p + 2:]]

    y = X - mu
    zi = np.zeros((max(len(phi), len(theta)) - 1))
    for i in range(len_p): #TODO understand how it works
        zi[i] = sum(-phi[:i + 1][::-1] * y[:i + 1])
    eps = lfilter(phi, theta, y, zi=zi)[0][len_p:]
    if errors:
        return eps
    else:
        ssr = np.dot(eps, eps)
        sigma2 = ssr / nobs
        loglikelihood = -nobs / 2. * (np.log(2 * np.pi * sigma2)) - ssr / (2. * sigma2)
        return -loglikelihood


def _polynomial_roots(x, len_p):
    phi = x[2:2 + len_p]
    theta = x[2 + len_p:]
    phi_roots = np.abs(np.roots(np.r_[-phi[::-1], 1.0]))
    theta_roots = np.abs(np.roots(np.r_[theta[::-1], 1.0]))
    return np.r_[2.0, 2.0, phi_roots, theta_roots] # TODO refactor 2.0


class MLEModel:

    def __init__(self, order, method='mle'):

        self.length = max(order[0], order[1] + 1)
        self.order = order
        self.method = method
        p0 = np.random.random(order[0])
        q0 = np.random.random(order[1])
        self.parameters = np.r_[0.0, 0.0, p0, q0]

    def fit(self, X):

        start = time.time()


        mu = X.mean(keepdims=True)
        sigma = X.std(keepdims=True)
        self.parameters[0] = mu
        self.parameters[1] = sigma

        constraints = NonlinearConstraint(lambda x: _polynomial_roots(x, self.order[0]),
                                     lb=np.ones(len(self.parameters)),
                                     ub=np.inf * np.ones(len(self.parameters))
                                     )
        if self.method == 'mle':
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]),
                            x0=self.parameters, method='SLSQP', constraints=constraints)
        elif self.method == 'css':
            Xmin = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0], transform=False),
                            x0=self.parameters, method='SLSQP', constraints=constraints)
        else:
            x0_css = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]),
                              x0=self.parameters, method='SLSQP', constraints=constraints)['x']
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]),
                            x0=x0_css, method='SLSQP', constraints=constraints)

        fitted_params = Xmin['x']
        self.parameters = fitted_params
        self.mu = fitted_params[0]
        self.sigma = fitted_params[1]
        self.phi = fitted_params[2:self.order[0] + 2]
        self.theta = fitted_params[-self.order[1]:] if self.order[1] > 0 else np.array([])
        return self

    def get_errors(self, X):
        if self.method in ['mle', 'css-mle']:
            errors = _run_mle(self.parameters, X, self.order[0], errors=True)
        else:
            errors = _run_css(self.parameters, X, self.order[0], errors=True)
        return errors



import numpy as np
from numpy.linalg import multi_dot
from gtime.stat_tools.kalman_filter import KalmanFilter
from gtime.stat_tools.tools import mat_square
from scipy.optimize import minimize, Bounds
from scipy.signal import lfilter
import time


def _likelihood(X, mu, sigma, phi, theta, errors=False):

    loglikelihood = 0.0
    m = phi.size
    kalman = KalmanFilter(mu, sigma, phi, theta)
    p = np.identity(m)
    a = np.zeros((m, 1))
    eps = np.zeros(len(X))
    for i, x in enumerate(X):
        a_hat, p_hat, x_hat, F, nu = kalman.predict(a, p, x)
        LL_last = -0.5 * (np.log(2 * np.pi * np.abs(F)) + mat_square(np.linalg.inv(F), nu))
        a, p = kalman.update(a_hat, p_hat, F, nu)
        eps[i] = nu
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


def _ar_transparams(params): # TODO do not copy directly!!!!!

    newparams = np.tanh(params/2)
    tmp = np.tanh(params/2)
    for j in range(1,len(params)):
        a = newparams[j]
        for kiter in range(j):
            tmp[kiter] -= a * newparams[j-kiter-1]
        newparams[:j] = tmp[:j]
    return newparams

def _ma_transparams(params):

    newparams = ((1-np.exp(-params))/(1+np.exp(-params))).copy()
    tmp = ((1-np.exp(-params))/(1+np.exp(-params))).copy()

    # levinson-durbin to get macf
    for j in range(1,len(params)):
        b = newparams[j]
        for kiter in range(j):
            tmp[kiter] += b * newparams[j-kiter-1]
        newparams[:j] = tmp[:j]
    return newparams

def _run_css(params, X, len_p, errors=False, transform=True):


    mu = params[0]
    nobs = len(X) - len_p
    phi = np.r_[1, _ar_transparams(params[2:len_p + 2])]
    theta = np.r_[1, _ma_transparams(params[len_p + 2:])]

    y = X - mu
    eps = lfilter(phi, theta, y)
    if errors:
        return eps
    else:
        ssr = np.dot(eps, eps)
        sigma2 = ssr / nobs
        loglikelihood = -nobs / 2. * (np.log(2 * np.pi * sigma2)) - ssr / (2. * sigma2)
        return -loglikelihood


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

        if self.method == 'mle':
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]), x0=self.parameters, tol=0.001, method='L-BFGS-B')
        elif self.method == 'css':
            upper_bound = np.r_[np.inf, np.inf, np.ones(len(self.parameters) - 2)]  # TODO a very simple stationarity restriction, could be better
            lower_bound = np.r_[-np.inf, 0.0, -np.ones(len(self.parameters) - 2)]
            bounds = Bounds(lower_bound, upper_bound)
            Xmin = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=self.parameters,
                            method='L-BFGS-B', bounds=bounds)
        else:
            upper_bound = np.r_[np.inf, np.inf, np.ones(len(self.parameters) - 2)]
            lower_bound = np.r_[-np.inf, 0.0, -np.ones(len(self.parameters) - 2)]
            bounds = Bounds(lower_bound, upper_bound)
            x0_css = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=self.parameters, bounds=bounds)['x']
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]), x0=x0_css, tol=0.001,
                            method='L-BFGS-B')

        # print(f'Time: {time.time() - start:.2f} s')
        # print(Xmin['fun'])
        fitted_params = Xmin['x']
        self.parameters = fitted_params
        self.mu = fitted_params[0]
        self.sigma = fitted_params[1]
        self.phi = _ar_transparams(fitted_params[2:self.order[0] + 2])
        self.theta = _ma_transparams(fitted_params[-self.order[1]:] if self.order[1] > 0 else np.array([]))

        return self

    def get_errors(self, X):
        if self.method in ['mle', 'css-mle']:
            errors = _run_mle(self.parameters, X, self.order[0], errors=True)
        else:
            errors = _run_css(self.parameters, X, self.order[0], errors=True)
        return errors



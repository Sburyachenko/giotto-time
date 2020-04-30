import numpy as np
from gtime.stat_tools.kalman_filter import KalmanFilter
from gtime.stat_tools.tools import loglikelihood_ns, arma_polynomial_roots, mat_square
from scipy.optimize import minimize, NonlinearConstraint, basinhopping
from scipy.signal import lfilter
from numpy.linalg import multi_dot
from .kalman import roll, llf_c


def kalman_init(sigma, phi, theta):

    m = len(phi)
    R = np.array([np.r_[1.0, theta[:-1]]]).T
    K = np.concatenate((phi[:-1].reshape((-1, 1)), np.identity(m - 1)), axis=1)
    K = np.concatenate((K, np.concatenate((phi[-1:], np.zeros(m - 1))).reshape(1, -1)))
    Q = np.array([[sigma ** 2]])

    vec_RR = np.matmul(R, R.T).ravel()
    vec_K = np.linalg.inv(np.identity(m ** 2) - np.kron(K, K))
    p = sigma ** 2 * np.matmul(vec_K, vec_RR).reshape((m, m))
    rqr = mat_square(Q, R)
    rqr = np.array(rqr, order='F')
    a = np.zeros(m)
    return a, p, K, rqr

# Cython engine wrapper
def alt_likelihood(X, mu, sigma, phi, theta, errors=False):

    a, p, K, rqr = kalman_init(sigma, phi, theta)
    eps = np.zeros(len(X))
    x_hat = a[0] + mu
    nu = float(X[0] - x_hat)
    F = p[0, 0]

    p = np.array(p, order='F')
    K = np.array(K, order='F')

    loglikelihood = llf_c(nu, F)
    eps[0] = nu

    r = p.shape[0]
    n = len(X)
    ll, eps = roll(X, a, p, eps, mu, F, nu, rqr, K, r, n)
    if errors:
        return np.array(eps)
    else:
        return -float(loglikelihood + ll)
    # return loglikelihood + ll, eps


def _run_mle(params, X, len_p, errors=False, transform=True):
    mu = params[0]
    sigma = params[1]
    len_q = len(params) - len_p - 2
    max_lag = max(len_p, len_q + 1)
    if transform:
        phi = _ar_transparams(params[2:len_p + 2])
        theta = _ma_transparams(params[len_p + 2:])
    else:
        phi = params[2:len_p + 2]
        theta = params[len_p + 2:]

    phi = np.pad(phi, (0, max_lag - len_p), mode='constant', constant_values=(0, 0))
    theta = np.pad(theta, (0, max_lag - len_q), mode='constant', constant_values=(0, 0))
    return alt_likelihood(X, mu, sigma, phi, theta, errors)


def _run_css(params, X, len_p, errors=False, transform=True):

    mu = params[0]
    nobs = len(X) - len_p
    if transform:
        phi = np.r_[1, -_ar_transparams(params[2:len_p + 2])]
        theta = np.r_[1, _ma_transparams(params[len_p + 2:])]
    else:
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


def _ar_transparams(params):  # TODO do not copy directly!!!!!

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


class MLEModel:

    def __init__(self, order, method='mle'):

        self.length = max(order[0], order[1] + 1)
        self.order = order
        self.method = method
        p0 = np.random.random(order[0]) #TODO can be better?
        q0 = np.random.random(order[1])
        self.parameters = np.r_[0.0, 0.0, p0, q0]

    def fit(self, X):

        mu = X.mean(keepdims=True)
        sigma = X.std(keepdims=True) / np.sqrt(len(X))
        self.parameters[0] = mu
        self.parameters[1] = sigma

        # Constrained optimization
        # constraints = NonlinearConstraint(lambda x: arma_polynomial_roots(x, self.order[0]),
        #                              lb=np.ones(len(self.parameters)),
        #                              ub=np.inf * np.ones(len(self.parameters))
        #                              )
        # minimizer_kwargs = {"method": "SLSQP", "constraints": constraints}
        # if self.method == 'mle':
        #     # Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]),
        #     #                 x0=self.parameters, method='SLSQP', constraints=constraints)
        #     Xmin = basinhopping(lambda phi: _run_mle(phi, X, len_p=self.order[0]),
        #                     x0=self.parameters, minimizer_kwargs=minimizer_kwargs)
        # elif self.method == 'css':
        #     Xmin = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]),
        #                     x0=self.parameters, method='SLSQP', constraints=constraints)
        # else:
        #     x0_css = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]),
        #                       x0=self.parameters, method='SLSQP', constraints=constraints)['x']
        #     Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]),
        #                     x0=x0_css, method='SLSQP', constraints=constraints)

        # Transformed LBFGS
        if self.method == 'mle':
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]), x0=self.parameters, method='L-BFGS-B')
        elif self.method == 'css':
            Xmin = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=self.parameters, method='L-BFGS-B')
        else:
            x0_css = minimize(lambda phi: _run_css(phi, X, len_p=self.order[0]), x0=self.parameters, method='L-BFGS-B')['x']
            Xmin = minimize(lambda phi: _run_mle(phi, X, len_p=self.order[0]), x0=x0_css, method='L-BFGS-B')

        fitted_params = Xmin['x']
        self.ml = Xmin['fun']
        self.mu = fitted_params[0]
        self.sigma = fitted_params[1]
        self.phi = _ar_transparams(fitted_params[2:self.order[0] + 2])
        self.theta = _ma_transparams(fitted_params[-self.order[1]:] if self.order[1] > 0 else np.array([]))
        self.parameters = np.r_[self.mu, self.sigma, self.phi, self.theta]
        # self.phi = fitted_params[2:self.order[0] + 2]
        # self.theta = fitted_params[-self.order[1]:] if self.order[1] > 0 else np.array([])
        return self

    def get_errors(self, X):
        if self.method in ['mle', 'css-mle']:
            errors = _run_mle(self.parameters, X, self.order[0], errors=True, transform=False)
        else:
            errors = _run_css(self.parameters, X, self.order[0], errors=True, transform=False)
        return errors



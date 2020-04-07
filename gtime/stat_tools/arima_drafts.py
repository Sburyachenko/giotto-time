import pandas as pd
import numpy as np
from numpy.linalg import multi_dot
from scipy.optimize import minimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter


log_yhat = []


class Kalman:

    def __init__(self, p, q, sigma):
        len_p = len(p)
        len_q = len(q)
        m = max(len_p, len_q+1)
        self.p = np.pad(p, (0, m - len_p), mode='constant')
        self.q = np.pad(q, (0, m - len_q), mode='constant')
        R = np.concatenate((np.ones(1), self.q[:-1]))
        self.R = np.diag(R)
        K = np.concatenate((self.p[:-1].reshape((-1, 1)), np.identity(m - 1)), axis=1)
        self.K = np.concatenate((K, np.concatenate((self.p[-1:], np.zeros(m - 1))).reshape(1, -1)))
        self.Q = sigma ** 2 * np.identity(m)
        self.Z = np.zeros((1, m))
        self.Z[0, 0] = 1.0

    def update(self, a_hat, p_hat, F, nu):
        gain = multi_dot([p_hat, np.transpose(self.Z), np.linalg.inv(F)])
        a = a_hat + multi_dot([gain, nu])
        p = p_hat - multi_dot([gain, self.Z, np.transpose(p_hat)])
        return a, p

    def predict(self, a, p, y):
        a_hat = np.matmul(self.K, a)
        p_hat = mat_square(p, self.K) + mat_square(self.Q, self.R)
        # p_hat = multi_dot([self.K, p, np.transpose(self.K)]) + self.Q
        y_hat = np.matmul(self.Z, a_hat)
        F = mat_square(p_hat, self.Z)
        nu = y - y_hat
        return a_hat, p_hat, y_hat, F, nu


def mat_square(X, M):
    return multi_dot([M, X, np.transpose(M)])

def mle(X, sigma, phi, theta):
    global log_yhat
    LL = 0.0
    m = max(len(phi), len(theta) + 1)
    kalman = Kalman(phi, theta, sigma)
    p = np.identity(m)
    a = np.zeros((m, 1))
    log_yhat = []
    for x in X.flatten():
        a_hat, p_hat, y_hat, F, nu = kalman.predict(a, p, x)
        log_yhat.append(y_hat)
        LL_last = -0.5 * (np.log(np.abs(F)) + multi_dot([nu, np.linalg.inv(F), np.transpose(nu)]) + np.log(2 * np.pi))
        a, p = kalman.update(a_hat, p_hat, F, nu)
        LL += LL_last

    return LL


def calc_coef(param, X, order):

    param = param.flatten()
    sigma = param[0]
    p = np.array([]) if order[0] == 0 else param[1:order[0] + 1]
    q = np.array([]) if order[2] == 0 else param[-order[2]:]
    ll = mle(X, sigma, p, q)
    print(param, ll)
    return -ll



"""
Steps:
 - Initiate
 - MLE function
 - Optimize MLE (scipy)
 
"""

# https://faculty.washington.edu/eeholmes/Files/Intro_to_kalman.pdf
# https://www.utsc.utoronto.ca/~sdamouras/courses/STAD57H3_W13/Lecture%2014/Lecture%2014%20Annotated.pdf
# https://towardsdatascience.com/the-kalman-filter-and-maximum-likelihood-9861666f6742
# https://uh.edu/~bsorense/kalman.pdf
# https://arxiv.org/pdf/1204.0375.pdf
# https://www.stat.purdue.edu/~chong/stat520/ps/statespace.pdf



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
    # y = np.zeros(100)

    # m = SARIMAX(y, order=(2, 0, 0))
    # res = m.fit()
    # print(res.summary())
    x = minimize(lambda x: calc_coef(x, y, (1, 0, 1)), x0=np.array([0.5, 0.2, 0.2]))


    pd.DataFrame(np.array([np.array(log_yhat).flatten(), y])).T.to_clipboard()
    ll = mle(y, 0.1, p, q)
    print(ll)
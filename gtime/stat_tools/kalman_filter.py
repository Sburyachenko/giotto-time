import numpy as np
from gtime.stat_tools.tools import mat_square
from numpy.linalg import multi_dot


# https://faculty.washington.edu/eeholmes/Files/Intro_to_kalman.pdf
# https://www.utsc.utoronto.ca/~sdamouras/courses/STAD57H3_W13/Lecture%2014/Lecture%2014%20Annotated.pdf
# https://towardsdatascience.com/the-kalman-filter-and-maximum-likelihood-9861666f6742
# https://uh.edu/~bsorense/kalman.pdf
# https://arxiv.org/pdf/1204.0375.pdf
# https://www.stat.purdue.edu/~chong/stat520/ps/statespace.pdf
# http://www.stat.ucla.edu/~frederic/221/W17/221ch3.pdf

class KalmanFilter:

    def __init__(self, mu, sigma, phi, theta):
        self.phi = phi
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        m = len(phi)
        R = np.concatenate((np.ones(1), self.theta[:-1]))
        self.R = np.diag(R)
        K = np.concatenate((self.phi[:-1].reshape((-1, 1)), np.identity(m - 1)), axis=1)
        self.K = np.concatenate((K, np.concatenate((self.phi[-1:], np.zeros(m - 1))).reshape(1, -1)))
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
        y_hat = np.matmul(self.Z, a_hat) + self.mu
        F = mat_square(p_hat, self.Z)
        nu = y - y_hat
        return a_hat, p_hat, y_hat, F, nu

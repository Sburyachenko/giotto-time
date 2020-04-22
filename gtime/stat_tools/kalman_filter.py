import numpy as np
from gtime.stat_tools.tools import mat_square
from numpy.linalg import multi_dot


class KalmanFilter:

    def __init__(self, mu, sigma, phi, theta):
        self.phi = phi
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        m = len(phi)
        self.m = m
        self.R = np.array([np.r_[1.0, self.theta[:-1]]]).T
        K = np.concatenate((self.phi[:-1].reshape((-1, 1)), np.identity(m - 1)), axis=1)
        self.K = np.concatenate((K, np.concatenate((self.phi[-1:], np.zeros(m - 1))).reshape(1, -1)))
        self.Q = np.array([[sigma ** 2]])
        self.Z = np.zeros((1, m))
        self.Z[0, 0] = 1.0

    def update(self, a_hat, p_hat, F, nu):
        gain = multi_dot([p_hat, np.transpose(self.Z), np.linalg.inv(F)])
        a = a_hat + multi_dot([gain, nu])
        p = p_hat - multi_dot([gain, self.Z, np.transpose(p_hat)])
        return a, p

    def predict(self, a, p, x):
        a_hat = np.matmul(self.K, a)
        p_hat = mat_square(p, self.K) + mat_square(self.Q, self.R)
        x_hat = np.matmul(self.Z, a_hat) + self.mu
        F = mat_square(p_hat, self.Z)
        nu = x - x_hat
        return a_hat, p_hat, x_hat, F, nu

    def initial_predict(self, x):
        vec_RR = np.matmul(self.R, self.R.T).ravel()
        vec_K = np.linalg.inv(np.identity(self.m ** 2) - np.kron(self.K, self.K))
        p = self.sigma**2 * np.matmul(vec_K, vec_RR).reshape((self.m, self.m))
        a = np.zeros((self.m, 1))
        x_hat = np.matmul(self.Z, a) + self.mu
        nu = x - x_hat
        F = mat_square(p, self.Z)
        return a, p, x_hat, F, nu

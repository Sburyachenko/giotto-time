import numpy as np
import pandas as pd
from gtime.forecasting.simple_models import SimpleForecaster
from gtime.stat_tools.mle_estimate import MLEModel # TODO better import


class ARMA(SimpleForecaster):

    def __init__(self, order, method='MLE'):
        self.order = order
        self.method = method

    def fit(self, X, y):

        if self.method == 'MLE':
            self.model = MLEModel(self.order)
        self.model.fit(X)
        super().fit(X, y)
        return self

    def _predict(self, X):
        pass



if __name__ == '__main__':

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

    p = [0.3, -0.3]
    q = [-0.1]
    mu = 5.0
    sigma = 0.7
    y = gen_y(100, (mu, sigma, p, q))

    model = ARMA((2, 1))
    model.fit(y, None)
    print('ok')
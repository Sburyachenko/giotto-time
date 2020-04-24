import numpy as np
import pandas as pd
from gtime.forecasting.simple_models import SimpleForecaster
from gtime.stat_tools.mle_estimate import MLEModel # TODO better import


def _arma_forecast(n, x0, eps0, mu, phi, theta):
    len_ar = len(phi)
    len_ma = len(theta)
    x = np.r_[x0, np.zeros(n)]
    eps = np.r_[eps0, np.zeros(n)]
    mu = mu * (1 - phi.sum())  # TODO Why???
    for i in range(n):
        x[i + len_ar] = mu + np.dot(phi, x[i:i + len_ar]) + np.dot(theta, eps[i:i + len_ma])
    return x[len_ar:]


class ARIMAForecaster(SimpleForecaster):

    def __init__(self, order, method='css-mle'):
        self.order = order
        self.max_degree = max(order[0], order[2])
        self.n_ar = order[0]
        self.n_ma = order[2]
        self.method = method
        self.model = None

    def _deintegrate(self, X):
        n = len(X)
        i_order = self.order[1]
        self.diff_vals = np.zeros((n, i_order))
        for i in range(i_order):
            self.diff_vals[:, i] = np.diff(X, n=i)[-n:]
        X = np.diff(X, n=i_order)
        return X[i_order:]

    def _integrate(self, X):
        for i in range(self.order[1]):
            X = np.concatenate([self.diff_vals[self.n_ar + self.order[1]:, [-i-1]], X], axis=1).cumsum(axis=1)
        return X

    def fit(self, X, y):
        self.last_train_values_ = X.iloc[-self.n_ar-self.order[1]:]
        np_x = X.to_numpy().flatten()
        np_x = self._deintegrate(np_x)
        model = MLEModel((self.n_ar, self.n_ma), self.method)
        model.fit(np_x)
        self.errors_ = model.get_errors(np_x)
        self.mu_ = model.mu
        self.phi_ = model.phi
        self.theta_ = model.theta
        self.model = model
        super().fit(X, y)
        return self

    def _predict(self, X):

        n = len(X)
        train_test_diff = X.index.min().start_time - self.last_train_values_.index.max().end_time
        if train_test_diff.value == 1:
            X = pd.concat([self.last_train_values_, X])
            errors = np.r_[self.errors_[-self.n_ma:], np.zeros(n)]
        else:
            last_index = pd.period_range(periods=self.n_ar + self.order[1] + 1, end=X.index[0])[:-1]
            last_values = pd.DataFrame(X.iloc[0], index=last_index)
            X = pd.concat([last_values, X])
            errors = np.zeros(n+self.n_ma)

        np_x = X.values.flatten()
        np_x = self._deintegrate(np_x)

        res = [_arma_forecast(n=self.horizon_,
                              x0=np_x[i:i+self.n_ar],
                              eps0=errors[i:i+self.n_ma],
                              mu=self.model.mu,
                              phi=self.model.phi,
                              theta=self.model.theta
                              )
               for i in range(n)]
        y_pred = self._integrate(np.array(res))

        return y_pred[:, self.order[1]:]



# if __name__ == '__main__':
# # #
# # # # TODO testing, to remove later
# #
# #     import numpy as np
# #
# #     np.set_printoptions(precision=2)
# #     import pandas as pd
#     from gtime.preprocessing import TimeSeriesPreparation
#     from gtime.time_series_models import ARIMA, AR
# #     from sklearn.compose import make_column_selector
# #     from gtime.feature_extraction import Shift
# #     from sklearn.metrics import mean_squared_error
# #     from scipy.stats import normaltest
# #     import matplotlib.pyplot as plt
# #
#     df_sp = pd.read_csv('https://storage.googleapis.com/l2f-open-models/giotto-time/examples/data/^GSPC.csv', parse_dates=['Date'])
#     df_close = df_sp.set_index('Date')['Close']
#     time_series_preparation = TimeSeriesPreparation()
#     df_real = time_series_preparation.transform(df_close)
#     model = ARIMA(horizon=100, order=(2, 1, 3), method='css')
#     model.fit(df_real)
#     pred = model.predict()
#     print('A')

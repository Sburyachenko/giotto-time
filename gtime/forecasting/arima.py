import numpy as np
import pandas as pd
from gtime.forecasting.simple_models import SimpleForecaster
from gtime.stat_tools.mle_estimate import MLEModel # TODO better import


def _forecast_arma(n, phi, theta, x0, eps0):
    len_init = len(x0)
    len_ar = len(phi)
    len_ma = len(theta)
    x = np.r_[x0, np.zeros(n)]
    eps = np.r_[eps0, np.zeros(n)]
    for i in range(len_init, n + len_init):
        x[i] = np.dot(phi, x[i - len_ar:i]) + np.dot(theta, eps[i - len_ma:i])
    return x[len_init:]

def _fit_predict_one(X, horizon, order, method):
    X = X.copy()
    model = MLEModel((order[0], order[2]), method)
    model.fit(X)
    print(model.phi, model.theta)
    errors = model.get_errors(X)
    x0 = X[-order[0]:] if order[0] > 0 else np.array([])
    mu = x0[-1]
    X -= mu
    eps0 = errors[-order[2]:] if order[2] > 0 else np.array([])
    forecast = _forecast_arma(horizon, model.phi, model.theta, x0, eps0) + mu
    return forecast

class ARIMA(SimpleForecaster):

    def __init__(self, order, method='css-mle'):
        self.order = order
        self.method = method
        self.model = None

    def fit(self, X, y):
        self.train = X
        self.len_train = len(X)
        super().fit(X, y)
        return self

    def _predict(self, X):
        n = len(X)
        i_order = self.order[1]
        x_np = pd.concat([self.train, X]).values.flatten()
        diff_vals = np.zeros((n, i_order))
        for i in range(i_order):
            diff_vals[:, i] = np.diff(x_np, n=i)[-n:]

        x_np = np.diff(x_np)
        y_pred = np.array(list(map(lambda x: _fit_predict_one(x, self.horizon_, self.order, self.method),
                              [x_np[:self.len_train+i+1] for i in range(n)])))
        y_pred = np.concatenate([diff_vals, y_pred], axis=1)
        y_pred = y_pred.cumsum(axis=1)[:, 1:]
        return y_pred


if __name__ == '__main__':
#
# # TODO testing, to remove later
    import numpy as np
    import pandas as pd
    from gtime.preprocessing import TimeSeriesPreparation
    from gtime.time_series_models import TimeSeriesForecastingModel
    from sklearn.compose import make_column_selector
    from gtime.feature_extraction import Shift
    df_sp = pd.read_csv('https://storage.googleapis.com/l2f-open-models/giotto-time/examples/data/^GSPC.csv')
    df_close = df_sp.set_index('Date')['Close']
    time_series_preparation = TimeSeriesPreparation()
    df = time_series_preparation.transform(df_close)
    df_train = df.iloc[:-100]
    df_test = df.iloc[-100:]
    features = [
        ("s1", Shift(0), make_column_selector()),
    ]
    model = TimeSeriesForecastingModel(features=features, horizon=100, model=ARIMA((2, 1, 2), method='css'))

    model.fit(df_train, None)
    d = model.predict(df_test)
    print('A')
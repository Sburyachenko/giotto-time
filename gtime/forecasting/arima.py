import numpy as np
import pandas as pd
from gtime.forecasting.simple_models import SimpleForecaster
from gtime.stat_tools.mle_estimate import MLEModel # TODO better import


def _forecast_arma(n, mu, phi, theta, x0, eps0):
    phi = phi[::-1]
    theta = theta[::-1]
    len_ar = len(phi)
    len_ma = len(theta)
    x = np.r_[x0, np.zeros(n)]
    eps = np.r_[eps0, np.zeros(n)]
    mu = mu * (1-phi.sum())  # TODO Why???
    for i in range(n):
        x[i + len_ar] = mu + np.dot(phi, x[i:i + len_ar]) + np.dot(theta, eps[i:i + len_ma])
    return x[len_ar:]


def _fit_predict_one(X, horizon, order, method):
    X = X.copy()
    model = MLEModel((order[0], order[2]), method)
    model.fit(X)
    # print(model.mu, model.phi, model.theta)
    errors = model.get_errors(X)
    mu = model.mu
    # X -= mu
    x0 = X[-order[0]:] if order[0] > 0 else np.array([])
    eps0 = errors[-order[2]:] if order[2] > 0 else np.array([])
    # print(x0, eps0)
    forecast = _forecast_arma(horizon, mu, model.phi, model.theta, x0, eps0)
    # print('Forecast: ', forecast.sum())
    # forecast = _forecast_arma(horizon, model.phi, model.theta, x0, eps0)
    param_dict = {'mu': mu,
                  'phi': model.phi,
                  'theta': model.theta,
                  'errors': errors
                  }
    return forecast, param_dict

class ARIMAForecaster(SimpleForecaster):

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

        x_np = np.diff(x_np, n=i_order)
        res = list(map(lambda x: _fit_predict_one(x, self.horizon_, self.order, self.method),
                              [x_np[:self.len_train+i+1] for i in range(n)]))
        y_pred = np.array([x[0] for x in res])
        self.params = pd.DataFrame([x[1] for x in res], index=X.index)
        # print('Dvals: ', diff_vals)
        for i in range(i_order):
            y_pred = np.concatenate([diff_vals[:, [-i-1]], y_pred], axis=1).cumsum(axis=1)
        # y_pred = y_pred.cumsum(axis=1)
        return y_pred[:, i_order:]


if __name__ == '__main__':
#
# # TODO testing, to remove later

    import numpy as np

    np.set_printoptions(precision=2)
    import pandas as pd
    from statsmodels.tsa.arima_model import ARIMA as ARIMA_sm
    from gtime.preprocessing import TimeSeriesPreparation
    from gtime.time_series_models import ARIMA, AR
    from sklearn.compose import make_column_selector
    from gtime.feature_extraction import Shift
    from sklearn.metrics import mean_squared_error
    from scipy.stats import normaltest
    import matplotlib.pyplot as plt

    df_sp = pd.read_csv('https://storage.googleapis.com/l2f-open-models/giotto-time/examples/data/^GSPC.csv', parse_dates=['Date'])
    df_close = df_sp.set_index('Date')['Close']
    time_series_preparation = TimeSeriesPreparation()
    df_real = time_series_preparation.transform(df_close)

    def run_giotto_arima(df, test_size, order, method='css-mle'):
        model = ARIMA(horizon=test_size, order=order, method=method)
        df_train = df
        df_test = df.iloc[-test_size:]
        model.fit(df_train)
        pred_g = model.predict(df_test.iloc[[0]])
        y_pred = pd.DataFrame(pred_g.values[0], index=df_test.index, columns=['time_series'])
        phi = model.model.params['phi'].values[0]
        theta = model.model.params['theta'].values[0]
        mu = model.model.params['mu'].values[0]
        train_errors = model.model.params['errors'].values[0]
        print(f'Fitted parameters: mu={mu:.2f}, p={phi}, q={theta}')
        print(f'AR roots abs:{np.abs(np.roots(np.r_[-phi[::-1], 1.0]))}')
        print(f'MA roots abs:{np.abs(np.roots(np.r_[theta[::-1], 1.0]))}')
        print(f'Train error mean: {train_errors.mean():.2f}, std: {train_errors.std():.2f}')
        print(f'RMSE: {mean_squared_error(y_pred, df_test.values):.2f}')
        print(normaltest(train_errors))
        return mu, phi, theta


    def run_sm(df, test_size, order, method='css-mle'):
        df_train = df.iloc[:-test_size]
        df_test = df.iloc[-test_size:]
        m2 = ARIMA_sm(df_train, order)
        f = m2.fit(method=method)
        y2, _, _ = f.forecast(test_size)
        y_pred = pd.DataFrame(y2, index=df_test.index, columns=['Forecast'])

        print(f'Fitted parameters: mu={f.params.const:.2f}, p={f.arparams}, q={f.maparams}')
        print(f'AR roots abs:{np.abs(f.arroots)}')
        print(f'MA roots abs:{np.abs(f.maroots)}')
        train_errors = m2.geterrors(f.params)
        print(f'Train error mean: {train_errors.mean():.2f}, std: {train_errors.std():.2f}')
        print(f'RMSE: {mean_squared_error(y2, df_test.values):.2f}')
        print(normaltest(train_errors))
        return f, m2

    # run_giotto_arima(df_real, 101, (3, 1, 2), 'css')
    from gtime.stat_tools.mle_estimate import _run_css, _run_mle
    f, m = run_sm(df_real, 100, (3, 0, 2), 'mle')


    # mu_grid = np.linspace(1500, 3500, 100)
    # ar_grid = np.linspace(0.5, 1, 50)
    l_g = _run_mle(np.r_[f.params[0], np.sqrt(f.sigma2), f.params[1:]], df_real[:-100].values.flatten(), 3)
    es = m.geterrors(f.params)
    l_s = -m.loglike_kalman(f.params)

    # g = [[_run_css(np.array([x, np.sqrt(f.sigma2), y]), df_real.iloc[:-100].values.flatten(), 2) for x in mu_grid] for y in ar_grid]
    # h = [[-m.loglike_css(np.array([x, y])) for x in mu_grid] for y in ar_grid]
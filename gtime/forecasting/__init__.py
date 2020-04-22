"""
The :mod:`gtime.forecasting` module contains a collection of machine learning models,
for dealing with time series data.
"""

from .gar import GAR, GARFF, MultiFeatureMultiOutputRegressor, MultiFeatureGAR
from .trend_models import TrendForecaster
from .online import HedgeForecaster
from .simple_models import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    DriftForecaster,
    AverageForecaster,
)
from .arima import ARIMAForecaster

__all__ = [
    "GAR",
    "GARFF",
    "MultiFeatureMultiOutputRegressor",
    "MultiFeatureGAR" "TrendForecaster",
    "HedgeForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "DriftForecaster",
    "AverageForecaster",
    "ARIMAForecaster",
]



Welcome to giotto-time's API reference!
========================================

Causality Tests
_________________

.. automodule:: gtime.causality
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   causality.ShiftedLinearCoefficient
   causality.ShiftedPearsonCorrelation
   causality.GrangerCausality


Compose
_________

.. automodule:: gtime.compose
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   compose.FeatureCreation


Feature Extraction
____________________

.. automodule:: gtime.feature_extraction
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

    feature_extraction.Shift
    feature_extraction.MovingAverage
    feature_extraction.MovingCustomFunction
    feature_extraction.Polynomial
    feature_extraction.Exogenous
    feature_extraction.CustomFeature
    feature_extraction.Detrender
    feature_extraction.SortedDensity
    feature_extraction.CrestFactorDetrending


Feature Generation
____________________

.. automodule:: gtime.feature_generation
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst


   feature_generation.PeriodicSeasonal
   feature_generation.Constant
   feature_generation.Calendar


Forecasting
_________________

.. automodule:: gtime.forecasting
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

    forecasting.GAR
    forecasting.GARFF
    forecasting.MultiFeatureGAR
    forecasting.TrendForecaster
    forecasting.HedgeForecaster
    forecasting.NaiveForecaster
    forecasting.SeasonalNaiveForecaster
    forecasting.DriftForecaster
    forecasting.AverageForecaster
    forecasting.MultiFeatureMultiOutputRegressor


Hierarchical
_________________

.. automodule:: gtime.hierarchical
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

    hierarchical.HierarchicalBase
    hierarchical.HierarchicalNaive
    hierarchical.HierarchicalBottomUp
    hierarchical.HierarchicalTopDown
    hierarchical.HierarchicalMiddleOut


Metrics
________

.. automodule:: gtime.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.non_zero_smape
   metrics.smape
   metrics.max_error
   metrics.mse
   metrics.log_mse
   metrics.r_square
   metrics.mae
   metrics.mape
   metrics.rmse
   metrics.rmsle
   metrics.gmae


Model Selection
_________________

.. automodule:: gtime.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.FeatureSplitter

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.horizon_shift
   model_selection.time_series_split
   model_selection.blocking_time_series_split


Plotting
_________________

.. automodule:: gtime.plotting
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plotting.seasonal_plot
   plotting.seasonal_subplots
   plotting.acf_plot
   plotting.lag_plot


Preprocessing
_________________

.. automodule:: gtime.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.TimeSeriesPreparation


Regressors
_________________

.. automodule:: gtime.regressors
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   regressors.LinearRegressor
   regressors.MultiFeatureMultiOutputRegressor
   regressors.ExplainableRegressor


Time series models
___________________

.. automodule:: gtime.time_series_models
   :no-members:
   :no-inherited-members:

.. currentmodule:: gtime

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series_models.TimeSeriesForecastingModel
   time_series_models.AR
   time_series_models.Naive
   time_series_models.SeasonalNaive
   time_series_models.Average
   time_series_models.Drift
   time_series_models.CVPipeline

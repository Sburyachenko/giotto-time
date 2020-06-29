===============
Theory Glossary
===============

.. contents::
   :depth: 3
..

Hierarchical models
===================

Basic concept
-------------

In many pracrical applications a there exists some hierarchical
relationship between time series. Most commonly, a time series can be
represented as a sum of its components which represent a split location,
type or categories. It can bet illustrated with a tree:

|image1|

In this case time series **T** is a sum of **A** and **B** which in turn both
can be split it two subseries. There are three main approaches to model such relationships.

Bottom-up approach
------------------
In the bottom-up aproach the leaves of the tree (**a1**, **a2**, **b1** and **b2**) are modelled
and forecasted separatelly with some time series model, while the remaining nodes are constructed
with followind matrix equation:

|image2|

Such approach is useful when the leaf subseries have sufficient data and fit the available models well.

Top-down approach
-----------------

The opposite top-down approach can be applied when the subseries have limited or unreliable data available.
In this case the time series model is applied to the root time series while the rest are derived from it
with a simple formula:

.. math::

   x_i = p_i * X

The proportion coefficients can be calculated during model fitting with one of the following methods:

* **Average historical proportion** where the ratios between two series are calculated for each point and averaged out:

.. math::
   p_i = \frac{1}{T} \sum_{t=1}^{T}{\frac{x_i(t)}{X(t)}}

* **Proportion of historical averages** which is the ratio between average values of two series:

.. math::
   p_i = \frac{\sum_{t=1}^{T}{x_i(t)}} {\sum_{t=1}^{T}{X(t)}}

* **Forecast proportion** where separate forecasts are done for each leaf of the node and then their share
  in the sum of all leaves of a particular node is calculated by one of the methods described above. The method
  is described with more details and examples in (Hyndman, Athanasopoulos, and OTexts.com 2014).


Middle-out approach
-------------------

A middle-out approach is a combination of the previous two. In this case stand-alone forecasts are generated
for one particular level of the hierarchy tree, after which all the levels above it are generated with bottom-up
aproach while top-down approach is applied to all the levels below.

Explainability
==============
The goal of model explainability methods is to summarize key principles of the model in a human-readable form,
and, in particular, to show how different features affect model predictions. Two most common explainability
frameworks are `SHAP <https://github.com/slundberg/shap/>`_ and `LIME <https://github.com/marcotcr/lime>`_, both
are supported by giotto-time.

LIME
-------------------
LIME technique was presented in (Ribeiro, Singh & Guestrin, 2016). It proposes an algorithm that allows to
locally approximate any classifier or regressor with a simpler interpretable model (Ridge regressor by default).
LIME algorithm guaranties additivity of the explanation values, i. e. the sum of individual explanation values
would add up to the total difference between predicted value and model baseline.
However, due to its local nature, LIME explanation might not be stable and consistent.

SHAP values
-------------------
SHAP approach was proposed in (Lundberg & Lee, 2017) aas an attempt to combine local explanations of LIME
with optimal credit allocation approach using the concept of Shapley values, which represent gain allocation
scheme in a cooperative game.
SHAP value explanations are consistent but they are also more computationally expensive than LIME.



Bibliography
============


.. raw:: html

   <div id="ref-nla.cat-vn6450419">

Hyndman, Rob J., Athanasopoulos George, and OTexts.com. 2014.
*Forecasting : Principles and Practice / Rob J Hyndman and George
Athanasopoulos*. Book. Print edition. OTexts.com [Heathmont?, Victoria].

Ribeiro, Marco & Singh, Sameer & Guestrin, Carlos. (2016). “Why Should I Trust You?”:
Explaining the Predictions of Any Classifier. 97-101. 10.18653/v1/N16-3020.

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in
Neural Information Processing Systems, pages 4768–4777, 2017.

.. |image1| image:: tree.png
   :width: 400

.. |image2| image:: matrix_form.png
   :width: 400

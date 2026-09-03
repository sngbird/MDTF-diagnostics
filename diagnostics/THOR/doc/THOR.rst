THOR Ocean Dynamical Regime Classifier
=======================================

Last update: 2026-09-02

THOR classifies each ocean grid point into one of six dynamical regimes
using a pretrained ensemble of 50 small feedforward neural networks, driven
by eight physical features derived from sea surface height (SSH), surface
wind stress, and bathymetry:
curl(:math:`\tau`), SSH, :math:`\partial_x` SSH, :math:`\partial_y` SSH,
depth, :math:`\partial_x` depth, :math:`\partial_y` depth, and the Coriolis
parameter :math:`f`.

Alongside the regime map, this POD reports the per-point Shannon entropy of
the ensemble's mean class-probability vector (normalized to [0, 1]) - a
confidence map showing where the 50 models agree versus where the
classification is ambiguous, e.g. near regime boundaries.

Version & Contact info
-----------------------

- Method: Yik et al. 2023, THOR-MOM6 (`<https://github.com/yikwill/THOR-MOM6>`__)
- Adapted for arbitrary CMIP6 models and MDTF: see
  `<https://github.com/sngbird/MDTF-diagnostics>`__

Functionality
-------------

``THOR.py`` is a thin driver: it stages this case's SSH/wind-stress/
bathymetry, builds the 8-feature matrix (``thor_lib/features.py``), runs
the ensemble and computes the entropy map (``thor_lib/ensemble.py``), and
plots both maps whole-domain and per named basin with an optional SSH-
gradient front contour (``thor_lib/plotting.py``).

If ``RUN_EXPLAIN`` is enabled, it also runs LRP (Layer-wise Relevance
Propagation, whole domain) and SHAP (Shapley values, one basin) feature
attribution (``thor_lib/explain.py``), answering which of the 8 input
features drove THOR's classification at a given point. LRP is cheap (a
single backward pass, comparable cost to the ensemble inference itself).
SHAP's exact algorithm is not - it evaluates all 256 feature coalitions per
sample, which is intractable over the whole ensemble/whole domain, so it
defaults to the ``fastshap`` package's approximate ``KernelExplainer``
(~4x faster, 0.9985 correlation to the exact algorithm on this model) and
stays scoped to a single basin (``SHAP_BASIN``) even then. Set
``SHAP_METHOD = "exact"`` to use the real ``shap`` package instead.

Required programming language and libraries
--------------------------------------------

Python 3, with ``numpy``, ``xarray``, ``scikit-learn``, ``matplotlib``,
``tensorflow``/``keras``, and (only when ``RUN_EXPLAIN`` is enabled)
``shap`` and ``fastshap``.

Required model output variables
---------------------------------

- ``zos``: sea surface height above geoid (ocean, monthly)
- ``tauuo``, ``tauvo``: surface downward x/y stress (ocean, monthly)
- ``deptho``: sea floor depth below geoid (ocean, static/fx) - a fixed
  property of the model's ocean grid, not a diagnosed quantity

References
----------

.. _ref-Yik:

1. W. Yik et al. (2023): Physics-informed machine learning for dynamical
   regime classification in the North Atlantic and Kuroshio Extension.
   *THOR-MOM6*, `<https://arxiv.org/abs/2310.13916>`__.

More about this diagnostic
----------------------------

This POD was adapted from a standalone interactive notebook pipeline (in
the source project's ``THOR/`` folder) that also supports browsing/fetching
arbitrary CMIP6 models from ESGF, building features from a custom dataset,
and exploring precomputed regime maps outside the MDTF framework.
``thor_lib/`` here is a direct port of that project's
``build_features.ipynb``, ``apply_thor_fresh.ipynb``, and
``apply_thor_explain.ipynb`` - see that project's own ``README.md`` for the
full notebook-based workflow.

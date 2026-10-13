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
gradient front contour (``thor_lib/plotting.py``). It then compares the
case's regime map against a shipped observational THOR reference
(``thor_lib/obs_compare.py`` - see below), unless ``RUN_OBS_COMPARE`` is
disabled.

If ``RUN_EXPLAIN`` is enabled, it also runs SHAP (Shapley values, one
basin) feature attribution (``thor_lib/explain.py``), answering which of
the 8 input features drove THOR's classification at a given point. SHAP's
exact algorithm evaluates all 256 feature coalitions per sample, which is
intractable over the whole ensemble/whole domain, so it defaults to the
``fastshap`` package's approximate ``KernelExplainer`` (~4x faster, 0.9985
correlation to the exact algorithm on this model) and stays scoped to a
single basin (``SHAP_BASIN``, a named basin or a user-defined custom
window - see ``CUSTOM_BASIN_*`` in settings.jsonc) even then. Set
``SHAP_METHOD = "exact"`` to use the real ``shap`` package instead.

SHAP's runtime grows in proportion to the number of ocean grid points in
that window (about 0.1 s per point for the 50-member ensemble, so ~6 min for
the default window on GFDL-ESM4 and ~21 min on the finer GFDL-CM4) and, more
slowly, with ``SHAP_BACKGROUND_SIZE`` (roughly 0.85 + 0.0015 × background
relative to the default of 100). settings.jsonc lists per-model costs, and
the POD logs an estimate, with a warning for long runs, before SHAP starts.

Unless ``RUN_OBS_COMPARE`` is disabled, the case's regime map is also
compared cell-by-cell against a shipped observational THOR reference
(``thor_lib/obs_compare.py``): THOR run on DUACS altimetry SSH and
scatterometer wind stress (ETOPO1 bathymetry), 2000-2014, on the same 0.5 deg
grid THOR was trained on (ECCO). That reference was built outside this POD
by the standalone ``THOR/obs_thor.py`` pipeline (it needs an external ERDDAP
fetch this POD's runtime environment has no access to) and ships as a small
digested file, ``obs_data/obs_scat_thor.npz``. The case's own regime map
stays on its native grid; agreement is computed by nearest-neighbor-matching
it onto the observational grid, reported globally and per basin in
``BASINS`` (``<CASENAME>_obs_compare_stats.txt``), alongside an obs-vs-case
regime map per basin (``<CASENAME>_obs_compare_<basin>.eps``). Since the
reference only spans 2000-2014, treat this comparison as informative mainly
for historical-like forcing, not an idealized/future scenario such as
1pctCO2 or abrupt-4xCO2.

Required programming language and libraries
--------------------------------------------

Python 3.12, with ``numpy``, ``scipy``, ``xarray``, ``netCDF4``, ``h5py``,
``scikit-learn``, ``matplotlib``, ``contourpy``, ``tensorflow``/``keras``,
and (only when ``RUN_EXPLAIN`` is enabled) ``shap`` and ``fastshap``. THOR
builds its own environment, ``_MDTF_python3_THOR``, from
``src/conda/env_python3_THOR.yml`` (or ``env_python3_THOR_micromamba.yml``),
with versions pinned to the ones the POD was validated with. ``fastshap`` is
installed separately (see doc/running_locally.md), since its package
metadata depends on the deprecated ``sklearn`` PyPI alias.

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

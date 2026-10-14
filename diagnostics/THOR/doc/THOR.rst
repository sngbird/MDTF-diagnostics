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

``THOR.ipynb`` is a thin driver notebook (run via ``jupyter nbconvert
--execute``, same as any notebook-driven MDTF POD): it stages this case's
SSH/wind-stress/bathymetry, builds the 8-feature matrix (``thor_lib/features.py``), runs
the ensemble and computes the entropy map (``thor_lib/ensemble.py``), and
plots both maps whole-domain and per named basin with an optional SSH-
gradient front contour (``thor_lib/plotting.py``). It then compares the
case's regime map against a shipped observational THOR reference
(``thor_lib/obs_compare.py``), unless ``RUN_OBS_COMPARE`` is disabled - see
"Observational THOR reference" below for how that reference is built.

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

Observational THOR reference
------------------------------

Unless ``RUN_OBS_COMPARE`` is disabled (on by default), the case's regime
map is also compared cell-by-cell against THOR run on real-world
observations rather than another model. That reference is built from three
independent products, all averaged over 2000-2014 (the span common to the
wind-stress record below):

- **SSH**: Copernicus DUACS DT2021 absolute dynamic topography, fetched
  from NOAA AOML's ERDDAP mirror, with the global-area mean removed from
  each snapshot before averaging - ``zos`` is a relative field, not
  absolute sea level, so the two need to be on the same footing.
- **Wind stress**: monthly scatterometer stress composites from NOAA
  CoastWatch's ERDDAP - QuikSCAT through Oct 2009 (its last full month),
  then Metop-A ASCAT from Nov 2009, summed into the same per-year files the
  SSH average uses. Coverage is limited to within 75 deg of the equator.
- **Bathymetry**: ETOPO1, also from NOAA CoastWatch's ERDDAP.
- **Coriolis parameter**: computed from latitude, same as the model side.

THOR was trained on ECCO v4r3 (LLC90, ~1 deg) interpolated onto a 0.5 deg
grid, and three of its eight features are spatial gradients, which grow at
finer resolution - so each of the three source fields above is first
block-averaged to 1 deg, then linearly interpolated onto ECCO's *exact*
0.5 deg grid, mirroring how ECCO's own training inputs were built. THOR's
features and ensemble then run on that grid exactly as they would on a
model's native grid. The case's own (native-grid) regime map is compared to
the result by nearest-neighbor-matching it onto that same 0.5 deg grid
(within 0.75 deg of a valid cell), and reported as percent agreement
globally and per basin in ``BASINS`` (``<CASENAME>_obs_compare_stats.txt``),
alongside an obs-vs-case regime map per basin
(``<CASENAME>_obs_compare_<basin>.eps``).

This reference isn't built by the POD itself - it needs ERDDAP fetches the
runtime environment has no access to - so it ships as a small, already-built
digested file, ``obs_data/obs_scat_thor.npz`` (lat/lon, regime class, and
land/no-data masks only - none of the raw input fields), produced once by
the standalone ``THOR/obs_thor.py`` pipeline in the source project (see
"More about this diagnostic" below) and committed alongside the POD's
pretrained model weights. Altimetry and scatterometer coverage both drop
out near coasts and poleward of 75 deg; those cells are reported as "ocean,
no data" rather than outlined as coastline. Since the reference only spans
2000-2014, treat this comparison as informative mainly for a case run under
historical-like forcing, not an idealized/future scenario such as 1pctCO2
or abrupt-4xCO2.

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

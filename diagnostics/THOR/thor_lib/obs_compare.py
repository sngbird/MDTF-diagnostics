"""Observational THOR vs. this case's own model THOR.

The observational side is a precomputed regime map, shipped with the POD
as a small digested file (``obs_data/obs_<source>_thor.npz``) rather than
built here - it comes from THOR run on DUACS altimetry SSH and
scatterometer wind stress (ETOPO1 for bathymetry), fetched from NOAA ERDDAP
by the standalone ``THOR/obs_thor.py`` pipeline in the source project, which
this POD's runtime environment has no access to. See that script's own
docstring for the fetch/build details and ``make_obs_thor_plots.py`` for the
reference implementation this module's matching/agreement logic mirrors.

Caveats intrinsic to this comparison, beyond the usual model-vs-observation
ones:
  - the observational record spans 2000-2014, so it's only a meaningful
    reference for a case run under similar (historical-like) forcing, not a
    future/idealized scenario such as 1pctCO2 or abrupt-4xCO2
  - altimetry and scatterometer wind stress both drop out near coasts and
    (for the scatterometer) poleward of 75 deg, so the observational field
    has real gaps distinct from land - shaded as "ocean, no data" rather
    than outlined as coast
  - the observational field sits on a 0.5 deg grid (matching ECCO, which
    THOR was trained on); the case's own regime map stays on its native
    grid, so agreement is computed by nearest-neighbor-matching the case's
    regime at each observational grid cell, not by regridding either field
"""
import os

import numpy as np
from scipy.spatial import cKDTree

OBS_DIR = os.path.join(os.path.dirname(__file__), "..", "obs_data")
OBS_SOURCE_LABELS = {"scat": "DUACS SSH + scatterometer stress, 2000-2014"}


def load_obs(source="scat"):
    """The shipped observational regime map and its 0.5 deg grid, plus
    land_mask/nodata_mask (both False where the regime map has a value)."""
    path = os.path.join(OBS_DIR, f"obs_{source}_thor.npz")
    with np.load(path) as d:
        return dict(
            lat=d["lat"], lon180=d["lon180"], regime=d["regime"],
            land_mask=d["land_mask"], nodata_mask=d["nodata_mask"],
        )


def _xyz(lat, lon180):
    la, lo = np.deg2rad(lat), np.deg2rad(lon180)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def match_to_grid(src_field, src_lat, src_lon180, dst_lat, dst_lon180, max_match_deg=0.75):
    """Nearest-neighbor-sample src_field (any grid, e.g. a model's native
    curvilinear grid) onto dst_lat/dst_lon180 - the same method
    make_obs_thor_plots.py uses to put each model's regime map onto the
    observations' grid, so the agreement numbers here match that pipeline's.
    A destination cell farther than max_match_deg from the nearest finite
    source cell is left NaN rather than matched to a distant value."""
    ok = np.isfinite(src_field) & np.isfinite(src_lat) & np.isfinite(src_lon180)
    if not ok.any():
        return np.full(dst_lat.shape, np.nan)
    tree = cKDTree(_xyz(src_lat[ok], src_lon180[ok]))
    dist, idx = tree.query(_xyz(dst_lat, dst_lon180).reshape(-1, 3))
    out = src_field[ok][idx].reshape(dst_lat.shape).astype(float)
    out[(dist > 2 * np.sin(np.deg2rad(max_match_deg) / 2)).reshape(dst_lat.shape)] = np.nan
    return out


def agreement_pct(a, b, box=None):
    """% of cells where discrete fields a and b agree, among cells where
    both are finite (and, if given, box is also True there). Returns
    (pct, n_cells); pct is nan when n_cells is 0."""
    both = np.isfinite(a) & np.isfinite(b)
    if box is not None:
        both &= box
    n = int(both.sum())
    return (100.0 * (a[both] == b[both]).mean() if n else float("nan")), n


def basin_box(lat, lon180, basin_def):
    return (
        (lon180 >= basin_def["lon"][0]) & (lon180 <= basin_def["lon"][1])
        & (lat >= basin_def["lat"][0]) & (lat <= basin_def["lat"][1])
    )

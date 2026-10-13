# THOR ocean dynamical regime classifier - MDTF POD driver
# ================================================================================
# Builds THOR's 8-feature matrix (curl(tau), ssh, grad_ssh_x/y, depth,
# grad_depth_x/y, coriolis f) from this case's staged SSH, wind stress, and
# bathymetry, runs THOR's pretrained 50-model MLP ensemble, and plots the
# resulting regime map plus a per-point prediction-entropy (confidence) map -
# both whole-domain and per named basin, with an SSH-gradient front (jet/
# current core) contour on the basin plots.
#
# A SHAP (one basin, expensive) feature-attribution plot runs afterward if
# RUN_EXPLAIN is set - see thor_lib/explain.py for why SHAP defaults to an
# approximate method and stays basin-scoped even then, and settings.jsonc
# for why this POD uses SHAP rather than LRP.
#
# Unless RUN_OBS_COMPARE is disabled, the case's regime map is also compared
# cell-by-cell against a shipped observational THOR reference (DUACS
# altimetry + scatterometer wind stress) - see thor_lib/obs_compare.py.
#
# See doc/THOR.rst for the underlying method (Yik et al. 2023, THOR-MOM6)
# and references. This file is intentionally a thin orchestrator - the
# actual feature/model/plotting code lives in thor_lib/, shared with
# THOR/apply_thor_fresh.ipynb and THOR/apply_thor_explain.ipynb, the
# interactive notebooks this POD was adapted from.
# ================================================================================
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-X windows backend

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from thor_lib import ensemble, features, plotting

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

WORK_DIR = os.environ["WORK_DIR"]
CASENAME = os.environ["CASENAME"]
PS_DIR = os.path.join(WORK_DIR, "model", "PS")
NC_DIR = os.path.join(WORK_DIR, "model", "netCDF")
assert os.path.isdir(PS_DIR), f"{PS_DIR} not found"
assert os.path.isdir(NC_DIR), f"{NC_DIR} not found"

# --- pod_env_vars (see settings.jsonc for defaults/descriptions) ---
BASINS = [b for b in os.environ.get("BASINS", "").split(",") if b]
FRONT_CONTOUR = os.environ.get("FRONT_CONTOUR", "true").lower() == "true"
FRONT_PERCENTILE = float(os.environ.get("FRONT_PERCENTILE", "90"))
RUN_EXPLAIN = os.environ.get("RUN_EXPLAIN", "false").lower() == "true"
RUN_OBS_COMPARE = os.environ.get("RUN_OBS_COMPARE", "true").lower() == "true"
OBS_SOURCE = os.environ.get("OBS_SOURCE", "scat")
SHAP_BASIN = os.environ.get("SHAP_BASIN", "custom")
SHAP_METHOD = os.environ.get("SHAP_METHOD", "fast")
SHAP_BACKGROUND_SIZE = int(os.environ.get("SHAP_BACKGROUND_SIZE", "100"))
N_MODELS = int(os.environ.get("N_MODELS", "50"))
TIME_COORD = os.environ.get("time_coord", "time")

# A user-defined basin window, registered into plotting.BASIN_DEFS under the
# key "custom" - used wherever "custom" appears in BASINS, or where
# SHAP_BASIN="custom". All four bounds are required together; left blank
# (the settings.jsonc default) means no custom basin is defined, and using
# "custom" in BASINS/SHAP_BASIN without them is a real configuration error,
# not something to silently ignore.
_custom_bounds = {
    k: os.environ.get(f"CUSTOM_BASIN_{k}", "")
    for k in ("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX")
}
if any(_custom_bounds.values()):
    if not all(_custom_bounds.values()):
        raise ValueError(
            "CUSTOM_BASIN_LON_MIN/LON_MAX/LAT_MIN/LAT_MAX must all be set together to define a custom "
            f"basin - got {_custom_bounds!r}"
        )
    _lon_min, _lon_max = float(_custom_bounds["LON_MIN"]), float(_custom_bounds["LON_MAX"])
    _lat_min, _lat_max = float(_custom_bounds["LAT_MIN"]), float(_custom_bounds["LAT_MAX"])
    _jet_lat_min = float(os.environ.get("CUSTOM_BASIN_JET_LAT_MIN") or _lat_min)
    _jet_lat_max = float(os.environ.get("CUSTOM_BASIN_JET_LAT_MAX") or _lat_max)
    plotting.BASIN_DEFS["custom"] = {
        "label": os.environ.get("CUSTOM_BASIN_LABEL", "Custom"),
        "lon": (_lon_min, _lon_max),
        "lat": (_lat_min, _lat_max),
        "jet_lat": (_jet_lat_min, _jet_lat_max),
    }
elif "custom" in BASINS or SHAP_BASIN == "custom":
    raise ValueError(
        "BASINS or SHAP_BASIN references \"custom\", but CUSTOM_BASIN_LON_MIN/LON_MAX/LAT_MIN/LAT_MAX "
        "aren't set - see settings.jsonc"
    )

# 1) Build features from this case's staged model output ####################
#
# The framework stages one file per varlist entry and sets a <VARNAME>_FILE
# env var pointing to it (plus a <varname>_var env var with the variable's
# actual name in that file), same as any other POD.
print(f"THOR: building features for {CASENAME}")
X_raw, valid_mask, lat, lon = features.build(
    zos_file=os.environ["ZOS_FILE"], zos_var=os.environ["zos_var"],
    tauuo_file=os.environ["TAUUO_FILE"], tauuo_var=os.environ["tauuo_var"],
    tauvo_file=os.environ["TAUVO_FILE"], tauvo_var=os.environ["tauvo_var"],
    deptho_file=os.environ["DEPTHO_FILE"], deptho_var=os.environ["deptho_var"],
    time_dim=TIME_COORD,
)
lon180 = plotting.lon_to_180(lon)

# columns 2 and 3 of X are the SSH gradient - reuse them (rather than
# recomputing) for the basin plots' jet/front contour
ssh_grad_mag = np.full(valid_mask.shape, np.nan)
ssh_grad_mag[valid_mask] = np.hypot(X_raw[:, 2], X_raw[:, 3])

# 2) Run the ensemble #########################################################
print("THOR: running the 50-model ensemble")
scaler = ensemble.load_scaler(MODEL_DIR)
X = ensemble.scale(X_raw, scaler)
regime_map, entropy_map = ensemble.run(X, valid_mask, MODEL_DIR, n_models=N_MODELS)

np.save(os.path.join(NC_DIR, f"{CASENAME}_regime_map.npy"), regime_map)
np.save(os.path.join(NC_DIR, f"{CASENAME}_entropy_map.npy"), entropy_map)

# 3) Plot #####################################################################
print("THOR: plotting regime + entropy maps")
plotting.plot_whole_domain(
    regime_map, lat, lon180, f"THOR regime prediction - {CASENAME} (whole domain)",
    os.path.join(PS_DIR, f"{CASENAME}_regimes.eps"),
    cmap=plotting.REGIME_CMAP, vmin=0, vmax=5, cbar_label="THOR regime class",
)
plotting.plot_whole_domain(
    entropy_map, lat, lon180, f"THOR ensemble entropy - {CASENAME} (whole domain)",
    os.path.join(PS_DIR, f"{CASENAME}_entropy.eps"),
    cmap="viridis", vmin=0, vmax=1,
    cbar_label="normalized entropy (0 = confident, 1 = maximally uncertain)",
)

if BASINS:
    plotting.plot_basins(
        regime_map, lat, lon180, BASINS, "THOR regimes",
        os.path.join(PS_DIR, f"{CASENAME}_basins.eps"),
        cmap=plotting.REGIME_CMAP, vmin=0, vmax=5, cbar_label="THOR regime class",
        front_field=ssh_grad_mag, front_field_label="SSH gradient",
        front_contour=FRONT_CONTOUR, front_percentiles=(FRONT_PERCENTILE,),
    )
    plotting.plot_basins(
        entropy_map, lat, lon180, BASINS, "THOR ensemble entropy",
        os.path.join(PS_DIR, f"{CASENAME}_entropy_basins.eps"),
        cmap="viridis", vmin=0, vmax=1, cbar_label="normalized entropy",
    )
else:
    print("BASINS is empty - skipping basin plots.")

# 4) Explainability (optional, expensive) ####################################
if RUN_EXPLAIN:
    from thor_lib import explain

    print(f"THOR: RUN_EXPLAIN is on - running SHAP (method={SHAP_METHOD!r}, basin={SHAP_BASIN!r})")
    shap_consist, in_basin = explain.run_shap(
        X, valid_mask, lat, lon180, SHAP_BASIN, MODEL_DIR, N_MODELS, SHAP_METHOD,
        background_size=SHAP_BACKGROUND_SIZE,
    )
    explain.plot_shap(shap_consist, in_basin, regime_map, lat, lon180, SHAP_BASIN, PS_DIR, CASENAME)

# 5) Observational THOR comparison (shipped reference, see thor_lib/obs_compare.py) #
if RUN_OBS_COMPARE:
    from thor_lib import obs_compare

    print(f"THOR: comparing against observational THOR (source={OBS_SOURCE!r})")
    obs = obs_compare.load_obs(OBS_SOURCE)
    model_on_obs_grid = obs_compare.match_to_grid(regime_map, lat, lon180, obs["lat"], obs["lon180"])
    source_label = obs_compare.OBS_SOURCE_LABELS[OBS_SOURCE]

    global_pct, global_n = obs_compare.agreement_pct(model_on_obs_grid, obs["regime"])
    stats = [
        f"Observational THOR ({source_label}) vs. {CASENAME}: % of shared cells in the same regime", "",
        f"global: {global_pct:.1f}% ({global_n} cells)",
    ]

    compare_basins = [b for b in BASINS if b in plotting.BASIN_DEFS]
    for key in compare_basins:
        b = plotting.BASIN_DEFS[key]
        basin_pct, basin_n = obs_compare.agreement_pct(
            model_on_obs_grid, obs["regime"], obs_compare.basin_box(obs["lat"], obs["lon180"], b),
        )
        stats.append(f"{key} ({b['label']}): {basin_pct:.1f}% ({basin_n} cells)")

        out_path = os.path.join(PS_DIR, f"{CASENAME}_obs_compare_{key}.eps")
        plotting.plot_multi_basin(
            key,
            [
                dict(field=obs["regime"], lat=obs["lat"], lon180=obs["lon180"],
                     cmap=plotting.REGIME_CMAP, vmin=0, vmax=5, cbar_label="THOR regime class",
                     tick_labels=plotting.REGIME_LABELS, title=f"Observations ({source_label})",
                     front_contour=False, land_mask=obs["land_mask"], nodata_mask=obs["nodata_mask"]),
                dict(field=regime_map, lat=lat, lon180=lon180,
                     cmap=plotting.REGIME_CMAP, vmin=0, vmax=5, cbar_label="THOR regime class",
                     tick_labels=plotting.REGIME_LABELS, title=f"{CASENAME}",
                     front_contour=False),
            ],
            f"THOR regimes - {b['label']}: {basin_pct:.0f}% of shared cells agree with observations",
            out_path, chrome=0.6,
        )
        print(f"THOR: saved {out_path}")

    stats_path = os.path.join(NC_DIR, f"{CASENAME}_obs_compare_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats) + "\n")
    print("\n".join(stats))

print("THOR: finished successfully")
sys.exit(0)

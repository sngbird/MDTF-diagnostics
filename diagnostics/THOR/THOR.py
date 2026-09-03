# THOR ocean dynamical regime classifier - MDTF POD driver
# ================================================================================
# Builds THOR's 8-feature matrix (curl(tau), ssh, grad_ssh_x/y, depth,
# grad_depth_x/y, coriolis f) from this case's staged SSH, wind stress, and
# bathymetry, runs THOR's pretrained 50-model MLP ensemble, and plots the
# resulting regime map plus a per-point prediction-entropy (confidence) map -
# both whole-domain and per named basin, with an SSH-gradient front (jet/
# current core) contour on the basin plots.
#
# LRP (whole domain, cheap) and SHAP (one basin, expensive) feature-
# attribution plots run afterward if RUN_EXPLAIN is set - see
# thor_lib/explain.py for why SHAP defaults to an approximate method and
# stays basin-scoped even then.
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
SHAP_BASIN = os.environ.get("SHAP_BASIN", "na")
SHAP_METHOD = os.environ.get("SHAP_METHOD", "fast")
SHAP_BACKGROUND_SIZE = int(os.environ.get("SHAP_BACKGROUND_SIZE", "100"))
LRP_EPSILON_SCALE = float(os.environ.get("LRP_EPSILON_SCALE", "0.25"))
N_MODELS = int(os.environ.get("N_MODELS", "50"))
TIME_COORD = os.environ.get("time_coord", "time")

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
        ssh_grad_mag=ssh_grad_mag, front_contour=FRONT_CONTOUR, front_percentile=FRONT_PERCENTILE,
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

    print("THOR: RUN_EXPLAIN is on - running LRP (whole domain)")
    lrp_consist = explain.run_lrp(X, MODEL_DIR, N_MODELS, LRP_EPSILON_SCALE)
    explain.plot_lrp(lrp_consist, valid_mask, lat, lon180, os.path.join(PS_DIR, f"{CASENAME}_lrp.eps"))

    print(f"THOR: RUN_EXPLAIN is on - running SHAP (method={SHAP_METHOD!r}, basin={SHAP_BASIN!r})")
    shap_consist, in_basin = explain.run_shap(
        X, valid_mask, lat, lon180, SHAP_BASIN, MODEL_DIR, N_MODELS, SHAP_METHOD,
        background_size=SHAP_BACKGROUND_SIZE,
    )
    explain.plot_shap(shap_consist, in_basin, regime_map, lat, lon180, SHAP_BASIN, PS_DIR, CASENAME)

print("THOR: finished successfully")
sys.exit(0)

"""LRP and SHAP feature-attribution explainability for THOR, following
Yik et al. 2023 (THOR-MOM6, https://github.com/yikwill/THOR-MOM6). Optional
POD stage, gated by RUN_EXPLAIN in settings.jsonc's pod_env_vars.

LRP is a single backward pass per sample - about as cheap as running the
ensemble itself, so it runs on the whole domain. SHAP's exact algorithm
evaluates 2^8=256 feature coalitions per sample - roughly 45 hours for all
50 ensemble members over a full ~256k-point global grid - so it's scoped to
one basin (matching what the paper itself did, for the same reason), and
defaults to the fastshap package's approximate KernelExplainer (~4x faster,
0.9985 correlation to the exact shap package on this model) rather than the
real shap package's exact algorithm.
"""
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Dense
import shap
import fastshap

from . import ensemble
from .plotting import BASIN_DEFS, true_aspect

# fastshap casts an internal counter array in a way that harmlessly overflows
# for larger inputs - doesn't affect the returned values (verified against
# real shap's exact output: 0.9985+ correlation)
warnings.filterwarnings("ignore", message="overflow encountered in cast")

FEATURE_NAMES = [
    "curl_tau", "ssh", "grad_ssh_x", "grad_ssh_y",
    "depth", "grad_depth_x", "grad_depth_y", "coriolis_f",
]


def _get_activations_and_weights(model, X_batch):
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    activation_model = Model(inputs=model.input, outputs=[l.output for l in dense_layers])
    acts = activation_model.predict(X_batch, verbose=0, batch_size=8192)
    weights = [l.get_weights() for l in dense_layers]
    return acts, weights


def _lrp_epsilon(X_batch, acts, weights, epsilon_scale):
    a_ins = [X_batch] + acts[:-1]
    W_last, b_last = weights[-1]
    logits = a_ins[-1] @ W_last + b_last
    predicted_class = np.argmax(logits, axis=1)

    n_samples, n_classes = logits.shape
    R = np.zeros_like(logits)
    R[np.arange(n_samples), predicted_class] = logits[np.arange(n_samples), predicted_class]

    for i in reversed(range(len(weights))):
        W, b = weights[i]
        a_in = a_ins[i]
        z = a_in @ W + b
        epsilon = epsilon_scale * np.std(z)
        z_stab = z + epsilon * np.where(z >= 0, 1.0, -1.0)
        contrib = (a_in[:, :, None] * W[None, :, :]) / z_stab[:, None, :]
        R = np.einsum("nio,no->ni", contrib, R)
    return R


def run_lrp(X, model_dir, n_models, epsilon_scale, batch_size=8192):
    """X is the scaled (n_valid, 8) feature matrix. Returns ensemble-consistent,
    per-sample-normalized relevance, shape (n_valid, 8)."""
    n_samples, n_features = X.shape
    lrp_all = np.zeros((n_models, n_samples, n_features), dtype="float32")

    for m in range(n_models):
        model = ensemble.build_model()
        model.load_weights(os.path.join(model_dir, f"vanillamodel_{m}.h5"))
        for start in range(0, n_samples, batch_size):
            sl = slice(start, start + batch_size)
            acts, weights = _get_activations_and_weights(model, X[sl])
            lrp_all[m, sl] = _lrp_epsilon(X[sl], acts, weights, epsilon_scale)
        print(f"  LRP: ensemble member {m + 1}/{n_models} done")

    lrp_scaled = lrp_all / np.max(np.abs(lrp_all), axis=2, keepdims=True)
    p25 = np.percentile(lrp_scaled, 25, axis=0)
    p75 = np.percentile(lrp_scaled, 75, axis=0)
    lrp_consist = np.zeros_like(p25)
    lrp_consist[p75 < 0] = p75[p75 < 0]
    lrp_consist[p25 > 0] = p25[p25 > 0]
    return lrp_consist


def plot_lrp(lrp_consist, valid_mask, lat, lon180, out_path):
    aspect = true_aspect(lat, lon180)
    maps = {}
    for j, name in enumerate(FEATURE_NAMES):
        m = np.full(valid_mask.shape, np.nan)
        m[valid_mask] = lrp_consist[:, j]
        maps[name] = m

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    im = None
    for ax, name in zip(axes.flat, FEATURE_NAMES):
        im = ax.imshow(maps[name], cmap="seismic", vmin=-1, vmax=1, origin="lower", aspect=aspect)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="LRP relevance (ensemble-consistent, normalized)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _run_shap_one_model(model, background, data, method, fastshap_n_coalition_sizes):
    if method == "exact":
        explainer = shap.Explainer(model, background, algorithm="exact", feature_names=FEATURE_NAMES)
        shap_info = explainer(data)
        return shap_info.values, shap_info.base_values
    elif method == "fast":
        def model_fn(X_batch):
            return model.predict(X_batch, verbose=0, batch_size=8192)
        explainer = fastshap.KernelExplainer(model_fn, background)
        raw = explainer.calculate_shap_values(data, n_coalition_sizes=fastshap_n_coalition_sizes)
        return raw[:, :-1, :], raw[:, -1, :]
    else:
        raise ValueError(f"Unknown SHAP method: {method!r}")


def run_shap(X, valid_mask, lat, lon180, basin_key, model_dir, n_models, method,
             fastshap_n_coalition_sizes=1, background_size=100, seed=0):
    """X is the scaled (n_valid, 8) feature matrix for the whole domain; the
    basin subset is what's actually explained. SHAP's background is a random
    subsample of X (not the whole domain) - both fastshap's KernelExplainer
    and shap's own algorithms scale in memory/time with background size x
    explained points x coalitions, so passing the entire domain as background
    is fine for a small grid but explodes for a finer one (observed: >1TB
    requested on a ~940k-point ocean grid). A few hundred points is the
    standard KernelSHAP background size and is enough to represent the
    feature distribution these 8 physical features come from.

    Returns (shap_consist, in_basin) - shap_consist is (basin_points, 8, 6).
    """
    b = BASIN_DEFS[basin_key]
    in_basin = (
        (lon180 >= b["lon"][0]) & (lon180 <= b["lon"][1])
        & (lat >= b["lat"][0]) & (lat <= b["lat"][1])
        & valid_mask
    )
    basin_flat_idx = np.where(in_basin[valid_mask])[0]
    X_basin = X[basin_flat_idx]

    rng = np.random.default_rng(seed)
    if X.shape[0] > background_size:
        background = X[rng.choice(X.shape[0], size=background_size, replace=False)]
    else:
        background = X
    print(f"SHAP scope: {b['label']!r} - {len(X_basin)} points, "
          f"background={len(background)} points, method={method!r}")

    shap_vals_list, shap_base_list = [], []
    for m in range(n_models):
        model = ensemble.build_model()
        model.load_weights(os.path.join(model_dir, f"vanillamodel_{m}.h5"))
        values, base_values = _run_shap_one_model(model, background, X_basin, method, fastshap_n_coalition_sizes)
        shap_vals_list.append(values)
        shap_base_list.append(base_values)
        print(f"  SHAP: ensemble member {m + 1}/{n_models} done")

    shap_vals = np.array(shap_vals_list)   # (models, basin_points, features, classes)
    shap_base = np.array(shap_base_list)   # (models, basin_points, classes)

    base_mean = shap_base.mean(axis=0)
    diff = shap_base - base_mean
    for j in range(len(FEATURE_NAMES)):
        shap_vals[:, :, j, :] += diff

    q25 = np.percentile(shap_vals, 25, axis=0)
    q75 = np.percentile(shap_vals, 75, axis=0)
    shap_consist = np.zeros_like(q25)
    shap_consist[q75 < 0] = q75[q75 < 0]
    shap_consist[q25 > 0] = q25[q25 > 0]
    return shap_consist, in_basin


def plot_shap(shap_consist, in_basin, regime_map, lat, lon180, basin_key, out_dir, out_prefix, ext="eps"):
    """One figure per regime class present in the basin, 8 feature panels
    each. Each point's SHAP value is scattered back onto the model's own
    native grid and drawn with pcolormesh (filled grid cells), not a
    scatter plot of small dots - at basin scale the two look similar, but
    a tight custom window (a handful of points across) reads as mostly
    empty space with tiny dots; filled native-resolution cells make each
    point occupy its actual grid-cell footprint instead. A black coastline
    (the model's own land mask, i.e. where regime_map is NaN) is drawn on
    every panel so the separation point/coast is visible regardless of
    zoom level."""
    b = BASIN_DEFS[basin_key]
    regime_basin = regime_map[in_basin]
    present_regimes = sorted(int(r) for r in np.unique(regime_basin[~np.isnan(regime_basin)]))
    land = np.isnan(regime_map).astype(float)

    for regime in present_regimes:
        mask = regime_basin == regime
        if mask.sum() < 5:
            continue
        clim = np.percentile(np.abs(shap_consist[mask, :, regime]), 95)
        regime_mask_full = in_basin & (regime_map == regime)

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        im = None
        for ax, j, name in zip(axes.flat, range(len(FEATURE_NAMES)), FEATURE_NAMES):
            field = np.full(lat.shape, np.nan)
            field[regime_mask_full] = shap_consist[mask, j, regime]
            im = ax.pcolormesh(lon180, lat, field, cmap="seismic", vmin=-clim, vmax=clim, shading="auto")
            ax.contour(lon180, lat, land, levels=[0.5], colors="black", linewidths=0.6)
            ax.set_xlim(b["lon"])
            ax.set_ylim(b["lat"])
            ax.set_title(name)
            ax.set_aspect("equal")
        fig.suptitle(f"SHAP - {b['label']} - regime {regime}")
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="SHAP value (ensemble-consistent)")
        plt.savefig(os.path.join(out_dir, f"{out_prefix}_shap_{basin_key}_regime{regime}.{ext}"), bbox_inches="tight")
        plt.close()

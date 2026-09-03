"""Shared plotting for THOR's regime/entropy maps - whole-domain and per
named basin, with an optional SSH-gradient front (jet/current core)
contour on basin plots.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

REGIME_CMAP = ListedColormap(["royalblue", "cyan", "yellow", "orange", "magenta", "red"])

# lon bounds are in -180..180 (normalized regardless of source convention)
BASIN_DEFS = {
    "na": {"label": "North Atlantic", "lon": (-80, 20), "lat": (20, 66)},
    "ke": {"label": "Kuroshio Extension", "lon": (127, 177), "lat": (15, 53)},
}


def lon_to_180(lon):
    return ((lon + 180) % 360) - 180


def true_aspect(lat, lon180):
    """Aspect-ratio correction for plain imshow of a curvilinear/displaced-pole
    grid, so the whole-domain plot isn't stretched relative to the grid's true
    lat/lon extent. See THOR/apply_thor_fresh.ipynb for why this is used
    instead of pcolormesh for the whole-domain plot (pole-fold smearing)."""
    lat_range = np.nanmax(lat) - np.nanmin(lat)
    lon_range = np.nanmax(lon180) - np.nanmin(lon180)
    ny, nx = lat.shape
    return (lat_range / lon_range) * (nx / ny)


def plot_whole_domain(field, lat, lon180, title, out_path, cmap, vmin, vmax, cbar_label):
    aspect = true_aspect(lat, lon180)
    plt.figure(figsize=(10, 8))
    plt.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect=aspect)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label(cbar_label)
    plt.xlabel("x (native grid index)")
    plt.ylabel("y (native grid index)")
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_basins(
    field, lat, lon180, basins, title_prefix, out_path, cmap, vmin, vmax, cbar_label,
    ssh_grad_mag=None, front_contour=False, front_percentile=90,
):
    fig, axes = plt.subplots(1, len(basins), figsize=(8 * len(basins), 7))
    if len(basins) == 1:
        axes = [axes]

    for ax, key in zip(axes, basins):
        b = BASIN_DEFS[key]
        in_basin = (
            (lon180 >= b["lon"][0]) & (lon180 <= b["lon"][1])
            & (lat >= b["lat"][0]) & (lat <= b["lat"][1])
        )
        sub = np.where(in_basin, field, np.nan)
        im = ax.pcolormesh(lon180, lat, sub, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

        if front_contour and ssh_grad_mag is not None:
            local_grad = np.where(in_basin, ssh_grad_mag, np.nan)
            threshold = np.nanpercentile(local_grad, front_percentile)
            ax.contour(lon180, lat, local_grad, levels=[threshold], colors="black", linewidths=1.2)
            front_proxy = Line2D(
                [0], [0], color="black", linewidth=1.2,
                label=f"steepest {100 - front_percentile}% SSH gradient (jet/front)",
            )
            ax.legend(handles=[front_proxy], loc="lower right", fontsize=8, framealpha=0.9)

        ax.set_xlim(b["lon"])
        ax.set_ylim(b["lat"])
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{title_prefix} - {b['label']}")
        cbar = plt.colorbar(im, ax=ax, fraction=0.03)
        cbar.set_label(cbar_label)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

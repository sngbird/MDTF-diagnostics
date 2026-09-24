"""Shared plotting for THOR's regime/entropy maps - whole-domain and per
named basin, with an optional front (jet/current core) overlay on basin
plots: contour lines from a scalar field (real current speed when
available, else an SSH-gradient geostrophic proxy) and, when real current
velocity is available, a quiver of flow direction.
"""
import contourpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008) - chosen over the
# original saturated royalblue/cyan/yellow/orange/magenta/red set for being
# both easier on the eyes and accessible to color-vision-deficient viewers.
REGIME_CMAP = ListedColormap(["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"])

# lon bounds are in -180..180 (normalized regardless of source convention).
# jet_lat is the narrower core band the jet/current itself is expected to sit
# within, used only by the SSH-isoline jet method below (not the full basin
# box) - e.g. the Kuroshio Extension basin spans 15-53N, but the jet core is
# tighter, ~28-45N. The North Atlantic range is an analogous first guess for
# the Gulf Stream/North Atlantic Current, not yet validated the way 28-45N
# was for the Kuroshio.
BASIN_DEFS = {
    "na": {"label": "North Atlantic", "lon": (-80, 20), "lat": (20, 66), "jet_lat": (30, 45)},
    "ke": {"label": "Kuroshio Extension", "lon": (127, 177), "lat": (15, 53), "jet_lat": (28, 45)},
}


def _basin_aspect(basin_key):
    """lon-span / lat-span for a basin - the data aspect ratio a plot of it
    (with set_aspect("equal")) actually renders at."""
    b = BASIN_DEFS[basin_key]
    return (b["lon"][1] - b["lon"][0]) / (b["lat"][1] - b["lat"][0])


def _figsize_for_aspects(aspects, panel_width=7.0, chrome=2.2):
    """A (width, height) sized to match one row of len(aspects)
    aspect-locked panels (each panel's own lon-span/lat-span), so there's
    no big leftover vertical space for constrained_layout to turn into a
    gap between the suptitle and the axes (an aspect-locked axes renders
    at its data aspect regardless of the figure's shape - if the figure is
    taller than that, the shortfall shows up as blank space above the row,
    not a shrunk figure). `chrome` is a rough per-row allowance for the
    subplot title, x-label, and ticks (on top of the data aspect itself).
    Panels sharing a row can have different aspect ratios; the tallest
    one's requirement wins so no panel ends up forced shorter than it
    needs."""
    needed_height = max(panel_width / a for a in aspects) + chrome
    return (panel_width * len(aspects), needed_height)


def _basin_figsize(basin_keys, panel_width=7.0, chrome=2.2):
    return _figsize_for_aspects([_basin_aspect(k) for k in basin_keys], panel_width, chrome)


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


def compute_jet_ssh_isoline(ssh, lat, lon180, in_basin, lat_min, lat_max):
    """Find the jet/front as a single SSH isoline, rather than thresholding
    gradient magnitude - matches how most papers actually locate a jet from
    altimetry: pick the latitude with the strongest average meridional SSH
    gradient within a core band, then trace the SSH contour at that
    latitude's own SSH value. That contour is one continuous, physically
    anchored curve (an actual SSH isoline) rather than a set of
    locally-steep-gradient patches.

    The grid here is curvilinear (2D lat/lon, not a clean lat x lon
    rectangle), so "latitude band" and "zonal mean" are approximated using
    each grid row's mean latitude - a good approximation away from a
    pole-fold, which every basin used here is.

    Returns (jet_lat, jet_level), or (None, None) if the band has no valid
    data (e.g. this basin/model combination doesn't cover it).
    """
    row_mean_lat = np.nanmean(np.where(in_basin, lat, np.nan), axis=1)
    row_mask = (row_mean_lat >= lat_min) & (row_mean_lat <= lat_max)
    if not row_mask.any():
        return None, None

    ssh_masked = np.where(in_basin, ssh, np.nan)
    z_band = ssh_masked[row_mask, :]
    y_band = row_mean_lat[row_mask]

    with np.errstate(all="ignore"):
        dz_dy = np.gradient(z_band, y_band, axis=0)
        zonal_mean_grad = np.nanmean(np.abs(dz_dy), axis=1)
    if np.all(np.isnan(zonal_mean_grad)):
        return None, None

    jet_idx = np.nanargmax(zonal_mean_grad)
    jet_level = np.nanmean(z_band[jet_idx, :])
    jet_lat = y_band[jet_idx]
    return jet_lat, jet_level


def widest_contour_path(lon180, lat, field, level):
    """The single connected contour line for `level` in `field` spanning the
    widest longitude range, as an (N, 2) array of (lon, lat) points, or None
    if the level isn't crossed anywhere. A field that crosses `level` in
    more than one place (a real, if noisy, possibility for anything derived
    from observations - mesoscale eddies and recirculation gyres can sit at
    the same SSH value as the jet itself) would otherwise draw every one of
    those disconnected loops, not just the jet.

    Longitude span, not raw arc length, is what picks out the jet: a jet
    stretches most of the basin's width, while a tight recirculation
    meander can rack up more total arc length while covering only a
    fraction of that span."""
    cg = contourpy.contour_generator(x=lon180, y=lat, z=field)
    lines = cg.lines(level)
    if not lines:
        return None

    def lon_span(pts):
        return pts[:, 0].max() - pts[:, 0].min()

    return max(lines, key=lon_span)


def _draw_basin_panel(
    ax, key, field, lat, lon180, title, cmap, vmin, vmax, cbar_label,
    front_field=None, front_field_label="SSH gradient", front_contour=True,
    front_percentiles=(75, 85, 90, 95),
    front_lat=None, front_lon180=None,
    front_u=None, front_v=None, quiver_stride=3, quiver_scale=3,
    ssh_field=None,
):
    """Draw one basin's field (plus optional front overlay) into `ax`. Shared
    by plot_basins() (one row of these, one per basin, same field) and
    plot_obs_vs_sim_basin() (two of these side by side, one for observed
    altimetry and one for the simulated field, same basin).

    front_field is whatever scalar field should be thresholded to trace the
    jet/front (real current speed sqrt(uo^2+vo^2) when available - the
    physically direct choice, since the front is where the current is
    fastest - or SSH gradient magnitude, a geostrophic-velocity proxy, as a
    fallback when real current velocity isn't published for a model).
    front_percentiles draws one contour line per percentile (increasing
    linewidth with percentile) instead of a single threshold, so the jet's
    structure shows as a gradient rather than one line that can look
    arbitrary/noisy wherever that one threshold happens to fall.
    front_field_label only affects the legend text.

    front_lat/front_lon180 let the front field (and front_u/front_v below)
    live on their own grid, separate from the main field's (lat, lon180) -
    needed for current velocity, which CMIP6 models publish on the ocean
    C-grid's U/V points, offset by about half a grid cell from the T-grid
    points zos/deptho (and this plot's main field) live on. Defaults to the
    main grid when the front field is on the same one (e.g. the SSH-gradient
    fallback, computed from the same T-grid features).

    front_u/front_v (optional): raw current velocity components, same grid
    as front_lat/front_lon180 - when given, overlays a subsampled quiver of
    flow direction/magnitude (every quiver_stride-th point) alongside the
    contour. Only meaningful for real current velocity, not the SSH-gradient
    fallback (which has no natural vector to draw).

    ssh_field (optional): raw SSH, on the same grid as `field`/lat/lon180 -
    when given (and the basin has a "jet_lat" range in BASIN_DEFS), also
    draws the jet as a single dashed SSH isoline via
    compute_jet_ssh_isoline(), a second, differently-colored contour
    alongside front_field's. This is closer to how altimetry-based papers
    actually locate a jet (one physically anchored SSH contour) than
    thresholding gradient magnitude is - kept as an addition rather than a
    replacement since both are informative in different ways."""
    b = BASIN_DEFS[key]
    in_basin = (
        (lon180 >= b["lon"][0]) & (lon180 <= b["lon"][1])
        & (lat >= b["lat"][0]) & (lat <= b["lat"][1])
    )
    sub = np.where(in_basin, field, np.nan)
    im = ax.pcolormesh(lon180, lat, sub, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    f_lat = lat if front_lat is None else front_lat
    f_lon180 = lon180 if front_lon180 is None else front_lon180
    in_basin_front = (
        (f_lon180 >= b["lon"][0]) & (f_lon180 <= b["lon"][1])
        & (f_lat >= b["lat"][0]) & (f_lat <= b["lat"][1])
    )

    legend_handles = []

    if front_contour and front_field is not None:
        local_field = np.where(in_basin_front, front_field, np.nan)
        levels = sorted(set(np.nanpercentile(local_field, p) for p in front_percentiles))
        if len(levels) > 1:
            linewidths = np.linspace(0.6, 2.0, len(levels))
            ax.contour(
                f_lon180, f_lat, local_field, levels=levels,
                colors="black", linewidths=linewidths, alpha=0.85,
            )
        pcts = ", ".join(str(p) for p in sorted(front_percentiles))
        legend_handles.append(Line2D(
            [0], [0], color="black", linewidth=1.2,
            label=f"{front_field_label} percentiles [{pcts}]",
        ))

    if ssh_field is not None and "jet_lat" in b:
        lat_min, lat_max = b["jet_lat"]
        jet_lat, jet_level = compute_jet_ssh_isoline(ssh_field, lat, lon180, in_basin, lat_min, lat_max)
        if jet_lat is not None:
            # Contour only within a padded band around the jet core, not the
            # whole basin - the same SSH value the jet sits at can recur far
            # away (elsewhere in a wide basin, or in mesoscale eddies in
            # noisier observational fields), producing disconnected loops
            # unrelated to the actual jet if left unrestricted.
            band_pad = 5
            in_band = in_basin & (lat >= lat_min - band_pad) & (lat <= lat_max + band_pad)
            ssh_masked = np.where(in_band, ssh_field, np.nan)
            # The same SSH value can be crossed in more than one place (a
            # basin-wide recirculation gyre, a mesoscale eddy in noisier
            # observational fields, ...) - contouring naively would draw
            # every one of those as a separate loop. The jet is a single
            # continuous front, so only the longest connected contour line
            # at this level is drawn.
            jet_path = widest_contour_path(lon180, lat, ssh_masked, jet_level)
            if jet_path is not None:
                ax.plot(
                    jet_path[:, 0], jet_path[:, 1],
                    color="lime", linewidth=2.5, linestyle="dashed",
                )
                legend_handles.append(Line2D(
                    [0], [0], color="lime", linewidth=2.5, linestyle="dashed",
                    label=f"SSH isoline at jet latitude ({jet_lat:.1f}°)",
                ))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    if front_u is not None and front_v is not None:
        sl = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
        qlat = f_lat[sl]
        qlon = f_lon180[sl]
        qu = np.where(in_basin_front, front_u, np.nan)[sl]
        qv = np.where(in_basin_front, front_v, np.nan)[sl]
        ax.quiver(qlon, qlat, qu, qv, color="dimgray", scale=quiver_scale, width=0.0025)

    ax.set_xlim(b["lon"])
    ax.set_ylim(b["lat"])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label(cbar_label)


def plot_basins(
    field, lat, lon180, basins, title_prefix, out_path, cmap, vmin, vmax, cbar_label,
    front_field=None, front_field_label="SSH gradient", front_contour=False,
    front_percentiles=(75, 85, 90, 95),
    front_lat=None, front_lon180=None,
    front_u=None, front_v=None, quiver_stride=3, quiver_scale=3,
    ssh_field=None,
):
    """One row of _draw_basin_panel(), one panel per basin, all showing the
    same field. See _draw_basin_panel() for what each parameter does."""
    fig, axes = plt.subplots(1, len(basins), figsize=_basin_figsize(basins), constrained_layout=True)
    if len(basins) == 1:
        axes = [axes]

    for ax, key in zip(axes, basins):
        b = BASIN_DEFS[key]
        _draw_basin_panel(
            ax, key, field, lat, lon180, f"{title_prefix} - {b['label']}",
            cmap, vmin, vmax, cbar_label,
            front_field=front_field, front_field_label=front_field_label,
            front_contour=front_contour, front_percentiles=front_percentiles,
            front_lat=front_lat, front_lon180=front_lon180,
            front_u=front_u, front_v=front_v,
            quiver_stride=quiver_stride, quiver_scale=quiver_scale,
            ssh_field=ssh_field,
        )

    # no bbox_inches="tight" - it fights with constrained_layout (the two
    # together can reintroduce the exact whitespace constrained_layout was
    # meant to remove, e.g. above a suptitle over an aspect-locked subplot)
    plt.savefig(out_path)
    plt.close()


def plot_obs_vs_sim_basin(basin_key, obs, sim, out_path):
    """Side-by-side comparison of observed altimetry (left) and the
    simulated field (right) for one basin, each drawn with its own front
    overlay via _draw_basin_panel() so the jet/front contours are computed
    identically on both sides and are directly comparable.

    `obs` and `sim` are dicts of _draw_basin_panel() keyword args, plus
    "field"/"lat"/"lon180"/"cmap"/"vmin"/"vmax"/"cbar_label"/"title" for the
    panel itself - both sides plot raw SSH (see color_bounds() for the
    diverging colormap), with the front/jet overlay (current speed or
    SSH-gradient percentile contours + the dashed jet isoline) drawn on top
    via the same front_field/front_lat/... keys _draw_basin_panel() already
    takes."""
    b = BASIN_DEFS[basin_key]
    fig, axes = plt.subplots(1, 2, figsize=_basin_figsize([basin_key, basin_key]), constrained_layout=True)

    for ax, panel, default_title in (
        (axes[0], obs, f"Observed - {b['label']}"),
        (axes[1], sim, f"Simulated - {b['label']}"),
    ):
        _draw_basin_panel(
            ax, basin_key, panel["field"], panel["lat"], panel["lon180"],
            panel.get("title", default_title),
            panel["cmap"], panel["vmin"], panel["vmax"], panel["cbar_label"],
            front_field=panel.get("front_field"),
            front_field_label=panel.get("front_field_label", "SSH gradient"),
            front_contour=panel.get("front_contour", True),
            front_percentiles=panel.get("front_percentiles", (75, 85, 90, 95)),
            front_lat=panel.get("front_lat"), front_lon180=panel.get("front_lon180"),
            front_u=panel.get("u"), front_v=panel.get("v"),
            quiver_stride=panel.get("quiver_stride", 3), quiver_scale=panel.get("quiver_scale", 3),
            ssh_field=panel.get("ssh_field"),
        )

    fig.suptitle(f"Observed altimetry vs. simulated - {b['label']}")
    plt.savefig(out_path)  # no bbox_inches="tight" - fights with constrained_layout
    plt.close()


def plot_multi_basin(basin_key, panels, title, out_path):
    """Side-by-side comparison of len(panels) basin panels - the N-panel
    generalization of plot_obs_vs_sim_basin's fixed 2 (obs, sim), for an
    intermodel comparison (one panel per model) instead. Each entry in
    `panels` is a dict of _draw_basin_panel() keyword args, plus
    "field"/"lat"/"lon180"/"cmap"/"vmin"/"vmax"/"cbar_label"/"title" for the
    panel itself - works for any field _draw_basin_panel() can draw
    (continuous, e.g. SSH variability; discrete, e.g. REGIME_CMAP class
    maps) since nothing here is specific to what's being compared."""
    b = BASIN_DEFS[basin_key]
    fig, axes = plt.subplots(
        1, len(panels), figsize=_basin_figsize([basin_key] * len(panels)), constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        _draw_basin_panel(
            ax, basin_key, panel["field"], panel["lat"], panel["lon180"],
            panel.get("title", b["label"]),
            panel["cmap"], panel["vmin"], panel["vmax"], panel["cbar_label"],
            front_field=panel.get("front_field"),
            front_field_label=panel.get("front_field_label", "SSH gradient"),
            front_contour=panel.get("front_contour", True),
            front_percentiles=panel.get("front_percentiles", (75, 85, 90, 95)),
            front_lat=panel.get("front_lat"), front_lon180=panel.get("front_lon180"),
            front_u=panel.get("u"), front_v=panel.get("v"),
            quiver_stride=panel.get("quiver_stride", 3), quiver_scale=panel.get("quiver_scale", 3),
            ssh_field=panel.get("ssh_field"),
        )

    fig.suptitle(title)
    plt.savefig(out_path)  # no bbox_inches="tight" - fights with constrained_layout
    plt.close()


def plot_model_scenario_grid(basin_key, grid_panels, title, out_path, panel_width=6.0, chrome_per_row=1.8):
    """Grid comparison for one basin: rows are (typically) models, columns
    are (typically) scenarios - the 2-D generalization of plot_multi_basin's
    single row. `grid_panels` is a list of rows, each a list of n_cols
    panel dicts (same schema as plot_multi_basin's `panels`); every row
    must have the same length. Each cell gets its own axis via
    _draw_basin_panel(), so it works for the same range of fields
    (continuous, e.g. SSH variability; discrete, e.g. REGIME_CMAP class
    maps).

    chrome_per_row is a rough per-row allowance (subplot title, x-label,
    ticks) added on top of the data-aspect height for each row - see
    _figsize_for_aspects's docstring for why a mismatched figure height
    otherwise leaves a gap above the grid instead of shrinking to fit."""
    n_rows = len(grid_panels)
    n_cols = len(grid_panels[0])
    panel_height = panel_width / _basin_aspect(basin_key)
    figsize = (panel_width * n_cols, n_rows * (panel_height + chrome_per_row))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True, squeeze=False)

    for row, panel_row in zip(axes, grid_panels):
        for ax, panel in zip(row, panel_row):
            _draw_basin_panel(
                ax, basin_key, panel["field"], panel["lat"], panel["lon180"],
                panel.get("title", basin_key),
                panel["cmap"], panel["vmin"], panel["vmax"], panel["cbar_label"],
                front_field=panel.get("front_field"),
                front_field_label=panel.get("front_field_label", "SSH gradient"),
                front_contour=panel.get("front_contour", True),
                front_percentiles=panel.get("front_percentiles", (75, 85, 90, 95)),
                front_lat=panel.get("front_lat"), front_lon180=panel.get("front_lon180"),
                front_u=panel.get("u"), front_v=panel.get("v"),
                quiver_stride=panel.get("quiver_stride", 3), quiver_scale=panel.get("quiver_scale", 3),
                ssh_field=panel.get("ssh_field"),
            )

    fig.suptitle(title)
    plt.savefig(out_path)  # no bbox_inches="tight" - fights with constrained_layout
    plt.close()


def plot_whole_domain_grid(grid_panels, title, out_path, cmap="seismic", vmin=-1, vmax=1, cbar_label=""):
    """N rows x M cols whole-domain grid via imshow on each panel's own
    native grid index (like plot_whole_domain/explain.plot_lrp) - for LRP,
    where a per-feature attribution map is whole-domain, not basin-cropped,
    so plot_model_scenario_grid's basin-panel machinery doesn't apply.
    Each entry in `grid_panels` is a dict with "field" (2-D, on the native
    grid), "lat"/"lon180" (for true_aspect), and "title"."""
    n_rows = len(grid_panels)
    n_cols = len(grid_panels[0])
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), constrained_layout=True, squeeze=False,
    )
    im = None
    for row, panel_row in zip(axes, grid_panels):
        for ax, panel in zip(row, panel_row):
            aspect = true_aspect(panel["lat"], panel["lon180"])
            im = ax.imshow(panel["field"], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect=aspect)
            ax.set_title(panel.get("title", ""))
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=cbar_label)
    fig.suptitle(title)
    plt.savefig(out_path)
    plt.close()


def regrid_to_regular(lon180, lat, field, lon_step=1.0, lat_step=1.0, max_gap_deg=2.0):
    """Nearest-neighbor-resample a curvilinear-grid field onto a regular
    lon/lat grid, via a single cKDTree query (not scipy.interpolate.griddata's
    default Delaunay-triangulation-based linear method, which is prone to
    segfaulting on the large, quasi-structured point sets a full native
    ocean grid produces - a known flaky failure mode of Qhull on Windows,
    not something try/except can catch since it crashes the interpreter).
    Nearest-neighbor is a fine substitute for true interpolation here
    anyway, since source and target are comparable resolution (~1 degree).

    A curvilinear ocean model grid's index-space seam (where it wraps from
    ~360 back to ~0) and its pole-fold (extra poles inserted mid-grid, e.g.
    a tripolar grid's Arctic cap) both make pcolormesh draw badly distorted
    polygons if plotted directly, on some grids badly enough to blank out
    or visibly smear a large chunk of the map - masking out individual
    seam-crossing cells (a prior approach here) still left the pole-cap
    distortion behind. Resampling onto an ordinary regular grid first
    sidesteps both problems at once: the output has no seam or fold to
    begin with, and is directly comparable in resolution/extent to the
    observational product's own regular grid.

    Nearest-neighbor has no notion of land, though, and will happily carry
    an ocean value straight across a continent to the far shore - any
    output point farther than max_gap_deg from the nearest real ocean
    point (a couple of native grid cells, at most) is masked back to NaN so
    land reappears instead of being smoothed over.

    Returns (regridded_field, grid_lat, grid_lon180), each 2-D via
    np.meshgrid, ready for a normal pcolormesh."""
    valid = np.isfinite(field) & np.isfinite(lon180) & np.isfinite(lat)
    points = np.column_stack([lon180[valid], lat[valid]])
    values = field[valid]

    grid_lon = np.arange(-180, 180, lon_step)
    grid_lat = np.arange(-90, 90, lat_step)
    glon, glat = np.meshgrid(grid_lon, grid_lat)

    tree = cKDTree(points)
    dist, idx = tree.query(np.column_stack([glon.ravel(), glat.ravel()]), k=1)
    regridded = values[idx].reshape(glon.shape)
    regridded[dist.reshape(glon.shape) > max_gap_deg] = np.nan

    return regridded, glat, glon


def color_bounds(field):
    """(vmin, vmax, cmap) for a field that might be signed (SSH: an anomaly
    around 0) or not (current speed: always >= 0) - a few extreme pixels
    (boundary currents; SSH highs/lows) would otherwise dominate the scale,
    so bounds are percentile-clipped either way. A signed field gets a
    zero-centered diverging colormap instead of viridis's implicit
    "0 = nothing here" reading, which would be wrong for SSH."""
    lo, hi = (float(v) for v in np.nanpercentile(field, [1, 99]))
    if lo < 0 < hi:
        m = max(abs(lo), abs(hi))
        return -m, m, "RdBu_r"
    return 0, hi, "viridis"


def plot_obs_vs_sim_whole_domain(obs_field, obs_lat, obs_lon180, sim_field, sim_lat, sim_lon180, field_label, title, out_path):
    """Whole-domain (global) observed-altimetry vs. simulated SSH comparison
    - not a THOR classification of either side. sim_field is first
    regridded onto a regular lat/lon grid via regrid_to_regular(): its
    curvilinear native grid, pcolormeshed directly, distorts badly at
    global scale (an index-space seam and, for a tripolar grid, a pole-fold
    can each smear or blank out a large chunk of the map) - regridding
    sidesteps that and makes the two panels directly comparable in
    resolution/extent too."""
    fig, axes = plt.subplots(1, 2, figsize=_figsize_for_aspects([2.0, 2.0], panel_width=8.0), constrained_layout=True)

    obs_vmin, obs_vmax, obs_cmap = color_bounds(obs_field)
    im0 = axes[0].pcolormesh(
        obs_lon180, obs_lat, obs_field, cmap=obs_cmap, vmin=obs_vmin, vmax=obs_vmax, shading="auto",
    )
    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(-90, 90)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title(f"Observed {field_label} (NOAA/AVISO altimetry)")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.03)
    cbar0.set_label(field_label)

    sim_regridded, sim_glat, sim_glon180 = regrid_to_regular(sim_lon180, sim_lat, sim_field)
    sim_vmin, sim_vmax, sim_cmap = color_bounds(sim_regridded)
    im1 = axes[1].pcolormesh(
        sim_glon180, sim_glat, sim_regridded, cmap=sim_cmap, vmin=sim_vmin, vmax=sim_vmax, shading="auto",
    )
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(-90, 90)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title(f"Simulated raw {field_label}")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.03)
    cbar1.set_label(field_label)

    fig.suptitle(title)
    plt.savefig(out_path)  # no bbox_inches="tight" - fights with constrained_layout
    plt.close()

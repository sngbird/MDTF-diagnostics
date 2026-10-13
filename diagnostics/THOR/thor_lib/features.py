"""Build THOR's 8-feature matrix (curl(tau), ssh, grad_ssh_x/y, depth,
grad_depth_x/y, coriolis f) from a case's staged SSH, wind stress, and
bathymetry files.

Regular near-CMIP-grid math only - the only grid type the framework's
staged CMIP6 Omon/Ofx output needs. (THOR/build_features.ipynb also has a
NEMO curvilinear-grid variant, for custom, non-CMIP model output outside
the POD.)
"""
import numpy as np
import xarray as xr


def _time_mean(path, var, time_dim):
    da = xr.open_dataset(path)[var]
    if time_dim in da.dims:
        da = da.mean(dim=time_dim)
    return da.values


LAT_VAR_CANDIDATES = ("lat", "latitude", "nav_lat")
LON_VAR_CANDIDATES = ("lon", "longitude", "nav_lon")


def find_coord_var(ds, candidates, kind):
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(
        f"None of {candidates} found as a {kind} variable in {list(ds.variables)} - "
        f"pass lat_var/lon_var explicitly for this model's naming convention."
    )


def _grid_spacing(lat, lon):
    """dy/dx (in meters) for np.gradient() on this (curvilinear) grid -
    shared by build() and any other caller needing an SSH/depth/tau gradient
    on the same grid (e.g. run_batch_local's obs-comparison, which needs an
    SSH gradient computed the exact same way, just on a different time
    window). dx comes from longitude steps along axis 1, as in the original
    THOR notebooks, taken mod 360 so a grid seam or the dateline doesn't
    produce a ~-360 deg step."""
    lat_diff = 1.111774765625000e05
    lon_diff = (np.roll(lon, axis=1, shift=-1) - lon) % 360
    lat_y = np.gradient(lat, axis=0) * lat_diff
    lon_x = np.abs(np.cos(lat * np.pi / 180)) * lat_diff * lon_diff
    return lat_y, lon_x


def _boxcar_kernel(width_cells):
    """Symmetric boxcar exactly width_cells wide, a fractional width going
    into the two end weights (4 -> [.5, 1, 1, 1, .5]), so the filter stays
    centred and nulls a period of width_cells."""
    if not np.isfinite(width_cells):  # degenerate row spacing
        return np.ones(1)
    half = (max(width_cells, 1.0) - 1) / 2
    m = int(np.ceil(half))
    k = np.ones(2 * m + 1)
    if m > half:
        k[0] = k[-1] = half - (m - 1)
    return k


def smooth_stress(tau, lat, lon, width_deg):
    """Boxcar-average a wind stress field over width_deg of latitude and of
    longitude, skipping land (NaN) cells and leaving them NaN.

    For coupled models whose atmosphere is coarser than their ocean: the
    coupler maps the atmosphere's stress onto the ocean grid, which leaves a
    small step at every atmosphere cell edge. The 20-year mean keeps it
    (the grid doesn't move), and the curl turns each step into a spike, so
    regime maps pick up a waffle pattern at the atmosphere's grid spacing.
    Averaging over one atmosphere cell removes it, and loses nothing real,
    since the atmosphere never resolved anything finer. The window is set in
    degrees and converted to cells row by row, so it follows grids whose
    row spacing varies with latitude (Mercator, ORCA's equatorial band)."""
    from scipy.ndimage import convolve1d

    ok = np.isfinite(tau)
    num, den = np.where(ok, tau, 0.0), ok.astype(float)
    dlat = np.nanmedian(np.abs(np.gradient(lat, axis=0)), axis=1)
    dlon = np.nanmedian((np.roll(lon, axis=1, shift=-1) - lon) % 360, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cells, y_cells = width_deg / dlon, width_deg / dlat

    for i in range(tau.shape[0]):  # along x: periodic in longitude
        k = _boxcar_kernel(x_cells[i])
        num[i] = convolve1d(num[i], k, mode="wrap")
        den[i] = convolve1d(den[i], k, mode="wrap")

    out_num, out_den = np.zeros_like(num), np.zeros_like(den)
    ny = tau.shape[0]
    for i in range(ny):  # along y: kernel width set by this row's spacing
        k = _boxcar_kernel(y_cells[i])
        m = len(k) // 2
        for j, w in zip(range(-m, m + 1), k):
            r = min(max(i + j, 0), ny - 1)
            out_num[i] += w * num[r]
            out_den[i] += w * den[r]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, out_num / out_den, np.nan)


def _grad(d, y, x):
    grad_y = np.gradient(d, axis=0) / y
    grad_x = np.gradient(d, axis=1) / x
    return grad_y, grad_x


def ssh_gradient(ssh, lat, lon):
    """grad_ssh_y, grad_ssh_x for an SSH field already on this module's
    (ny, nx) lat/lon grid - the same computation build() uses internally,
    exposed so callers with their own SSH field (e.g. a differently
    time-windowed mean) can compute a directly comparable gradient."""
    lat_y, lon_x = _grid_spacing(lat, lon)
    return _grad(ssh, lat_y, lon_x)


def build(
    zos_file, tauuo_file, tauvo_file, deptho_file,
    zos_var="zos", tauuo_var="tauuo", tauvo_var="tauvo", deptho_var="deptho",
    lat_var=None, lon_var=None, time_dim="time", land_fill_threshold=1e19,
    stress_smooth_deg=None,
):
    """Returns X (n_valid, 8), valid_mask (ny, nx), lat (ny, nx), lon (ny, nx).

    lat/lon are read from deptho_file's own coordinates - CMIP6 Ofx output
    carries them alongside the bathymetry field. Different modeling centers
    use different variable names for these (GFDL: lat/lon, UKESM: latitude/
    longitude, NEMO-family: nav_lat/nav_lon) - auto-detected from
    LAT_VAR_CANDIDATES/LON_VAR_CANDIDATES unless overridden explicitly.

    stress_smooth_deg: the atmosphere's grid spacing, for models whose
    stress carries its imprint (see smooth_stress); None leaves it as is.
    """
    ssh = _time_mean(zos_file, zos_var, time_dim)
    tauu = _time_mean(tauuo_file, tauuo_var, time_dim)
    tauv = _time_mean(tauvo_file, tauvo_var, time_dim)

    depth_ds = xr.open_dataset(deptho_file)
    depth = depth_ds[deptho_var].values.copy()
    depth[depth > land_fill_threshold] = np.nan
    resolved_lat_var = lat_var or find_coord_var(depth_ds, LAT_VAR_CANDIDATES, "latitude")
    resolved_lon_var = lon_var or find_coord_var(depth_ds, LON_VAR_CANDIDATES, "longitude")
    lat = depth_ds[resolved_lat_var].values
    lon = depth_ds[resolved_lon_var].values

    lat_y, lon_x = _grid_spacing(lat, lon)
    if stress_smooth_deg:
        tauu = smooth_stress(tauu, lat, lon, stress_smooth_deg)
        tauv = smooth_stress(tauv, lat, lon, stress_smooth_deg)

    omega = 7.2921e-5
    f = 2 * omega * np.sin(lat * np.pi / 180)

    grad_tauu_y, grad_tauu_x = _grad(tauu, lat_y, lon_x)
    grad_tauv_y, grad_tauv_x = _grad(tauv, lat_y, lon_x)
    curl_tau = (grad_tauv_x - grad_tauu_y) / 1032

    grad_ssh_y, grad_ssh_x = _grad(ssh, lat_y, lon_x)
    grad_depth_y, grad_depth_x = _grad(depth, lat_y, lon_x)

    missing = np.isnan(curl_tau * ssh * grad_ssh_x * grad_ssh_y * depth * grad_depth_x * grad_depth_y)
    valid_mask = ~missing

    X = np.stack(
        (
            curl_tau[valid_mask], ssh[valid_mask],
            grad_ssh_x[valid_mask], grad_ssh_y[valid_mask],
            depth[valid_mask], grad_depth_x[valid_mask], grad_depth_y[valid_mask],
            f[valid_mask],
        ),
        axis=1,
    )
    return X, valid_mask, lat, lon

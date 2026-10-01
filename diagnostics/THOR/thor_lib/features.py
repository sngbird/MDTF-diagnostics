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


def _grid_spacing(lat):
    """dy/dx (in meters) for np.gradient() on this (curvilinear) lat grid -
    shared by build() and any other caller needing an SSH/depth/tau gradient
    on the same grid (e.g. run_batch_local's obs-comparison, which needs an
    SSH gradient computed the exact same way, just on a different time
    window)."""
    lat_diff = 1.111774765625000e05
    lat_roll = np.roll(lat, axis=0, shift=-1)
    lon_diff = lat_roll - lat
    lat_y = np.gradient(lat, axis=0) * lat_diff
    lon_x = np.abs(np.cos(lat * np.pi / 180)) * lat_diff * lon_diff
    return lat_y, lon_x


def _grad(d, y, x):
    grad_y = np.gradient(d, axis=0) / y
    grad_x = np.gradient(d, axis=1) / x
    return grad_y, grad_x


def ssh_gradient(ssh, lat):
    """grad_ssh_y, grad_ssh_x for an SSH field already on this module's
    (ny, nx) lat grid - the same computation build() uses internally,
    exposed so callers with their own SSH field (e.g. a differently
    time-windowed mean) can compute a directly comparable gradient."""
    lat_y, lon_x = _grid_spacing(lat)
    return _grad(ssh, lat_y, lon_x)


def build(
    zos_file, tauuo_file, tauvo_file, deptho_file,
    zos_var="zos", tauuo_var="tauuo", tauvo_var="tauvo", deptho_var="deptho",
    lat_var=None, lon_var=None, time_dim="time", land_fill_threshold=1e19,
):
    """Returns X (n_valid, 8), valid_mask (ny, nx), lat (ny, nx), lon (ny, nx).

    lat/lon are read from deptho_file's own coordinates - CMIP6 Ofx output
    carries them alongside the bathymetry field. Different modeling centers
    use different variable names for these (GFDL: lat/lon, UKESM: latitude/
    longitude, NEMO-family: nav_lat/nav_lon) - auto-detected from
    LAT_VAR_CANDIDATES/LON_VAR_CANDIDATES unless overridden explicitly.
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

    lat_y, lon_x = _grid_spacing(lat)

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

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


def build(
    zos_file, tauuo_file, tauvo_file, deptho_file,
    zos_var="zos", tauuo_var="tauuo", tauvo_var="tauvo", deptho_var="deptho",
    lat_var="lat", lon_var="lon", time_dim="time", land_fill_threshold=1e19,
):
    """Returns X (n_valid, 8), valid_mask (ny, nx), lat (ny, nx), lon (ny, nx).

    lat/lon are read from deptho_file's own coordinates - CMIP6 Ofx output
    carries them alongside the bathymetry field.
    """
    ssh = _time_mean(zos_file, zos_var, time_dim)
    tauu = _time_mean(tauuo_file, tauuo_var, time_dim)
    tauv = _time_mean(tauvo_file, tauvo_var, time_dim)

    depth_ds = xr.open_dataset(deptho_file)
    depth = depth_ds[deptho_var].values.copy()
    depth[depth > land_fill_threshold] = np.nan
    lat = depth_ds[lat_var].values
    lon = depth_ds[lon_var].values

    lat_diff = 1.111774765625000e05
    lat_roll = np.roll(lat, axis=0, shift=-1)
    lon_diff = lat_roll - lat
    lat_y = np.gradient(lat, axis=0) * lat_diff
    lon_x = np.abs(np.cos(lat * np.pi / 180)) * lat_diff * lon_diff

    omega = 7.2921e-5
    f = 2 * omega * np.sin(lat * np.pi / 180)

    def grad(d, y, x):
        grad_y = np.gradient(d, axis=0) / y
        grad_x = np.gradient(d, axis=1) / x
        return grad_y, grad_x

    grad_tauu_y, grad_tauu_x = grad(tauu, lat_y, lon_x)
    grad_tauv_y, grad_tauv_x = grad(tauv, lat_y, lon_x)
    curl_tau = (grad_tauv_x - grad_tauu_y) / 1032

    grad_ssh_y, grad_ssh_x = grad(ssh, lat_y, lon_x)
    grad_depth_y, grad_depth_x = grad(depth, lat_y, lon_x)

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

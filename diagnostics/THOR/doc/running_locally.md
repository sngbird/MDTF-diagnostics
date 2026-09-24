# Running the THOR POD by hand (WSL2)

This walks through everything the automated setup does, so you can run a
single case yourself without the batch script. It assumes the environment
already built once (below) - re-running envs isn't needed after the first
time.

## One-time setup (already done on this machine)

```bash
# 1. Clone your fork
mkdir -p ~/mdtf && cd ~/mdtf
git clone https://github.com/sngbird/MDTF-diagnostics.git
cd MDTF-diagnostics

# 2. Install micromamba (no sudo needed - see below if bzip2 is missing)
mkdir -p ~/micromamba-bin && cd ~/micromamba-bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o mm.tar.bz2
python3 -c "
import bz2, tarfile, io
data = bz2.decompress(open('mm.tar.bz2','rb').read())
tarfile.open(fileobj=io.BytesIO(data)).extract('bin/micromamba', '.')
"
mv bin/micromamba . && rmdir bin && rm mm.tar.bz2 && chmod +x micromamba

# 3. Make micromamba's default env lookup point at where envs actually get built -
#    the framework's own `micromamba activate <name>` only searches
#    <micromamba_root>/envs by default
mkdir -p ~/conda_envs
ln -s ~/conda_envs ~/micromamba/envs

# 4. Build the framework's own env (~2GB)
cd ~/mdtf/MDTF-diagnostics
./src/conda/micromamba_env_setup.sh -e base \
    --micromamba_root ~/micromamba --micromamba_exe ~/micromamba-bin/micromamba \
    --env_dir ~/conda_envs

# 5. Build THOR's env. NOTE: this framework version's pod_setup.py always checks
#    the SHARED `_MDTF_python3_base` env for every POD's runtime_requirements -
#    despite src/conda/env_python3_<name>.yml files existing for some PODs (e.g.
#    WBC_var), there's no actual code path that picks a POD-specific env by name.
#    Until that's resolved one way or another (see the open question in this
#    repo's PR/commit history), THOR's deps get installed as their own env and
#    ALIASED to the name the framework expects:
cat > src/conda/env_python3_THOR_micromamba.yml <<'EOF'
name: _MDTF_python3_THOR
channels:
- conda-forge
dependencies:
- python=3.12
- pip=23.3.1
- pip:
    - numpy
    - scipy
    - xarray
    - netCDF4
    - scikit-learn
    - matplotlib
    - contourpy
    - pandas
    - h5py
    - tensorflow
    - keras
    - shap
EOF
./src/conda/micromamba_env_setup.sh -e python3_THOR \
    --micromamba_root ~/micromamba --micromamba_exe ~/micromamba-bin/micromamba \
    --env_dir ~/conda_envs
# fastshap needs this env var - its own metadata still references the deprecated
# "sklearn" PyPI alias, which refuses to install on its own otherwise
~/micromamba-bin/micromamba run -r ~/micromamba -p ~/conda_envs/_MDTF_python3_THOR \
    bash -c "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install fastshap"

# alias it as the env the framework actually checks (see note above)
ln -s ~/conda_envs/_MDTF_python3_THOR ~/conda_envs/_MDTF_python3_base
```

Everything below assumes this is already done.

## Running one case

### 1. Get real CMIP6 data locally

Use `~/mdtf_batch/fetch_and_run_batch.py` (see below) or fetch by hand with
`fetch_cmip_data_intake.ipynb` from the `THOR/` project - either way you need
local `.nc` files for `zos`, `tauuo`, `tauvo` (Omon) and `deptho` (Ofx,
possibly from a different experiment of the same model - it's a fixed grid
property).

### 2. Build a local intake-esm catalog

The framework's catalog query (see `src/data_sources.py`
`DataSourceBase.set_query` and `src/preprocessor.py` `query_catalog`) filters
**only** on `standard_name`, `frequency`, `realm`, and a regex built from the
case name matched against the `path` column - not on `source_id`/
`experiment_id`/`institution_id`. So:

- every row needs `standard_name`/`units`/`realm`/`frequency` matching
  `diagnostics/THOR/settings.jsonc`'s varlist **exactly**:

  | variable | standard_name | units | frequency |
  |---|---|---|---|
  | zos | `sea_surface_height_above_geoid` | m | mon |
  | tauuo | `downward_x_stress_at_sea_water_surface` | Pa | mon |
  | tauvo | `downward_y_stress_at_sea_water_surface` | Pa | mon |
  | deptho | `sea_floor_depth_below_geoid` | m | *(leave frequency off entirely for POD varlist; fine as "fx" in the catalog itself)* |

- the local directory holding a case's files should literally contain the
  case name string you use as the `case_list` key (e.g. name it
  `GFDL-ESM4.historical/`), so the path-regex match succeeds.
- a long CMIP6 timeseries is split across many files - include **every**
  chunk file as its own catalog row, not just one.

Minimal catalog CSV columns that matter: `standard_name, units, realm,
frequency, variable_id, path` (plus `version`, `table_id`, `member_id`,
`source_id`, `institution_id`, `experiment_id`, `grid_label`,
`variant_label`, `activity_id` for completeness/downstream use, though they
aren't part of the actual first-pass filter). Wrap the CSV in an intake-esm
JSON header (`catalog_file` pointing at the CSV, `esmcat_version`,
`attributes` listing the columns, `assets.column_name: "path"`).

### 3. Write a runtime config

```jsonc
{
  "pod_list": ["THOR"],
  "case_list": {
    "GFDL-ESM4.historical": {
      "convention": "CMIP",
      "startdate": "18500116",   // must match the data's actual first timestamp
      "enddate": "20141216"      // exactly - see the note below
    }
  },
  "DATA_CATALOG": "/home/you/mdtf_batch/catalogs/GFDL-ESM4.historical.json",
  "OBS_DATA_ROOT": "/home/you/mdtf_batch/obs_data",
  "WORK_DIR": "/home/you/mdtf_batch/wkdir/GFDL-ESM4.historical",
  "OUTPUT_DIR": "/home/you/mdtf_batch/wkdir/GFDL-ESM4.historical",
  "conda_root": "/home/you/micromamba",
  "micromamba_exe": "/home/you/micromamba-bin/micromamba",
  "conda_env_root": "/home/you/conda_envs",
  "run_pp": true,
  "translate_data": true,
  "save_ps": false,
  "large_file": true,
  "save_pp_data": true,
  "make_variab_tar": false,
  "make_multicase_figure_html": false,
  "overwrite": true,
  "user_pp_scripts": []
}
```

**On `startdate`/`enddate`:** the framework's monthly-frequency date-range
check (`src/preprocessor.py` `check_time_bounds`) has a real bug - it's
*supposed* to only compare year/month for `"mon"`/`"year"` frequency data,
but the comparison it actually runs (`freq in ['mon', 'year']`) always
evaluates `False` because `freq` there is a `DateFrequency` object (a
`timedelta` subclass), not a plain string, so it silently falls through to
an exact-datetime comparison instead. Practically: set `startdate`/`enddate`
to the **exact** first/last timestamp in your downloaded `zos` file (day
included), not just the 1st/end of a calendar month, or the run will fail
with `"dataset start (...) is after requested date range start (...)"` even
though your data does cover the requested period.

### 4. Run it

```bash
cd ~/mdtf/MDTF-diagnostics
./mdtf -f /path/to/your_config.jsonc
```

Output lands in `WORK_DIR/MDTF_output/THOR/`: `THOR.html`, `THOR.log`,
`model/*.png` (regime map, entropy map, both whole-domain and per-basin),
`model/netCDF/*.npy` (the raw regime/entropy arrays).

### 5. Turning on SHAP

`RUN_EXPLAIN` defaults to `false` in `settings.jsonc`'s `pod_env_vars`
because SHAP is expensive - even after fixing the SHAP background-sampling
bug (see the repo's commit history), a full 50-model run over one basin
still takes real time. There's currently no supported way to override a
`pod_env_vars` value from the runtime config `.jsonc` for a single run - the
only way to change it is editing the default in `settings.jsonc` itself.
`SHAP_BASIN` defaults to `"custom"`, a small user-defined window (see
`CUSTOM_BASIN_*` in `settings.jsonc`) rather than one of the large named
basins, to keep a full 50-model SHAP run practical.

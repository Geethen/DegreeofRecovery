# Agent Instructions

These notes are for any coding agent working in this repository, regardless of editor, host OS, or agent framework.

## Python Environment

- Prefer a reusable pixi environment named `geo` for new Python commands, tests, and smoke checks.
- Use `geo-python` instead of system `python`, `python3`, or a broken local virtualenv when the command is meant to run project code.
- Use the `earthengine` command exposed from the same pixi environment for Google Earth Engine authentication and CLI work.
- Do not assume `.venv/` or `venv-3.13/` exists. They were removed on one Linux workstation because they were incomplete. Versioned virtualenv directories are ignored with `venv-*/`.
- This repository still contains `pyproject.toml` and `uv.lock`; use `uv` only when intentionally rebuilding or validating the project-managed environment.

## Expected `geo` Packages

The shared `geo` pixi environment should include:

- `python=3.12`
- `earthengine-api`
- `duckdb`
- `pandas`
- `numpy`
- `geopandas`
- `shapely`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `requests`
- `cartopy`
- `pyarrow`
- `typer`
- `pip`
- `ipython`
- `jupyterlab`

## Create The Shared Pixi Environment

Run this once on a new machine.

Linux/macOS shell:

```bash
pixi global install \
  --environment geo \
  --expose geo-python=python \
  --expose geo-pip=pip \
  --expose geo-ipython=ipython \
  --expose geo-jupyter=jupyter \
  --expose earthengine=earthengine \
  python=3.12 \
  pip \
  ipython \
  jupyterlab \
  earthengine-api \
  duckdb \
  pandas \
  numpy \
  geopandas \
  shapely \
  scikit-learn \
  matplotlib \
  tqdm \
  requests \
  cartopy \
  pyarrow \
  typer
```

Windows PowerShell:

```powershell
pixi global install `
  --environment geo `
  --expose geo-python=python `
  --expose geo-pip=pip `
  --expose geo-ipython=ipython `
  --expose geo-jupyter=jupyter `
  --expose earthengine=earthengine `
  python=3.12 `
  pip `
  ipython `
  jupyterlab `
  earthengine-api `
  duckdb `
  pandas `
  numpy `
  geopandas `
  shapely `
  scikit-learn `
  matplotlib `
  tqdm `
  requests `
  cartopy `
  pyarrow `
  typer
```

## Verify The Environment

Linux/macOS shell:

```bash
command -v geo-python earthengine
geo-python --version
geo-python -c "import ee, duckdb, geopandas, shapely; print(ee.__version__, duckdb.__version__, geopandas.__version__, shapely.__version__)"
```

Windows PowerShell:

```powershell
where.exe geo-python
where.exe earthengine
geo-python --version
geo-python -c "import ee, duckdb, geopandas, shapely; print(ee.__version__, duckdb.__version__, geopandas.__version__, shapely.__version__)"
```

## Common Project Commands

Compile-check representative scripts:

```bash
geo-python -m py_compile v1-ecoregion/scripts/sampling/sample_reference_states.py v1/scripts/extraction/extract_alphaearth_embeddings.py v1/scripts/extraction/extract_test_site_embeddings.py
```

Run the ecoregion sampler:

```bash
geo-python v1-ecoregion/scripts/sampling/sample_reference_states.py --all
```

Authenticate Google Earth Engine:

```bash
earthengine authenticate
```

On Windows, run the same commands from PowerShell. Use forward slashes in repository-relative paths when possible; Python accepts them on Windows.

## Notes For Local Windows Agents

- Do not copy Linux absolute paths such as `/home/...` or `/data/...`; resolve paths relative to the local checkout.
- If `geo-python` is not found, check whether Pixi's global binary directory is on `PATH`. Common Windows location: `%USERPROFILE%\.pixi\bin`.
- If the `geo` environment exists but commands are missing, run `pixi global sync` or recreate the environment with the command above.
- Prefer PowerShell syntax for multiline commands on Windows; use backticks for line continuation.
# Agent Instructions

These notes are for any coding agent working in this repository, regardless of editor, host OS, or agent framework.

## Python Environment

- Prefer a reusable pixi environment named `geo` for new Python commands, tests, and smoke checks.
- Use `geo-python` instead of system `python`, `python3`, or a broken local virtualenv when the command is meant to run project code.
- Use `pandoc` (exposed from the same `geo` environment) for all document conversions. Do not use a system pandoc or download one via `pypandoc.download_pandoc()`.
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
- `pandoc`
- `pypandoc`

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
  --expose pandoc=pandoc \
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
  typer \
  pandoc \
  pypandoc
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
  --expose pandoc=pandoc `
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
  typer `
  pandoc `
  pypandoc
```

## PATH Setup

The pixi global binary directory must be on `PATH` for `geo-python`, `pandoc`, and `earthengine` to be callable without a full path.

**Windows** — add to your user or system PATH:
```
%USERPROFILE%\.pixi\bin
```

**Linux/macOS** — add to `~/.bashrc` or `~/.zshrc`:
```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

On the Linux VDI, if pixi itself is not yet installed:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
Then add `~/.pixi/bin` to `PATH` as above and recreate the `geo` environment with the Linux command above.

## Verify The Environment

Linux/macOS shell:

```bash
command -v geo-python earthengine pandoc
geo-python --version
pandoc --version
geo-python -c "import ee, duckdb, geopandas, shapely, pypandoc; print(ee.__version__, duckdb.__version__, geopandas.__version__, shapely.__version__, pypandoc.get_pandoc_version())"
```

Windows PowerShell:

```powershell
where.exe geo-python
where.exe earthengine
where.exe pandoc
geo-python --version
pandoc --version
geo-python -c "import ee, duckdb, geopandas, shapely, pypandoc; print(ee.__version__, duckdb.__version__, geopandas.__version__, shapely.__version__, pypandoc.get_pandoc_version())"
```

## Common Project Commands

Build the methods docx (skipping figure regeneration):

```bash
geo-python scripts/build_docx.py --skip-figures
```

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
- If `geo-python` or `pandoc` is not found, check whether `%USERPROFILE%\.pixi\bin` is on `PATH`.
- If the `geo` environment exists but commands are missing, run `pixi global sync` or recreate the environment with the command above.
- Prefer PowerShell syntax for multiline commands on Windows; use backticks for line continuation.
- Do not run `pypandoc.download_pandoc()` — pandoc is provided by the `geo` pixi environment.

## Notes For Linux VDI Agents

- The Linux VDI is the primary remote execution environment. Pixi and the `geo` environment should be set up under `~/.pixi/` exactly as on Windows.
- If `geo-python` or `pandoc` is not found, verify `~/.pixi/bin` is on `PATH` for the current shell session (`echo $PATH`).
- The Linux VDI may not persist shell profile changes across sessions; confirm `~/.bashrc` or `~/.profile` sources the PATH addition.
- Use `pixi global sync` if the manifest exists but binaries are missing (e.g. after a home-directory restore).
- Do not use `/home/geethen.singh/.pixi/envs/geo/bin/python` as a direct path in scripts — use `geo-python` so the command works on both Windows and Linux without modification.

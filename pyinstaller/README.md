# PyInstaller builds

## CLI

```
pip install -r requirements-dev.txt
pyinstaller pyinstaller/cli.spec
```

Binary is emitted into `dist/combine_equity/`. It runs the headless CLI entry (`scripts/combine_equity.py`).

## Streamlit dashboard

```
pip install -r requirements-dev.txt
pyinstaller pyinstaller/streamlit.spec
```

This uses the helper runner `scripts/run_streamlit_dashboard.py` to launch `app.py` without shelling out to the `streamlit` CLI. The resulting bundle lives in `dist/streamlit_dashboard/`.

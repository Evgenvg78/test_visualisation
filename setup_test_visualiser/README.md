# Packaging workspace for test_visualiser

This folder mirrors the runtime snapshot that becomes the standalone executable.

## Updating before a rebuild
1. Copy `app.py`, `src/`, and any assets (`help_data/`, additional configs) from the main repo into this directory so the packaged inputs reflect your latest changes.
2. Keep `help_data/dashboard_state.json` or regenerate it with the running app if you need to reset saved state.

## Preparing the environment
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -r build_requirements.txt
```

## Building the executable
```powershell
# run from inside setup_test_visualiser
.\build_exe.ps1 -Clean
```

The script calls PyInstaller with `--onefile` and includes `help_data/`. The resulting executable lands in `dist/test_visualiser.exe`.

## Quick iterative flow
1. Update sources (copy from the repo or edit directly inside this folder).
2. Run `.\build_exe.ps1` (add `-Clean` if PyInstaller artifacts should be dropped first).
3. Share `dist/test_visualiser.exe` with colleagues; it embeds the bundled python runtime.

If you need to tweak hidden imports or extra data directories, edit `build_exe.ps1` and rerun.

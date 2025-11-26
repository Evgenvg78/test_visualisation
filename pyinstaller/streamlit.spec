# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPEC).resolve().parents[1]
datas = []
help_dir = project_root / "help_data"
if help_dir.exists():
    datas.append((str(help_dir), "help_data"))

block_cipher = None

a = Analysis(
    [str(project_root / "scripts" / "run_streamlit_dashboard.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=["streamlit.web.bootstrap"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "flake8", "black"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="streamlit_dashboard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="streamlit_dashboard",
)

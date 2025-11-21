# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\user\\Documents\\spreader_pro\\code\\test_visualisation\\help_data', 'help_data'), ('C:\\Users\\user\\Documents\\spreader_pro\\code\\test_visualisation\\app.py', 'app_bundle'), ('C:\\Users\\user\\Documents\\spreader_pro\\code\\test_visualisation\\src', 'app_bundle\\src'), ('C:\\Users\\user\\Documents\\spreader_pro\\code\\test_visualisation\\help_data', 'app_bundle\\help_data')]
binaries = []
hiddenimports = ['moexalgo']
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('importlib_metadata')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('moexalgo')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\user\\Documents\\spreader_pro\\code\\test_visualisation\\setup_test_visualiser\\launch_app.py'],
    pathex=['C:\\Users\\user\\Documents\\spreader_pro\\code\\test_visualisation'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='test_visualiser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

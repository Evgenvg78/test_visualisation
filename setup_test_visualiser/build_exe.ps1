param(
    [switch]$Clean,
    [string]$ProjectRoot = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) '..'),
    [string]$EntryPoint = 'setup_test_visualiser/launch_app.py',
    [string]$PythonExe = 'python'
)

$setupDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$resolvedRoot = (Resolve-Path $ProjectRoot).Path
$entryPointFull = if ([System.IO.Path]::IsPathRooted($EntryPoint)) {
    (Resolve-Path $EntryPoint).Path
} else {
    (Resolve-Path (Join-Path $resolvedRoot $EntryPoint)).Path
}
$helpDataPath = Join-Path $resolvedRoot 'help_data'

Push-Location $setupDir
try {
    if ($Clean) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist,build,__pycache__
    }
    if (-not (Test-Path 'dist')) {
        New-Item -ItemType Directory -Path 'dist' | Out-Null
    }
    if (-not (Test-Path 'build')) {
        New-Item -ItemType Directory -Path 'build' | Out-Null
    }

    $additionalData = if (Test-Path $helpDataPath) { "$helpDataPath;help_data" } else { $null }

    $pyinstallerArgs = @(
        '--noconfirm',
        '--clean',
        '--onefile',
        '--name', 'test_visualiser',
        '--distpath', 'dist',
        '--workpath', 'build',
        '--paths', $resolvedRoot
    )

    if ($additionalData) {
        $pyinstallerArgs += @('--add-data', $additionalData)
    }

    $pyinstallerArgs += @(
        '--collect-all', 'streamlit',
        '--collect-all', 'importlib_metadata',
        '--collect-all', 'moexalgo',
        '--hidden-import', 'moexalgo'
    )

    $bundlePairs = @(
        @('app.py', 'app_bundle'),
        @('src', 'app_bundle\src'),
        @('help_data', 'app_bundle\help_data')
    )

    foreach ($pair in $bundlePairs) {
        $source = Join-Path $resolvedRoot $pair[0]
        if (Test-Path $source) {
            $pyinstallerArgs += @('--add-data', "$source;$($pair[1])")
        }
    }

    $pyinstallerArgs += $entryPointFull

    Write-Host "Running: $PythonExe -m PyInstaller $($pyinstallerArgs -join ' ')"
    & $PythonExe -m PyInstaller @pyinstallerArgs
}
finally {
    Pop-Location
}

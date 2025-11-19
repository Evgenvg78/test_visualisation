param(
    [switch]$Clean,
    [string]$EntryPoint = 'app.py'
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $root
try {
    if ($Clean) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist,build,__pycache__
    }

    $pyinstallerArgs = @(
        '--noconfirm',
        '--clean',
        '--onefile',
        '--name', 'test_visualiser',
        '--distpath', 'dist',
        '--workpath', 'build',
        '--add-data', 'help_data;help_data',
        '--paths', '.',
        $EntryPoint
    )

    Write-Host "Running: python -m PyInstaller $($pyinstallerArgs -join ' ')"
    python -m PyInstaller @pyinstallerArgs
}
finally {
    Pop-Location
}

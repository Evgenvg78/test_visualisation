# init_project.ps1
# Инициализация Python-проекта

# 2. Переходим в папку проекта (если ты запускаешь из неё — шаг просто отработает)
$projectPath = "C:\Users\user\Documents\spreader_pro\code\test_visualisation"
Set-Location $projectPath

Write-Host "Working in $projectPath"

# 3. Создаем venv, если его нет
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
} else {
    Write-Host "Virtual environment .venv already exists, skipping..."
}

# 4–5. Ставим пакеты ВНУТРИ venv без ручной активации
# используем .venv\Scripts\python
$venvPython = ".\.venv\Scripts\python.exe"

# Обновляем pip
& $venvPython -m pip install --upgrade pip

# БАЗОВЫЕ: pytest, black, isort, flake8, pre-commit
& $venvPython -m pip install pytest black isort flake8 pre-commit

# 6. Инициализация git, если его нет
if (-Not (Test-Path ".git")) {
    git init
    Write-Host "Git repo initialized."
} else {
    Write-Host "Git repo already exists, skipping git init..."
}

# 7. Создаем .gitignore, если нет
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.swp
.venv/
.env
.ipynb_checkpoints/
.DS_Store

# VSCode / Cursor
.vscode/
.history/
"@

if (-Not (Test-Path ".gitignore")) {
    $gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host ".gitignore created."
} else {
    Write-Host ".gitignore already exists, skipping..."
}

# 8. Создаем requirements.txt, если нет
if (-Not (Test-Path "requirements.txt")) {
    & $venvPython -m pip freeze | Out-File -FilePath "requirements.txt" -Encoding UTF8
    Write-Host "requirements.txt created."
} else {
    Write-Host "requirements.txt already exists, updating from venv..."
    & $venvPython -m pip freeze | Out-File -FilePath "requirements.txt" -Encoding UTF8
}

# 9. Базовая структура проекта
if (-Not (Test-Path "src")) {
    New-Item -ItemType Directory -Path "src" | Out-Null
    New-Item -ItemType File -Path "src\__init__.py" | Out-Null
}
if (-Not (Test-Path "tests")) {
    New-Item -ItemType Directory -Path "tests" | Out-Null
    'def test_basic(): assert True' | Out-File -FilePath "tests\test_basic.py" -Encoding UTF8
}
if (-Not (Test-Path "README.md")) {
    "# test_visualisation" | Out-File -FilePath "README.md" -Encoding UTF8
}

# 11. pre-commit конфиг
$preCommitContent = @"
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/PyCQA/flake8
    rev: 7.1.0
    hooks:
      - id: flake8
"@

if (-Not (Test-Path ".pre-commit-config.yaml")) {
    $preCommitContent | Out-File -FilePath ".pre-commit-config.yaml" -Encoding UTF8
    # ставим хуки
    & $venvPython -m pre_commit install
    Write-Host "pre-commit installed."
} else {
    Write-Host ".pre-commit-config.yaml already exists, skipping..."
}

# 12. VS Code / Cursor settings
if (-Not (Test-Path ".vscode")) {
    New-Item -ItemType Directory -Path ".vscode" | Out-Null
}
$settingsContent = @"
{
  "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
  "python.testing.pytestEnabled": true
}
"@
$settingsContent | Out-File -FilePath ".vscode\settings.json" -Encoding UTF8

Write-Host "Project initialized successfully."

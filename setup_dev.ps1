$venv = "venv"

if (-not (Test-Path -Path $venv)) {
    Write-Output "Creating virtual environment"
    & python -m venv "$venv"
}

# Check error
if (-not (Test-Path -Path $venv)) {
    Write-Error "Failed create virtual environment"
    Exit 1
}

Write-Output "Activate virtual env"
& "$venv\Scripts\Activate.ps1"


Write-Output "Upgrade Pip"

python -m pip install --upgrade pip
# Check error
if (-not $LASTEXITCODE -eq 0){
    Write-Error "Upgrade Pip failed."
    Exit $LASTEXITCODE
}

Write-Output "Install python packages"
pip install wheel
pip install -r requirements.txt

# Check error
if (-not $LASTEXITCODE -eq 0){
    Write-Error "Package installation failed."
    Exit $LASTEXITCODE
}

Write-Output "Running tests"
& .\run_tests.ps1

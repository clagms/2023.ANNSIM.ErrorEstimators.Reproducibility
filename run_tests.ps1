& venv/Scripts/Activate.ps1
$Env:NONSTOP = "ON"
$Env:PYTHONPATH = ".;src"
& unittest-parallel -s test -p '*.py' -v

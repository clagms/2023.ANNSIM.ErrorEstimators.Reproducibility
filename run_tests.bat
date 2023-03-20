CALL conda activate cosimulation
set NONSTOP=ON
set PYTHONPATH=".\src"
CALL unittest-parallel -s test -p '*.py' -v 
rem CALL python -m unittest discover test -p '*.py' -v
CALL conda deactivate
PAUSE

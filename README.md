# 2023.ANNSIM.ErrorEstimators.Reproducibility

## Contacts

For help setting up and running this reproducibility package, please open an issue, or contact the contributors directly.


## Code Description

### Parameter Set Results

* The selection of the initial conditions (`X0`) and the parameter values (`COSIM_PARAMS`) can be done in [./src/sys_params.py](./src/sys_params.py).  
There you can find the parameter sets that are listed in the paper. However, you can still modify them and play with system parameters of the mass-spring-damper systems. 

* The linear MSD system model is simulated in [./test/adaptive_cosim_tests.py](./test/adaptive_cosim_tests.py) file. 

* The function `test_run_adaptive_cosim_power_input` benchmarks adaptive scheduling algorithms (input estimation and power transfer) against static co-simulation sequences for the given linear problem.

* The requested plots can be switched on/off by setting the boolean variables `True` or `False`.

### Parameter Sweep Results

The parameter sweep results can be obtained by running the [./src/adaptive_cosim/parameter_sweep.py](./src/adaptive_cosim/parameter_sweep.py).
This file runs the system with random parameters to obtain the effect of the parameters on co-simulation accuracy and sequence selection.
However, this file produces only the raw simulation result data. 
In order to post-process the data and obtain the figures in the paper, the user must run [./src/adaptive_cosim/analyze_parameter_sweep_results.py](./src/adaptive_cosim/analyze_parameter_sweep_results.py).
This python file can post_process the data into meaningful expressions, e.g. extracting the best sequence at a time, as well as plotting them, e.g. plotting the frequency of adaptive algorithm staying as the best sequence.

## Running the Code

**Note:** Some plots require Latex. Make sure to have it installed on your system: https://www.latex-project.org/get/

### From command line

Setup environment (see [setup_dev.ps1](./setup_dev.ps1)):
1. Open command line in experiments folder.
2. Create a python virtual environment: `python -m venv venv`
3. Activate the python virtual environment. Run one of the scripts in `./venv/scripts/activate*`
4. Install packages required with pip: `pip install -r requirements.txt`

Run the unit test desired: 
1. Activate virtual environment
2. `python -m unittest test.cosimulation_solution.MyTestCase.test_show_l2_error`

Run all tests (see the file [run_tests.ps1](./run_tests.ps1)):\
1. Activate virtual environment
2. Set environment variables:
   1. NONSTOP = "ON"
   2. PYTHONPATH = ".;src"
3. Run unit tests: `unittest-parallel -s test -p '*.py' -v`

Run a python script from the src folder (e.g., [./src/adaptive_cosim/parameter_sweep.py](./src/adaptive_cosim/parameter_sweep.py) as mentioned above): 
1. Activate virtual environment
2. Run python script: `python -m src.adaptive_cosim.parameter_sweep`
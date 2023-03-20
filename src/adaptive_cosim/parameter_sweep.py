import os
import pickle
import shutil
from itertools import product
from multiprocessing import Pool
from numpy.random import default_rng
import numpy as np
from tqdm import tqdm

from adaptive_cosim.adaptive_cosim_msd import store_results_cosim_compare, store_results_cosim_all_compare
from fsutils import resource_file_path
from sys_params import X0


def run_experiments(x0, H, tf, total_cosims, param_per_task, out_dir, allp=False, parallel=False):
    with tqdm(total=total_cosims) as bar:
        if not parallel:
            for params in param_per_task:
                if allp:
                    store_results_cosim_all_compare(x0, H, tf, out_dir, params)
                else:
                    store_results_cosim_compare(x0, H, tf, out_dir, params)
                bar.update()
        else:
            with Pool(processes=2) as p:
                if allp:
                    multiple_results = [p.apply_async(store_results_cosim_all_compare, (x0, H, tf, out_dir, params)) for params in param_per_task]
                else:
                    multiple_results = [p.apply_async(store_results_cosim_compare, (x0, H, tf, out_dir, params)) for params in param_per_task]
                for res in multiple_results:
                    success = res.get()
                    assert success
                    bar.update()

def prepare_directory(out_dir, clean_dir, parameters=None):
    if os.path.exists(out_dir) and clean_dir:
        shutil.rmtree(out_dir)

    # Checks if there's been a mismatch in the parameters used to generate the experiment.
    readme_file = os.path.join(out_dir, "README.md")
    parameters_used_path = os.path.join(out_dir, "parameters.pickle")

    if parameters is not None:
        if os.path.exists(parameters_used_path):
            with open(parameters_used_path, "rb") as f:
                parameters_used = pickle.load(f)
                assert parameters_used == parameters
        else:
            with open(parameters_used_path, "wb") as f:
                pickle.dump(parameters, f)

    # with open(readme_file, 'w') as f:
    #     f.write(f"These experiments have been produced with the following parameters:\n"
    #             f"N = {parameters['N']}\n"
    #             f"max_wait_after_change = {parameters['max_wait_after_change']}\n"
    #             f"global_error = {parameters['global_error']}\n"
    #             f"fmus_only_extrapolate = {parameters['fmus_only_extrapolate']}\n"
    #             f"tf = {parameters['tf']}\n"
    #             f"H = {parameters['H']}")

def random_parameter(nsamples, N):
    # Partitions the exploration space.
    # Starts the process that will carried the individual simulations

    rng = default_rng()

    # Create a list of spaces for each category of parameters.
    m_spaces = [np.round(rng.uniform(1.0, 100.0, nsamples)) for _ in range(N)]
    d_spaces = [np.round(rng.uniform(1.0, 1000.0, nsamples)) for _ in range(N)]
    c_spaces = [np.round(rng.uniform(1.0, 10000.0, nsamples)) for _ in range(N)]
    ## Spaces for MSD_RIGHT are singletons
    dw_space = [np.round(rng.uniform(1.0, 1000.0, nsamples))]
    cw_space = [np.round(rng.uniform(1.0, 10000.0, nsamples))]

    # Then create a flat list by appending all previous lists. This is needed so that the product can be applied to that list.
    total_space = m_spaces + d_spaces + c_spaces + dw_space + cw_space

    param_per_task = list(product(*total_space))

    total_cosims = nsamples ** len(total_space)

    assert len(param_per_task) == total_cosims

    return param_per_task, total_cosims


if __name__ == '__main__':
    tf = 5.0
    H = 0.01
    N = 2
    nsamples = 3
    load_params = False
    filepath = "datasets/parameter_random_all_estimators_s2s1"
    out_dir = resource_file_path(filepath)
    clean_dir = False
    all_estimators = True
    prepare_directory(out_dir, clean_dir)
    param_filename = "past_parameters.pickle"
    if load_params:
        param_filepath = out_dir + "/" +param_filename
        FILE = open(param_filepath, 'rb')
        param_per_task = pickle.load(FILE)
        FILE.close()
        total_cosims = len(param_per_task)
    else:
        param_per_task, total_cosims = random_parameter(nsamples, N)
        param_filepath = out_dir + "/" +param_filename
        FILE = open(param_filepath,'wb')
        pickle.dump(param_per_task,FILE)
        FILE.close()

    current_iter = len(os.listdir(out_dir))-1
    param_per_task = param_per_task[current_iter:]
    run_experiments(X0, H, tf, total_cosims, param_per_task, out_dir
        , allp=all_estimators, parallel=True)
    # shutil.copy(param_filename, out_dir)

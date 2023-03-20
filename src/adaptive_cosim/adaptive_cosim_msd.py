import os
import pickle

from PyCosimLibrary.scenario import Connection, VarType, SignalType, OutputConnection, CosimScenario
from tqdm import tqdm

from adaptive_cosim.AdaptiveGSRunner import AdaptiveGSRunner
from adaptive_cosim.MSD1Adaptive import MSD1Adaptive
from adaptive_cosim.MSD2Adaptive import MSD2Adaptive
from cosim_msd_utils import CoupledMSD


def run_adaptive_cosim(msd1, msd2, sol, params, x0, H, tf, static_mode, msd1_first,
                       mass_normalize=False, progress=True, filter=False, cooldown=0, estimator='input'):
    order = 1

    # Clean up state of msd1 and msd2
    msd1.reset()
    msd2.reset()
    sol.reset()

    # MSD1 parameters
    m1_cons = params[0]
    d1_cons = params[1]
    c1_cons = params[2]
    # MSD2 parameters
    m2_cons = params[3]
    d2_cons = params[4]
    c2_cons = params[5]
    # coupling parameters
    dc_cons = params[6]
    cc_cons = params[7]

    # Output of msd1 connects to the input of msd2.
    msd1_out = Connection(value_type=VarType.REAL,
                          signal_type=SignalType.CONTINUOUS,
                          source_fmu=msd1,
                          target_fmu=msd2,
                          source_vr=[msd1.x1, msd1.v1],
                          target_vr=[msd2.x1, msd2.v1])

    # Output of msd2 connects to the input of msd1
    msd1_in = Connection(value_type=VarType.REAL,
                         signal_type=SignalType.CONTINUOUS,
                         source_fmu=msd2,
                         target_fmu=msd1,
                         source_vr=[msd2.fk],
                         target_vr=[msd1.fk])

    # This connection informs the library that we are also interested in plotting the values of msd2
    msd2_in = OutputConnection(value_type=VarType.REAL,
                                signal_type=SignalType.CONTINUOUS,
                                source_fmu=msd2,
                                source_vr=[msd2.x1, msd2.v1])


    msd2_out = OutputConnection(value_type=VarType.REAL,
                                signal_type=SignalType.CONTINUOUS,
                                source_fmu=msd2,
                                source_vr=[msd2.x2, msd2.v2])

    sol_out = OutputConnection(value_type=VarType.REAL,
                               signal_type=SignalType.CONTINUOUS,
                               source_fmu=sol,
                               source_vr=[sol.x1, sol.v1, sol.fk,
                                          sol.x2, sol.v2])

    msd1_fk_out = OutputConnection(value_type=VarType.REAL,
                                   signal_type=SignalType.CONTINUOUS,
                                   source_fmu=msd1,
                                   source_vr=[msd1.extrapolated_fk, msd1.error_fk_direct, msd1.error_fk_direct_filter,
                                              msd1.fk])

    msd2_fk_out = OutputConnection(value_type=VarType.REAL,
                                   signal_type=SignalType.CONTINUOUS,
                                   source_fmu=msd2,
                                   source_vr=[msd2.extrapolated_fk, msd2.error_fk_indirect,
                                              msd2.error_fk_indirect_filter, msd2.extrapolated_x1,
                                              msd2.extrapolated_v1])

    msd1_energy_out = OutputConnection(value_type=VarType.REAL,
                                   signal_type=SignalType.CONTINUOUS,
                                   source_fmu=msd1,
                                   source_vr=[msd1.total_energy, msd1.dissipated_energy, msd1.work_done])

    msd2_energy_out = OutputConnection(value_type=VarType.REAL,
                                   signal_type=SignalType.CONTINUOUS,
                                   source_fmu=msd2,
                                   source_vr=[msd2.total_energy, msd2.dissipated_energy, msd2.work_done])

    connections = [msd1_out, msd1_in]
    output_connections = [msd1_out, msd1_in, msd2_in, msd2_out, sol_out, msd1_fk_out, msd2_fk_out,
                          msd1_energy_out, msd2_energy_out]  # Controls which signals are plotted

    # Sets initial values and parameters

    real_parameters = {msd1:
                           ([msd1.c1,
                             msd1.d1,
                             msd1.m1,
                             msd1.x1,
                             msd1.v1,
                             msd1.input_approximation_order],
                            [c1_cons, d1_cons, m1_cons, x0[0], x0[1], order]),
                       msd2:
                           ([msd2.c2,
                             msd2.d2,
                             msd2.m2,
                             msd2.cc,
                             msd2.dc,
                             msd2.x2,
                             msd2.v2,
                             msd2.input_approximation_order],
                            [c2_cons, d2_cons, m2_cons, cc_cons, dc_cons, x0[2], x0[3], order]),
                       sol:
                           ([sol.c1,
                             sol.d1,
                             sol.m1,
                             sol.x1,
                             sol.v1,
                             sol.c2,
                             sol.d2,
                             sol.m2,
                             sol.cc,
                             sol.dc,
                             sol.x2,
                             sol.v2],
                            [c1_cons, d1_cons, m1_cons, x0[0], x0[1],
                             c2_cons, d2_cons, m2_cons, cc_cons, dc_cons, x0[2], x0[3]])
                       }

    fmus = [msd1, msd2, sol]

    scenario = CosimScenario(
        fmus=fmus,
        connections=connections,
        step_size=H,
        print_interval=H,
        stop_time=tf,
        outputs=output_connections,
        real_parameters=real_parameters)

    # Instantiate FMUs
    msd1.instantiate()
    msd2.instantiate()
    sol.instantiate()

    # Run cosim
    cooldown_period = cooldown
    master = AdaptiveGSRunner(msd1, msd2, sol, msd1_out, msd1_in, static_mode, msd1_first,
                              mass_normalize=mass_normalize, switch_cooldown_timer=cooldown_period,
                              filter=filter, error_estimator=estimator)

    if progress:
        with tqdm(total=int(tf / H)) as bar:
            def status(_):
                bar.update()

            results = master.run_cosim(scenario, status)
    else:
        results = master.run_cosim(scenario, None)

    msd1.terminate()
    msd2.terminate()
    sol.terminate()

    return results



def store_results_cosim_compare(x0, H, tf, out_dir, params):
    # Run 3 co-simulations, and stores the experiment results in a pickled file, whose name is derived from the parameters used.

    x0_string = '_'.join(map(lambda f: "{:.2f}".format(f), x0))
    params_string = '_'.join(map(lambda f: "{:.2f}".format(f), params))
    filename = "r_x0_{}_H_{:.2f}_{:.2f}_{}.pickle".format(x0_string, H, tf, params_string)

    filepath = os.path.join(out_dir, filename)

    assert not os.path.exists(filepath), f"File {filepath} already exists."

    global_error = True

    msd1 = MSD1Adaptive("msd1", global_error)
    msd2 = MSD2Adaptive("msd2", global_error)
    sol = CoupledMSD("sol")

    # static simulations
    results_s1_s2 = run_adaptive_cosim(msd1, msd2, sol,
                                       params, x0, H, tf,
                                       static_mode=True, msd1_first=True, progress=False)

    results_s2_s1 = run_adaptive_cosim(msd1, msd2, sol,
                                       params, x0, H, tf,
                                       static_mode=True, msd1_first=False, progress=False)

    # adaptive simulations
    error_estimator = 'input'
    mass_normalize = False
    filtered = True
    cooldown_period = 1
    msd1first = True
    results_adaptive = run_adaptive_cosim(msd1, msd2, sol,
                                                     params, x0, H, tf,
                                                     static_mode=False, msd1_first=msd1first,
                                                     mass_normalize=mass_normalize,
                                                     filter=filtered,
                                                     cooldown=cooldown_period,
                                                     estimator=error_estimator)

    results = {
        "s1_s2": results_s1_s2,
        "s2_s1": results_s2_s1,
        "adaptive": results_adaptive,
        "x0": x0,
        "H": H,
        "tf": tf,
        "params": params,
    }

    with open(filepath, "wb") as f:
        pickle.dump(results, f)

    return True

def store_results_cosim_all_compare(x0, H, tf, out_dir, params):
    # Run 3 co-simulations, and stores the experiment results in a pickled file, whose name is derived from the parameters used.

    x0_string = '_'.join(map(lambda f: "{:.2f}".format(f), x0))
    params_string = '_'.join(map(lambda f: "{:.2f}".format(f), params))
    filename = "r_x0_{}_H_{:.2f}_{:.2f}_{}.pickle".format(x0_string, H, tf, params_string)

    filepath = os.path.join(out_dir, filename)

    assert not os.path.exists(filepath), f"File {filepath} already exists."

    global_error = True

    # initialize msds
    msd1 = MSD1Adaptive("msd1", global_error)
    msd2 = MSD2Adaptive("msd2", global_error)
    sol = CoupledMSD("sol")

    # static simulations
    results_s1_s2 = run_adaptive_cosim(msd1, msd2, sol,
                                       params, x0, H, tf,
                                       static_mode=True, msd1_first=True, progress=False)

    results_s2_s1 = run_adaptive_cosim(msd1, msd2, sol,
                                       params, x0, H, tf,
                                       static_mode=True, msd1_first=False, progress=False)

    # initialize msds
    msd1 = MSD1Adaptive("msd1", global_error)
    msd2 = MSD2Adaptive("msd2", global_error)
    sol = CoupledMSD("sol")
    # adaptive simulations
    error_estimator = 'input'
    mass_normalize = False
    filtered = True
    cooldown_period = 1
    msd1first = False
    results_input = run_adaptive_cosim(msd1, msd2, sol,
                                             params, x0, H, tf,
                                             static_mode=False, msd1_first=msd1first,
                                             mass_normalize=mass_normalize,
                                             filter=filtered,
                                             cooldown=cooldown_period,
                                             estimator=error_estimator)

    # error_estimator = 'input'
    # mass_normalize = True
    # filtered = True
    # cooldown_period = 1
    # msd1first = True
    # results_input_norm = run_adaptive_cosim(msd1, msd2, sol,
    #                                          params, x0, H, tf,
    #                                          static_mode=False, msd1_first=msd1first,
    #                                          mass_normalize=mass_normalize,
    #                                          filter=filtered,
    #                                          cooldown=cooldown_period,
    #                                          estimator=error_estimator)

    # initialize msds
    msd1 = MSD1Adaptive("msd1", global_error)
    msd2 = MSD2Adaptive("msd2", global_error)
    sol = CoupledMSD("sol")
    # simulation settings
    error_estimator = 'power'
    mass_normalize = False
    filtered = True
    cooldown_period = 1
    msd1first = False
    results_power = run_adaptive_cosim(msd1, msd2, sol,
                                             params, x0, H, tf,
                                             static_mode=False, msd1_first=msd1first,
                                             mass_normalize=mass_normalize,
                                             filter=filtered,
                                             cooldown=cooldown_period,
                                             estimator=error_estimator)

    results = {
        "s1_s2": results_s1_s2,
        "s2_s1": results_s2_s1,
        "adap_input": results_input,
        # "adap_norm": results_input_norm,
        "adap_power": results_power,
        "x0": x0,
        "H": H,
        "tf": tf,
        "params": params,
    }

    with open(filepath, "wb") as f:
        pickle.dump(results, f)

    return True

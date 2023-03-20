import os.path
import unittest
import pickle

import matplotlib.pyplot as plt

from adaptive_cosim.MSD1Adaptive import MSD1Adaptive
from adaptive_cosim.MSD2Adaptive import MSD2Adaptive
from adaptive_cosim.adaptive_cosim_msd import run_adaptive_cosim
from cosim_msd_utils import get_analytical_error, CoupledMSD, extrapolation_error_aligned
from main import COSIM_PARAMS, X0
from test import NONSTOP
from test.TimedTest import TimedTest
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


class AdaptiveCosimTests(TimedTest):

    def test_microstep_convergence(self):

        # settings
        global_error = True
        load_ref = False
        load_sim = False
        msd1first = True

        tf = 5.0 if not NONSTOP else .10
        H = 0.01
        nsteps_msd1 = [10, 25, 50, 100]
        nsteps_msd2 = nsteps_msd1

        ref_filename = './ref_msd2_results.pickle'
        if load_ref and os.path.exists(ref_filename):
            msd1 = MSD1Adaptive("msd1", global_error, nstep=200)
            msd2 = MSD2Adaptive("msd2", global_error, nstep=200)
            # load static results
            with open(ref_filename, 'rb') as f:
                reference = pickle.load(f)
        else:
            msd1 = MSD1Adaptive("msd1", global_error, nstep=200)
            msd2 = MSD2Adaptive("msd2", global_error, nstep=200)
            sol = CoupledMSD("sol")
            reference = run_adaptive_cosim(msd1, msd2, sol,
                                       COSIM_PARAMS, X0, H, tf,
                                       static_mode=True, msd1_first=msd1first)

            # save static results
            file = open(ref_filename, 'wb')
            pickle.dump(reference, file)
            file.close()

        results = []
        if load_sim:
            for i in range(len(nsteps_msd1)):
                msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1[i])
                msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2[i])
                sol = CoupledMSD("sol")
            # load static results
            with open('./static_msd2_results.pickle', 'rb') as f:
                results = pickle.load(f)
        else:
            for i in range(len(nsteps_msd1)):
                msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1[i])
                msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2[i])
                sol = CoupledMSD("sol")
                results.append({})
                # obtain static results
                results[i] = run_adaptive_cosim(msd1, msd2, sol,
                                               COSIM_PARAMS, X0, H, tf,
                                               static_mode=True, msd1_first=msd1first)

            # save static results
            file = open('./static_msd2_results.pickle', 'wb')
            pickle.dump(results, file)
            file.close()

        plt.figure()
        plt.title('L2 error x1')
        lines = ['solid','dashed','dotted','dashdot']
        labels = []
        for i in range(len(nsteps_msd1)):
            labels.append(str(nsteps_msd1[i]))
            x1_cosim = np.array(results[i].out_signals[msd1.instanceName][msd1.x1])
            x1_sol = np.array(results[i].out_signals[sol.instanceName][sol.x1])
            # x1_sol = np.array(reference.out_signals[msd1.instanceName][msd1.x1])
            error = np.divide(np.abs(x1_cosim-x1_sol), np.abs(x1_sol)+1)
            plt.plot(results[i].timestamps, error, color='black', linestyle=lines[i], label='nstep='+labels[i])
        plt.xlabel('t [s]')
        plt.yscale('log')
        plt.legend()

        if not NONSTOP:
            plt.show()

    def test_standalone_adaptive_cosim(self):

        # settings
        global_error = True
        load_sim = False

        tf = 5.0 if not NONSTOP else .10
        H = 0.01
        nsteps_msd1 = 100
        nsteps_msd2 = 100

        msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
        msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
        sol = CoupledMSD("sol")

        results = {}
        static_filename = './static_msd2_results.pickle'
        if load_sim and os.path.exists(static_filename):
            # load static results
            with open(static_filename, 'rb') as f:
                results = pickle.load(f)
        else:
            # obtain static results
            results['MSD1->MSD2'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=True)

            results['MSD2->MSD1'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=False)
            # save static results
            file = open(static_filename, 'wb')
            pickle.dump(results, file)
            file.close()

        results['Adaptive'] = run_adaptive_cosim(msd1, msd2, sol,
                                                   COSIM_PARAMS, X0, H, tf,
                                                   static_mode=False, msd1_first=True)

        x1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.v1]
        x1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.v1]
        x1_cosim_adaptive = results['Adaptive'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim_adaptive = results['Adaptive'].out_signals[msd1.instanceName][msd1.v1]

        x1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x1]
        v1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v1]
        x2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x2]
        v2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v2]

        plt.figure()
        plt.title('x1')
        plt.plot(results['MSD1->MSD2'].timestamps, x1_cosim, color='blue', label="s1_s2")
        plt.plot(results['MSD1->MSD2'].timestamps, x1_cosim_flip, color='orange', label="s2_s1")
        plt.plot(results['Adaptive'].timestamps, x1_cosim_adaptive, color='green', label="adap")
        plt.plot(results['MSD1->MSD2'].timestamps, x1_sol, color='red', label="ref")
        plt.legend()

        plt.figure()
        plt.title('v1')
        plt.plot(results['MSD1->MSD2'].timestamps, v1_cosim, color='blue', label="s1_s2")
        plt.plot(results['MSD1->MSD2'].timestamps, v1_cosim_flip, color='orange', label="s2_s1")
        plt.plot(results['MSD1->MSD2'].timestamps, v1_sol, color='red', label="ref")
        plt.legend()

        # error wrt sol
        errors = {}
        x1_sol = np.array(x1_sol)
        x1_cosim = np.array(x1_cosim)
        x1_cosim_flip = np.array(x1_cosim_flip)
        errors['MSD1->MSD2'] = np.divide(np.abs(x1_cosim-x1_sol), np.abs(x1_sol)+1)
        errors['MSD2->MSD1'] = np.divide(np.abs(x1_cosim_flip-x1_sol), np.abs(x1_sol)+1)
        plt.figure()
        plt.title('L2 error x1')
        plt.plot(results['MSD1->MSD2'].timestamps, errors['MSD1->MSD2'], color='blue', label="s1_s2")
        plt.plot(results['MSD1->MSD2'].timestamps, errors['MSD2->MSD1'], color='orange', label="s2_s1")
        plt.xlabel('t [s]')
        plt.yscale('log')

        v1_sol = np.array(v1_sol)
        v1_cosim = np.array(v1_cosim)
        v1_cosim_flip = np.array(v1_cosim_flip)
        errors['MSD1->MSD2'] = np.divide(np.abs(v1_cosim - v1_sol), np.abs(v1_sol)+1)
        errors['MSD2->MSD1'] = np.divide(np.abs(v1_cosim_flip - v1_sol), np.abs(v1_sol)+1)
        plt.figure()
        plt.title('L2 error v1')
        plt.plot(results['MSD1->MSD2'].timestamps, errors['MSD1->MSD2'], color='blue', label="s1_s2")
        plt.plot(results['MSD1->MSD2'].timestamps, errors['MSD2->MSD1'], color='orange', label="s2_s1")
        plt.xlabel('t [s]')
        plt.yscale('log')

        if not NONSTOP:
            plt.show()

    def test_run_adaptive_cosim_input(self):

        # settings
        error_estimator = 'input'
        global_error = True
        logscale = True
        load_sim = False
        mass_normalize = False
        filtered = True
        cooldown_period = 1
        msd1first = False

        # static plots
        plot_actual_input_error = True
        plot_actual_state_error = True
        plot_actual_sequence = True
        # extrapolation
        plot_extrapolation = True
        plot_opportunity = False
        plot_opportunity_actual = False
        plot_validation = False
        plot_opportunity_adaptive = True
        # energy plots
        plot_energy_conservation = True
        plot_energy_deviation = False
        plot_energy_deviation_percentage = not plot_energy_deviation
        plot_energy_sequence = True
        # adaptive plots
        plot_adaptive_state = True
        plot_adaptive_place = False
        plot_adaptive_mode = True

        # simulation settings
        tf = 5.0 if not NONSTOP else 1.0
        H = 0.01
        nsteps_msd1 = 10
        nsteps_msd2 = nsteps_msd1

        # initalize msds
        msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
        msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
        sol = CoupledMSD("sol")

        results = {}
        static_filename = './static_msd2_results.pickle'
        if load_sim and os.path.exists(static_filename):
            # load static results
            with open(static_filename, 'rb') as f:
                results = pickle.load(f)
        else:
            # obtain static results
            results['MSD1->MSD2'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=True)

            results['MSD2->MSD1'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=False)
            # save static results
            file = open(static_filename, 'wb')
            pickle.dump(results, file)
            file.close()

        # initalize msds
        msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
        msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
        sol = CoupledMSD("sol")
        results['Adaptive'] = run_adaptive_cosim(msd1, msd2, sol,
                                                 COSIM_PARAMS, X0, H, tf,
                                                 static_mode=False, msd1_first=msd1first,
                                                 mass_normalize=mass_normalize,
                                                 filter=filtered,
                                                 cooldown=cooldown_period,
                                                 estimator=error_estimator)

        # ERROR WRT. ANALYTICAL SOLUTION
        x1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'x1')
        v1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'v1')
        x2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'x2')
        v2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'v2')

        x1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'x1')
        v1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'v1')
        x2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'x2')
        v2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'v2')

        x1_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd1, sol, 'x1')
        v1_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd1, sol, 'v1')
        x2_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd2, sol, 'x2')
        v2_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd2, sol, 'v2')

        x1_cosim_error_filt             = savgol_filter(x1_cosim_error, 51, 1)
        x1_cosim_error_flipped_filt     = savgol_filter(x1_cosim_error_flipped, 51, 1)
        x1_cosim_error_adaptive_filt    = savgol_filter(x1_cosim_adaptive_error, 51, 1)
        v1_cosim_error_filt             = savgol_filter(v1_cosim_error, 51, 1)
        v1_cosim_error_flipped_filt     = savgol_filter(v1_cosim_error_flipped, 51, 1)
        v1_cosim_error_adaptive_filt    = savgol_filter(v1_cosim_adaptive_error, 51, 1)

        x1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.v1]
        x1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.v1]
        x1_cosim_adaptive = results['Adaptive'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim_adaptive = results['Adaptive'].out_signals[msd1.instanceName][msd1.v1]

        x1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x1]
        v1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v1]
        x2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x2]
        v2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v2]
        Fk_sol = np.array([msd2.compute_F(X[0], X[1], X[2], X[3])
                  for X in zip(x1_sol, v1_sol, x2_sol, v2_sol)])

        # plt.figure()
        # plt.title('x')
        # plt.plot(results['MSD1->MSD2'].timestamps,x1_cosim,color='blue', label="s1_s2")
        # plt.plot(results['MSD1->MSD2'].timestamps,x1_cosim_flip,color='orange', label="s2_s1")
        # plt.plot(results['MSD1->MSD2'].timestamps,x1_sol,color='red', label="ref")
        # plt.legend()
        #
        # plt.figure()
        # plt.title('v')
        # plt.plot(results['MSD1->MSD2'].timestamps,v1_cosim,color='blue', label="s1_s2")
        # plt.plot(results['MSD1->MSD2'].timestamps,v1_cosim_flip,color='orange', label="s2_s1")
        # plt.plot(results['MSD1->MSD2'].timestamps,v1_sol,color='red', label="ref")
        # plt.legend()

        # MSD1 -> MSD2
        # Extrapolated inputs
        # Get Force signals that are being extrapolated.
        Fk_extrapolation = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.extrapolated_fk])
        # Compare extrapolation with error.
        Fk_cosim_direct_error_extrapolation =  Fk_extrapolation - Fk_sol

        # Interpolated inputs
        x1_inputs = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.x1])
        v1_inputs = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.v1])
        # x1_sol = results['MSD1->MSD2'].in_signals_before_step[sol.instanceName][sol.x1]
        # v1_sol = results['MSD1->MSD2'].in_signals_before_step[sol.instanceName][sol.v1]
        Fk_cosim_indirect = np.array([msd2.compute_F(X[0], X[1], X[2], X[3])
                                               for X in zip(x1_inputs, v1_inputs, x2_sol, v2_sol)])
        Fk_cosim_indirect_error = Fk_cosim_indirect - Fk_sol

        # MSD2 -> MSD1
        # Extrapolated inputs
        # Do the same as before, but for the opposite order.
        # Now x1 and v1 are the extrapolated values, so we start from those.
        x1_extrapolation = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.extrapolated_x1])
        v1_extrapolation = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.extrapolated_v1])

        # x1_extrapolation_error = x1_sol - x1_extrapolation
        # v1_extrapolation_error = v1_sol - v1_extrapolation

        Fk_cosim_indirect_flipped = np.array([msd2.compute_F(X[0], X[1], X[2], X[3])
                                               for X in zip(x1_extrapolation, v1_extrapolation, x2_sol, v2_sol)])
        Fk_cosim_indirect_error_flipped = Fk_cosim_indirect_flipped - Fk_sol

        # Fk_cosim_indirect_flipped = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.extrapolated_fk])
        # Fk_cosim_indirect_error_flipped = Fk_sol - Fk_cosim_indirect_flipped

        # Interpolated inputs
        Fk_cosim_direct = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.fk])
        Fk_cosim_direct_error_flipped = Fk_cosim_direct - Fk_sol
        
        Fk_cosim_direct_error_extrapolation[0] = 0.0
        Fk_cosim_indirect_error_flipped[0] = 0.0

        # Fk_direct_difference = Fk_cosim_direct_error_extrapolation - Fk_cosim_direct_error_flipped
        # Fk_indirect_difference = Fk_cosim_indirect_error_flipped - Fk_cosim_indirect_error

        Fk_cosim_direct_error_extrapolation = np.abs(Fk_cosim_direct_error_extrapolation)
        Fk_cosim_indirect_error = np.abs(Fk_cosim_indirect_error)
        Fk_cosim_direct_error_flipped = np.abs(Fk_cosim_direct_error_flipped)
        Fk_cosim_indirect_error_flipped = np.abs(Fk_cosim_indirect_error_flipped)

        Fk_direct_difference = Fk_cosim_direct_error_extrapolation - Fk_cosim_direct_error_flipped
        Fk_indirect_difference = Fk_cosim_indirect_error_flipped - Fk_cosim_indirect_error

        # Fk_direct_difference = np.abs(Fk_direct_difference )
        # Fk_indirect_difference = np.abs(Fk_indirect_difference )

        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)
        if plot_actual_input_error:
            plt.figure()

            plt.plot(results['MSD1->MSD2'].timestamps,
                     Fk_cosim_direct_error_extrapolation,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2 $F_{direct}$ (extrapolated)")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     Fk_cosim_indirect_error,
                     color='tab:green',
                     label=r"MSD1$\rightarrow$MSD2 $F_{indirect}$ (interpolated)")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     Fk_cosim_direct_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1 $F_{direct}$ (interpolated)")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     Fk_cosim_indirect_error_flipped,
                     color='tab:orange',
                     label=r"MSD2$\rightarrow$MSD1 $F_{indirect}$ (extrapolated)")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          Fk_cosim_indirect_error_flipped_2,
            #          label="MSD2->MSD1 Fk indirect (alternative)")

            # plt.title('Actual Input Error')
            plt.ylim([1e-4,1e-0])
            plt.legend(loc='lower center')
            plt.xlabel('t [s]')
            plt.ylabel('F [N]')
            plt.xlim(0.0, tf)
            if logscale:
                plt.ylabel(r'$\vert F_{sim} - F_{ref} \vert$ [N]')
                plt.yscale('log')
            plt.tight_layout()

        if plot_actual_state_error:
            plt.figure()

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_cosim_error_filt,
                     color='tab:blue',
                     linestyle='dashed',
                     label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_cosim_error_flipped_filt,
                     color='tab:red',
                     linestyle='dashed',
                     label=r"MSD2$\rightarrow$MSD1 SGF")

            if plot_adaptive_state:
                plt.plot(results['Adaptive'].timestamps,
                         x1_cosim_adaptive_error,
                         color='tab:green',
                         label="Adaptive")

                plt.plot(results['Adaptive'].timestamps,
                         x1_cosim_error_adaptive_filt,
                         color='tab:green',
                         linestyle='dashed',
                         label="Adaptive SGF")

            # plt.title('Absolute Error x1')
            plt.legend()
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(1e-5, 1e-2) # set B
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $x_1$ [m]')
            plt.tight_layout()

            plt.figure()

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_cosim_error_filt,
                     color='tab:blue',
                     linestyle='dashed',
                     label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_cosim_error_flipped_filt,
                     color='tab:red',
                     linestyle='dashed',
                     label=r"MSD2$\rightarrow$MSD1 SGF")

            if plot_adaptive_state:
                plt.plot(results['Adaptive'].timestamps,
                         v1_cosim_adaptive_error,
                         color='tab:green',
                         label="Adaptive")

                plt.plot(results['Adaptive'].timestamps,
                         v1_cosim_error_adaptive_filt,
                         color='tab:green',
                         linestyle='dashed',
                         label="Adaptive SGF")

            # plt.title('Absolute Error v1')
            plt.legend()
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(1e-5, 1e-2) # set B
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $v_1$ [m/s]')
            plt.tight_layout()

        x_seq_diff = x1_cosim_error_filt - x1_cosim_error_flipped_filt
        x_actual_sequence = np.zeros_like(x_seq_diff)
        v_seq_diff = v1_cosim_error_filt - v1_cosim_error_flipped_filt
        v_actual_sequence = np.zeros_like(v_seq_diff)
        for i in range(len(x_seq_diff)):
            if np.sign(x_seq_diff[i]) < 0:
                x_actual_sequence[i] = 0
            # elif np.sign(x_seq_diff[i]) == 0 > 1:
            #     x_actual_sequence[i] = x_actual_sequence[i-1]
            else:
                x_actual_sequence[i] = 1
            if np.sign(v_seq_diff[i]) < 0:
                v_actual_sequence[i] = 0
            else:
                v_actual_sequence[i] = 1
                
        if plot_actual_sequence:
            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['MSD1->MSD2'].timestamps,
                     x_actual_sequence,
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Actual Sequence for $x_1$")
            plt.tight_layout()

            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['MSD1->MSD2'].timestamps,
                     v_actual_sequence,
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            # plt.title(r"Actual Sequence for $v_1$")
            plt.tight_layout()

            # plt.figure()
            # plt.title('State Trajectories')
            #
            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.x1],
            #          label="~x1")
            #
            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.x2],
            #          label="~x2")
            #
            # plt.legend()
            # plt.xlabel('t [s]')

        x1_min_error = np.minimum(x1_cosim_error_filt,  x1_cosim_error_flipped_filt, x1_cosim_error_adaptive_filt)
        x1_max_error = np.maximum(x1_cosim_error_filt,  x1_cosim_error_flipped_filt, x1_cosim_error_adaptive_filt)
        x1_adap_place = np.zeros_like(x1_cosim_error_adaptive_filt)
        v1_min_error = np.minimum(v1_cosim_error_filt,  v1_cosim_error_flipped_filt, v1_cosim_error_adaptive_filt)
        v1_max_error = np.maximum(v1_cosim_error_filt,  v1_cosim_error_flipped_filt, v1_cosim_error_adaptive_filt)
        v1_adap_place = np.zeros_like(v1_cosim_error_adaptive_filt)
        for i in range(len(x1_adap_place)):
            # adaptive place for x1
            if x1_min_error[i] == x1_cosim_error_adaptive_filt[i]:
                x1_adap_place[i] = 0
            elif x1_max_error[i] == x1_cosim_error_adaptive_filt[i]:
                x1_adap_place[i] = 2
            else:
                x1_adap_place[i] = 1
            # adaptive place for v1
            if v1_min_error[i] == v1_cosim_error_adaptive_filt[i]:
                v1_adap_place[i] = 0
            elif v1_max_error[i] == v1_cosim_error_adaptive_filt[i]:
                v1_adap_place[i] = 2
            else:
                v1_adap_place[i] = 1

        if plot_adaptive_place:
            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['Adaptive'].timestamps,
                     x1_adap_place,
                     color='tab:blue')
            plt.yticks(np.arange(3), ['best', 'middle','worst'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Adaptive Place for $x_1$")
            plt.tight_layout()

            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['Adaptive'].timestamps,
                     v1_adap_place,
                     color='tab:blue')
            plt.yticks(np.arange(3), ['best', 'middle','worst'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Adaptive Place for $v_1$")
            plt.tight_layout()


        # total energy at the system
        total_energy_msd1       = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2       = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_flip  = np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_flip  = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_adap  = np.array(results['Adaptive'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_adap  = np.array(results['Adaptive'].out_signals[msd2.instanceName][msd2.total_energy])

        # energy lost at the system by dissipation
        energy_loss_msd1        = np.cumsum(np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2        = np.cumsum(np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_flip   = np.cumsum(np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_flip   = np.cumsum(np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_adap   = np.cumsum(np.array(results['Adaptive'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_adap   = np.cumsum(np.array(results['Adaptive'].out_signals[msd2.instanceName][msd2.dissipated_energy]))

        # energy conservation
        total_energy = total_energy_msd1 + total_energy_msd2 + energy_loss_msd1 + energy_loss_msd2
        total_energy_flip = total_energy_msd1_flip + total_energy_msd2_flip + energy_loss_msd1_flip + energy_loss_msd2_flip
        total_energy_adap = total_energy_msd1_adap + total_energy_msd2_adap + energy_loss_msd1_adap + energy_loss_msd2_adap

        # energy conservation filt
        total_energy_filt = savgol_filter(total_energy, 51, 1)
        total_energy_flip_filt = savgol_filter(total_energy_flip, 51, 1)
        total_energy_adap_filt = savgol_filter(total_energy_adap, 51, 1)

        if plot_energy_conservation:
            if plot_energy_deviation:
                # think about putting np.abs !!!
                total_energy_filt = total_energy_filt - total_energy[0]
                total_energy = total_energy - total_energy[0]
                total_energy_flip_filt = total_energy_flip_filt - total_energy_flip[0]
                total_energy_flip = total_energy_flip - total_energy_flip[0]
                total_energy_adap_filt = total_energy_adap_filt - total_energy_adap[0]
                total_energy_adap = total_energy_adap - total_energy_adap[0]
            elif plot_energy_deviation_percentage:
                total_energy_filt = (total_energy_filt - total_energy[0]) / total_energy[0]
                total_energy = (total_energy - total_energy[0]) / total_energy[0]
                total_energy_flip_filt = (total_energy_flip_filt - total_energy_flip[0]) / total_energy_flip[0]
                total_energy_flip = (total_energy_flip - total_energy_flip[0]) / total_energy_flip[0]
                total_energy_adap_filt = (total_energy_adap_filt - total_energy_adap[0]) / total_energy_adap[0]
                total_energy_adap = (total_energy_adap - total_energy_adap[0]) / total_energy_adap[0]


            plt.figure()
            plt.title('Energy Conservation')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     total_energy,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          total_energy_filt,
            #          color='tab:blue',
            #          linestyle='dashed',
            #          label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     total_energy_flip,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          total_energy_flip_filt,
            #          color='tab:red',
            #          linestyle='dashed',
            #          label=r"MSD2$\rightarrow$MSD1")

            if plot_adaptive_state:
                plt.plot(results['Adaptive'].timestamps,
                         total_energy_adap,
                         color='tab:green',
                         label=r"Adaptive")

                # plt.plot(results['Adaptive'].timestamps,
                #          total_energy_adap_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label=r"MSD1$\rightarrow$MSD2")

            plt.xlabel('t [s]')
            if plot_energy_deviation:
                plt.ylabel(r'E [J]')
            else:
                plt.ylabel(r'E [\%]')
            plt.xlim(0.0, tf)
            plt.legend()
            plt.tight_layout()

        # eng_seq_diff = np.abs(total_energy) - np.abs(total_energy_flip)
        eng_seq_diff = np.abs(total_energy) - np.abs(total_energy_flip)
        eng_actual_sequence = np.zeros_like(x_seq_diff)
        for i in range(len(x_seq_diff)):
            if np.sign(eng_seq_diff[i]) < 0:
                eng_actual_sequence[i] = 0
            # elif np.sign(x_seq_diff[i]) == 0 > 1:
            #     x_actual_sequence[i] = x_actual_sequence[i-1]
            else:
                eng_actual_sequence[i] = 1

        if plot_energy_sequence:
            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['MSD1->MSD2'].timestamps,
                     eng_actual_sequence,
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Sequence for energy conservation")
            plt.tight_layout()

        def get_extrapolated_input_estimation_error(actual, xtr, fmu, ind):
            local_input_error = actual - xtr
            if global_error:
                global_input_error = [local_input_error[0]]
                B = fmu.getSystemMatrix()[1]
                B = B[1, :]
                # print('B for system'+str(sys+1)+':')
                # print(B)
                for j in range(1, len(local_input_error)):
                    global_input_error.append(local_input_error[j] * H + global_input_error[j - 1] * B[ind] * H)
                err = np.array(global_input_error)
            else:
                err = local_input_error
            return err

        # MSD1->MSD2 order
        # S1
        Fk_direct = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.fk])
        Fk_extrapolation = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.extrapolated_fk])

        index = 0
        fk_direct_error1 = np.abs(get_extrapolated_input_estimation_error(Fk_direct, Fk_extrapolation, msd1, index))

        # S2
        index = 0
        x1_inputs = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.x1])
        x1_extrapolation = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.extrapolated_x1])
        x1_extrapolation[0] = x1_inputs[0]
        x1_extrapolation[1] = x1_inputs[1]
        x1_error = get_extrapolated_input_estimation_error(x1_inputs, x1_extrapolation, msd2, index)

        index = 1
        v1_inputs = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.v1])
        v1_extrapolation = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.extrapolated_v1])
        v1_extrapolation[0] = v1_inputs[0]
        v1_extrapolation[1] = v1_inputs[1]
        v1_error = get_extrapolated_input_estimation_error(v1_inputs, v1_extrapolation, msd2, index)

        cc = COSIM_PARAMS[7]
        dc = COSIM_PARAMS[6]

        def get_fk_indirect(x1, v1):
            return - cc * x1 - dc * v1

        fk_indirect_error1 = np.abs(get_fk_indirect(x1_error, v1_error))

        # MSD2-> MSD1 order
        # S1
        Fk_direct_flipped = np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.fk])
        Fk_interpolation = np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.extrapolated_fk])
        index = 0
        fk_flipped_direct_error1 = np.abs(get_extrapolated_input_estimation_error(Fk_direct, Fk_extrapolation, msd1, index))

        # S2
        index = 0
        x1_inputs = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.x1])
        # x1_inputs = np.insert(x1_inputs[:-1], 0, 0.0)
        x1_extrapolation = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.extrapolated_x1])
        x1_extrapolation[0] = x1_inputs[0]
        x1_flipped_error = get_extrapolated_input_estimation_error(x1_inputs, x1_extrapolation, msd2, index)

        index = 1
        v1_inputs = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.v1])
        # v1_inputs = np.insert(x1_inputs[:-1], 0, 0.0)
        v1_extrapolation = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.extrapolated_v1])
        v1_extrapolation[0] = v1_inputs[0]
        v1_flipped_error = get_extrapolated_input_estimation_error(v1_inputs, v1_extrapolation, msd2, index)

        fk_flipped_indirect_error1 = np.abs(get_fk_indirect(x1_flipped_error, v1_flipped_error))

        fk_direct_error = np.abs(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.error_fk_direct])
        fk_indirect_error = np.abs(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.error_fk_indirect])
        fk_flipped_indirect_error = np.abs(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.error_fk_indirect])
        fk_flipped_direct_error = np.abs(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.error_fk_direct])
        # normalization with mass (acceleration effect)
        if mass_normalize:
            fk_direct_error = fk_direct_error / msd1.state[msd1.m1]
            fk_indirect_error = fk_indirect_error / msd2.state[msd2.m2]
            fk_flipped_indirect_error = fk_flipped_indirect_error / msd2.state[msd2.m2]
            fk_flipped_direct_error = fk_flipped_direct_error / msd1.state[msd1.m1]

        if filtered:
            fk_direct_error_filter = np.abs(results['Adaptive'].out_signals[msd1.instanceName][msd1.error_fk_direct_filter])
            fk_indirect_error_filter = np.abs(results['Adaptive'].out_signals[msd2.instanceName][msd2.error_fk_indirect_filter])
            if mass_normalize:
                fk_direct_error_filter = fk_direct_error_filter / msd1.state[msd1.m1]
                fk_indirect_error_filter = fk_indirect_error_filter / msd2.state[msd2.m2]

        ## Adaptive extrapolation error
        Fk_actual = []
        Fk_hypo = []
        for i in range(len(results['Adaptive'].timestamps)):
            if results['Adaptive'].out_signals["_ma"][0][i]:
                Fk_actual.append(fk_direct_error[i])
                Fk_hypo.append(fk_indirect_error[i])
            else:
                Fk_actual.append(fk_flipped_indirect_error[i])
                Fk_hypo.append(fk_flipped_direct_error[i])

        if filtered:
            Fk_actual_filt = []
            Fk_hypo_filt = []
            for i in range(len(results['Adaptive'].timestamps)):
                if results['Adaptive'].out_signals["_ma"][0][i]:
                    Fk_actual_filt.append(fk_direct_error_filter[i])
                    Fk_hypo_filt.append(fk_indirect_error_filter[i])
                else:
                    Fk_actual_filt.append(fk_indirect_error_filter[i])
                    Fk_hypo_filt.append(fk_direct_error_filter[i])

        def rms_calc(res):
            ms = []
            summ = 0
            for i in range(len(res)):
                summ += res[i]**2
                rms = summ / (i+1)
                ms.append(np.sqrt(rms))
            return ms

        def func(x, a, b):
            x = np.array(x)
            return a*np.exp(-x)+b

        popt, pcov = curve_fit(func, results['MSD1->MSD2'].timestamps, fk_direct_error)
        popt1, pcov1 = curve_fit(func, results['MSD2->MSD1'].timestamps, fk_flipped_indirect_error)

        fk_direct_error_filt = savgol_filter(fk_direct_error, 51, 1)
        fk_flipped_indirect_error_filt = savgol_filter(fk_flipped_indirect_error, 51, 1)

        if plot_extrapolation:
            plt.figure()

            # if global_error:
            #     plt.title('Extrapolation - Global')
            # else:
            #     plt.title('Extrapolation - Local')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_direct_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2 F direct" )

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #         func(results['MSD1->MSD2'].timestamps, *popt),
            #         color='tab:blue',
            #         linestyle='dashed')

            plt.plot(results['MSD1->MSD2'].timestamps,
                    fk_direct_error_filt,       
                    color='tab:blue',
                    linestyle='dashed',
                    label=r"MSD1$\rightarrow$MSD2 F direct SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     fk_flipped_indirect_error,
                     color='tab:orange',
                     label=r"MSD2$\rightarrow$MSD1 F indirect")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          func(results['MSD2->MSD1'].timestamps, *popt1),
            #          color='tab:orange',
            #          linestyle='dashed')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_flipped_indirect_error_filt,
                     color='tab:orange',
                     linestyle='dashed',
                    label = r"MSD2$\rightarrow$MSD1 F indirect SGF")

            plt.legend()
            plt.xlabel('t [s]')
            if mass_normalize:
                ylabel_text = r'a $[m/s^2]$'
            else:
                ylabel_text = 'F [N]'

            plt.ylabel(ylabel_text)
            plt.xlim(0.0, tf)
            if logscale:
                plt.yscale('log')
            plt.tight_layout()

        if plot_opportunity:
            plt.figure(figsize=(7, 4))
            if global_error:
                plt.title('Opportunity Cost - Global')
            else:
                plt.title('Opportunity Cost - Local')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_direct_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2 F direct (Actual)")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     fk_flipped_indirect_error,
                     color='tab:orange',
                     label=r"MSD2$\rightarrow$MSD1 F indirect (Actual)")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_indirect_error,
                     color='tab:green',
                     linestyle='dashed',
                     label=r"MSD1$\rightarrow$MSD2 F indirect (Hypo)")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     fk_flipped_direct_error,
                     color='tab:red',
                     linestyle='dashed',
                     label=r"MSD2$\rightarrow$MSD1 F direct (Hypo)")

            plt.legend()
            plt.xlabel('t [s]')
            if mass_normalize:
                ylabel_text = 'a [m/s2]'
            else:
                ylabel_text = 'F [N]'
            plt.ylabel(ylabel_text)
            plt.xlim(0.0, tf)
            if logscale:
                plt.ylabel('log '+ylabel_text)
                plt.yscale('log')
            plt.tight_layout()

        # DIFF BETWEEN NEW AND OLD INPUT ESTIMATION CALCULATION
        if plot_validation:
            plt.figure()
            if global_error:
                plt.title('Opportunity Cost - Global')
            else:
                plt.title('Opportunity Cost - Local')
            # 1
            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_direct_error,
                     color='tab:blue',
                     label="MSD1->MSD2 F direct (Actual)")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_direct_error1,
                     color='tab:red',
                     label="MSD1->MSD2 F direct (Actual) Old")

            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel('F [N]')

            # 2
            plt.figure()
            if global_error:
                plt.title('Opportunity Cost - Global')
            else:
                plt.title('Opportunity Cost - Local')
            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_indirect_error,
                     color='tab:blue',
                     linestyle='dashed',
                     label="MSD1->MSD2 F indirect (Hypo)")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     fk_indirect_error1,
                     color='tab:red',
                     linestyle='dashed',
                     label="MSD1->MSD2 F indirect (Hypo) Old")

            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel('F [N]')

            # 3
            plt.figure()
            if global_error:
                plt.title('Opportunity Cost - Global')
            else:
                plt.title('Opportunity Cost - Local')
            plt.plot(results['MSD2->MSD1'].timestamps,
                     fk_flipped_indirect_error,
                     color='tab:orange',
                     label="MSD2->MSD1 F indirect (Actual)")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     fk_flipped_indirect_error1,
                     color='tab:green',
                     label="MSD2->MSD1 F indirect (Actual) Old")

            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel('F [N]')

            # 4
            plt.figure()
            if global_error:
                plt.title('Opportunity Cost - Global')
            else:
                plt.title('Opportunity Cost - Local')
            plt.plot(results['MSD2->MSD1'].timestamps,
                     fk_flipped_direct_error,
                     color='tab:orange',
                     linestyle='dashed',
                     label="MSD2->MSD1 F direct (Hypo)")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     fk_flipped_direct_error,
                     color='tab:green',
                     linestyle='dashed',
                     label="MSD2->MSD1 F direct (Hypo) Old")

            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel('F [N]')


        if plot_opportunity_adaptive:
            plt.figure(figsize=(7, 4))
            if global_error:
                plt.title('Opportunity Cost - Global')
            else:
                plt.title('Opportunity Cost - Local')

            plt.plot(results['Adaptive'].timestamps,
                     Fk_actual,
                     color='tab:blue',
                     label="Actual Error")

            plt.plot(results['Adaptive'].timestamps,
                     Fk_hypo,
                     color='tab:orange',
                     label="Hypo Error")

            if filtered:
                plt.plot(results['Adaptive'].timestamps,
                         Fk_actual_filt,
                         color='tab:blue',
                         linestyle='dashed',
                         label="Actual Error Filtered")

                plt.plot(results['Adaptive'].timestamps,
                         Fk_hypo_filt,
                         color='tab:orange',
                         linestyle='dashed',
                         label="Hypo Error Filtered")

            plt.legend()
            plt.xlabel('t [s]')
            if mass_normalize:
                ylabel_text = r'a $[m/s^2]$'
            else:
                ylabel_text = 'F [N]'
            plt.ylabel(ylabel_text)
            plt.xlim(0.0, tf)
            if logscale:
                plt.ylabel('log '+ylabel_text)
                plt.yscale('log')
            plt.tight_layout()

        if plot_adaptive_mode:
            plt.figure(num=None, figsize=(7, 2))
            # plt.title('Cosimulation Sequence')

            plt.plot(results['Adaptive'].timestamps,
                     np.abs(1-np.array(results['Adaptive'].out_signals["_ma"][0])),
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])

            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.tight_layout()

        if not NONSTOP:
            plt.show()

    def test_run_adaptive_cosim_power(self):

        # settings
        error_estimator = 'power'
        global_error = True
        logscale = True
        load_sim = False
        mass_normalize = False
        filtered = True
        cooldown_period = 1
        msd1first = False

        # static plots
        plot_actual_state_error = True
        plot_actual_sequence = True

        # energy plots
        plot_energy_conservation = True
        plot_energy_deviation = False
        plot_energy_deviation_percentage = not plot_energy_deviation
        plot_energy_sequence = True
        plot_power_transfer = True

        # adaptive plots
        plot_adaptive_state = True
        plot_adaptive_energy = True
        plot_adaptive_mode = True
        plot_adaptive_place = True

        # simulation settings
        tf = 5.0 if not NONSTOP else 1.0
        H = 0.01
        nsteps_msd1 = 10
        nsteps_msd2 = nsteps_msd1

        # initalize msds
        msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
        msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
        sol = CoupledMSD("sol")

        static_filename = './static_msd2_results.pickle'
        results = {}
        if load_sim and os.path.exists(static_filename):
            # load static results
            with open(static_filename, 'rb') as f:
                results = pickle.load(f)
        else:
            # obtain static results
            results['MSD1->MSD2'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=True)

            results['MSD2->MSD1'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=False)
            # save static results
            file = open(static_filename, 'wb')
            pickle.dump(results, file)
            file.close()

        results['Adaptive'] = run_adaptive_cosim(msd1, msd2, sol,
                                                 COSIM_PARAMS, X0, H, tf,
                                                 static_mode=False, msd1_first=msd1first,
                                                 mass_normalize=mass_normalize,
                                                 filter=filtered,
                                                 cooldown=cooldown_period,
                                                 estimator=error_estimator)

        # ERROR WRT. ANALYTICAL SOLUTION
        x1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'x1')
        v1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'v1')
        x2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'x2')
        v2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'v2')

        x1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'x1')
        v1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'v1')
        x2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'x2')
        v2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'v2')

        x1_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd1, sol, 'x1')
        v1_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd1, sol, 'v1')
        x2_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd2, sol, 'x2')
        v2_cosim_adaptive_error = get_analytical_error(results['Adaptive'], msd2, sol, 'v2')

        x1_cosim_error_filt             = savgol_filter(x1_cosim_error, 51, 1)
        x1_cosim_error_flipped_filt     = savgol_filter(x1_cosim_error_flipped, 51, 1)
        x1_cosim_error_adaptive_filt    = savgol_filter(x1_cosim_adaptive_error, 51, 1)
        v1_cosim_error_filt             = savgol_filter(v1_cosim_error, 51, 1)
        v1_cosim_error_flipped_filt     = savgol_filter(v1_cosim_error_flipped, 51, 1)
        v1_cosim_error_adaptive_filt    = savgol_filter(v1_cosim_adaptive_error, 51, 1)

        x1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.v1]
        x1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.v1]
        x1_cosim_adaptive = results['Adaptive'].out_signals[msd1.instanceName][msd1.x1]
        v1_cosim_adaptive = results['Adaptive'].out_signals[msd1.instanceName][msd1.v1]

        x1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x1]
        v1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v1]
        x2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x2]
        v2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v2]
        Fk_sol = np.array([msd2.compute_F(X[0], X[1], X[2], X[3])
                  for X in zip(x1_sol, v1_sol, x2_sol, v2_sol)])

        def get_L2_error(res, sim_system, var, ref_system, refvar, type='absolute'):
            sim = np.array(res.out_signals[sim_system][var])
            ref = np.array(res.out_signals[ref_system][refvar])
            if type == 'absolute':
                err = np.linalg.norm(sim - ref)
            else:
                err = np.linalg.norm(sim - ref) / np.linalg.norm(ref)
            return err

        state_var = 'v1'
        l2_error = get_L2_error(results['MSD1->MSD2'], msd1.instanceName, msd1.v1, sol.instanceName, sol.v1)
        l2_error_flip = get_L2_error(results['MSD2->MSD1'], msd1.instanceName, msd1.v1, sol.instanceName, sol.v1)
        l2_error_adap = get_L2_error(results['Adaptive'], msd1.instanceName, msd1.v1, sol.instanceName, sol.v1)
        print('L2-Error '+ state_var +': ')
        print('MSD1->MSD2: ' + format(l2_error,'.2E'))
        print('MSD2->MSD1: ' + format(l2_error_flip,'.2E'))
        print('Adaptive' + format(l2_error_adap,'.2E'))

        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)
        if plot_actual_state_error:
            plt.figure()

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_cosim_error_filt,
                     color='tab:blue',
                     linestyle='dashed',
                     label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_cosim_error_flipped_filt,
                     color='tab:red',
                     linestyle='dashed',
                     label=r"MSD2$\rightarrow$MSD1 SGF")
            if plot_adaptive_state:
                plt.plot(results['Adaptive'].timestamps,
                         x1_cosim_adaptive_error,
                         color='tab:green',
                         label="Adaptive")

                plt.plot(results['Adaptive'].timestamps,
                         x1_cosim_error_adaptive_filt,
                         color='tab:green',
                         linestyle='dashed',
                         label="Adaptive SGF")

            # plt.title('Absolute Error x1')
            plt.legend()
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            # plt.ylim(1e-5, 1e-2)
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $x_1$ [m]')
            plt.tight_layout()

            plt.figure()

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_cosim_error_filt,
                     color='tab:blue',
                     linestyle='dashed',
                     label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_cosim_error_flipped_filt,
                     color='tab:red',
                     linestyle='dashed',
                     label=r"MSD2$\rightarrow$MSD1 SGF")

            if plot_adaptive_state:
                plt.plot(results['Adaptive'].timestamps,
                         v1_cosim_adaptive_error,
                         color='tab:green',
                         label="Adaptive")

                plt.plot(results['Adaptive'].timestamps,
                         v1_cosim_error_adaptive_filt,
                         color='tab:green',
                         linestyle='dashed',
                         label="Adaptive SGF")

            # plt.title('Absolute Error v1')
            plt.legend()
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            # plt.ylim(1e-5, 1e-2)
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $v_1$ [m/s]')
            plt.tight_layout()

        x_seq_diff = x1_cosim_error_filt - x1_cosim_error_flipped_filt
        x_actual_sequence = np.zeros_like(x_seq_diff)
        v_seq_diff = v1_cosim_error_filt - v1_cosim_error_flipped_filt
        v_actual_sequence = np.zeros_like(v_seq_diff)
        for i in range(len(x_seq_diff)):
            if np.sign(x_seq_diff[i]) < 0:
                x_actual_sequence[i] = 0
            # elif np.sign(x_seq_diff[i]) == 0 > 1:
            #     x_actual_sequence[i] = x_actual_sequence[i-1]
            else:
                x_actual_sequence[i] = 1
            if np.sign(v_seq_diff[i]) < 0:
                v_actual_sequence[i] = 0
            else:
                v_actual_sequence[i] = 1
                
        if plot_actual_sequence:
            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['MSD1->MSD2'].timestamps,
                     x_actual_sequence,
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Actual Sequence for $x_1$")
            plt.tight_layout()

            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['MSD1->MSD2'].timestamps,
                     v_actual_sequence,
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            # plt.title(r"Actual Sequence for $v_1$")
            plt.tight_layout()

        x1_min_error = np.minimum(x1_cosim_error_filt, x1_cosim_error_flipped_filt, x1_cosim_error_adaptive_filt)
        x1_max_error = np.maximum(x1_cosim_error_filt, x1_cosim_error_flipped_filt, x1_cosim_error_adaptive_filt)
        x1_adap_place = np.zeros_like(x1_cosim_error_adaptive_filt)
        v1_min_error = np.minimum(v1_cosim_error_filt, v1_cosim_error_flipped_filt, v1_cosim_error_adaptive_filt)
        v1_max_error = np.maximum(v1_cosim_error_filt, v1_cosim_error_flipped_filt, v1_cosim_error_adaptive_filt)
        v1_adap_place = np.zeros_like(v1_cosim_error_adaptive_filt)
        for i in range(len(x1_adap_place)):
            # adaptive place for x1
            if x1_min_error[i] == x1_cosim_error_adaptive_filt[i]:
                x1_adap_place[i] = 0
            elif x1_max_error[i] == x1_cosim_error_adaptive_filt[i]:
                x1_adap_place[i] = 2
            else:
                x1_adap_place[i] = 1
            # adaptive place for v1
            if v1_min_error[i] == v1_cosim_error_adaptive_filt[i]:
                v1_adap_place[i] = 0
            elif v1_max_error[i] == v1_cosim_error_adaptive_filt[i]:
                v1_adap_place[i] = 2
            else:
                v1_adap_place[i] = 1

        if plot_adaptive_place:
            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['Adaptive'].timestamps,
                     x1_adap_place,
                     color='tab:blue')
            plt.yticks(np.arange(3), ['best', 'middle', 'worst'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Adaptive Place for $x_1$")
            plt.tight_layout()

            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['Adaptive'].timestamps,
                     v1_adap_place,
                     color='tab:blue')
            plt.yticks(np.arange(3), ['best', 'middle', 'worst'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Adaptive Place for $v_1$")
            plt.tight_layout()

        # total energy at the system
        total_energy_msd1       = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2       = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_flip  = np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_flip  = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_adap  = np.array(results['Adaptive'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_adap  = np.array(results['Adaptive'].out_signals[msd2.instanceName][msd2.total_energy])

        # energy lost at the system by dissipation
        energy_loss_msd1        = np.cumsum(np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2        = np.cumsum(np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_flip   = np.cumsum(np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_flip   = np.cumsum(np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_adap   = np.cumsum(np.array(results['Adaptive'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_adap   = np.cumsum(np.array(results['Adaptive'].out_signals[msd2.instanceName][msd2.dissipated_energy]))

        # energy conservation
        total_energy = total_energy_msd1 + total_energy_msd2 + energy_loss_msd1 + energy_loss_msd2
        total_energy_flip = total_energy_msd1_flip + total_energy_msd2_flip + energy_loss_msd1_flip + energy_loss_msd2_flip
        total_energy_adap = total_energy_msd1_adap + total_energy_msd2_adap + energy_loss_msd1_adap + energy_loss_msd2_adap

        # energy conservation filt
        total_energy_filt = savgol_filter(total_energy, 51, 1)
        total_energy_flip_filt = savgol_filter(total_energy_flip, 51, 1)
        total_energy_adap_filt = savgol_filter(total_energy_adap, 51, 1)

        if plot_energy_conservation:
            if plot_energy_deviation:
                # think about putting np.abs !!!
                total_energy = total_energy - total_energy[0]
                total_energy_filt = total_energy_filt - total_energy_filt[0]
                total_energy_flip = total_energy_flip - total_energy_flip[0]
                total_energy_flip_filt = total_energy_flip_filt - total_energy_flip_filt[0]
                total_energy_adap = total_energy_adap - total_energy_adap[0]
                total_energy_adap_filt = total_energy_adap_filt - total_energy_adap_filt[0]
            elif plot_energy_deviation_percentage:
                total_energy = (total_energy - total_energy[0]) / total_energy[0]
                total_energy_filt = (total_energy_filt - total_energy_filt[0]) / total_energy_filt[0]
                total_energy_flip = (total_energy_flip - total_energy_flip[0]) / total_energy_flip[0]
                total_energy_flip_filt = (total_energy_flip_filt - total_energy_flip_filt[0]) / total_energy_flip_filt[0]
                total_energy_adap = (total_energy_adap - total_energy_adap[0]) / total_energy_adap[0]
                total_energy_adap_filt = (total_energy_adap_filt - total_energy_adap_filt[0]) / total_energy_adap_filt[0]


            plt.figure()
            plt.title('Energy Conservation')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     total_energy,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     total_energy_flip,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")
            if plot_adaptive_state:
                plt.plot(results['Adaptive'].timestamps,
                         total_energy_adap,
                         color='tab:green',
                         label=r"Adaptive")

            plt.xlabel('t [s]')
            if plot_energy_deviation:
                plt.ylabel(r'E [J]')
            else:
                plt.ylabel(r'E [\%]')
            plt.xlim(0.0, tf)
            plt.legend()
            plt.tight_layout()

        # eng_seq_diff = np.abs(total_energy) - np.abs(total_energy_flip)
        eng_seq_diff = np.abs(total_energy) - np.abs(total_energy_flip)
        eng_actual_sequence = np.zeros_like(x_seq_diff)
        for i in range(len(x_seq_diff)):
            if np.sign(eng_seq_diff[i]) < 0:
                eng_actual_sequence[i] = 0
            # elif np.sign(x_seq_diff[i]) == 0 > 1:
            #     x_actual_sequence[i] = x_actual_sequence[i-1]
            else:
                eng_actual_sequence[i] = 1

        if plot_energy_sequence:
            plt.figure(num=None, figsize=(7, 2))
            plt.plot(results['MSD1->MSD2'].timestamps,
                     eng_actual_sequence,
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])
            plt.xlim(0.0, tf)
            plt.xlabel('t [s]')
            plt.title(r"Sequence for energy conservation")
            plt.tight_layout()

        # work done by the system
        work_done_msd1          = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.work_done]) / H
        work_done_msd2          = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.work_done]) / H
        work_done_msd1_flip     = np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.work_done]) / H
        work_done_msd2_flip     = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.work_done]) / H
        work_done_msd1_adap     = np.array(results['Adaptive'].out_signals[msd1.instanceName][msd1.work_done]) / H
        work_done_msd2_adap     = np.array(results['Adaptive'].out_signals[msd2.instanceName][msd2.work_done]) / H

        if plot_power_transfer:
            plt.figure()
            # plt.title(r'Power Transfer MSD1$\rightarrow$MSD2')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     work_done_msd1,
                     color='tab:blue',
                     label=r"S1$\rightarrow$S2")

            plt.plot(results['MSD1->MSD2'].timestamps,
                     work_done_msd2,
                     color='tab:red',
                     label=r"S2$\rightarrow$S1")

            plt.xlabel('t [s]')
            plt.ylabel('W [Watt]')
            plt.xlim(0.0, tf)
            plt.legend()
            plt.tight_layout()

            plt.figure()
            # plt.title(r'Power Transfer MSD2$\rightarrow$MSD1')

            plt.plot(results['MSD2->MSD1'].timestamps,
                     work_done_msd1_flip,
                     color='tab:blue',
                     label=r"S1$\rightarrow$S2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     work_done_msd2_flip,
                     color='tab:red',
                     label=r"S2$\rightarrow$S1")

            plt.xlabel('t [s]')
            plt.ylabel('W [Watt]')
            plt.xlim(0.0, tf)
            plt.legend()
            plt.tight_layout()

            plt.figure(figsize=(7, 4))
            plt.title('Power Transfer Adaptive')

            plt.plot(results['Adaptive'].timestamps,
                     work_done_msd1_adap,
                     color='tab:blue',
                     label=r"S1$\rightarrow$S2")

            plt.plot(results['Adaptive'].timestamps,
                     work_done_msd2_adap,
                     color='tab:red',
                     label=r"S2$\rightarrow$S1")

            plt.xlabel('t [s]')
            plt.ylabel('W [Watt]')
            plt.xlim(0.0, tf)
            plt.tight_layout()

        if plot_adaptive_mode:
            plt.figure(num=None, figsize=(7, 2))
            # plt.title('Cosimulation Sequence')

            plt.plot(results['Adaptive'].timestamps,
                     np.abs(1-np.array(results['Adaptive'].out_signals["_ma"][0])),
                     color='tab:blue')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])

            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.tight_layout()

        if not NONSTOP:
            plt.show()

    def test_run_adaptive_cosim_all(self):
        # simulation settings
        global_error = True
        load_sim = False
        load_adap = False
        logscale = True

        # static plots
        plot_actual_input_error = False
        plot_actual_state_error = True
        plot_actual_sequence = True
        # extrapolation
        plot_extrapolation = True
        plot_opportunity = False
        plot_opportunity_actual = False
        plot_validation = False
        plot_opportunity_adaptive = True
        # energy plots
        plot_energy_conservation = True
        plot_energy_deviation = False
        plot_energy_deviation_percentage = not plot_energy_deviation
        plot_energy_sequence = True
        # adaptive plots
        plot_adaptive_state = True
        plot_adaptive_mode = True
        plot_solution_benchmark = True

        # simulation settings
        tf = 5.0 if not NONSTOP else 1.0
        H = 0.01
        nsteps_msd1 = 10
        nsteps_msd2 = nsteps_msd1

        # initalize msds
        msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
        msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
        sol = CoupledMSD("sol")

        results = {}
        static_filename = './static_msd2_results.pickle'
        if load_sim and os.path.exists(static_filename):
            # load static results
            with open(static_filename, 'rb') as f:
                results = pickle.load(f)
        else:
            # obtain static results
            results['MSD1->MSD2'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=True)

            results['MSD2->MSD1'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=False)
            # save static results
            file = open(static_filename, 'wb')
            pickle.dump(results, file)
            file.close()

        # simulation settings
        adap_filename = './adap_msd2_results.pickle'
        if load_adap and os.path.exists(adap_filename):
            # load adap results
            with open(adap_filename, 'rb') as f:
                results = pickle.load(f)
        else:
            error_estimator = 'input'
            logscale = True
            mass_normalize = False
            filtered = True
            cooldown_period = 1
            msd1first = True
            results['Adaptive Input'] = run_adaptive_cosim(msd1, msd2, sol,
                                                     COSIM_PARAMS, X0, H, tf,
                                                     static_mode=False, msd1_first=msd1first,
                                                     mass_normalize=mass_normalize,
                                                     filter=filtered,
                                                     cooldown=cooldown_period,
                                                     estimator=error_estimator)

            # simulation settings
            error_estimator = 'input'
            logscale = True
            mass_normalize = True
            filtered = True
            cooldown_period = 1
            msd1first = True
            results['Adaptive Input Norm'] = run_adaptive_cosim(msd1, msd2, sol,
                                                     COSIM_PARAMS, X0, H, tf,
                                                     static_mode=False, msd1_first=msd1first,
                                                     mass_normalize=mass_normalize,
                                                     filter=filtered,
                                                     cooldown=cooldown_period,
                                                     estimator=error_estimator)

            # simulation settings
            error_estimator = 'power'
            logscale = True
            mass_normalize = False
            filtered = True
            cooldown_period = 1
            msd1first = True
            results['Adaptive Power'] = run_adaptive_cosim(msd1, msd2, sol,
                                                     COSIM_PARAMS, X0, H, tf,
                                                     static_mode=False, msd1_first=msd1first,
                                                     mass_normalize=mass_normalize,
                                                     filter=filtered,
                                                     cooldown=cooldown_period,
                                                     estimator=error_estimator)
            # save adap results
            file = open(adap_filename, 'wb')
            pickle.dump(results, file)
            file.close()

        # ERROR WRT. ANALYTICAL SOLUTION
        x1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'x1')
        v1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'v1')
        x2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'x2')
        v2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'v2')

        x1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'x1')
        v1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'v1')
        x2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'x2')
        v2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'v2')

        x1_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd1, sol, 'x1')
        v1_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd1, sol, 'v1')
        x2_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd2, sol, 'x2')
        v2_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd2, sol, 'v2')

        x1_cosim_adaptive_norm_error = get_analytical_error(results['Adaptive Input Norm'], msd1, sol, 'x1')
        v1_cosim_adaptive_norm_error = get_analytical_error(results['Adaptive Input Norm'], msd1, sol, 'v1')
        x2_cosim_adaptive_norm_error = get_analytical_error(results['Adaptive Input Norm'], msd2, sol, 'x2')
        v2_cosim_adaptive_norm_error = get_analytical_error(results['Adaptive Input Norm'], msd2, sol, 'v2')

        x1_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd1, sol, 'x1')
        v1_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd1, sol, 'v1')
        x2_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd2, sol, 'x2')
        v2_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd2, sol, 'v2')

        x1_cosim_error_filt             = savgol_filter(x1_cosim_error, 51, 1)
        x1_cosim_error_flipped_filt     = savgol_filter(x1_cosim_error_flipped, 51, 1)
        x1_cosim_error_adaptive_input_filt      = savgol_filter(x1_cosim_adaptive_input_error, 51, 1)
        x1_cosim_error_adaptive_norm_filt       = savgol_filter(x1_cosim_adaptive_norm_error, 51, 1)
        x1_cosim_error_adaptive_power_filt      = savgol_filter(x1_cosim_adaptive_power_error, 51, 1)

        v1_cosim_error_filt             = savgol_filter(v1_cosim_error, 51, 1)
        v1_cosim_error_flipped_filt     = savgol_filter(v1_cosim_error_flipped, 51, 1)
        v1_cosim_error_adaptive_input_filt    = savgol_filter(v1_cosim_adaptive_input_error, 51, 1)
        v1_cosim_error_adaptive_norm_filt    = savgol_filter(v1_cosim_adaptive_norm_error, 51, 1)
        v1_cosim_error_adaptive_power_filt    = savgol_filter(v1_cosim_adaptive_power_error, 51, 1)

        # x1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.x1]
        # v1_cosim = results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.v1]
        # x1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.x1]
        # v1_cosim_flip = results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.v1]
        # x1_cosim_adaptive_input = results['Adaptive Input'].out_signals[msd1.instanceName][msd1.x1]
        # v1_cosim_adaptive_input = results['Adaptive Input'].out_signals[msd1.instanceName][msd1.v1]
        # x1_cosim_adaptive_norm = results['Adaptive Input Norm'].out_signals[msd1.instanceName][msd1.x1]
        # v1_cosim_adaptive_norm = results['Adaptive Input Norm'].out_signals[msd1.instanceName][msd1.v1]
        # x1_cosim_adaptive_power = results['Adaptive Power'].out_signals[msd1.instanceName][msd1.x1]
        # v1_cosim_adaptive_power = results['Adaptive Power'].out_signals[msd1.instanceName][msd1.v1]

        x1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x1]
        v1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v1]
        x2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x2]
        v2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v2]

        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)
        if plot_actual_state_error:
            plt.figure(figsize=(7, 6))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          x1_cosim_error_filt,
            #          color='tab:blue',
            #          linestyle='dashed',
            #          label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          x1_cosim_error_flipped_filt,
            #          color='tab:red',
            #          linestyle='dashed',
            #          label=r"MSD2$\rightarrow$MSD1 SGF")

            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         x1_cosim_adaptive_input_error,
                         color='tab:green',
                         label="Adap Input")

                # plt.plot(results['Adaptive Input'].timestamps,
                #          x1_cosim_error_adaptive_input_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label="Adap Input SGF")

                plt.plot(results['Adaptive Input Norm'].timestamps,
                         x1_cosim_adaptive_norm_error,
                         color='tab:orange',
                         label="Adap Input Norm")

                # plt.plot(results['Adaptive Input Norm'].timestamps,
                #          x1_cosim_error_adaptive_norm_filt,
                #          color='tab:orange',
                #          linestyle='dashed',
                #          label="Adap Input SGF")

                plt.plot(results['Adaptive Power'].timestamps,
                         x1_cosim_adaptive_power_error,
                         color='tab:olive',
                         label="Adap Power")

                # plt.plot(results['Adaptive Power'].timestamps,
                #          x1_cosim_error_adaptive_power_filt,
                #          color='tab:olive',
                #          linestyle='dashed',
                #          label="Adap Power SGF")
            # plt.title('Absolute Error x1')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                       ncol=3, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            # plt.ylim(1e-5, 1e-2)
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $x_1$ [m]')
            plt.tight_layout()

            plt.figure(figsize=(7, 6))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          v1_cosim_error_filt,
            #          color='tab:blue',
            #          linestyle='dashed',
            #          label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          v1_cosim_error_flipped_filt,
            #          color='tab:red',
            #          linestyle='dashed',
            #          label=r"MSD2$\rightarrow$MSD1 SGF")

            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         v1_cosim_adaptive_input_error,
                         color='tab:green',
                         label="Adap Input")

                # plt.plot(results['Adaptive Input'].timestamps,
                #          v1_cosim_error_adaptive_input_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label="Adap Input SGF")

                plt.plot(results['Adaptive Input Norm'].timestamps,
                         v1_cosim_adaptive_norm_error,
                         color='tab:orange',
                         label="Adap Input Norm")

                # plt.plot(results['Adaptive Input Norm'].timestamps,
                #          v1_cosim_error_adaptive_norm_filt,
                #          color='tab:orange',
                #          linestyle='dashed',
                #          label="Adap Input SGF")

                plt.plot(results['Adaptive Power'].timestamps,
                         v1_cosim_adaptive_power_error,
                         color='tab:olive',
                         label="Adap Power")

                # plt.plot(results['Adaptive Power'].timestamps,
                #          v1_cosim_error_adaptive_power_filt,
                #          color='tab:olive',
                #          linestyle='dashed',
                #          label="Adap Power SGF")

            # plt.title('Absolute Error v1')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                       ncol=3, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            # plt.ylim(1e-5, 1e-2)
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $v_1$ [m/s]')
            plt.tight_layout()

        x1_normal_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        x1_flip_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        x1_input_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        x1_norm_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        x1_power_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        for i in range(len(results['MSD1->MSD2'].timestamps)):
            seq = np.array([
                x1_cosim_error_filt[i],
                x1_cosim_error_flipped_filt[i],
                x1_cosim_error_adaptive_input_filt[i],
                v1_cosim_error_adaptive_norm_filt[i],
                v1_cosim_error_adaptive_power_filt[i]
            ])
            sort_index = np.argsort(seq)
            x1_normal_place[i] = sort_index[0]
            x1_flip_place[i] = sort_index[1]
            x1_input_place[i] = sort_index[2]
            x1_norm_place[i] = sort_index[3]
            x1_power_place[i] = sort_index[4]


        v1_normal_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        v1_flip_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        v1_input_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        v1_norm_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        v1_power_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        for i in range(len(results['MSD1->MSD2'].timestamps)):
            seq = np.array([
                v1_cosim_error_filt[i],
                v1_cosim_error_flipped_filt[i],
                v1_cosim_error_adaptive_input_filt[i],
                v1_cosim_error_adaptive_norm_filt[i],
                v1_cosim_error_adaptive_power_filt[i]
            ])
            sort_index = np.argsort(seq)
            v1_normal_place[i] = sort_index[0]
            v1_flip_place[i] = sort_index[1]
            v1_input_place[i] = sort_index[2]
            v1_norm_place[i] = sort_index[3]
            v1_power_place[i] = sort_index[4]

        if plot_solution_benchmark:
            plt.figure(figsize=(7, 5))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_normal_place,
                     color='tab:blue',
                     linewidth=3,
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_flip_place,
                     color='tab:red',
                     linewidth=3,
                     label=r"MSD2$\rightarrow$MSD1")
            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         x1_input_place,
                         color='tab:green',
                         linewidth=3,
                         label="Adap Input")

                plt.plot(results['Adaptive Input Norm'].timestamps,
                         x1_norm_place,
                         color='tab:orange',
                         linewidth=3,
                         label="Adap Input Norm")

                plt.plot(results['Adaptive Power'].timestamps,
                         x1_power_place,
                         color='tab:olive',
                         linewidth=3,
                         label="Adap Power")

            plt.title(r'Solution Benchmark for $x_1$')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),
                       ncol=3, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(-0.2, 4.2)
            plt.yticks([0,4], [r'Best', r'Worst'])
            plt.tight_layout()
            
            plt.figure(figsize=(7, 5))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_normal_place,
                     color='tab:blue',
                     linewidth=3,
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_flip_place,
                     color='tab:red',
                     linewidth=3,
                     label=r"MSD2$\rightarrow$MSD1")
            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         v1_input_place,
                         color='tab:green',
                         linewidth=3,
                         label="Adap Input")

                plt.plot(results['Adaptive Input Norm'].timestamps,
                         v1_norm_place,
                         color='tab:orange',
                         linewidth=3,
                         label="Adap Input Norm")

                plt.plot(results['Adaptive Power'].timestamps,
                         v1_power_place,
                         color='tab:olive',
                         linewidth=3,
                         label="Adap Power")

            plt.title(r'Solution Benchmark for $v_1$')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),
                       ncol=3, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(-0.2, 4.2)
            plt.yticks([0,4], [r'Best', r'Worst'])
            plt.tight_layout()

        # total energy at the system
        total_energy_msd1       = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2       = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_flip  = np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_flip  = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_adap_input    = np.array(results['Adaptive Input'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_adap_input    = np.array(results['Adaptive Input'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_adap_norm     = np.array(results['Adaptive Input'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_adap_norm     = np.array(results['Adaptive Input'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_adap_power    = np.array(results['Adaptive Power'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_adap_power    = np.array(results['Adaptive Power'].out_signals[msd2.instanceName][msd2.total_energy])

        # energy lost at the system by dissipation
        energy_loss_msd1        = np.cumsum(np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2        = np.cumsum(np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_flip   = np.cumsum(np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_flip   = np.cumsum(np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_adap_input   = np.cumsum(np.array(results['Adaptive Input'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_adap_input   = np.cumsum(np.array(results['Adaptive Input'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_adap_norm   = np.cumsum(np.array(results['Adaptive Input Norm'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_adap_norm   = np.cumsum(np.array(results['Adaptive Input Norm'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_adap_power   = np.cumsum(np.array(results['Adaptive Power'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_adap_power   = np.cumsum(np.array(results['Adaptive Power'].out_signals[msd2.instanceName][msd2.dissipated_energy]))

        # energy conservation
        total_energy = total_energy_msd1 + total_energy_msd2 + energy_loss_msd1 + energy_loss_msd2
        total_energy_flip = total_energy_msd1_flip + total_energy_msd2_flip + energy_loss_msd1_flip + energy_loss_msd2_flip
        total_energy_adap_input = total_energy_msd1_adap_input + total_energy_msd2_adap_input + energy_loss_msd1_adap_input + energy_loss_msd2_adap_input
        total_energy_adap_norm = total_energy_msd1_adap_norm + total_energy_msd2_adap_norm + energy_loss_msd1_adap_norm + energy_loss_msd2_adap_norm
        total_energy_adap_power = total_energy_msd1_adap_power + total_energy_msd2_adap_power + energy_loss_msd1_adap_power + energy_loss_msd2_adap_power

        # energy conservation filt
        total_energy_filt = savgol_filter(total_energy, 51, 1)
        total_energy_flip_filt = savgol_filter(total_energy_flip, 51, 1)
        total_energy_adap_input_filt = savgol_filter(total_energy_adap_input, 51, 1)
        total_energy_adap_norm_filt = savgol_filter(total_energy_adap_norm, 51, 1)
        total_energy_adap_power_filt = savgol_filter(total_energy_adap_power, 51, 1)

        if plot_energy_conservation:
            if plot_energy_deviation:
                # think about putting np.abs !!!
                total_energy_filt = total_energy_filt - total_energy[0]
                total_energy = total_energy - total_energy[0]
                total_energy_flip_filt = total_energy_flip_filt - total_energy_flip[0]
                total_energy_flip = total_energy_flip - total_energy_flip[0]
                total_energy_adap_input_filt = total_energy_adap_input_filt - total_energy_adap_input[0]
                total_energy_adap_input = total_energy_adap_input - total_energy_adap_input[0]
                total_energy_adap_power_filt = total_energy_adap_power_filt - total_energy_adap_power[0]
                total_energy_adap_power = total_energy_adap_power - total_energy_adap_power[0]
                total_energy_adap_norm_filt = total_energy_adap_norm_filt - total_energy_adap_norm[0]
                total_energy_adap_norm = total_energy_adap_norm - total_energy_adap_norm[0]
            elif plot_energy_deviation_percentage:
                total_energy_filt = (total_energy_filt - total_energy[0]) / total_energy[0]
                total_energy = (total_energy - total_energy[0]) / total_energy[0]
                total_energy_flip_filt = (total_energy_flip_filt - total_energy_flip[0]) / total_energy_flip[0]
                total_energy_flip = (total_energy_flip - total_energy_flip[0]) / total_energy_flip[0]
                total_energy_adap_input_filt = (total_energy_adap_input_filt - total_energy_adap_input[0]) / total_energy_adap_input[0]
                total_energy_adap_input = (total_energy_adap_input - total_energy_adap_input[0]) / total_energy_adap_input[0]
                total_energy_adap_norm_filt = (total_energy_adap_norm_filt - total_energy_adap_norm[0]) / total_energy_adap_norm[0]
                total_energy_adap_norm = (total_energy_adap_norm - total_energy_adap_norm[0]) / total_energy_adap_norm[0]
                total_energy_adap_power_filt = (total_energy_adap_power_filt - total_energy_adap_power[0]) / total_energy_adap_power[0]
                total_energy_adap_power = (total_energy_adap_power - total_energy_adap_power[0]) / total_energy_adap_power[0]


            plt.figure(figsize=(7, 6))
            plt.title('Energy Conservation')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     total_energy,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          total_energy_filt,
            #          color='tab:blue',
            #          linestyle='dashed',
            #          label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     total_energy_flip,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          total_energy_flip_filt,
            #          color='tab:red',
            #          linestyle='dashed',
            #          label=r"MSD2$\rightarrow$MSD1")

            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         total_energy_adap_input,
                         color='tab:green',
                         label=r"Adap Input")

                # plt.plot(results['Adaptive'].timestamps,
                #          total_energy_adap_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label=r"MSD1$\rightarrow$MSD2")

                plt.plot(results['Adaptive Input Norm'].timestamps,
                         total_energy_adap_norm,
                         color='tab:orange',
                         label=r"Adap Input Norm")

                # plt.plot(results['Adaptive'].timestamps,
                #          total_energy_adap_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label=r"MSD1$\rightarrow$MSD2")
                
                plt.plot(results['Adaptive Power'].timestamps,
                         total_energy_adap_power,
                         color='tab:olive',
                         label=r"Adap Power")

                # plt.plot(results['Adaptive'].timestamps,
                #          total_energy_adap_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label=r"MSD1$\rightarrow$MSD2")
                

            plt.xlabel('t [s]')
            if plot_energy_deviation:
                plt.ylabel(r'E [J]')
            else:
                plt.ylabel(r'E [\%]')
            plt.xlim(0.0, tf)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),
                       ncol=3, fancybox=True, shadow=True, fontsize=12)
            plt.tight_layout()


        energy_normal_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        energy_flip_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        energy_input_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        energy_norm_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        energy_power_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        for i in range(len(results['MSD1->MSD2'].timestamps)):
            seq = np.array([
                total_energy[i],
                total_energy_flip[i],
                total_energy_adap_input[i],
                total_energy_adap_norm[i],
                total_energy_adap_power[i]
            ])
            sort_index = np.argsort(seq)
            energy_normal_place[i] = sort_index[0]
            energy_flip_place[i] = sort_index[1]
            energy_input_place[i] = sort_index[2]
            energy_norm_place[i] = sort_index[3]
            energy_power_place[i] = sort_index[4]

        if plot_solution_benchmark:
            plt.figure(figsize=(7, 5))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     energy_normal_place,
                     color='tab:blue',
                     linewidth=3,
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     energy_flip_place,
                     color='tab:red',
                     linewidth=3,
                     label=r"MSD2$\rightarrow$MSD1")
            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         energy_input_place,
                         color='tab:green',
                         linewidth=3,
                         label="Adap Input")

                plt.plot(results['Adaptive Input Norm'].timestamps,
                         energy_norm_place,
                         color='tab:orange',
                         linewidth=3,
                         label="Adap Input Norm")

                plt.plot(results['Adaptive Power'].timestamps,
                         energy_power_place,
                         color='tab:olive',
                         linewidth=3,
                         label="Adap Power")

            plt.title(r'Solution Benchmark for energy')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),
                       ncol=3, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(-0.2, 4.2)
            plt.yticks([0,4], [r'Best', r'Worst'])
            plt.tight_layout()
        
        if not NONSTOP:
            plt.show()

    def test_run_adaptive_cosim_power_input(self):
        # simulation settings
        global_error = False
        load_sim = False
        load_adap = False
        logscale = True
        msd1first = False

        # static plots
        plot_actual_input_error = False
        plot_actual_state_error = True
        plot_actual_sequence = True
        # extrapolation
        plot_extrapolation = True
        plot_opportunity = False
        plot_opportunity_actual = False
        plot_validation = False
        plot_opportunity_adaptive = True
        # energy plots
        plot_energy_conservation = True
        plot_energy_deviation = False
        plot_energy_deviation_percentage = not plot_energy_deviation
        plot_energy_sequence = True
        # adaptive plots
        plot_adaptive_state = True
        plot_adaptive_mode = True
        plot_solution_benchmark = True

        # simulation settings
        tf = 5.0 if not NONSTOP else 1.0
        H = 0.01
        nsteps_msd1 = 10
        nsteps_msd2 = nsteps_msd1

        # initalize msds
        msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
        msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
        sol = CoupledMSD("sol")

        results = {}
        if load_sim:
            # load static results
            with open('./static_msd2_results.pickle', 'rb') as f:
                results = pickle.load(f)
        else:
            # obtain static results
            results['MSD1->MSD2'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=True)
            results['MSD2->MSD1'] = run_adaptive_cosim(msd1, msd2, sol,
                                                       COSIM_PARAMS, X0, H, tf,
                                                       static_mode=True, msd1_first=False)
            # save static results
            file = open('./static_msd2_results.pickle', 'wb')
            pickle.dump(results, file)
            file.close()

        # simulation settings
        if load_adap:
            # load adap results
            with open('./adap_msd2_results.pickle', 'rb') as f:
                results = pickle.load(f)
        else:
            # initalize msds
            msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
            msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
            sol = CoupledMSD("sol")

            error_estimator = 'input'
            mass_normalize = False
            filtered = True
            cooldown_period = 1
            results['Adaptive Input'] = run_adaptive_cosim(msd1, msd2, sol,
                                                           COSIM_PARAMS, X0, H, tf,
                                                           static_mode=False, msd1_first=msd1first,
                                                           mass_normalize=mass_normalize,
                                                           filter=filtered,
                                                           cooldown=cooldown_period,
                                                           estimator=error_estimator)

            # initalize msds
            msd1 = MSD1Adaptive("msd1", global_error, nstep=nsteps_msd1)
            msd2 = MSD2Adaptive("msd2", global_error, nstep=nsteps_msd2)
            sol = CoupledMSD("sol")

            # simulation settings
            error_estimator = 'power'
            mass_normalize = False
            filtered = True
            cooldown_period = 1
            results['Adaptive Power'] = run_adaptive_cosim(msd1, msd2, sol,
                                                           COSIM_PARAMS, X0, H, tf,
                                                           static_mode=False, msd1_first=msd1first,
                                                           mass_normalize=mass_normalize,
                                                           filter=filtered,
                                                           cooldown=cooldown_period,
                                                           estimator=error_estimator)
            # save adap results
            file = open('./adap_msd2_results.pickle', 'wb')
            pickle.dump(results, file)
            file.close()

        # ERROR WRT. ANALYTICAL SOLUTION
        x1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'x1')
        v1_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd1, sol, 'v1')
        x2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'x2')
        v2_cosim_error = get_analytical_error(results['MSD1->MSD2'], msd2, sol, 'v2')

        x1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'x1')
        v1_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd1, sol, 'v1')
        x2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'x2')
        v2_cosim_error_flipped = get_analytical_error(results['MSD2->MSD1'], msd2, sol, 'v2')

        x1_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd1, sol, 'x1')
        v1_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd1, sol, 'v1')
        x2_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd2, sol, 'x2')
        v2_cosim_adaptive_input_error = get_analytical_error(results['Adaptive Input'], msd2, sol, 'v2')

        x1_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd1, sol, 'x1')
        v1_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd1, sol, 'v1')
        x2_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd2, sol, 'x2')
        v2_cosim_adaptive_power_error = get_analytical_error(results['Adaptive Power'], msd2, sol, 'v2')

        x1_cosim_error_filt = savgol_filter(x1_cosim_error, 51, 1)
        x1_cosim_error_flipped_filt = savgol_filter(x1_cosim_error_flipped, 51, 1)
        x1_cosim_error_adaptive_input_filt = savgol_filter(x1_cosim_adaptive_input_error, 51, 1)
        x1_cosim_error_adaptive_power_filt = savgol_filter(x1_cosim_adaptive_power_error, 51, 1)

        v1_cosim_error_filt = savgol_filter(v1_cosim_error, 51, 1)
        v1_cosim_error_flipped_filt = savgol_filter(v1_cosim_error_flipped, 51, 1)
        v1_cosim_error_adaptive_input_filt = savgol_filter(v1_cosim_adaptive_input_error, 51, 1)
        v1_cosim_error_adaptive_power_filt = savgol_filter(v1_cosim_adaptive_power_error, 51, 1)

        x1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x1]
        v1_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v1]
        x2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.x2]
        v2_sol = results['MSD1->MSD2'].out_signals[sol.instanceName][sol.v2]

        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)
        if plot_actual_state_error:
            plt.figure(figsize=(7, 6))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          x1_cosim_error_filt,
            #          color='tab:blue',
            #          linestyle='dashed',
            #          label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          x1_cosim_error_flipped_filt,
            #          color='tab:red',
            #          linestyle='dashed',
            #          label=r"MSD2$\rightarrow$MSD1 SGF")

            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         x1_cosim_adaptive_input_error,
                         color='tab:green',
                         label="Adap Input")

                # plt.plot(results['Adaptive Input'].timestamps,
                #          x1_cosim_error_adaptive_input_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label="Adap Input SGF")

                plt.plot(results['Adaptive Power'].timestamps,
                         x1_cosim_adaptive_power_error,
                         color='tab:olive',
                         label="Adap Power")

                # plt.plot(results['Adaptive Power'].timestamps,
                #          x1_cosim_error_adaptive_power_filt,
                #          color='tab:olive',
                #          linestyle='dashed',
                #          label="Adap Power SGF")
            # plt.title('Absolute Error x1')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                       ncol=2, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            # plt.ylim(1e-5, 1e-2)
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $x_1$ [m]')
            plt.tight_layout()

            plt.figure(figsize=(7, 6))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_cosim_error,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          v1_cosim_error_filt,
            #          color='tab:blue',
            #          linestyle='dashed',
            #          label=r"MSD1$\rightarrow$MSD2 SGF")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_cosim_error_flipped,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          v1_cosim_error_flipped_filt,
            #          color='tab:red',
            #          linestyle='dashed',
            #          label=r"MSD2$\rightarrow$MSD1 SGF")

            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         v1_cosim_adaptive_input_error,
                         color='tab:green',
                         label="Adap Input")

                # plt.plot(results['Adaptive Input'].timestamps,
                #          v1_cosim_error_adaptive_input_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label="Adap Input SGF")

                plt.plot(results['Adaptive Power'].timestamps,
                         v1_cosim_adaptive_power_error,
                         color='tab:olive',
                         label="Adap Power")

                # plt.plot(results['Adaptive Power'].timestamps,
                #          v1_cosim_error_adaptive_power_filt,
                #          color='tab:olive',
                #          linestyle='dashed',
                #          label="Adap Power SGF")

            # plt.title('Absolute Error v1')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                       ncol=2, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(1e-6, 1e-2)
            if logscale:
                plt.yscale('log')
            plt.ylabel(r'Absolute Error $v_1$ [m/s]')
            plt.tight_layout()

        x1_normal_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        x1_flip_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        x1_input_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        x1_power_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        for i in range(len(results['MSD1->MSD2'].timestamps)):
            seq = np.array([
                x1_cosim_error_filt[i],
                x1_cosim_error_flipped_filt[i],
                x1_cosim_error_adaptive_input_filt[i],
                v1_cosim_error_adaptive_power_filt[i]
            ])
            sort_index = np.argsort(seq)
            for j in range(len(seq)):
                if sort_index[j] == 0:
                    x1_normal_place[i] = j
                elif sort_index[j] == 1:
                    x1_flip_place[i] = j
                elif sort_index[j] == 2:
                    x1_input_place[i] = j
                else:
                    x1_power_place[i] = j

        v1_normal_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        v1_flip_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        v1_input_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        v1_power_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        for i in range(len(results['MSD1->MSD2'].timestamps)):
            seq = np.array([
                v1_cosim_error_filt[i],
                v1_cosim_error_flipped_filt[i],
                v1_cosim_error_adaptive_input_filt[i],
                v1_cosim_error_adaptive_power_filt[i]
            ])
            sort_index = np.argsort(seq)
            for j in range(len(seq)):
                if sort_index[j] == 0:
                    v1_normal_place[i] = j
                elif sort_index[j] == 1:
                    v1_flip_place[i] = j
                elif sort_index[j] == 2:
                    v1_input_place[i] = j
                else:
                    v1_power_place[i] = j
        if plot_solution_benchmark:
            plt.figure(figsize=(7, 4))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     x1_normal_place,
                     color='tab:blue',
                     linewidth=3,
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     x1_flip_place,
                     color='tab:red',
                     linewidth=3,
                     label=r"MSD2$\rightarrow$MSD1")
            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         x1_input_place,
                         color='tab:green',
                         linewidth=3,
                         label="Adap Input")

                plt.plot(results['Adaptive Power'].timestamps,
                         x1_power_place,
                         color='tab:olive',
                         linewidth=3,
                         label="Adap Power")

            # plt.title(r'Solution Benchmark for $x_1$')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
                       ncol=2, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(-0.2, 3.2)
            plt.yticks([0, 3], [r'Best', r'Worst'])
            plt.tight_layout()

            plt.figure(figsize=(7, 4))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     v1_normal_place,
                     color='tab:blue',
                     linewidth=3,
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     v1_flip_place,
                     color='tab:red',
                     linewidth=3,
                     label=r"MSD2$\rightarrow$MSD1")
            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         v1_input_place,
                         color='tab:green',
                         linewidth=3,
                         label="Adap Input")

                plt.plot(results['Adaptive Power'].timestamps,
                         v1_power_place,
                         color='tab:olive',
                         linewidth=3,
                         label="Adap Power")

            # plt.title(r'Solution Benchmark for $v_1$')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
                       ncol=2, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(-0.2, 3.2)
            plt.yticks([0, 3], [r'Best', r'Worst'])
            plt.tight_layout()

        # total energy at the system
        total_energy_msd1 = np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2 = np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_flip = np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_flip = np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_adap_input = np.array(
            results['Adaptive Input'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_adap_input = np.array(
            results['Adaptive Input'].out_signals[msd2.instanceName][msd2.total_energy])
        total_energy_msd1_adap_power = np.array(
            results['Adaptive Power'].out_signals[msd1.instanceName][msd1.total_energy])
        total_energy_msd2_adap_power = np.array(
            results['Adaptive Power'].out_signals[msd2.instanceName][msd2.total_energy])

        # energy lost at the system by dissipation
        energy_loss_msd1 = np.cumsum(
            np.array(results['MSD1->MSD2'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2 = np.cumsum(
            np.array(results['MSD1->MSD2'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_flip = np.cumsum(
            np.array(results['MSD2->MSD1'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_flip = np.cumsum(
            np.array(results['MSD2->MSD1'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_adap_input = np.cumsum(
            np.array(results['Adaptive Input'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_adap_input = np.cumsum(
            np.array(results['Adaptive Input'].out_signals[msd2.instanceName][msd2.dissipated_energy]))
        energy_loss_msd1_adap_power = np.cumsum(
            np.array(results['Adaptive Power'].out_signals[msd1.instanceName][msd1.dissipated_energy]))
        energy_loss_msd2_adap_power = np.cumsum(
            np.array(results['Adaptive Power'].out_signals[msd2.instanceName][msd2.dissipated_energy]))

        # energy conservation
        total_energy = total_energy_msd1 + total_energy_msd2 + energy_loss_msd1 + energy_loss_msd2
        total_energy_flip = total_energy_msd1_flip + total_energy_msd2_flip + energy_loss_msd1_flip + energy_loss_msd2_flip
        total_energy_adap_input = total_energy_msd1_adap_input + total_energy_msd2_adap_input + energy_loss_msd1_adap_input + energy_loss_msd2_adap_input
        total_energy_adap_power = total_energy_msd1_adap_power + total_energy_msd2_adap_power + energy_loss_msd1_adap_power + energy_loss_msd2_adap_power

        # energy conservation filt
        total_energy_filt = savgol_filter(total_energy, 51, 1)
        total_energy_flip_filt = savgol_filter(total_energy_flip, 51, 1)
        total_energy_adap_input_filt = savgol_filter(total_energy_adap_input, 51, 1)
        total_energy_adap_power_filt = savgol_filter(total_energy_adap_power, 51, 1)

        if plot_energy_conservation:
            if plot_energy_deviation:
                # think about putting np.abs !!!
                total_energy_filt = total_energy_filt - total_energy[0]
                total_energy = total_energy - total_energy[0]
                total_energy_flip_filt = total_energy_flip_filt - total_energy_flip[0]
                total_energy_flip = total_energy_flip - total_energy_flip[0]
                total_energy_adap_input_filt = total_energy_adap_input_filt - total_energy_adap_input[0]
                total_energy_adap_input = total_energy_adap_input - total_energy_adap_input[0]
                total_energy_adap_power_filt = total_energy_adap_power_filt - total_energy_adap_power[0]
                total_energy_adap_power = total_energy_adap_power - total_energy_adap_power[0]
            elif plot_energy_deviation_percentage:
                total_energy_filt = (total_energy_filt - total_energy[0]) / total_energy[0]
                total_energy = (total_energy - total_energy[0]) / total_energy[0]
                total_energy_flip_filt = (total_energy_flip_filt - total_energy_flip[0]) / total_energy_flip[0]
                total_energy_flip = (total_energy_flip - total_energy_flip[0]) / total_energy_flip[0]
                total_energy_adap_input_filt = (total_energy_adap_input_filt - total_energy_adap_input[0]) / \
                                               total_energy_adap_input[0]
                total_energy_adap_input = (total_energy_adap_input - total_energy_adap_input[0]) / \
                                          total_energy_adap_input[0]
                total_energy_adap_power_filt = (total_energy_adap_power_filt - total_energy_adap_power[0]) / \
                                               total_energy_adap_power[0]
                total_energy_adap_power = (total_energy_adap_power - total_energy_adap_power[0]) / \
                                          total_energy_adap_power[0]

            plt.figure(figsize=(7, 6))
            plt.title('Energy Conservation')

            plt.plot(results['MSD1->MSD2'].timestamps,
                     total_energy,
                     color='tab:blue',
                     label=r"MSD1$\rightarrow$MSD2")

            # plt.plot(results['MSD1->MSD2'].timestamps,
            #          total_energy_filt,
            #          color='tab:blue',
            #          linestyle='dashed',
            #          label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     total_energy_flip,
                     color='tab:red',
                     label=r"MSD2$\rightarrow$MSD1")

            # plt.plot(results['MSD2->MSD1'].timestamps,
            #          total_energy_flip_filt,
            #          color='tab:red',
            #          linestyle='dashed',
            #          label=r"MSD2$\rightarrow$MSD1")

            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         total_energy_adap_input,
                         color='tab:green',
                         label=r"Adap Input")

                # plt.plot(results['Adaptive'].timestamps,
                #          total_energy_adap_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label=r"MSD1$\rightarrow$MSD2")

                plt.plot(results['Adaptive Power'].timestamps,
                         total_energy_adap_power,
                         color='tab:olive',
                         label=r"Adap Power")

                # plt.plot(results['Adaptive'].timestamps,
                #          total_energy_adap_filt,
                #          color='tab:green',
                #          linestyle='dashed',
                #          label=r"MSD1$\rightarrow$MSD2")

            plt.xlabel('t [s]')
            if plot_energy_deviation:
                plt.ylabel(r'E [J]')
            else:
                plt.ylabel(r'E [\%]')
            plt.xlim(0.0, tf)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),
                       ncol=2, fancybox=True, shadow=True, fontsize=12)
            plt.tight_layout()

        energy_normal_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        energy_flip_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        energy_input_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        energy_power_place = np.zeros_like(results['MSD1->MSD2'].timestamps)
        for i in range(len(results['MSD1->MSD2'].timestamps)):
            seq = np.array([
                abs(total_energy[i]),
                abs(total_energy_flip[i]),
                abs(total_energy_adap_input[i]),
                abs(total_energy_adap_power[i])
            ])
            sort_index = np.argsort(seq)
            for j in range(len(seq)):
                if sort_index[j] == 0:
                    energy_normal_place[i] = j
                elif sort_index[j] == 1:
                    energy_flip_place[i] = j
                elif sort_index[j] == 2:
                    energy_input_place[i] = j
                else:
                    energy_power_place[i] = j

        if plot_solution_benchmark:
            plt.figure(figsize=(7, 4))

            plt.plot(results['MSD1->MSD2'].timestamps,
                     energy_normal_place,
                     color='tab:blue',
                     linewidth=3,
                     label=r"MSD1$\rightarrow$MSD2")

            plt.plot(results['MSD2->MSD1'].timestamps,
                     energy_flip_place,
                     color='tab:red',
                     linewidth=3,
                     label=r"MSD2$\rightarrow$MSD1")
            if plot_adaptive_state:
                plt.plot(results['Adaptive Input'].timestamps,
                         energy_input_place,
                         color='tab:green',
                         linewidth=3,
                         label="Adap Input")

                plt.plot(results['Adaptive Power'].timestamps,
                         energy_power_place,
                         color='tab:olive',
                         linewidth=3,
                         label="Adap Power")

            plt.title(r'Solution Benchmark for energy')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
                       ncol=2, fancybox=True, shadow=True, fontsize=12)
            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.ylim(-0.2, 3.2)
            plt.yticks([0, 3], [r'Best', r'Worst'])
            plt.tight_layout()

        if plot_adaptive_mode:
            plt.figure(num=None, figsize=(7, 2))
            # plt.title('Cosimulation Sequence')

            plt.plot(results['Adaptive Input'].timestamps,
                     np.abs(1-np.array(results['Adaptive Input'].out_signals["_ma"][0])),
                     color='tab:blue', label='input')
            plt.plot(results['Adaptive Power'].timestamps,
                     np.abs(1-np.array(results['Adaptive Power'].out_signals["_ma"][0])),
                     color='tab:red', label='power')
            plt.yticks(np.arange(2), [r's1$\rightarrow$s2', r's2$\rightarrow$s1'])

            plt.xlabel('t [s]')
            plt.xlim(0.0, tf)
            plt.tight_layout()

        if not NONSTOP:
            plt.show()

if __name__ == '__main__':
    unittest.main()

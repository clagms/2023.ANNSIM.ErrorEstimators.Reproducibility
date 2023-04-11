import os.path
import unittest
import pickle

import matplotlib.pyplot as plt

from adaptive_cosim.MSD1Adaptive import MSD1Adaptive
from adaptive_cosim.MSD2Adaptive import MSD2Adaptive
from adaptive_cosim.adaptive_cosim_msd import run_adaptive_cosim
from cosim_msd_utils import get_analytical_error, CoupledMSD, extrapolation_error_aligned
from sys_params import COSIM_PARAMS, X0
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
        nsteps_msd1 = 10
        nsteps_msd2 = 10

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
    
    def test_run_adaptive_cosim_power_input(self):
        # simulation settings
        global_error = False
        load_sim = True
        load_adap = True
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
        plot_energy_conservation = False
        plot_energy_deviation = False
        plot_energy_deviation_percentage = plot_energy_deviation
        plot_energy_sequence = False
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
        adaptive_filename = "./adap_msd2_results.pickle"
        if load_adap and os.path.exists(adaptive_filename):
            # load adap results
            with open(adaptive_filename, 'rb') as f:
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
            file = open(adaptive_filename, 'wb')
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

            plt.title(r'Solution Benchmark for $x_1$')
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

            plt.title(r'Solution Benchmark for $v_1$')
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

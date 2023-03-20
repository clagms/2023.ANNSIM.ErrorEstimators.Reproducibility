# from src.cosimulation_solution import CosimMSD

#              ['m1', 'd1', 'c1', 'm2', 'd2', 'c2', 'dc', 'cc']
# COSIM_PARAMS = [1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]  # Param set A
COSIM_PARAMS = [100.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  100.0]  # Param set B
# COSIM_PARAMS = [100.0, 1.0, 1.0,  1.0,  1.0,  1000.0,  1.0,  1.0]  # Param set B2
# COSIM_PARAMS = [10.0, 1.0, 100.0,  10.0,  1.0,  100.0,  1.0,  100.0]  # Param set C
# COSIM_PARAMS = [100.0, 300.0, 20000.0,  1700.0,  700.0,  20000.0,  1.0,  100.0]  # Param set D
# COSIM_PARAMS = [1000.0, 1.0, 1.0, 10.0, 1.0,  1000.0, 1.0, 1000.0]  # param set B (of SMTP journal paper)


# initial conditions: x1, v1, x2, v2
X0 = [1.0, 0.0, 0.0, 0.0]
# X0 = [0.0, 0.0, 1.0, 0.0]
# cosim_msd = CosimMSD(COSIM_PARAMS, X0)
# sensitivities_filename = 'resources/sensitivities.pickle'
#
# if __name__ == '__main__':
#     # cosim_msd.test_sensitivities_all_params(sensitivities_filename)
#     # cosim_msd.test_compare_input_approximation_order()
#     # cosim_msd.test_show_error_single_param('m1', [1.0, 10.0])
#     # cosim_msd.test_local_error_diff_initial_conditions()
#     # cosim_msd.test_input_rate_difference(normalize=False,force=False)
#     # cosim_msd.test_power_transfer()
#     # cosim_msd.test_energy_difference()
#     # cosim_msd.test_input_error(normalize=False)
#     # cosim_msd.test_show_results()
#     cosim_msd.test_input_estimation_error(gte=False, absolute=False)

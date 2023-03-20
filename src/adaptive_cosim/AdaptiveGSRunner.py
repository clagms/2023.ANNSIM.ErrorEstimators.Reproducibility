from PyCosimLibrary.results import CosimResults
from PyCosimLibrary.runner import CosimRunner
from PyCosimLibrary.scenario import CosimScenario


class AdaptiveGSRunner(CosimRunner):
    def __init__(self, msd1, msd2, sol, msd1_out, msd1_in, static_mode, msd1_first,
                 results_id="_ma",
                 switch_cooldown_timer=1,
                 mass_normalize=False,
                 filter=False,
                 error_estimator='input'):
        self.msd1 = msd1
        self.msd2 = msd2
        self.sol = sol
        self.msd1_out = [msd1_out]
        self.msd1_in = [msd1_in]
        self.msd1_first = msd1_first
        self.msd1_first_res_id = 0
        self.msd1_mode_in = [0.0]
        self.msd2_mode_in = [1.0]
        self.msd1_input_mode = [self.msd1.input_is_from_future]
        self.msd2_input_mode = [self.msd2.input_is_from_future]
        self.static_mode = static_mode
        self.results_id = results_id
        self.switch_cooldown_timer_max = switch_cooldown_timer
        self.switch_cooldown_timer = switch_cooldown_timer
        self.mass_normalize = mass_normalize
        self.filter = filter
        self.error_estimator = error_estimator
        self.skip_init = 2
    #     self.extrapolate_msd1_file = open("msd1_extrapolate.txt", "w")
    #     self.extrapolate_msd2_file = open("msd2_extrapolate.txt", "w")
    #
    # def __del__(self):
    #     self.extrapolate_msd1_file.close()
    #     self.extrapolate_msd2_file.close()

    def update_fmu_mode(self, mode):
        self.msd1_mode_in[0] = 0.0 if mode else 1.0
        self.msd2_mode_in[0] = 1.0 - self.msd1_mode_in[0]
        self.msd1.setReal(self.msd1_input_mode, self.msd1_mode_in)
        self.msd2.setReal(self.msd2_input_mode, self.msd2_mode_in)

    def propagate_initial_outputs(self, scenario: CosimScenario):
        super(AdaptiveGSRunner, self).propagate_initial_outputs(scenario)
        self.update_fmu_mode(self.msd1_first)

    def run_cosim_step(self, time, scenario: CosimScenario):
        # compute energy
        prev_energy_msd1 = self.msd1.compute_energy()
        prev_energy_msd2 = self.msd2.compute_energy()
        prev_states = [self.msd1.state[self.msd1.x1], self.msd1.state[self.msd1.v1],
                            self.msd2.state[self.msd2.x2], self.msd2.state[self.msd2.v2]]

        # run cosim for H
        if self.msd1_first:
            # MSD1 -> MSD2
            self.msd1.doStep(time, scenario.step_size)
            self.propagate_outputs(self.msd1_out)
            self.msd2.doStep(time, scenario.step_size)
            self.propagate_outputs(self.msd1_in)
        else:
            # MSD2 -> MSD1
            self.msd2.doStep(time, scenario.step_size)
            self.propagate_outputs(self.msd1_in)
            self.msd1.doStep(time, scenario.step_size)
            self.propagate_outputs(self.msd1_out)

        # Now that both fmus have received the right inputs
        self.msd1.compute_extrapolated_error()
        self.msd2.compute_extrapolated_error()
        # compute energy transfer
        total_energy_msd1 = self.msd1.compute_energy()
        total_energy_msd2 = self.msd2.compute_energy()
        # energy lost at the system by dissipation
        energy_loss_msd1 = self.msd1.get_dissipated_energy(prev_states)
        energy_loss_msd2 = self.msd2.get_dissipated_energy(prev_states)
        # work done by the system
        work_msd1 = prev_energy_msd1 - total_energy_msd1 - energy_loss_msd1
        work_msd2 = prev_energy_msd2 - total_energy_msd2 - energy_loss_msd2
        self.msd1.state[self.msd1.work_done] = work_msd1
        self.msd2.state[self.msd2.work_done] = work_msd2

        self.sol.doStep(time, scenario.step_size)

        if not self.static_mode and self.skip_init == 0:
            # Adaptive algorithm
            if self.switch_cooldown_timer == 0:
                if self.error_estimator == 'input':
                    if self.filter:
                        if self.mass_normalize:
                            err_msd1 = abs(self.msd1.get_extrapolated_error_filtered()) / self.msd1.state[self.msd1.m1]
                            err_msd2 = abs(self.msd2.get_extrapolated_error_filtered()) / self.msd2.state[self.msd2.m2]
                        else:
                            err_msd1 = abs(self.msd1.get_extrapolated_error_filtered())
                            err_msd2 = abs(self.msd2.get_extrapolated_error_filtered())
                    else:
                        if self.mass_normalize:
                            err_msd1 = abs(self.msd1.get_extrapolated_error()) / self.msd1.state[self.msd1.m1]
                            err_msd2 = abs(self.msd2.get_extrapolated_error()) / self.msd2.state[self.msd2.m2]
                        else:
                            err_msd1 = abs(self.msd1.get_extrapolated_error())
                            err_msd2 = abs(self.msd2.get_extrapolated_error())

                    # self.extrapolate_msd1_file.write(f"{err_msd1}\n")
                    # self.extrapolate_msd2_file.write(f"{err_msd2}\n")

                    error_side = err_msd1 < err_msd2
                elif 'power': # adapt wrt power transfer -> causality
                    error_side = work_msd1 > work_msd2

                prev_order = self.msd1_first
                self.msd1_first = error_side
                self.update_fmu_mode(self.msd1_first)
                if prev_order != self.msd1_first:
                    if error_side:
                        # flush forward extrapolating side
                        self.msd1.state[self.msd1.prev_fk] = self.msd1.state[self.msd1.prev_prev_fk]
                        # flush backward interpolating side
                        self.msd2.state[self.msd2.prev_prev_x1] = self.msd2.state[self.msd2.prev_x1]
                        self.msd2.state[self.msd2.prev_x1] = self.msd2.state[self.msd2.x1]
                        self.msd2.state[self.msd2.prev_prev_v1] = self.msd2.state[self.msd2.prev_v1]
                        self.msd2.state[self.msd2.prev_v1] = self.msd2.state[self.msd2.v1]
                    else:
                        # flush forward extrapolating side
                        self.msd2.state[self.msd2.prev_x1] = self.msd2.state[self.msd2.prev_prev_x1]
                        self.msd2.state[self.msd2.prev_v1] = self.msd2.state[self.msd2.prev_prev_v1]
                        # flush backward interpolating side
                        self.msd1.state[self.msd1.prev_prev_fk] = self.msd1.state[self.msd1.prev_fk]
                        self.msd1.state[self.msd1.prev_fk] = self.msd1.state[self.msd1.fk]

                self.switch_cooldown_timer = self.switch_cooldown_timer_max
            else:
                self.switch_cooldown_timer -= 1

        if self.skip_init > 0:
            self.skip_init -= 1

            # if self.switch_cooldown_timer == 0:
            #     prev_order = self.msd1_first
            #     error_ratio = abs(self.msd1.get_extrapolated_error()) / abs(self.msd2.get_extrapolated_error())
            #     error_side =  abs(self.msd1.get_extrapolated_error()) < abs(self.msd2.get_extrapolated_error())
            #     threshold_power = 0
            #     if error_ratio > 10**threshold_power or error_ratio < 10**(-threshold_power):
            #         self.msd1_first = error_side
            #     self.update_fmu_mode(self.msd1_first)
            #     self.switch_cooldown_timer = self.switch_cooldown_timer_max
            # else:
            #     self.switch_cooldown_timer -= 1

        # if not self.static_mode:
        #     # Adaptive algorithm
        #     prev_order = self.msd1_first
        #     self.msd1_first = abs(self.msd1.get_extrapolated_error()) < abs(self.msd2.get_extrapolated_error())
        #     if (prev_order != self.msd1_first) and self.switch_cooldown_timer < 1:
        #         self.update_fmu_mode(self.msd1_first)
        #         self.switch_cooldown_timer = self.switch_cooldown_timer_max
        #     else:
        #         self.switch_cooldown_timer -= 1

    def init_results(self, scenario: CosimScenario, results=None):
        initialized_results = super(AdaptiveGSRunner, self).init_results(scenario, results)

        # Add adaptive cosim data
        initialized_results.out_signals[self.results_id] = {
            self.msd1_first_res_id: []
        }

        return initialized_results

    def snapshot(self, time: float, scenario: CosimScenario, results: CosimResults):
        super(AdaptiveGSRunner, self).snapshot(time, scenario, results)

        # Add adaptive cosim data
        results.out_signals[self.results_id][self.msd1_first_res_id].append(self.msd1_first)

# Implementation of the virtual fmu MSD2
from PyCosimLibrary.virtual_fmus import VirtualFMU
from fmpy.fmi2 import fmi2OK

from linear_sys_dynamics import form_jac_matrix, analytical_solution_matrix_affine
from system import *
from scipy.stats.mstats import gmean

class MSD2Adaptive(VirtualFMU):
    def __init__(self, instanceName, global_error, nstep=10):
        ref = 0
        self.x2 = ref
        ref += 1
        self.v2 = ref
        ref += 1
        self.m2 = ref
        ref += 1
        self.c2 = ref
        ref += 1
        self.d2 = ref
        ref += 1
        self.cc = ref
        ref += 1
        self.dc = ref
        ref += 1
        self.fk = ref
        ref += 1
        self.x1 = ref
        ref += 1
        self.v1 = ref
        ref += 1
        self.prev_x1 = ref
        ref += 1
        self.prev_prev_x1 = ref
        ref += 1
        self.prev_v1 = ref
        ref += 1
        self.prev_prev_v1 = ref
        ref += 1
        self.extrapolated_fk = ref
        ref += 1
        self.extrapolated_x1 = ref
        ref += 1
        self.extrapolated_v1 = ref
        ref += 1
        self.error_fk_indirect = ref
        ref += 1
        self.error_fk_indirect_filter = ref
        ref += 1
        self.error_x1 = ref
        ref += 1
        self.error_v1 = ref
        ref += 1
        self.input_is_from_future = ref
        ref += 1
        self.input_approximation_order = ref
        ref += 1
        self.kinetic_energy = ref
        ref += 1
        self.potential_energy = ref
        ref += 1
        self.total_energy = ref
        ref += 1
        self.dissipated_energy = ref
        ref += 1
        self.work_done = ref
        ref += 1

        self._global_error = global_error
        assert isinstance(nstep, int)
        assert nstep >= 1
        self._nstep = nstep
        self._A = None
        self._B = None
        self._H = 0.0
        self.error_fk_indirect_hist = []
        self.hist_size = 31

        super().__init__(instanceName, ref)

    def replace_constants(self, eq):
        return eq.subs(cc, self.state[self.cc]) \
            .subs(dc, self.state[self.dc]) \
            .subs(m2, self.state[self.m2]) \
            .subs(c2, self.state[self.c2]) \
            .subs(d2, self.state[self.d2])

    def compute_F(self, x1_num, v1_num, x2_num, v2_num):
        F_exp_num = self.replace_constants(F_c)
        return float(F_exp_num.subs(x2, x2_num) \
                     .subs(v2, v2_num) \
                     .subs(x1, x1_num) \
                     .subs(v1, v1_num))

    def exitInitializationMode(self):
        super(MSD2Adaptive, self).exitInitializationMode()

        self.state[self.prev_prev_x1] = self.state[self.x1]
        self.state[self.prev_x1] = self.state[self.x1]
        self.state[self.prev_prev_v1] = self.state[self.v1]
        self.state[self.prev_v1] = self.state[self.v1]

        self.state[self.extrapolated_fk] = self.state[self.fk]
        self.state[self.total_energy] = self.compute_energy()

        # Pre-compute system matrix
        F_exp_num = self.replace_constants(F_c)

        dx2_dt_num = dx2_dt
        dv2_dt_num = self.replace_constants(dv2_dt.subs(F, F_exp_num))

        # State matrix
        self._A = form_jac_matrix([dx2_dt_num, dv2_dt_num], [x2, v2])
        # Input matrix B, so that we end up with a system \der{x} = Ax + Bu
        self._B = form_jac_matrix([dx2_dt_num, dv2_dt_num], [x1, v1])

    def _calcF(self):
        self.state[self.fk] = self.compute_F(self.state[self.x1], self.state[self.v1],
                                             self.state[self.x2], self.state[self.v2])

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=fmi2True):
        # if currentCommunicationPoint == 0.0:
        #     print(f"MSD2.input_is_from_future = {self.state[self.input_is_from_future]}")
        self._H = communicationStepSize
        h = communicationStepSize / self._nstep  # Internal step size

        # Compute internal solution, taking into account approximation of the input.
        t = 0.0

        # If self.input_is_from_future, then self.state[self.fk] is the input from the future, and the interpolation must be done starting with self.prev_fk.
        # Except at the initial step of the simulation, where we don't have enough information.
        if self.state[self.input_approximation_order] == 1:
            if (self.state[self.input_is_from_future] < 0.5):
                # x1_approx = self.state[self.x1]  # extrapolation
                # v1_approx = self.state[self.v1]
                x1_init = self.state[self.x1]  # extrapolation
                v1_init = self.state[self.v1]
            else:
                # x1_approx = self.state[self.prev_x1] # interpolation
                # v1_approx = self.state[self.prev_v1]
                x1_init = self.state[self.prev_x1] # interpolation
                v1_init = self.state[self.prev_v1]

            x1_derivative = ((self.state[self.x1] - self.state[self.prev_x1]) / communicationStepSize)
            # x1_increment = x1_derivative * h
            v1_derivative = ((self.state[self.v1] - self.state[self.prev_v1]) / communicationStepSize)
            # v1_increment = v1_derivative * h
        else:
            assert self.state[self.input_approximation_order] == 0, "Expected order to be 0. Other orders are not supported."
            x1_approx = self.state[self.x1]
            v1_approx = self.state[self.v1]
            x1_derivative = 0.0
            v1_derivative = 0.0
            # x1_increment = 0.0
            # v1_increment = 0.0

        # for i in range(self._nstep):
        while t < communicationStepSize:
            if communicationStepSize - t < h:
                h = communicationStepSize - t
            x1_approx = x1_init + x1_derivative * t
            v1_approx = v1_init + v1_derivative * t
            # Take input into account
            u = np.array([[x1_approx], [v1_approx]])
            assert u.shape == (2, 1)

            B_approx = self._B.dot(u)

            x2_t = self.state[self.x2]
            v2_t = self.state[self.v2]
            x_t = np.array([[x2_t], [v2_t]])

            assert x_t.shape == (2, 1)

            (ts, step_solution) = analytical_solution_matrix_affine(self._A, B_approx, x_t, h, h)

            self.state[self.x2] = step_solution[-1][0, 0]
            self.state[self.v2] = step_solution[-1][1, 0]

            # x1_approx += x1_increment
            # v1_approx += v1_increment
            t += h

        if self.state[self.input_is_from_future] < 0.5:
            # Extrapolation. Calculate actual error
            self.state[self.extrapolated_x1] = 2 * self.state[self.x1] - self.state[self.prev_x1]
            self.state[self.extrapolated_v1] = 2 * self.state[self.v1] - self.state[self.prev_v1]
        else:
            # Calculate hypothetical error
            self.state[self.extrapolated_x1] = 2 * self.state[self.prev_x1] - self.state[self.prev_prev_x1]
            self.state[self.extrapolated_v1] = 2 * self.state[self.prev_v1] - self.state[self.prev_prev_v1]

        self.state[self.extrapolated_fk] = self.compute_F(self.state[self.extrapolated_x1],
                                                          self.state[self.extrapolated_v1],
                                                          self.state[self.x2],
                                                          self.state[self.v2])
        # Store previous inputs
        self.state[self.prev_prev_x1] = self.state[self.prev_x1]
        self.state[self.prev_x1] = self.state[self.x1]
        self.state[self.prev_prev_v1] = self.state[self.prev_v1]
        self.state[self.prev_v1] = self.state[self.v1]

        if self.state[self.input_is_from_future] == 0: # extrapolation
            self.state[self.x1] = self.state[self.extrapolated_x1]
            self.state[self.v1] = self.state[self.extrapolated_v1]
        else: # interpolation
            pass

        return fmi2OK

    def getReal(self, vr):
        self._calcF()
        return super(MSD2Adaptive, self).getReal(vr)

    def compute_extrapolated_error(self):
        # This is called after msd2 has received the new values for x1 and v1.
        self._calcF()

        m2_num = self.state[self.m2]
        cc_num = self.state[self.cc]
        dc_num = self.state[self.dc]
        H = self._H

        e_x1 = self.state[self.x1] - self.state[self.extrapolated_x1]
        g_x1 = self.state[self.error_x1]
        self.state[self.error_x1] = e_x1 if not self._global_error else e_x1 * H + g_x1 * (cc_num / m2_num) * H

        e_v1 = self.state[self.v1] - self.state[self.extrapolated_v1]
        g_v1 = self.state[self.error_v1]
        self.state[self.error_v1] = e_v1 if not self._global_error else e_v1 * H + g_v1 * (dc_num / m2_num) * H

        # for linear case -> it is OK
        # e_fk = - cc_num * e_x1 - dc_num * e_v1
        e_fk = - cc_num * self.state[self.error_x1] - dc_num * self.state[self.error_v1]
        self.state[self.error_fk_indirect] = e_fk

        self.error_fk_indirect_hist.append(self.state[self.error_fk_indirect])
        if len(self.error_fk_indirect_hist) > self.hist_size:
            self.error_fk_indirect_hist.pop(0)

        # filtered --> Take this under get_extrapolated_error_filtered function for acceleration
        window_ratio = 1
        current_hist_size = len(self.error_fk_indirect_hist)
        window = np.ceil(current_hist_size / window_ratio)
        if window > 2:
            # # savgol filter
            # if window % 2:
            #     efk_filt = savgol_filter(self.error_fk_indirect_hist, int(window), 1)
            # else:
            #     efk_filt = savgol_filter(self.error_fk_indirect_hist, int(window)-1, 1)
            # remove outliers
            temp_errors = np.abs(np.array(self.error_fk_indirect_hist))
            errors_to_filter = []
            for i in range(len(temp_errors)):
                if temp_errors[i] > np.mean(temp_errors)* 1e-6:
                    errors_to_filter.append(temp_errors[i])
            # geometric mean
            efk_filt = [gmean(errors_to_filter)]
            # # linear regression
            # model = LinearRegression()
            # x = np.array(list(range(current_hist_size))).reshape(-1, 1)
            # y = np.array(self.error_fk_indirect_hist).reshape(-1, 1)
            # model.fit(x, y)
            # efk_filt = model.predict(np.array([current_hist_size - 1]).reshape(-1, 1))[0]
        else:
            efk_filt = self.error_fk_indirect_hist
        self.state[self.error_fk_indirect_filter] = efk_filt[-1]

        # # for nonlinear case -> need to use this formulation
        # e_fk = self.state[self.fk] - self.state[self.extrapolated_fk]
        # g_fk = self.state[self.error_fk_indirect]
        # self.state[self.error_fk_indirect] = e_fk if not self._global_error else e_fk * H + g_fk * (
        #         1 / m2_num) * H

    def compute_energy(self):
        self.state[self.kinetic_energy] = 0.5 * self.state[self.m2] * (self.state[self.v2] ** 2)  
        self.state[self.potential_energy] = 0.5 * self.state[self.c2] * (self.state[self.x2] ** 2) \
                                    + 0.5 * self.state[self.cc] * ((self.state[self.x2] - self.state[self.x1]) ** 2)
        self.state[self.total_energy] = self.state[self.kinetic_energy] + self.state[self.potential_energy]
        return self.state[self.total_energy]

    def get_dissipated_energy(self, prev_states):
        prev_x1, prev_v1, prev_x2, prev_v2 = prev_states
        self.state[self.dissipated_energy] =  \
            0.5 * self.state[self.d2] * (self.state[self.v2] + prev_v2) * (self.state[self.x2] - prev_x2) \
            + 0.5 * self.state[self.dc] * ((self.state[self.v2] - self.state[self.v1]) + (prev_v2 - prev_v1)) \
            * ((self.state[self.x2] - self.state[self.x1]) - (prev_x2 - prev_x1))
        return self.state[self.dissipated_energy]


    def get_extrapolated_error(self):
        return self.state[self.error_fk_indirect]

    def get_extrapolated_error_filtered(self):
        return self.state[self.error_fk_indirect_filter]

    def getSystemMatrix(self):
        return self._A, self._B

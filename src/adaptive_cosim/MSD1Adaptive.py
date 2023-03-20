
# FMU class for MSD1
from PyCosimLibrary.virtual_fmus import VirtualFMU
from fmpy.fmi2 import fmi2OK

from linear_sys_dynamics import *
from system import *
from scipy.stats.mstats import gmean

class MSD1Adaptive(VirtualFMU):
    def __init__(self, instanceName, global_error, nstep=10):
        # Declare value references for variables
        ref = 0
        self.x1 = ref
        ref += 1
        self.v1 = ref
        ref += 1
        self.m1 = ref
        ref += 1
        self.c1 = ref
        ref += 1
        self.d1 = ref
        ref += 1
        self.fk = ref
        ref += 1
        self.prev_fk = ref
        ref += 1
        self.prev_prev_fk = ref
        ref += 1
        self.extrapolated_fk = ref
        ref += 1
        self.error_fk_direct = ref
        ref += 1
        self.error_fk_direct_filter = ref
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
        self.error_fk_direct_hist = []
        self.hist_size = 31

        super().__init__(instanceName, ref)

    def replace_constants(self, eq):
        return eq.subs(m1, self.state[self.m1]) \
            .subs(c1, self.state[self.c1]) \
            .subs(d1, self.state[self.d1])

    def exitInitializationMode(self):
        super(MSD1Adaptive, self).exitInitializationMode()
        self.state[self.prev_prev_fk] = self.state[self.fk]
        self.state[self.prev_fk] = self.state[self.fk]
        self.state[self.extrapolated_fk] = self.state[self.fk]

        self.state[self.total_energy] = self.compute_energy()

        # Simulation pre-computed stuff

        # Pre-compute and system matrices equations

        # Replace parameters: assumes that they have all been set already.
        dx1_dt_num = dx1_dt
        dv1_dt_num = self.replace_constants(dv1_dt)

        # State transition matrix
        self._A = form_jac_matrix([dx1_dt_num, dv1_dt_num], [x1, v1])
        # Input matrix B, so that we end up with a system \der{x} = Ax + Bu
        self._B = np.array([[0.0], [diff(dv1_dt_num, F)]])

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=fmi2True):
        # if currentCommunicationPoint == 0.0:
        #     print(f"MSD1.input_is_from_future = {self.state[self.input_is_from_future]}")

        self._H = communicationStepSize
        h = communicationStepSize / self._nstep  # Internal step size

        # Compute internal solution, taking into account approximation of the input.
        t = 0.0

        if self.state[self.input_approximation_order] == 1:
            # If self.input_is_from_future, then self.state[self.fk] is the input from the future, and the interpolation must be done starting with self.prev_fk.
            # Except at the initial step of the simulation, where we don't have enough information.
            # fk_approx = self.state[self.fk] if (self.state[self.input_is_from_future] < 0.5 or currentCommunicationPoint == 0.0) else \
            if (self.state[self.input_is_from_future] < 0.5):
                fk_init = self.state[self.fk]  # extrapolating
                # fk_approx = self.state[self.fk]
            else:
                fk_init = self.state[self.prev_fk] # interpolating
                # fk_approx = self.state[self.prev_fk]

            # fk_derivative = ((self.state[self.fk] - self.state[self.prev_fk]) / communicationStepSize) if currentCommunicationPoint > 0.0 else 0.0
            fk_derivative = ((self.state[self.fk] - self.state[self.prev_fk]) / communicationStepSize)
            # fk_increment = fk_derivative * h
        else:
            assert self.state[self.input_approximation_order] == 0, "Expected order to be 0. Other orders are not supported."
            fk_approx = self.state[self.fk]
            fk_derivative = 0.0
            # fk_increment = 0.0

        # for i in range(self._nstep):
        while t < communicationStepSize:
            if communicationStepSize - t < h:
                h = communicationStepSize - t
            fk_approx = fk_init + fk_derivative * t
            u = np.array([[fk_approx]])
            # Take input approximation into B
            B_approx = self._B.dot(u)

            x1_t = self.state[self.x1]
            v1_t = self.state[self.v1]
            x_t = np.array([[x1_t], [v1_t]])

            (ts, step_solution) = analytical_solution_matrix_affine(self._A, B_approx, x_t, h, h)

            self.state[self.x1] = step_solution[-1][0, 0]
            self.state[self.v1] = step_solution[-1][1, 0]

            # fk_approx += fk_increment
            t += h

        # Store extrapolation of fk, to calculate the error
        if self.state[self.input_is_from_future] > 0.5: # interpolation
            self.state[self.extrapolated_fk] = 2 * self.state[self.prev_fk] - self.state[self.prev_prev_fk]
        else: # extrapolation
            self.state[self.extrapolated_fk] = 2 * self.state[self.fk] - self.state[self.prev_fk]

        # Store previous inputs
        self.state[self.prev_prev_fk] = self.state[self.prev_fk]
        self.state[self.prev_fk] = self.state[self.fk]

        return fmi2OK

    def compute_extrapolated_error(self):
        # This is called after msd1 has received the new value for fk.
        e_fk = self.state[self.fk] - self.state[self.extrapolated_fk]
        g_fk = self.state[self.error_fk_direct]
        m1_num = self.state[self.m1]
        self.state[self.error_fk_direct] = e_fk if not self._global_error else e_fk*self._H + g_fk*(1/m1_num)*self._H

        self.error_fk_direct_hist.append(self.state[self.error_fk_direct])
        if len(self.error_fk_direct_hist) > self.hist_size:
            self.error_fk_direct_hist.pop(0)

        # filtered extrapolation
        window_ratio = 1
        current_hist_size = len(self.error_fk_direct_hist)
        window = np.ceil(current_hist_size / window_ratio)

        if window > 2:
            # # savgol_filter
            # if window % 2:
            #     efk_filt = savgol_filter(self.error_fk_direct_hist, int(window), 1)
            # else:
            #     efk_filt = savgol_filter(self.error_fk_direct_hist, int(window)-1, 1)
            # remove outliers
            temp_errors = np.abs(np.array(self.error_fk_direct_hist))
            errors_to_filter = []
            for i in range(len(temp_errors)):
                if temp_errors[i] > np.mean(temp_errors) * 1e-6:
                    errors_to_filter.append(temp_errors[i])
            # geometric mean
            efk_filt = [gmean(errors_to_filter)]
            # # linear regression
            # model = LinearRegression()
            # x = np.array(list(range(current_hist_size))).reshape(-1, 1)
            # y = np.array(self.error_fk_direct_hist).reshape(-1, 1)
            # model.fit(x, y)
            # efk_filt = model.predict(np.array([current_hist_size - 1]).reshape(-1, 1))[0]
        else:
            efk_filt = self.error_fk_direct_hist
        self.state[self.error_fk_direct_filter] = efk_filt[-1]

    def compute_energy(self):
        self.state[self.kinetic_energy] = 0.5 * self.state[self.m1] * (self.state[self.v1] ** 2)
        self.state[self.potential_energy] = 0.5 * self.state[self.c1] * (self.state[self.x1] ** 2)
        self.state[self.total_energy] = self.state[self.kinetic_energy] + self.state[self.potential_energy]
        return self.state[self.total_energy]

    def get_dissipated_energy(self, prev_states):
        prev_x1, prev_v1, prev_x2, prev_v2 = prev_states
        # Trapezoidal rule: \int_{x} d_1 v_1 dx = 1/2 d_1 (v_1_{i} + v_1_{i+1}) (x_1_{i+1} - x_1_{i})
        self.state[self.dissipated_energy] \
            = 0.5 * self.state[self.d1] * (self.state[self.v1] + prev_v1) * (self.state[self.x1] - prev_x1)
        return self.state[self.dissipated_energy]

    def get_extrapolated_error(self):
        return self.state[self.error_fk_direct]

    def get_extrapolated_error_filtered(self):
        return self.state[self.error_fk_direct_filter]

    def getSystemMatrix(self):
        # State matrix
        return self._A, self._B

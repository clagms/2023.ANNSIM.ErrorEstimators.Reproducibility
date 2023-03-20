import sys

from PyCosimLibrary.scenario import Connection, VarType, SignalType, OutputConnection, CosimScenario
from PyCosimLibrary.virtual_fmus import VirtualFMU
from fmpy.fmi2 import fmi2OK

from src import linear_sys_dynamics as lsd
from src.CustomGSRunner import CustomGSRunner
from src.system import *
from src.utils import *


# FMU class for MSD1
class MSD1(VirtualFMU):
    def __init__(self, instanceName):
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
        self.input_is_from_future = ref
        ref += 1
        self.input_approximation_order = ref
        ref += 1
        self.Фx1F = ref
        ref += 1
        self.Фv1F = ref
        ref += 1

        super().__init__(instanceName, ref)

    def replace_constants(self, eq):
        return eq.subs(m1, self.state[self.m1]) \
            .subs(c1, self.state[self.c1]) \
            .subs(d1, self.state[self.d1])

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=fmi2True):
        # if currentCommunicationPoint == 0.0:
        #     print(f"MSD1.input_is_from_future = {self.state[self.input_is_from_future]}")

        h = communicationStepSize / 10.0  # Internal step size
        tf = communicationStepSize  # Time to terminate the doStep

        # State matrix
        dx1_dt_num = dx1_dt
        dv1_dt_num = self.replace_constants(dv1_dt)

        A = lsd.form_jac_matrix([dx1_dt_num, dv1_dt_num], [x1, v1])
        # Input matrix B, so that we end up with a system \der{x} = Ax + Bu
        B = np.array([[0.0], [diff(dv1_dt_num, F)]])

        # Compute internal solution, taking into account approximation of the input.
        t = 0

        if self.state[self.input_approximation_order] == 1:
            # If self.input_is_from_future, then self.state[self.fk] is the input from the future, and the interpolation must be done starting with self.prev_fk.
            # Except at the initial step of the simulation, where we don't have enough information.
            fk_approx = self.state[self.fk] if (
                    self.state[self.input_is_from_future] < 0.5 or currentCommunicationPoint == 0.0) else \
                self.state[self.prev_fk]

            fk_derivative = ((self.state[self.fk] - self.state[
                self.prev_fk]) / communicationStepSize) if currentCommunicationPoint > 0.0 else 0.0
            fk_increment = fk_derivative * h
        else:
            assert self.state[
                       self.input_approximation_order] == 0, "Expected order to be 0. Other orders are not supported."
            fk_approx = self.state[self.fk]
            fk_increment = 0.0

        while t < communicationStepSize:
            fk_approx += fk_increment

            # Take input approximation into B
            B_approx = B * fk_approx

            x1_t = self.state[self.x1]
            v1_t = self.state[self.v1]
            x_t = np.array([[x1_t], [v1_t]])

            (ts, step_solution) = lsd.analytical_solution_matrix_affine(A, B_approx, x_t, h, h)

            self.state[self.x1] = step_solution[-1][0, 0]
            self.state[self.v1] = step_solution[-1][1, 0]

            t += h

        # Sensitivity System: put it in the form Ax + B
        dФx1F_dt_num = self.replace_constants(dФx1F_dt)
        dФv1F_dt_num = self.replace_constants(dФv1F_dt)

        AS = lsd.form_jac_matrix([dФx1F_dt_num, dФv1F_dt_num], [Фx1F, Фv1F])
        BS = np.array([[0.0], [dФv1F_dt_num.subs(Фx1F, 0).subs(Фv1F, 0)]])
        assert BS.shape == (2, 1)
        Ф_t = np.array([[0.0], [0.0]])
        assert Ф_t.shape == (2, 1)

        (ts, step_sensitivity) = lsd.analytical_solution_matrix_affine(AS, BS, Ф_t, h, tf)

        self.state[self.Фx1F] = step_sensitivity[-1][0, 0]
        self.state[self.Фv1F] = step_sensitivity[-1][1, 0]

        # Store previous inputs
        self.state[self.prev_fk] = self.state[self.fk]

        return fmi2OK

    def getSystemMatrix(self):
        # State matrix
        dx1_dt_num = dx1_dt
        dv1_dt_num = self.replace_constants(dv1_dt)

        A = lsd.form_jac_matrix([dx1_dt_num, dv1_dt_num], [x1, v1])
        # Input matrix B, so that we end up with a system \der{x} = Ax + Bu
        B = np.array([[0.0], [diff(dv1_dt_num, F)]])
        return A, B


# Implementation of the virtual fmu MSD2
class MSD2(VirtualFMU):
    def __init__(self, instanceName):
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
        self.prev_v1 = ref
        ref += 1
        self.input_is_from_future = ref
        ref += 1
        self.input_approximation_order = ref
        ref += 1
        self.Фx2x1 = ref
        ref += 1
        self.Фv2x1 = ref
        ref += 1
        self.Фx2v1 = ref
        ref += 1
        self.Фv2v1 = ref
        ref += 1

        super().__init__(instanceName, ref)

    def setReal(self, vr, value):
        super().setReal(vr, value)
        self._calcF()  # Feedthrough

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

    def _calcF(self):
        self.state[self.fk] = self.compute_F(self.state[self.x1], self.state[self.v1],
                                             self.state[self.x2], self.state[self.v2])

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=fmi2True):
        # if currentCommunicationPoint == 0.0:
        #     print(f"MSD2.input_is_from_future = {self.state[self.input_is_from_future]}")

        h = communicationStepSize / 10.0  # Internal step size
        tf = communicationStepSize  # Time to terminate the doStep

        F_exp_num = self.replace_constants(F_c)

        dx2_dt_num = dx2_dt
        dv2_dt_num = self.replace_constants(dv2_dt.subs(F, F_exp_num))

        # State matrix
        A = lsd.form_jac_matrix([dx2_dt_num, dv2_dt_num], [x2, v2])
        # Input matrix B, so that we end up with a system \der{x} = Ax + Bu
        B = lsd.form_jac_matrix([dx2_dt_num, dv2_dt_num], [x1, v1])

        # Compute internal solution, taking into account approximation of the input.
        t = 0

        # If self.input_is_from_future, then self.state[self.fk] is the input from the future, and the interpolation must be done starting with self.prev_fk.
        # Except at the initial step of the simulation, where we don't have enough information.

        if self.state[self.input_approximation_order] == 1:
            x1_approx = self.state[self.x1] if (
                    self.state[self.input_is_from_future] < 0.5 or currentCommunicationPoint == 0.0) else self.state[
                self.prev_x1]
            v1_approx = self.state[self.v1] if (
                    self.state[self.input_is_from_future] < 0.5 or currentCommunicationPoint == 0.0) else self.state[
                self.prev_v1]

            x1_derivative = ((self.state[self.x1] - self.state[
                self.prev_x1]) / communicationStepSize) if currentCommunicationPoint > 0.0 else 0.0
            x1_increment = x1_derivative * h

            v1_derivative = ((self.state[self.v1] - self.state[
                self.prev_v1]) / communicationStepSize) if currentCommunicationPoint > 0.0 else 0.0
            v1_increment = v1_derivative * h
        else:
            assert self.state[
                       self.input_approximation_order] == 0, "Expected order to be 0. Other orders are not supported."
            x1_approx = self.state[self.x1]
            v1_approx = self.state[self.v1]
            x1_increment = 0.0
            v1_increment = 0.0

        while t < communicationStepSize:
            x1_approx += x1_increment
            v1_approx += v1_increment

            # Take input into account
            u = np.array([[x1_approx], [v1_approx]])
            assert u.shape == (2, 1)

            B_approx = B.dot(u)

            x2_t = self.state[self.x2]
            v2_t = self.state[self.v2]
            x_t = np.array([[x2_t], [v2_t]])

            assert x_t.shape == (2, 1)

            (ts, step_solution) = lsd.analytical_solution_matrix_affine(A, B_approx, x_t, h, h)

            self.state[self.x2] = step_solution[-1][0, 0]
            self.state[self.v2] = step_solution[-1][1, 0]

            t += h

        # Sensitivity System: put it in the form Ax + B
        dФx2x1_dt_num = self.replace_constants(dФx2x1_dt)
        dФv2x1_dt_num = self.replace_constants(dФv2x1_dt)
        dФx2v1_dt_num = self.replace_constants(dФx2v1_dt)
        dФv2v1_dt_num = self.replace_constants(dФv2v1_dt)

        AS = lsd.form_jac_matrix([dФx2x1_dt_num, dФv2x1_dt_num, dФx2v1_dt_num, dФv2v1_dt_num],
                                 [Фx2x1, Фv2x1, Фx2v1, Фv2v1])
        BS = np.array([[0.0],
                       [dФv2x1_dt_num.subs(Фx2x1, 0).subs(Фv2x1, 0)],
                       [0.0],
                       [dФv2v1_dt_num.subs(Фx2v1, 0).subs(Фv2v1, 0)]])
        Ф_t = np.array([[0.0], [0.0], [0.0], [0.0]])

        (ts, step_sensitivity) = lsd.analytical_solution_matrix_affine(AS, BS, Ф_t, h, tf)

        self.state[self.Фx2x1] = step_sensitivity[-1][0, 0]
        self.state[self.Фv2x1] = step_sensitivity[-1][1, 0]
        self.state[self.Фx2v1] = step_sensitivity[-1][2, 0]
        self.state[self.Фv2v1] = step_sensitivity[-1][3, 0]

        self._calcF()

        # Store previous inputs
        self.state[self.prev_x1] = self.state[self.x1]
        self.state[self.prev_v1] = self.state[self.v1]

        return fmi2OK

    def getSystemMatrix(self):
        F_exp_num = self.replace_constants(F_c)

        dx2_dt_num = dx2_dt
        dv2_dt_num = self.replace_constants(dv2_dt.subs(F, F_exp_num))

        # State matrix
        A = lsd.form_jac_matrix([dx2_dt_num, dv2_dt_num], [x2, v2])
        # Input matrix B, so that we end up with a system \der{x} = Ax + Bu
        B = lsd.form_jac_matrix([dx2_dt_num, dv2_dt_num], [x1, v1])
        return A, B


class CoupledMSD(VirtualFMU):
    def __init__(self, instanceName):
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

        super().__init__(instanceName, ref)

    def setReal(self, vr, value):
        super().setReal(vr, value)
        self.calcF()  # Feedthrough

    def calcF(self):
        F_exp_num = F_c.subs(cc, self.state[self.cc]).subs(dc, self.state[self.dc])
        self.state[self.fk] = float(F_exp_num.subs(x2, self.state[self.x2]) \
                                            .subs(v2, self.state[self.v2]) \
                                            .subs(x1, self.state[self.x1]) \
                                            .subs(v1, self.state[self.v1]))

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=fmi2True):
        h = communicationStepSize  # Internal step size
        tf = communicationStepSize  # Time to terminate the doStep

        # State matrix
        F_exp_num = F_c.subs(cc, self.state[self.cc]).subs(dc, self.state[self.dc])

        dx1_dt_num = dx1_dt
        dv1_dt_num = dv1_dt.subs(m1, self.state[self.m1]) \
            .subs(c1, self.state[self.c1]) \
            .subs(d1, self.state[self.d1]) \
            .subs(F, F_exp_num)

        dx2_dt_num = dx2_dt
        dv2_dt_num = dv2_dt.subs(m2, self.state[self.m2]) \
            .subs(c2, self.state[self.c2]) \
            .subs(d2, self.state[self.d2]) \
            .subs(F, F_exp_num)

        A = lsd.form_jac_matrix([dx1_dt_num, dv1_dt_num, dx2_dt_num, dv2_dt_num], [x1, v1, x2, v2])

        x1_t = self.state[self.x1]
        v1_t = self.state[self.v1]
        x2_t = self.state[self.x2]
        v2_t = self.state[self.v2]

        x_t = np.array([[x1_t], [v1_t], [x2_t], [v2_t]])

        (ts, step_solution) = lsd.analytical_solution_matrix(A, x_t, h, tf)

        self.state[self.x1] = step_solution[-1][0, 0]
        self.state[self.v1] = step_solution[-1][1, 0]
        self.state[self.x2] = step_solution[-1][2, 0]
        self.state[self.v2] = step_solution[-1][3, 0]
        self.calcF()

        return fmi2OK


# Create cosim scenario with the two virtual FMUs
def run_cosim(msd1, msd2, sol, params, flip_order=False, x1_0=1.0, v1_0=0.0, x2_0=0.0, v2_0=0.0, H=0.01, tf=5.0,
              order=1):
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
    msd2_out = OutputConnection(value_type=VarType.REAL,
                                signal_type=SignalType.CONTINUOUS,
                                source_fmu=msd2,
                                source_vr=[msd2.x2, msd2.v2, msd2.Фx2x1, msd2.Фv2x1, msd2.Фx2v1, msd2.Фv2v1])

    msd1_sens = Connection(value_type=VarType.REAL,
                           signal_type=SignalType.CONTINUOUS,
                           source_fmu=msd1,
                           source_vr=[msd1.Фx1F, msd1.Фv1F])

    sol_out = OutputConnection(value_type=VarType.REAL,
                               signal_type=SignalType.CONTINUOUS,
                               source_fmu=sol,
                               source_vr=[sol.x1, sol.v1, sol.fk,
                                          sol.x2, sol.v2])

    connections = [msd1_out, msd1_in]
    output_connections = [msd1_out, msd1_in, msd1_sens, msd2_out, sol_out]  # Controls which signals are plotted

    # Sets initial values and parameters

    flip_order_as_float = 1.0 if flip_order else 0.0

    real_parameters = {msd1:
                           ([msd1.c1,
                             msd1.d1,
                             msd1.m1,
                             msd1.x1,
                             msd1.v1,
                             msd1.input_is_from_future,
                             msd1.input_approximation_order,
                             msd1.Фx1F,
                             msd1.Фv1F],
                            [c1_cons, d1_cons, m1_cons, x1_0, v1_0, flip_order_as_float, order, 0.0, 0.0]),
                       msd2:
                           ([msd2.c2,
                             msd2.d2,
                             msd2.m2,
                             msd2.cc,
                             msd2.dc,
                             msd2.x2,
                             msd2.v2,
                             msd2.input_is_from_future,
                             msd2.input_approximation_order,
                             msd2.Фx2x1,
                             msd2.Фv2x1,
                             msd2.Фx2v1,
                             msd2.Фv2v1],
                            [c2_cons, d2_cons, m2_cons, cc_cons, dc_cons, x2_0, v2_0, 1 - flip_order_as_float, order,
                             0.0, 0.0, 0.0, 0.0]),
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
                            [c1_cons, d1_cons, m1_cons, x1_0, v1_0,
                             c2_cons, d2_cons, m2_cons, cc_cons, dc_cons, x2_0, v2_0])
                       }

    fmus = [msd1, msd2, sol] if not flip_order else [msd2, msd1, sol]

    scenario = CosimScenario(
        fmus=fmus,
        connections=connections,
        step_size=H,
        print_interval=H,
        stop_time=tf,
        outputs=output_connections,
        real_parameters=real_parameters)

    # Instantiate FMUs and run cosim
    msd1.instantiate()
    msd2.instantiate()
    sol.instantiate()

    master = CustomGSRunner()

    def status(time):
        sys.stdout.write("\rTime: %.3f" % time)
        sys.stdout.flush()

    results = master.run_cosim(scenario, status)

    msd1.terminate()
    msd2.terminate()
    sol.terminate()

    return results


def get_states(msd1, msd2, results):
    sys_names = [msd1.instanceName, msd2.instanceName]
    states = [[msd1.x1, msd1.v1], [msd2.x2, msd2.v2]]
    res = []
    for sys in range(len(sys_names)):
        res.append([])
        for state in range(len(states[sys])):
            res[sys].append(np.array(results.out_signals[sys_names[sys]][states[sys][state]]))
    return res


def get_ref_states(sol, results):
    sol_name = sol.instanceName
    states = [[sol.x1, sol.v1], [sol.x2, sol.v2]]
    res = []
    for sys in range(len(states)):
        res.append([])
        for state in range(len(states[sys])):
            res[sys].append(np.array(results.out_signals[sol_name][states[sys][state]]))
    return res


# type 0: absolute error, type 1: relative error, type 2: L2 relative error
def compute_state_error(msd1, msd2, sol, results, typerr=0, ran='global'):
    states_sol_mapping = {
        msd1.instanceName: {
            msd1.x1: sol.x1,
            msd1.v1: sol.v1
        },
        msd2.instanceName: {
            msd2.x2: sol.x2,
            msd2.v2: sol.v2
        }
    }
    errors = {}
    for sys in states_sol_mapping:
        errors[sys] = {}
        for state in states_sol_mapping[sys]:
            res = results.out_signals[sys][state]
            ref = results.out_signals[sol.instanceName][state]
            if ran == 'global':
                errors[sys][state] = calc_error(res, ref, typerr)
            else:
                errors[sys][state] = calc_local_error(res, ref, typerr)
    return errors


def get_IO(msd1, msd2, results, before=True):
    sys_names = [msd1.instanceName, msd2.instanceName]
    inputs = [[msd1.fk], [msd2.x1, msd2.v1]]
    res = []
    for sys in range(2):
        res.append([])
        for inp in range(len(inputs[sys])):
            if before:
                res[sys].append(np.array(results.in_signals_before_step[sys_names[sys]][inputs[sys][inp]]))
            else:
                res[sys].append(np.array(results.in_signals_after_step[sys_names[sys]][inputs[sys][inp]]))
            # res[sys].append(np.array(results.out_signals[sys_names[sys]][inputs[sys][inp]]))
    return res


def get_ref_IO(sol, results):
    sol_name = sol.instanceName
    inputs = [[sol.fk], [sol.x1, sol.v1]]
    res = []
    for sys in range(len(inputs)):
        res.append([])
        for inp in range(len(inputs[sys])):
            res[sys].append(np.array(results.out_signals[sol_name][inputs[sys][inp]]))
    return res


def get_output(res, fmu, var_name):
    return np.array(res.out_signals[fmu.instanceName][fmu.__getattribute__(var_name)])


def get_input_before(res, fmu, var_name):
    return np.array(res.in_signals_before_step[fmu.instanceName][fmu.__getattribute__(var_name)])


def get_input_after(res, fmu, var_name):
    return np.array(res.in_signals_after_step[fmu.instanceName][fmu.__getattribute__(var_name)])


# type 0: absolute error, type 1: relative error, type 2: L2 relative error
def compute_IO_error(msd1, msd2, sol, results, typerr=0, ran='global'):
    sys_names = [msd1.instanceName, msd2.instanceName]
    sol_name = sol.instanceName
    inputs = [[msd1.x1, msd1.v1], [msd2.fk]]
    sol_inputs = [[sol.x1, sol.v1], [sol.fk]]
    res = []
    ref = []
    errors = []
    for sys in range(2):
        res.append([])
        ref.append([])
        for inp in range(len(inputs[sys])):
            res[sys].append(results.out_signals[sys_names[sys]][inputs[sys][inp]])
            ref[sys].append(results.out_signals[sol_name][sol_inputs[sys][inp]])
            if ran == 'global':
                errors.append(calc_error(res[sys][inp], ref[sys][inp], typerr))
            else:
                errors.append(calc_local_error(res[sys][inp], ref[sys][inp], typerr))
    # x1_e = max([math.fabs(a-b) for (a,b) in zip(results.out_signals[msd1.instanceName][msd1.x1], results.signals[sol.instanceName][sol.x1])])
    # v1_e = max([math.fabs(a-b) for (a,b) in zip(results.out_signals[msd1.instanceName][msd1.v1], results.signals[sol.instanceName][sol.v1])])
    # fk_e = max([math.fabs(a-b) for (a,b) in zip(results.out_signals[msd2.instanceName][msd2.fk], results.signals[sol.instanceName][sol.fk])])
    return errors


def get_momentum(msd1, msd2, params, results):
    momentum = []
    sys_names = [msd1.instanceName, msd2.instanceName]
    state_names = [[msd1.x1, msd1.v1], [msd2.x2, msd2.v2]]
    states = get_states(msd1, msd2, results)
    mass = [params[0], params[3]]
    for sys in range(len(sys_names)):
        momentum.append(mass[sys] * states[sys][1])

    return momentum


def get_force_transmission(msd1, msd2, params, results):
    force = []
    sys_names = [msd1.instanceName, msd2.instanceName]
    inputs = get_IO(msd1, msd2, results)  # [Fk],[x1, v1]
    force_weights = [[1], [params[6], params[7]]]  # dc, cc
    force_input = []
    for sys in range(len(sys_names)):
        force_input.append(np.zeros_like(inputs[sys][0]))
        for inp in range(len(inputs[sys])):
            force_input[sys] += inputs[sys][inp] * force_weights[sys][inp]

    return force_input


def get_energy(msd1, msd2, params, results):
    KE = []
    PE = []
    sys_names = [msd1.instanceName, msd2.instanceName]
    states = get_states(msd1, msd2, results)
    mass = [params[0], params[3]]
    spring = [params[2], params[5]]
    spring_coupling = params[7]
    for sys in range(len(sys_names)):
        KE.append(0.5 * mass[sys] * np.power(states[sys][1], 2))
        PE.append(0.5 * spring[sys] * np.power(states[sys][0], 2))
    PE[1] = np.add(PE[1], 0.5 * spring_coupling * np.power(np.subtract(states[1][0], states[0][0]), 2))

    return KE, PE


def get_ref_energy(sol, params, results):
    KE = []
    PE = []
    state_names = [[sol.x1, sol.v1], [sol.x2, sol.v2]]
    states = get_ref_states(sol, results)
    mass = [params[0], params[3]]
    spring = [params[2], params[5]]
    spring_coupling = params[7]
    for sys in range(len(state_names)):
        KE.append(0.5 * mass[sys] * np.power(states[sys][1], 2))
        PE.append(0.5 * spring[sys] * np.power(states[sys][0], 2))
    PE[1] = np.add(PE[1], 0.5 * spring_coupling * np.power(np.subtract(states[1][0], states[0][0]), 2))

    return KE, PE


def get_dissipation(msd1, msd2, params, results):
    loss = []
    sys_names = [msd1.instanceName, msd2.instanceName]
    states = get_states(msd1, msd2, results)
    damping = [params[1], params[4]]
    damping_coupling = params[6]
    for i in range(len(sys_names)):
        loss.append([])
        for j in range(len(results.timestamps) - 1):
            loss[i].append(
                0.5 * damping[i] * (states[i][1][j] + states[i][1][j + 1]) * (states[i][0][j + 1] - states[i][0][j]))
    for j in range(len(results.timestamps) - 1):
        loss[1][j] = np.add(loss[1][j], 0.5 * damping_coupling *
                            ((states[1][1][j] + states[1][1][j + 1]) - (states[0][1][j] + states[0][1][j + 1])) *
                            ((states[1][0][j + 1] - states[1][0][j]) - (states[0][0][j + 1] - states[0][0][j])))
    for i in range(len(sys_names)):
        loss[i].insert(0, 0.0)

    return loss


def get_ref_dissipation(sol, params, results):
    loss = []
    states = get_ref_states(sol, results)
    damping = [params[1], params[4]]
    damping_coupling = params[6]
    for i in range(len(states)):
        loss.append([])
        for j in range(len(results.timestamps) - 1):
            loss[i].append(
                0.5 * damping[i] * (states[i][1][j] + states[i][1][j + 1]) * (states[i][0][j + 1] - states[i][0][j]))
    for j in range(len(results.timestamps) - 1):
        loss[1][j] = np.add(loss[1][j], 0.5 * damping_coupling * (
                (states[1][1][j] + states[1][1][j + 1]) - (states[0][1][j] + states[0][1][j + 1])) * (
                                    (states[1][0][j + 1] - states[1][0][j]) - (
                                    states[0][0][j + 1] - states[0][0][j])))
    for i in range(len(states)):
        loss[i].insert(0, 0.0)

    return loss


# results display
def show_total_error(msd1, msd2, sol, results, typerr=0, var='all'):
    if var == 'all':
        for e in results.keys():
            state_errors = compute_state_error(msd1, msd2, sol, results[e], typerr)

            print(f"State errors for {e}:")
            print("msd1.x1={:.3e}".format(state_errors[msd1.instanceName][msd1.x1]))
            print("msd1.v1={:.3e}".format(state_errors[msd1.instanceName][msd1.v1]))
            print("msd2.x2={:.3e}".format(state_errors[msd2.instanceName][msd2.x2]))
            print("msd2.v2={:.3e}".format(state_errors[msd2.instanceName][msd2.v2]))

    if var == 'state':
        for e in results.keys():
            state_errors = compute_state_error(msd1, msd2, sol, results[e], typerr)
            print(f"State errors for {e}:")
            print("msd1.x1={:.3e}".format(state_errors[msd1.instanceName][msd1.x1]))
            print("msd1.v1={:.3e}".format(state_errors[msd1.instanceName][msd1.v1]))
            print("msd2.x2={:.3e}".format(state_errors[msd2.instanceName][msd2.x2]))
            print("msd2.v2={:.3e}".format(state_errors[msd2.instanceName][msd2.v2]))

    if var == 'input':
        for e in results.keys():
            input_errors = compute_IO_error(msd1, msd2, sol, results[e], typerr)
            print("Input Errors for " + e + " (x1, v1, fk): \t",
                  ", ".join("{:.3e}".format(j) for j in input_errors))


def show_local_state_error(msd1, msd2, sol, results, fmu, state, typerr=0):
    errors = {}
    for label in results:
        result = results[label]
        errors[label] = compute_state_error(msd1, msd2, sol, result, typerr, ran='local')
    for label in errors:
        plt.plot(results[label].timestamps, errors[label][fmu.instanceName][state], label=label)


def show_local_input_error(msd1, msd2, sol, label_res_map, inp, typerr=0):
    errors = {}
    for label in label_res_map:
        result = label_res_map[label]
        errors[label] = compute_IO_error(msd1, msd2, sol, result, typerr, ran='local')
    for label in errors:
        plt.plot(label_res_map[label].timestamps, errors[label][inp], label=label)


def compare_flip(msd1, msd2, sol, params, x0, H=0.01, order=1, tf=5.0):
    results_1 = run_cosim(msd1, msd2, sol, params, flip_order=False, x1_0=x0[0], v1_0=x0[1], x2_0=x0[2], v2_0=x0[3],
                          H=H,
                          order=order,
                          tf=tf)
    results_2 = run_cosim(msd1, msd2, sol, params, flip_order=True, x1_0=x0[0], v1_0=x0[1], x2_0=x0[2], v2_0=x0[3],
                          H=H,
                          order=order,
                          tf=tf)
    results_dictionary = {
        'MSD1->MSD2': results_1,
        'MSD2->MSD1': results_2
    }
    return results_dictionary


def sensitivityParams(msd1, msd2, sol, params, x0):
    numParams = len(params)
    tol = 1e-3
    initialStep = 1e2
    itermax = 10
    sens_noflip = [0] * numParams
    sens_flip = [0] * numParams
    sens_noflip_old = [0] * numParams
    sens_flip_old = [0] * numParams
    for i in range(numParams):
        err = tol + 1
        count = 0
        step = params[i] / initialStep
        while (err > tol and count < itermax):
            # central difference scheme
            # plus var
            params[i] = params[i] + step
            results = compare_flip(msd1, msd2, sol, params, x0)
            res1 = results['MSD1->MSD2']
            res2 = results['MSD2->MSD1']
            state_no_flip = get_states(msd1, msd2, res1)
            # input_no_flip = compute_IO_error(msd1,msd2,res1)
            state_flip = get_states(msd1, msd2, res2)
            # input_flip = compute_IO_error(msd1,msd2,res2)
            var_plus = [state_no_flip, state_flip]
            # minus var
            params[i] = params[i] - 2 * step
            results = compare_flip(msd1, msd2, sol, params, x0)
            res1 = results['MSD1->MSD2']
            res2 = results['MSD2->MSD1']
            state_no_flip = get_states(msd1, msd2, res1)
            # input_no_flip = compute_IO_error(msd1,msd2,res1)
            state_flip = get_states(msd1, msd2, res2)
            # input_flip = compute_IO_error(msd1,msd2,res2)
            var_minus = [state_no_flip, state_flip]
            params[i] = params[i] + step
            # calculation of sensitivities
            sens_noflip[i] = (np.array(var_plus[0]) - np.array(var_minus[0])) / 2 / step
            sens_flip[i] = (np.array(var_plus[1]) - np.array(var_minus[1])) / 2 / step
            if (count > 0):
                err = max(
                    [calc_error(sens_noflip[i], sens_noflip_old[i], 2), calc_error(sens_flip[i], sens_flip_old[i], 2)])
            sens_noflip_old[i] = sens_noflip[i]
            sens_flip_old[i] = sens_flip[i]
            count += 1
            step /= 2

    return res1.timestamps, sens_noflip, sens_flip


def get_abs_distance_inputs_before_after(res, fmu_name, input_var):
    return np.abs(get_diff_inputs_before_after(res, fmu_name, input_var))


def get_diff_inputs_before_after(res, fmu_name, input_var):
    array_a = np.array(res.in_signals_before_step[fmu_name][input_var])
    array_b = np.array(res.in_signals_after_step[fmu_name][input_var])
    return array_a - array_b


def compute_first_order_extrapolation(inputs_before_step):
    # Mimic the extrapolation done inside the FMU to estimate the point at the end of the co-sim step.
    """
    We assume that the first point of the extrapolation is correct in the sense that it matches the next input.
    This is similar to a zero-order hold from the right.
    It prevents a spike in the plots and is not that important otherwise.
    """
    extrapolation_ending_points = [inputs_before_step[1]] # -> to avoid peak | switch to [inputs_before_step[0]] 
    for i in range(1, len(inputs_before_step)):
        # linear extrapolation
        extrapolated = 2 * inputs_before_step[i] - inputs_before_step[i - 1]
        extrapolation_ending_points.append(extrapolated)

    extrapolation_ending_points = np.array(extrapolation_ending_points)
    return extrapolation_ending_points


def get_extrapolated_input_estimation_error(res, fmu, input_var, ind, H, gte=False):
    before_step = np.array(res.in_signals_before_step[fmu.instanceName][input_var])
    after_step = np.array(res.in_signals_after_step[fmu.instanceName][input_var])

    assert np.array_equal(before_step[1:], after_step[:-1])  # this holds for extrapolated values

    extrapolation_ending_points = compute_first_order_extrapolation(before_step)

    local_input_error = after_step - extrapolation_ending_points
    if gte:
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


def get_interpolated_input_estimation_error(res, fmu, input_var, ind, H, gte=False):
    before_step = np.array(res.in_signals_before_step[fmu.instanceName][input_var])
    after_step = np.array(res.in_signals_after_step[fmu.instanceName][input_var])

    assert np.array_equal(before_step, after_step)  # before_step and after_step is the same for interpolated values

    # The calculation we want is Extrapolation[i] = 2*inputs[i-1] - inputs[i-2]
    # So the first two elements are dummy
    extrapolation_ending_points = [before_step[0], before_step[1]]

    for i in range(2, len(before_step)):
        extrapolated = 2 * before_step[i-1] - before_step[i-2]
        extrapolation_ending_points.append(extrapolated)

    extrapolation_ending_points = np.array(extrapolation_ending_points)

    local_input_error = after_step - extrapolation_ending_points

    if gte:
        global_input_error = [local_input_error[0] * H]
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


def get_analytical_error(res, fmu, sol, var_name, type=0, ran='global'):
    cosim_res = get_output(res, fmu, var_name)
    sol_res = get_output(res, sol, var_name)
    return np.abs((cosim_res - sol_res))


def extrapolation_error_aligned(signal, sol):
    # Compute the extrapolation
    extrapolation = compute_first_order_extrapolation(signal)

    # Notice that the indexes have to be aligned, since extrapolation[0] corresponds to sol[1].
    # Insert a dummy value at index 0 and drop the last element, so we can match the timestamps.
    extrapolation = np.insert(extrapolation[:-1], 0, extrapolation[0])
    # Now extrapolation[1] corresponds to sol[1]
    extrapolation_error = sol - extrapolation
    return extrapolation, extrapolation_error
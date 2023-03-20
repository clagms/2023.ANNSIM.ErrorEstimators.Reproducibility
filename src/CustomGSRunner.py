from typing import Dict, List

from PyCosimLibrary.results import CosimResults
from PyCosimLibrary.runner import CosimRunner
from PyCosimLibrary.scenario import CosimScenario
from fmpy.fmi2 import fmi2OK

from src.CosimResultsInputs import CosimResultsInputs


class CustomGSRunner(CosimRunner):
    """
    This class implements the gauss seidel co-simulation algorithm customized to record input data
    before and after the cosim step.
    """
    def __init__(self):
        self.results = None

    def valid_scenario(self, scenario: CosimScenario):
        super(CustomGSRunner, self).valid_scenario(scenario)
        assert scenario.print_interval == scenario.step_size, "This runner assumes that these settings hold."

    def init_input_results(self, scenario: CosimScenario, input_results):
        # Go over connections and initialize the results.
        for connection in scenario.connections:
            if connection.target_fmu.instanceName not in input_results.keys():
                input_results[connection.target_fmu.instanceName] = {}
            for vr in connection.target_vr:
                assert vr not in input_results[connection.target_fmu.instanceName].keys(), \
                    "No inputs with more than one incoming output."
                input_results[connection.target_fmu.instanceName][vr] = []

    def init_results(self, scenario: CosimScenario) -> CosimResultsInputs:
        self.results = super(CustomGSRunner, self).init_results(scenario, results=CosimResultsInputs())
        self.results.in_signals_before_step = {}
        self.results.in_signals_after_step = {}

        self.init_input_results(scenario, self.results.in_signals_before_step)
        self.init_input_results(scenario, self.results.in_signals_after_step)

        return self.results

    def propagate_outputs_fmu(self, scenario, f):
        fmu_connections = filter(lambda c: c.source_fmu == f, scenario.connections)
        self.propagate_outputs(fmu_connections) # What does this do?

    def snapshot_fmu_inputs(self, fmu, scenario: CosimScenario, input_signals):
        fmu_in_connections = filter(lambda c: c.target_fmu == fmu, scenario.connections)
        for connection in fmu_in_connections:
            values = self.get_fmu_vars(fmu, connection.target_vr, connection.value_type)
            for i in range(len(connection.target_vr)):
                value_append = values[i]
                vr = connection.target_vr[i]
                signal = input_signals[fmu.instanceName][vr]
                signal.append(value_append)

    def run_cosim_step(self, time, scenario: CosimScenario):
        for f in scenario.fmus:
            # Record inputs to FMU f.
            self.snapshot_fmu_inputs(f, scenario, self.results.in_signals_before_step)
            res = f.doStep(time, scenario.step_size)
            assert res == fmi2OK, "Step failed."
            self.propagate_outputs_fmu(scenario, f)

        # Record fmu inputs after step.
        for f in scenario.fmus:
            # Record inputs to FMU f.
            self.snapshot_fmu_inputs(f, scenario, self.results.in_signals_after_step)

        # Check invariants on the results:
        #  there should be as many input snapshots as output snapshots as this point.
        self.check_result_invariants(self.results)

    def check_result_invariants(self, results: CosimResultsInputs):
        num_results = len(results.timestamps)

        def check(signal_results, signal_name):
            for fmu_name in signal_results.keys():
                for vr in signal_results[fmu_name].keys():
                    signal = signal_results[fmu_name][vr]
                    assert len(signal) == num_results, \
                        f"Signal {signal_name} of {fmu_name}.{vr} expected to have size {num_results} " \
                        f"but has instead {len(signal)}"

        check(results.out_signals, "out_signals")
        check(results.in_signals_before_step, "in_signals_before_step")
        check(results.in_signals_after_step, "in_signals_after_step")


    def terminate_cosim(self, scenario: CosimScenario):
        # Take another snapshot of the FMU inputs, to ensure consistency with results.timestamps.
        for f in scenario.fmus:
            # Record inputs to FMU f.
            self.snapshot_fmu_inputs(f, scenario, self.results.in_signals_before_step)
            self.snapshot_fmu_inputs(f, scenario, self.results.in_signals_after_step)

        self.check_result_invariants(self.results)
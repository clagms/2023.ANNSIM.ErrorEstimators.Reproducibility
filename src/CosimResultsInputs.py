from typing import Dict, List
from PyCosimLibrary.results import CosimResults


class CosimResultsInputs(CosimResults):
    in_signals_before_step: Dict[str, Dict[int, List[float]]]
    in_signals_after_step: Dict[str, Dict[int, List[float]]]

    def __init__(self):
        super(CosimResultsInputs, self).__init__()

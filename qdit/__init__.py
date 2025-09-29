from qdit.gates import *
from qdit.utils.verification import *
from qdit.noise import *
from qdit.utils.math import *
from qdit.benchmarking import *
from qdit.compiling import *
from qdit.gates.multi_qudit import (
    CZ, CX
)
from qdit.gates.single_qudit import (
    Id,
    X,
    Y,
    Z,
    WeylOperator,
    Rx, Ry, Rz,
    F, S, Ma, H, P, Id
)

__version__ = "0.1.0"
__all__ = [
    # Core utilities
    "multiplicative_group_mod_d",
    "clifford_card",
    "check_pauli_relation",
    "verify_clifford_group",
    
    # Multi-qudit gates
    "CZ",
    "CX",

    # Single-qudit gates
    "Id",
    "X",
    "Y",
    "Z",
    "WeylOperator",
    "Rx", "Ry", "Rz",
    "F",
    "S",
    "Ma",
    "H",
    "P",
    
    # Benchmarking and compilation
    "Tableau",
    "RandomizedBenchmarkingResult",
    "single_qutrit_randomized_benchmarking",
    "RandomCompiling",
    "CircuitDecomposer",
    "Compiler"
    
    # Examples and demos
    "example_qutrit_clifford_usage",
]
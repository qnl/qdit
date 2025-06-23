from qnlib.gates import *
from qnlib.utils.verification import *
from qnlib.noise import *
from qnlib.utils.math import *
from qnlib.benchmarking import *
from qnlib.compiling import *
from qnlib.gates.multi_qudit import (
    QuditFourierGate,
    QuditPhaseGate,
    QuditCZGate,
    QuditCXGate,
    QuditMultiplicationGate,
    H, S, CZ, CX
)
from qnlib.gates.single_qudit import (
    PauliXGate,
    PauliYGate,
    PauliZGate,
    WeylOperator,
    Rx, Ry, Rz
)

__version__ = "0.1.0"
__all__ = [
    # Core utilities
    "multiplicative_group_mod_d",
    "clifford_card",
    "check_pauli_relation",
    "verify_clifford_group",
    
    # Multi-qudit Clifford gates
    "QuditFourierGate",
    "QuditPhaseGate", 
    "QuditCZGate",
    "QuditCXGate",
    "QuditMultiplicationGate",
    "H", "S", "CZ", "CX",
    
    # Single-qudit gates
    "PauliXGate",
    "PauliYGate",
    "PauliZGate",
    "WeylOperator",
    "Rx", "Ry", "Rz",
    
    # Benchmarking and compilation
    "Tableau",
    "RandomizedBenchmarkingResult",
    "single_qutrit_randomized_benchmarking",
    "RandomCompiling",
    "CircuitDecomposer",
    
    # Examples and demos
    "example_qutrit_clifford_usage",
]
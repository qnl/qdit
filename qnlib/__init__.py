from qnlib.gates import *
from qnlib.utils.verification import *
from qnlib.noise import *
from qnlib.utils.math import *
from qnlib.benchmarking import *
from qnlib.compiling import *
from qnlib.gates.cliffords.quditCliffords import (
    QuditFourierGate,
    QuditPhaseGate,
    QuditCZGate,
    QuditCXGate,
    QuditMultiplicationGate
)
from qnlib.gates.paulis.quditPaulis import (
    PauliXGate,
    PauliYGate,
    PauliZGate,
    WeylOperator
)

__version__ = "0.1.0"
__all__ = [
    # Core utilities
    "multiplicative_group_mod_d",
    "clifford_card",
    "check_pauli_relation",
    "verify_clifford_group",
    
    # Clifford gates
    "QuditFourierGate",
    "QuditPhaseGate", 
    "QuditCZGate",
    "QuditCXGate",
    "QuditMultiplicationGate",
    "QuditCliffordGates",
    "single_qutrit_cliffords",
    
    # Pauli gates
    "PauliXGate",
    "PauliYGate",
    "PauliZGate",
    "WeylOperator",
    
    # Benchmarking and compilation
    "Tableau",
    "RandomizedBenchmarkingResult",
    "single_qutrit_randomized_benchmarking",
    "RandomCompiling",
    "CircuitDecomposer",
    
    # Examples and demos
    "example_qutrit_clifford_usage",
]
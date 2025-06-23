import cirq
import numpy as np
from typing import Tuple
from .utils import clifford_generator_mats
from ...utils.math import multiplicative_group_mod_d

class QuditFourierGate(cirq.Gate):
    """Quantum Fourier transform gate for qutrits (generalized Hadamard)."""
    
    def __init__(self, dimension: int, name: str = None):
        self.dimension = dimension
        self.name = name
        self._matrix = clifford_generator_mats(dimension)[0]  # F matrix
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension,)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return self.name if self.name else "F"
        
    def __pow__(self, exponent: int) -> 'QuditFourierGate':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = QuditFourierGate(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        new_gate.name = f"{self.name if self.name else 'F'}inv" if exponent == -1 else f"{self.name if self.name else 'F'}{exponent}"
        return new_gate

class QuditPhaseGate(cirq.Gate):
    """Phase gate for qutrits."""
    
    def __init__(self, dimension: int, name: str = None):
        self.dimension = dimension
        self.name = name
        self._matrix = clifford_generator_mats(dimension)[1]  # S matrix
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension,)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return self.name if self.name else "S"
        
    def __pow__(self, exponent: int) -> 'QuditPhaseGate':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = QuditPhaseGate(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        new_gate.name = f"{self.name if self.name else 'S'}inv" if exponent == -1 else f"{self.name if self.name else 'S'}{exponent}"
        return new_gate

class QuditMultiplicationGate(cirq.Gate):
    """Multiplication gate Ma for qudits."""
    
    def __init__(self, dimension: int, a: int, name: str = None):
        self.dimension = dimension
        self.a = a
        self.name = name
        M_gates = clifford_generator_mats(dimension)[2]
        a_idx = list(multiplicative_group_mod_d(dimension)).index(a)
        self._matrix = M_gates[a_idx]
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension,)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return self.name if self.name else f"M{self.a}"
        
    def __pow__(self, exponent: int) -> 'QuditMultiplicationGate':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = QuditMultiplicationGate(self.dimension, self.a, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        base_name = self.name if self.name else f"M{self.a}"
        new_gate.name = f"{base_name}inv" if exponent == -1 else f"{base_name}{exponent}"
        return new_gate

class QuditCZGate(cirq.Gate):
    """Controlled-Z gate for qutrits."""
    
    def __init__(self, dimension: int, control_idx: int = 0, target_idx: int = 1, name: str = None):
        self.dimension = dimension
        self._matrix = clifford_generator_mats(dimension)[3]  # CZ matrix
        self.control_idx, self.target_idx = control_idx, target_idx
        self.name = name
        
    def _num_qubits_(self) -> int:
        return 2
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension, self.dimension)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        if self.name:
            return f"{self.name}c", f"{self.name}t"
        if self.control_idx > self.target_idx:
            return f"CZ{self.control_idx}", f"CZ{self.target_idx}"
        else:
            return f"CZ{self.target_idx}", f"CZ{self.control_idx}"
        
    def __pow__(self, exponent: int) -> 'QuditCZGate':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = QuditCZGate(self.dimension, self.control_idx, self.target_idx, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        base_name = self.name if self.name else "CZ"
        new_gate.name = f"{base_name}inv" if exponent == -1 else f"{base_name}{exponent}"
        return new_gate

class QuditCXGate(cirq.Gate):
    """Controlled-X (SUM) gate for qutrits."""
    
    def __init__(self, dimension: int, control_idx: int = 0, target_idx: int = 1, name: str = None):
        self.dimension = dimension
        self._matrix = clifford_generator_mats(dimension)[4]  # CX matrix
        self.control_idx, self.target_idx = control_idx, target_idx
        self.name = name
        
    def _num_qubits_(self) -> int:
        return 2
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension, self.dimension)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        if self.name:
            return f"{self.name}c", f"{self.name}t"
        if self.control_idx > self.target_idx:
            return f"CX{self.control_idx}", f"CX{self.target_idx}"
        else:
            return f"CX{self.target_idx}", f"CX{self.control_idx}"
        
    def __pow__(self, exponent: int) -> 'QuditCXGate':
        """Returns the gate raised to a power.
        
        Args:
            exponent: The power to raise the gate to.
            
        Returns:
            A new QuditCXGate with the matrix raised to the given power.
        """
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
            
        new_gate = QuditCXGate(self.dimension, self.control_idx, self.target_idx, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        base_name = self.name if self.name else "CX"
        new_gate.name = f"{base_name}inv" if exponent == -1 else f"{base_name}{exponent}"
        return new_gate

# Convenience instances for qutrit gates
H = lambda d, name=None: QuditFourierGate(d, name)
S = lambda d, name=None: QuditPhaseGate(d, name)
CZ = lambda d, name=None: QuditCZGate(d, name=name)
CX = lambda d, name=None: QuditCXGate(d, name=name)
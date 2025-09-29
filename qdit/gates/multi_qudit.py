import cirq
import numpy as np
from typing import Tuple
from .utils import clifford_generator_mats

class CZ(cirq.Gate):
    """Controlled-Z gate for qutrits."""
    
    def __init__(self, dimension: int, name: str = None):
        self.dimension = dimension
        self._matrix = clifford_generator_mats(dimension)[3]  # CZ matrix
        self.name = name
        
    def _num_qubits_(self) -> int:
        return 2
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension, self.dimension)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return f"CZc", f"CZt"
        
    def __pow__(self, exponent: int) -> 'CZ':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = CZ(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        base_name = self.name if self.name else "CZ"
        new_gate.name = f"{base_name}inv" if exponent == -1 else f"{base_name}{exponent}"
        return new_gate

class CX(cirq.Gate):
    """Controlled-X (SUM) gate for qutrits."""
    
    def __init__(self, dimension: int, name: str = None):
        self.dimension = dimension
        self._matrix = clifford_generator_mats(dimension)[4]  # CX matrix
        self.name = name
        
    def _num_qubits_(self) -> int:
        return 2
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension, self.dimension)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return f"CXc", f"CXt"
        
    def __pow__(self, exponent: int) -> 'CX':
        """Returns the gate raised to a power.
        
        Args:
            exponent: The power to raise the gate to.
            
        Returns:
            A CX with the matrix raised to the given power.
        """
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
            
        new_gate = CX(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        base_name = self.name if self.name else "CX"
        new_gate.name = f"{base_name}inv" if exponent == -1 else f"{base_name}{exponent}"
        return new_gate
    

class iSWAP(cirq.Gate):
    def __init__(self, dimension: int=2, name: str=None):
        self.d = dimension
        self.name = name
        self._matrix = np.array([
            [1,0,0,0],
            [0,0,1j,0],
            [0,1j,0,0],
            [0,0,0,1]
        ])
    
    def _num_qubits_(self) -> int:
        return 2
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2, 2)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return f"⨂", f"⨂"
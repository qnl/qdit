import cirq
import numpy as np
from typing import Sequence, Union, Tuple
from .utils import qudit_pauli_mats

class PauliXGate(cirq.Gate):
    """A d-dimensional Pauli X gate."""
    
    def __init__(self, dimension: int, power: int = 1):
        """
        Args:
            dimension: Dimension of the qudit
            power: Power of the X operator (default=1)
        """
        super(PauliXGate, self)
        self.d = dimension
        self.power = power % dimension
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        _, X, _ = qudit_pauli_mats(self.d)
        return X
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.power == 1:
            return f"X({self.d})"
        return f"X{self.power}({self.d})"
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
        
class PauliZGate(cirq.Gate):
    """A d-dimensional Pauli Z gate."""
    
    def __init__(self, dimension: int, power: int = 1):
        """
        Args:
            dimension: Dimension of the qudit
            power: Power of the Z operator (default=1)
        """
        super(PauliZGate, self)
        self.d = dimension
        self.power = power % dimension
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        _, _, Z = qudit_pauli_mats(self.d)
        return Z
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.power == 1:
            return f"Z({self.d})"
        return f"Z{self.power}({self.d})"
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
        
class PauliYGate(cirq.Gate):
    """A d-dimensional Pauli Y gate."""
    
    def __init__(self, dimension: int, power: int = 1):
        """
        Args:
            dimension: Dimension of the qudit
            power: Power of the Y operator (default=1)
        """
        super(PauliYGate, self)
        self.d = dimension
        self.power = power % dimension
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        # Get X and Z matrices
        w_til, X, Z = qudit_pauli_mats(self.d)
        
        # Phase factor tau
        tau = np.power(w_til, 1/2, dtype=np.clongdouble)
        
        # Compute Y = tau * X^dagger * Z^dagger
        Y = X.conj().T @ tau @ Z.conj().T
        
        # Return the requested power of Y
        return np.linalg.matrix_power(Y, self.power)
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.power == 1:
            return f"Y({self.d})"
        return f"Y{self.power}({self.d})"
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
        
class WeylOperator(cirq.Gate):
    """A d-dimensional Weyl operator W(a,b)."""
    
    def __init__(self, a: Union[int, Sequence[int]], 
                 b: Union[int, Sequence[int]], 
                 dimension: int = 3,
                 num_qudits: int = 1):
        """
        Args:
            a: Power(s) of X operator
            b: Power(s) of Z operator
            dimension: Dimension of each qudit (default=3)
            num_qudits: Number of qudits (default=1)
        """
        super(WeylOperator, self)
        self.d = dimension
        self.nq = num_qudits
        self.a = np.atleast_1d(a) % dimension
        self.b = np.atleast_1d(b) % dimension
        
        if len(self.a) != len(self.b) or len(self.a) != num_qudits:
            raise ValueError("Length of a and b must match number of qudits")
            
    def _num_qubits_(self) -> int:
        return self.nq
        
    def _unitary_(self) -> np.ndarray:
        # Phase factor
        tau = np.power(-1, self.d) * np.exp(np.pi * 1j / self.d)
        phase = np.power(tau, -np.dot(self.a, self.b))
        
        # Generate X and Z matrices
        _, X, Z = qudit_pauli_mats(self.d)
        
        # Generate operators for each qudit
        operators = []
        for i in range(self.nq):
            op = (np.linalg.matrix_power(X, int(self.a[i])) @ 
                 np.linalg.matrix_power(Z, int(self.b[i])))
            operators.append(op)
            
        # Compute tensor product
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
            
        return phase * result
    
    def _circuit_diagram_info_(self, args) -> Tuple[str, ...]:
        labels = []
        for i in range(self.nq):
            if self.a[i] == 0 and self.b[i] == 0:
                labels.append(f"I({self.d})")
            elif self.b[i] == 0:
                labels.append(f"X{self.a[i]}({self.d})")
            elif self.a[i] == 0:
                labels.append(f"Z{self.b[i]}({self.d})")
            else:
                labels.append(f"W({self.a[i]},{self.b[i]})")
        return tuple(labels)
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,) * self.nq
import cirq
import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Union, Tuple
from .utils import qudit_pauli_mats
from .utils import clifford_generator_mats
from ..utils.math import multiplicative_group_mod_d


subspace_map = {
    '1,2': 'GE',
    '2,3': 'EF'
}

def embed_2x2(matrix: np.ndarray, dim: int, subspace: Tuple[int, int]) -> np.ndarray:
    """Embed a 2x2 matrix into a larger identity matrix at specified subspace indices.
    
    Args:
        matrix: 2x2 matrix to embed
        dim: Dimension of target space
        subspace: Tuple of (i,j) indices specifying the subspace location
        
    Returns:
        NDArray: dim x dim matrix with embedded 2x2 block
    """
    result = np.eye(dim, dtype=complex)
    result[np.ix_(subspace, subspace)] = matrix
    return result

def rx(theta: float) -> NDArray:
    """Unitary rotation about the x-axis.

    Args:
        theta (float): angle of rotation.

    Returns:
        NDArray: matrix expression of a unitary rotation about the x-axis by an
            angle theta.
    """
    return np.array([[np.cos(theta/2.), -1.j*np.sin(theta/2.)],
                     [-1.j*np.sin(theta/2.), np.cos(theta/2.)]])


def ry(theta: float) -> NDArray:
    """Unitary rotation about the y-axis.

    Args:
        theta (float): angle of rotation.

    Returns:
        NDArray: matrix expression of a unitary rotation about the y-axis by an
            angle theta.
    """
    return np.array([[np.cos(theta/2.), -np.sin(theta/2.)],
                     [np.sin(theta/2.), np.cos(theta/2.)]])


def rz(theta: float) -> NDArray:
    """Unitary rotation about the z-axis.

    Args:
        theta (float): angle of rotation.

    Returns:
        NDArray: matrix expression of a unitary rotation about the z-axis by an
            angle theta.
    """
    return np.array([[np.exp(1.j*theta/2), 0.],
                     [0., np.exp(-1.j*theta/2)]])


class Rx(cirq.Gate):
    """A rotation gate about the x-axis."""
    
    def __init__(self, rads: float, dimension: int = 2, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(Rx, self)
    
        self._rads = rads
        self.d = dimension

        if isinstance(subspace, str):
            i, j = map(int, subspace.split(","))
            self._subspace = (i, j)
        else:
            self._subspace = subspace

    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        if self.d == 2:
            return rx(self._rads)
        return embed_2x2(rx(self._rads), self.d, (self._subspace[0]-1, self._subspace[1]-1))
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.d == 2:
            return f"Rx({np.real(self._rads/np.pi):.2f}π)"
        return (f"Rx({np.real(self._rads/np.pi):.2f}π)", )
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
    
class X90(cirq.Gate):
    """A π/2 rotation gate about the x-axis."""
    def __init__(self, dimension: int = 2, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(X90, self)
        self.d = dimension

        if isinstance(subspace, str):
            i, j = map(int, subspace.split(","))
            self._subspace = (i, j)
        else:
            self._subspace = subspace

    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        if self.d == 2:
            return rx(np.pi/2)
        return embed_2x2(rx(np.pi/2), self.d, (self._subspace[0]-1, self._subspace[1]-1))

    def _circuit_diagram_info_(self, args) -> str:
        if self.d == 2:
            return f"X90"
        return (f"X90{self._subspace[0]}{self._subspace[1]}", )
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
    

class Ry(cirq.Gate):
    """A rotation gate about the y-axis."""
    
    def __init__(self, rads: float, dimension: int = 2, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(Ry, self)
        self._rads = rads
        self.d = dimension

        if isinstance(subspace, str):
            i, j = map(int, subspace.split(","))
            self._subspace = (i, j)
        else:
            self._subspace = subspace
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        if self.d == 2:
            return ry(self._rads)
        return embed_2x2(ry(self._rads), self.d, (self._subspace[0]-1, self._subspace[1]-1))
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.d == 2:
            return f"Ry({np.real(self._rads/np.pi):.2f}π)"
        return (f"Ry({np.real(self._rads/np.pi):.2f}π){self._subspace[0]}{self._subspace[1]}", )
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
    

class Rz(cirq.Gate):
    """A rotation gate about the z-axis."""
    
    def __init__(self, rads: float, dimension: int = 2, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(Rz, self)
        self._rads = rads
        self.d = dimension

        if isinstance(subspace, str):
            i, j = map(int, subspace.split(","))
            self._subspace = (i, j)
        else:
            self._subspace = subspace

    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        if self.d == 2:
            return rz(self._rads)
        return embed_2x2(rz(self._rads), self.d, (self._subspace[0]-1, self._subspace[1]-1))
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.d == 2:
            return f"Rz({np.real(self._rads/np.pi):.2f}π)"
        return (f"Rz({np.real(self._rads/np.pi):.2f}π){self._subspace[0]}{self._subspace[1]}", )
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
    
class Ph(cirq.Gate):
    """Qudit global phase gate."""
    
    def __init__(self, rads: float, dimension: int = 2):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
        """
        super(Ph, self)
        self._rads = rads
        self.d = dimension

    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        # return self._rads * np.eye(self.d)
        return np.array([
            [np.exp(1j*self._rads),0],
            [0,np.exp(1j*self._rads)]
        ])
    
    def _circuit_diagram_info_(self, args) -> str:
        return f"Ph({np.real(self._rads/np.pi):.2f}π)"
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)

class Id(cirq.Gate):
    """Identity gate for a d-dimensional qudit."""
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: Dimension of the qudit
        """
        super(Id, self)
        self.d = dimension
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        return np.eye(self.d)
    
    def _circuit_diagram_info_(self, args) -> str:
        return ""
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)

class X(cirq.Gate):
    """A d-dimensional Pauli X gate."""
    
    def __init__(self, dimension: int = 2, power: int = 1):
        """
        Args:
            dimension: Dimension of the qudit
            power: Power of the X operator (default=1)
        """
        super(X, self)
        self.d = dimension
        self.power = power % dimension
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        _, X, _ = qudit_pauli_mats(self.d)
        return np.linalg.matrix_power(X,self.power)
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.power == 1:
            return f"X"
        return f"X{self.power}"
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
        
class Z(cirq.Gate):
    """A d-dimensional Pauli Z gate."""
    
    def __init__(self, dimension: int = 2, power: int = 1):
        """
        Args:
            dimension: Dimension of the qudit
            power: Power of the Z operator (default=1)
        """
        super(Z, self)
        self.d = dimension
        self.power = power % dimension
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        _, _, Z = qudit_pauli_mats(self.d)
        return np.linalg.matrix_power(Z,self.power)
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.power == 1:
            return f"Z"
        return f"Z{self.power}"
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)
        
class Y(cirq.Gate):
    """A d-dimensional Pauli Y gate."""
    
    def __init__(self, dimension: int = 2, power: int = 1):
        """
        Args:
            dimension: Dimension of the qudit
            power: Power of the Y operator (default=1)
        """
        super(Y, self)
        self.d = dimension
        self.power = power % dimension
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        # Get X and Z matrices
        w_til, X, Z = qudit_pauli_mats(self.d)
        
        # Phase factor tau
        tau = np.power(w_til, 1/2)
        
        # Compute Y = tau * X^dagger * Z^dagger
        Y = X.conj().T @ tau @ Z.conj().T
        
        # Return the requested power of Y
        return np.linalg.matrix_power(Y, self.power)
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.power == 1:
            return f"Y"
        return f"Y{self.power}"
    
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
                labels.append(f"I")
            elif self.b[i] == 0:
                labels.append(f"X{self.a[i]}")
            elif self.a[i] == 0:
                labels.append(f"Z{self.b[i]}")
            else:
                labels.append(f"W({self.a[i]},{self.b[i]})")
        return tuple(labels)
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,) * self.nq

class F(cirq.Gate):
    """Quantum Fourier transform gate for qudits (generalized Hadamard)."""
    
    def __init__(self, dimension: int = 2, name: str = None):
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
        
    def __pow__(self, exponent: int) -> 'F':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = F(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        new_gate.name = f"{self.name if self.name else 'F'}inv" if exponent == -1 else f"{self.name if self.name else 'F'}{exponent}"
        return new_gate

class S(cirq.Gate):
    """Phase gate for qudits."""
    
    def __init__(self, dimension: int = 2, name: str = None):
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
        
    def __pow__(self, exponent: int) -> 'S':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = S(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        new_gate.name = f"{self.name if self.name else 'S'}inv" if exponent == -1 else f"{self.name if self.name else 'S'}{exponent}"
        return new_gate


class Sdg(cirq.Gate):
    """Phase gate adjoint for qudits."""
    
    def __init__(self, dimension: int = 2, name: str = None):
        self.dimension = dimension
        self.name = name
        self._matrix = np.transpose(np.conjugate(clifford_generator_mats(dimension)[1]))  # Sdg matrix
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension,)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return self.name if self.name else "Sdg"
        
    def __pow__(self, exponent: int) -> 'Sdg':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = S(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        new_gate.name = f"{self.name if self.name else 'Sdg'}inv" if exponent == -1 else f"{self.name if self.name else 'Sdg'}{exponent}"
        return new_gate

class Ma(cirq.Gate):
    """Multiplication gate Ma for qudits."""
    
    def __init__(self, dimension: int = 2, a: int = 1, name: str = None):
        self.dimension = dimension
        self.a = a
        self.name = name
        M_gates = clifford_generator_mats(dimension)[2]
        try:
            a_idx = list(multiplicative_group_mod_d(dimension)).index(a)
        except ValueError:
            group_list = list(multiplicative_group_mod_d(dimension))
            raise ValueError(f"a={a} is not in the multiplicative group mod {dimension}. Group: {group_list}")
        self._matrix = M_gates[a_idx]
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension,)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return self.name if self.name else f"M{self.a}"
        
    def __pow__(self, exponent: int) -> 'Ma':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = Ma(self.dimension, self.a, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        base_name = self.name if self.name else f"M{self.a}"
        new_gate.name = f"{base_name}inv" if exponent == -1 else f"{base_name}{exponent}"
        return new_gate

class C(cirq.Gate):
    """Cycling gate for qubits"""
    def __init__(self, dimension: int=2, name: str=None):
        self.dimension = dimension
        self.name = name
        self._matrix = 1/2*np.array([[1-1j,-1-1j],
                                     [1-1j,1+1j]])
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension,)
        
    def _unitary_(self) -> np.ndarray:
        return self._matrix
    
    def _circuit_diagram_info_(self, args):
        return self.name if self.name else f"C"
        
    def __pow__(self, exponent: int) -> 'C':
        if not isinstance(exponent, int):
            raise ValueError("Exponent must be an integer")
        new_gate = C(self.dimension, self.name)
        new_gate._matrix = np.linalg.matrix_power(self._matrix, exponent)
        base_name = self.name if self.name else f"C"
        new_gate.name = f"{base_name}inv" if exponent == -1 else f"{base_name}{exponent}"
        return new_gate

# Convenience instances for qutrit gates
H = lambda dimension, name=None: F(dimension, name)
P = lambda dimension, name=None: S(dimension, name)
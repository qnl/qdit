### Qubit X, Y, and Z rotations ###
import cirq
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union

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
    i, j = subspace
    result[i-1:i+1, j-1:j+1] = matrix
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
    return np.array([[np.exp(-1.j*theta/2), 0.],
                     [0., np.exp(1.j*theta/2)]])


class Rx(cirq.Gate):
    """A rotation gate about the x-axis."""
    
    def __init__(self, rads: float, dimension: int = 2, n_qudits: int=1, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(Rx, self)
    
        self._rads = rads
        self.d = dimension
        self.n = n_qudits
        self._subspace = subspace_map[subspace]
        self._connected = False

        if isinstance(subspace, str):
            i, j = map(int, subspace.split(","))
            self._subspace = (i, j)
        else:
            self._subspace = subspace

        # if not (1 <= self._subspace[0] <= dimension and 1 <= self._subspace[1] <= dimension):
        #     raise ValueError(f"Subspace indices {self._subspace} must be between 1 and dimension {dimension}")
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        if self.d == 2:
            return rx(self._rads)
        return embed_2x2(rx(self._rads), self.d**self.n, self._subspace)
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.d**self.n == 2:
            return f"Rx({np.real(self._rads/np.pi):.1f}π)"
        return (f"Rx({np.real(self._rads/np.pi):.1f}π)", ) * self.n
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)*self.n
    
class X90(cirq.Gate):
    """A π/2 rotation gate about the x-axis."""
    def __init__(self, dimension: int = 2, n_qudits: int=1, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(X90, self)
        self.d = dimension
        self.n = n_qudits
        self._subspace = subspace_map[subspace]
        self._connected = False

        if isinstance(subspace, str):
            i, j = map(int, subspace.split(","))
            self._subspace = (i, j)
        else:
            self._subspace = subspace

        # if not (1 <= self._subspace[0] <= dimension and 1 <= self._subspace[1] <= dimension):
        #     raise ValueError(f"Subspace indices {self._subspace} must be between 1 and dimension {dimension}")
        
    def _num_qubits_(self) -> int:
        return 1
        
    def _unitary_(self) -> np.ndarray:
        if self.d == 2:
            return rx(np.pi/2)
        return embed_2x2(rx(np.pi/2), self.d**self.n, self._subspace)

    def _circuit_diagram_info_(self, args) -> str:
        if self.d**self.n == 2:
            return f"X90"
        return (f"X90", ) * self.n
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)*self.n
    

class Ry(cirq.Gate):
    """A rotation gate about the y-axis."""
    
    def __init__(self, rads: float, dimension: int = 2, n_qudits: int=1, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(Ry, self)
        self._rads = rads
        self.d = dimension
        self.n = n_qudits
        self._subspace = subspace_map[subspace]
        self._connected = False

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
        return embed_2x2(ry(self._rads), self.d**self.n, self._subspace)
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.d**self.n == 2:
            return f"Ry({np.real(self._rads/np.pi):.1f}π)"
        return (f"Ry({np.real(self._rads/np.pi):.1f}π){self._subspace[0]}{self._subspace[1]}", ) * self.n
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)*self.n
    

class Rz(cirq.Gate):
    """A rotation gate about the z-axis."""
    
    def __init__(self, rads: float, dimension: int = 2, n_qudits: int=1, subspace: Union[str, Tuple[int, int]] = "1,2"):
        """
        Args:
            theta: Rotation angle in radians
            dimension: Dimension of the qudit (default=2)
            subspace: Two energy levels to rotate between, as "i,j" string or (i,j) tuple (default="1,2")
        """
        super(Rz, self)
        self._rads = rads
        self.d = dimension
        self.n = n_qudits
        self._subspace = subspace_map[subspace]
        self._connected = False

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
        return embed_2x2(rz(self._rads), self.d**self.n, self._subspace)
    
    def _circuit_diagram_info_(self, args) -> str:
        if self.d**self.n == 2:
            return f"Rz({np.real(self._rads/np.pi):.1f}π)"
        return (f"Rz", ) * self.n
    
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.d,)*self.n
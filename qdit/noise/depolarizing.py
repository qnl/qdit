import numpy as np
import cirq
from typing import Tuple, Sequence

class Depolarizing(cirq.Gate):
    """Single-qutrit symmetrically depolarizing error channel."""
    
    def __init__(self, p: float):
        """
        Args:
            p (float): Each qutrit Pauli operator is applied with probability p/8
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be in the interval [0,1]")
        self.p = p
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (3,)
        
    def _kraus_(self) -> Sequence[np.ndarray]:
        w = np.exp(2j * np.pi / 3)
        w2 = w * w
        
        K0 = np.sqrt(1 - self.p) * np.eye(3)
        normalization = np.sqrt(self.p / 8)
        
        K1 = normalization * np.array([[1, 0, 0], [0, w, 0], [0, 0, w2]], dtype=complex)
        K2 = normalization * np.array([[1, 0, 0], [0, w2, 0], [0, 0, w]], dtype=complex)
        K3 = normalization * np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
        K4 = normalization * np.array([[0, w, 0], [0, 0, w2], [1, 0, 0]], dtype=complex)
        K5 = normalization * np.array([[0, w2, 0], [0, 0, w], [1, 0, 0]], dtype=complex)
        K6 = normalization * np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
        K7 = normalization * np.array([[0, 0, w2], [1, 0, 0], [0, w, 0]], dtype=complex)
        K8 = normalization * np.array([[0, 0, w], [1, 0, 0], [0, w2, 0]], dtype=complex)
        
        return [K0, K1, K2, K3, K4, K5, K6, K7, K8]

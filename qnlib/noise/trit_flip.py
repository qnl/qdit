import numpy as np
import cirq
from typing import Tuple, Sequence

class TritFlip(cirq.Gate):
    """Single-qutrit trit flip error channel."""
    
    def __init__(self, p_01: float, p_02: float, p_12: float):
        """
        Args:
            p_01 (float): Probability of |0⟩ ↔ |1⟩ flip
            p_02 (float): Probability of |0⟩ ↔ |2⟩ flip
            p_12 (float): Probability of |1⟩ ↔ |2⟩ flip
        """
        for p in (p_01, p_02, p_12):
            if not 0 <= p <= 1:
                raise ValueError("All probabilities must be in the interval [0,1]")
        if not 0 <= p_01 + p_02 + p_12 <= 1:
            raise ValueError("Sum of probabilities must be in the interval [0,1]")
            
        self.p_01 = p_01
        self.p_02 = p_02
        self.p_12 = p_12
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (3,)
        
    def _kraus_(self) -> Sequence[np.ndarray]:
        K0 = np.sqrt(1 - (self.p_01 + self.p_02 + self.p_12)) * np.eye(3)
        
        K1 = np.sqrt(self.p_01) * np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        K2 = np.sqrt(self.p_02) * np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        
        K3 = np.sqrt(self.p_12) * np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        
        return [K0, K1, K2, K3]

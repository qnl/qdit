import numpy as np
import cirq
from typing import Tuple, Sequence

class AmplitudeDamping(cirq.Gate):
    """Single-qutrit amplitude damping error channel."""
    
    def __init__(self, gamma_10: float, gamma_20: float, gamma_21: float):
        """
        Args:
            gamma_10 (float): |1⟩ → |0⟩ amplitude damping probability
            gamma_20 (float): |2⟩ → |0⟩ amplitude damping probability
            gamma_21 (float): |2⟩ → |1⟩ amplitude damping probability
        """
        for gamma in (gamma_10, gamma_20, gamma_21):
            if not 0 <= gamma <= 1:
                raise ValueError("Each probability must be in the interval [0,1]")
        if not 0 <= gamma_20 + gamma_21 <= 1:
            raise ValueError("gamma_20 + gamma_21 must be in the interval [0,1]")
            
        self.gamma_10 = gamma_10
        self.gamma_20 = gamma_20
        self.gamma_21 = gamma_21
        
    def _qid_shape_(self) -> Tuple[int, ...]:
        return (3,)
        
    def _kraus_(self) -> Sequence[np.ndarray]:
        K0 = np.diag([1, np.sqrt(1 - self.gamma_10), 
                      np.sqrt(1 - (self.gamma_20 + self.gamma_21))])
        
        K1 = np.zeros((3, 3), dtype=complex)
        K1[0, 1] = np.sqrt(self.gamma_10)
        
        K2 = np.zeros((3, 3), dtype=complex)
        K2[0, 2] = np.sqrt(self.gamma_20)
        
        K3 = np.zeros((3, 3), dtype=complex)
        K3[1, 2] = np.sqrt(self.gamma_21)
        
        return [K0, K1, K2, K3]

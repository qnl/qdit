import numpy as np
from typing import List
from ..gates.utils import qudit_pauli_mats

def verify_clifford_group(matrices: List[np.ndarray], d: int = 3):
    """Verify that the generated matrices form a group and normalize the Pauli group.
    
    Args:
        matrices (List[np.ndarray]): List of matrices to verify
        d (int): Dimension of the qudit system (default: 3)
    """
    def is_unitary(m):
        return np.allclose(m @ m.conj().T, np.eye(d))
    
    _, X, Z = qudit_pauli_mats(d)
    paulis = [X, Z]
    
    for C in matrices:
        assert is_unitary(C), "Non-unitary matrix found"
        for P in paulis:
            conjugated = C @ P @ C.conj().T
            assert np.allclose(conjugated @ conjugated @ conjugated, np.eye(d)), \
                   "Matrix doesn't normalize Pauli group"
            

def verify_group_properties(d, elements):
    """
    Verifies that the elements form a group under multiplication mod d
    by checking closure and existence of inverses.
    
    Args:
        d (int): The modulus
        elements (list): List of group elements
        
    Returns:
        bool: True if group properties are satisfied
    """
    # Check closure
    for a in elements:
        for b in elements:
            product = (a * b) % d
            if product not in elements:
                return False
                
    # Check inverses
    for a in elements:
        has_inverse = False
        for b in elements:
            if (a * b) % d == 1:
                has_inverse = True
                break
        if not has_inverse:
            return False
            
    return True
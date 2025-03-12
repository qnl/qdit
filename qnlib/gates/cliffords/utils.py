from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import cirq
import itertools
from ...utils.math import multiplicative_group_mod_d

def quditCZ(d, t):
    """
    Creates a control-Z operator matrix for a d-dimensional system with parameter t.
    
    Parameters:
    d (int): Dimension of the system
    t (float): Parameter for the phase
    
    Returns:
    numpy.ndarray: d^2 x d^2 matrix representing the control-Z operator
    """
    # Initialize the operator matrix
    dim = d * d  # Total dimension of the tensor product space
    operator = np.zeros((dim, dim), dtype=complex)
    
    # Implement the double sum
    for q1 in range(d):
        for q2 in range(d):
            # Calculate the phase factor
            phase = t ** ((2 * q1 * q2)%d)
            
            # Calculate the index in the full tensor product space
            index = q1 * d + q2
            
            # Add the contribution to the operator
            operator[index, index] = phase
    
    return operator

def quditCX(d):
    """
    Creates a d-level CNOT (SUM) operator matrix.
    
    Parameters:
    d (int): Dimension of each subsystem
    
    Returns:
    numpy.ndarray: d^2 x d^2 matrix representing the d-level CNOT operator
    """
    # Initialize the operator matrix
    dim = d * d  # Total dimension of the tensor product space
    operator = np.zeros((dim, dim))
    
    # Implement the double sum
    for q1 in range(d):
        for q2 in range(d):
            # Calculate (q1 + q2) mod d for the target qudit
            target_state = (q1 + q2) % d
            
            # Calculate indices in the tensor product space
            input_index = q1 * d + q2 
            output_index = q1 * d + target_state
            
            # Add the matrix element
            operator[output_index, input_index] = 1
    
    return operator

def clifford_generator_mats(d: int, canonical: bool = False) -> tuple:
    """
    Returns the generators of the qudit Clifford group.
    
    This function implements the construction described in https://arxiv.org/pdf/1102.3354
    
    Parameters
    ----------
    d : int
        The dimension of the qudit system. Must be a positive integer ≥ 2.
    canonical : bool, optional
        If True, returns a canonical form of the matrices (currently not implemented).
        Default is False.
    
    Returns
    -------
    tuple
        A 5-tuple containing:
        - F : ndarray
            The Fourier gate matrix (d x d). Generalizes the Hadamard gate to d dimensions.
        - S : ndarray
            The phase gate matrix (d x d). Applies phase shifts based on qudit states.
        - M_gates : list
            List of multiplication gates Ma for each a in the multiplicative group mod d.
        - CZ : ndarray
            The controlled-Z gate matrix (d^2 x d^2).
        - CX : ndarray
            The controlled-X gate matrix (d^2 x d^2). Generalizes CNOT to d dimensions.
    
    Notes
    -----
    - The function uses τ = exp(2πi(d^2+1)/d) as the principal phase factor.
    - The Fourier gate F is defined as F[j,i] = τ^(2ij)/√d
    - The phase gate S is diagonal with entries S[i,i] = τ^(i^2)
    - The multiplication gates Ma are diagonal with entries (ai mod d)
    - CZ and CX are d^2 x d^2 matrices implementing controlled operations
    
    Examples
    --------
    >>> F, S, M_gates, CZ, CX = generate_n_qudit_clifford_mats(3)
    # Returns matrices for a qutrit (d=3) system
    
    References
    ----------
    For detailed mathematical background, see:
    https://arxiv.org/pdf/1102.3354
    
    Raises
    ------
    ValueError
        If d is less than 2 or not an integer.
    NotImplementedError
        If canonical=True is specified (currently not implemented).
    """
    if canonical:
        return NotImplementedError
        
    # Phase factor τ = exp(2πi(d^2+1)/d)
    tau = np.exp(1j*np.pi*(d**2+1)/d, dtype=np.clongdouble)
    
    # Fourier gate (generalized Hadamard)
    F = np.array([[np.power(tau,int(2*i*j), dtype=np.clongdouble)/np.sqrt(d, dtype=np.clongdouble) for i in range(d)] for j in range(d)], dtype=np.clongdouble)
    
    # Phase gate
    S = np.diag([np.power(tau,int(i**2), dtype=np.clongdouble) for i in range(d)])
    
    # Multiplication gates
    a_range = multiplicative_group_mod_d(d)
    M_gates = [np.array([[1 if (a*i) % d == j else 0 for i in range(d)] for j in range(d)], dtype=np.clongdouble) for a in a_range]    
    
    # Controlled gates
    CZ = quditCZ(d, tau)
    CX = quditCX(d)
    
    return F, S, M_gates, CZ, CX
    
@dataclass
class QuditCliffordGates:
    """Collection of qutrit Clifford gates decomposed into elementary operations."""
    gates: Sequence[Sequence[cirq.Gate]]

def single_qudit_clifford_mats(d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the fundamental d-dimensional Clifford matrices.
    
    For d=3, generates the quantum Fourier transform (H), phase gate (S), and shift gate (X).
    
    Args:
        d (int): Dimension of the qudit system (currently only supports d=3)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: H, S, and X matrices
    """
    assert d == 3, "Currently only implemented for qutrits (d=3)"
    
    w = np.exp(2j * np.pi / d)
    
    zi = np.round(np.exp(2j * np.pi / 9), 10)
    
    H = (1/np.sqrt(d))*np.array([[1,1,1],[1,w,w**2],[1,w**2,w]])
    
    S = zi**8*np.array([[1,0,0],[0,1,0],[0,0,w]])
    
    X = np.array([[int((i+1)%d==j) for i in range(d)] for j in range(d)])

    return H, S, X

def single_qutrit_cliffords(reduced: bool=True) -> QuditCliffordGates:
    """Generate all 216 elements of the single-qutrit Clifford group.
    
    Args:
        reduced (bool): If True, use reduced powers of S (mod 9), otherwise use all 27
        
    Returns:
        QuditCliffordGates: Object containing all Clifford gates
    """
    d = 3
    H, S, X = single_qudit_clifford_mats(d)
    I = np.eye(d)
    
    L = np.array([I, H, S@H, S@S@H])
    M = np.array([I, H@H])
    N = []
    
    phase_depth = 9 if reduced else 27
    for s_power in range(phase_depth):
        S_n = np.linalg.matrix_power(S, s_power)
        for x_power in range(3):
            X_n = np.linalg.matrix_power(X, x_power)
            N.append(np.dot(S_n, X_n))
    
    permutations = list(itertools.product(L, M, N))
    clifford_matrices = [c1@c2@c3 for c1,c2,c3 in permutations]
    
    assert len(clifford_matrices) == 216 if reduced else 648, \
           f"Generated {len(clifford_matrices)} gates instead of {216 if reduced else 648}"
    
    clifford_gates = []
    for matrix in clifford_matrices:
        gate = cirq.MatrixGate(matrix, qid_shape=[d])
        clifford_gates.append([gate])
    
    return QuditCliffordGates(gates=clifford_gates)
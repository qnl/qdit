import cirq
import numpy as np
from typing import List, Tuple, Optional, Sequence
from itertools import product
from dataclasses import dataclass
from ..utils.math import multiplicative_group_mod_d

def matrix_to_cirq_gate(
    matrix: np.ndarray,
    dimension: Optional[int] = None,
    name: Optional[str] = None,
    num_controls: int = 0
) -> cirq.Gate:
    """Convert a unitary matrix to a Cirq gate.
    
    Args:
        matrix: Unitary matrix representing the gate
        dimension: Qudit dimension. If None, inferred from matrix size
        name: Optional name for the gate
        
    Returns:
        cirq.MatrixGate that implements the unitary
    """
    controlled = (num_controls != 0)

    if dimension is None:
        dimension = matrix.shape[0]
        
    if not controlled and matrix.shape != (dimension, dimension):
        raise ValueError(f"Matrix shape {matrix.shape} incompatible with dimension {dimension}")
    
    gate= cirq.MatrixGate(
        matrix,
        name=name,
        qid_shape=[dimension for _ in range(num_controls+1)] if controlled else (dimension,)
    )   

    gate._circuit_diagram_info_(name)

    return gate

def qudit_pauli_mats(d: int) -> Tuple[np.matrix, np.matrix, np.matrix]:
    """Returns the d-dimensional Pauli matrices.
    
    Args:
        d (int): Dimension of the qudit system
        
    Returns:
        Tuple[np.matrix, np.matrix, np.matrix]: Phase factor (w_til), X matrix, and Z matrix
    """
    w = np.exp(2j * np.pi / d)

    X = np.matrix([[int((i + 1) % d == j) for i in range(d)] for j in range(d)])
    
    Z = np.matrix([[np.power(w,i,dtype=np.clongdouble) * int(i == j) for i in range(d)] for j in range(d)])
    
    w_til = w*np.eye(d, d) if d % 2 else np.power(w,(1/2)) * np.eye(d, d)
    
    return w_til, X, Z

def quditX_mat(d: int) -> np.matrix:
    """Generate the d-dimensional Pauli X gate.
    
    Args:
        d (int): Dimension of the qudit system
        
    Returns:
        np.matrix: X matrix
    """
    X = np.matrix([[int((i + 1) % d == j) for i in range(d)] for j in range(d)])
     
    return X

def quditZ_mat(d: int) -> np.matrix:
    """Generate the d-dimensional Pauli Z gate.
    
    Args:
        d (int): Dimension of the qudit system
        
    Returns:
        np.matrix: Z matrix
    """
    w = np.exp(2j * np.pi / d)

    Z = np.matrix([[w**i * int(i == j) for i in range(d)] for j in range(d)])
     
    return Z

def generate_valid_ops(d: int) -> List[str]:
    """Generate valid Pauli operators for a d-dimensional system.
    
    Args:
        d (int): Dimension of the qudit system
    
    Returns:
        List[str]: List of valid Pauli operator representations
    """
    # Start with identity
    ops = ['I']
    
    # Generate X and Z powers
    x_powers = [f'X{i}' if i > 1 else 'X' for i in range(1, d)]
    z_powers = [f'Z{i}' if i > 1 else 'Z' for i in range(1, d)]
    
    # Combine X and Z powers
    ops.extend(x_powers + z_powers)
    
    # Generate combined X and Z operators
    for x_pow in x_powers:
        for z_pow in z_powers:
            ops.append(f'{x_pow}{z_pow}')
    
    return ops

def qudit_pauli_group(nq: int, d: int):
    """
    Generates an iterator onto the Pauli group of n qudits,
    where n is given by nq and dimension is given by d.
    
    Args:
        nq (int): Number of qudits
        d (int): Dimension of each qudit
    
    Returns:
        Iterator of Pauli operators
    """
    valid_ops = generate_valid_ops(d)
    
    for op in product(valid_ops, repeat=nq):
        yield "⊗".join(op)

def weyl(a, b, nq: int = 1, d: int = 3):
    """
    Compute the Weyl operator W(a,b) for given X and Z powers.
    
    Args:
        a (list): Powers of X operators for each qudit
        b (list): Powers of Z operators for each qudit
        nq (int): Number of qudits
        d (int): Dimension of each qudit
        
    Returns:
        numpy.ndarray: The resulting Weyl operator
    """
    # Ensure powers are within [0,d-1]
    a = a%d
    b = b%d
    # Phase factor
    tau = np.power(-1,d) * np.exp(np.pi * 1j / d)
    phase = np.power(tau,(-np.dot(a, b)))
    # Generate single-qudit operator
    _,X,Z = qudit_pauli_mats(d)
    
    if hasattr(a,'__len__') and hasattr(b,'__len__'):
        assert len(a) == len(b) == nq, 'List of X and Z powers must match number of qudits and each other'
        # Generate single-qudit operators
        operators = []
        for i in range(nq):
            # Compute X^a * Z^b for this qudit
            op = np.linalg.matrix_power(X, a[i]) @ np.linalg.matrix_power(Z, b[i])
            operators.append(op)
        # Compute tensor product of all operators
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        return phase * result
    else:
        # Compute X^a * Z^b for this qudit
        op = np.linalg.matrix_power(X, int(a)) @ np.linalg.matrix_power(Z, int(b))
        return op
    
def rand_pauli(dim):
    """
    Generate a random generalized Pauli operator in a d-dimensional Hilbert space.

    The function creates a random Pauli operator by combining a random phase with
    a Weyl operator. The phase is chosen as exp(iπ*k/d) where k is random,
    and the Weyl operator parameters are randomly selected.

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space.

    Returns
    -------
    numpy.ndarray
        A dim × dim complex matrix representing the random generalized Pauli operator.
        The matrix is unitary and has the form exp(iπ*k/d)W(m,n) where W is a Weyl
        operator and k,m,n are random integers between 0 and dim-1.

    Notes
    -----
    The resulting operator is guaranteed to be unitary and is uniformly sampled
    from the set of generalized Pauli operators in the given dimension.
    """
    pauli_vec = np.round((dim-1)*np.random.rand(3))
    print(pauli_vec)
    return np.exp(1j*np.pi/dim*pauli_vec[0])*weyl(pauli_vec[1], pauli_vec[2], d=dim)

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
            phase = t ** ((2 * q1 * q2))
            
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
    tau = np.power(-1, d)*np.exp(1j*np.pi/d)
    
    # Fourier gate (generalized Hadamard)
    F = np.array([[np.power(tau,int(2*i*j))/np.sqrt(d) for i in range(d)] for j in range(d)])
    
    # Phase gate
    S = np.diag([np.power(tau,int(i**2)) for i in range(d)])
    
    # Multiplication gates
    a_range = multiplicative_group_mod_d(d)
    M_gates = [np.array([[1 if (a*i) % d == j else 0 for i in range(d)] for j in range(d)]) for a in a_range]    
    
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
    
    permutations = list(product(L, M, N))
    clifford_matrices = [c1@c2@c3 for c1,c2,c3 in permutations]
    
    assert len(clifford_matrices) == 216 if reduced else 648, \
           f"Generated {len(clifford_matrices)} gates instead of {216 if reduced else 648}"
    
    clifford_gates = []
    for matrix in clifford_matrices:
        gate = cirq.MatrixGate(matrix, qid_shape=[d])
        clifford_gates.append([gate])
    
    return QuditCliffordGates(gates=clifford_gates)
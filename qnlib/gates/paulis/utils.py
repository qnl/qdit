import numpy as np
from typing import List, Tuple
from itertools import product

def qudit_pauli_mats(d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the d-dimensional Pauli matrices.
    
    Args:
        d (int): Dimension of the qudit system
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Phase factor (w_til), X matrix, and Z matrix
    """
    w = np.exp(2j * np.pi / d, dtype=np.clongdouble)

    X = np.array([[int((i + 1) % d == j) for i in range(d)] for j in range(d)], dtype=np.clongdouble)
    
    Z = np.array([[np.power(w,i,dtype=np.clongdouble) * int(i == j) for i in range(d)] for j in range(d)], dtype=np.clongdouble)
    
    w_til = w*np.eye(d, d, dtype=np.clongdouble) if d % 2 else np.power(w,(1/2), dtype=np.clongdouble) * np.eye(d, d, dtype=np.clongdouble)
    
    return w_til, X, Z

def quditX_mat(d: int) -> np.ndarray:
    """Generate the d-dimensional Pauli X gate.
    
    Args:
        d (int): Dimension of the qudit system
        
    Returns:
        np.ndarray: X matrix
    """
    X = np.array([[int((i + 1) % d == j) for i in range(d)] for j in range(d)])
     
    return X

def quditZ_mat(d: int) -> np.ndarray:
    """Generate the d-dimensional Pauli Z gate.
    
    Args:
        d (int): Dimension of the qudit system
        
    Returns:
        np.ndarray: Z matrix
    """
    w = np.exp(2j * np.pi / d)

    Z = np.array([[w**i * int(i == j) for i in range(d)] for j in range(d)])
     
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
    tau = np.power(-1,d,dtype=np.clongdouble) * np.exp(np.pi * 1j / d, dtype=np.clongdouble)
    phase = np.power(tau,(-np.dot(a, b)), dtype=np.clongdouble)
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
    return np.exp(1j*np.pi/dim*pauli_vec[0])*weyl(pauli_vec[1], pauli_vec[2], d=dim)
import numpy as np
from typing import List, Tuple
from collections import defaultdict

def multiplicative_group_mod_d(d: int) -> List[int]:
    """
    Returns the multiplicative group of integers modulo d.
    This group consists of numbers that are coprime to d.
    
    Args:
        d (int): The modulus
        
    Returns:
        list: List of integers that form the multiplicative group mod d
    """
    if d <= 0:
        raise ValueError("Modulus must be positive")
        
    # A number is in the multiplicative group if and only if
    # it is coprime to the modulus
    return np.array([i for i in range(1, d) if np.gcd(i, d) == 1], dtype=np.float64)

def clifford_card(d, k):
    return d**(2*k+1)*(d**(2*k)-1)

def find_inverse_mat(mat):
    """
    Find the inverse of a matrix by repeated multiplication, with robust complex number handling.
    
    Args:
        mat (np.ndarray): Input matrix
    
    Returns:
        np.ndarray: Inverse matrix if found, None otherwise
    """
    n = np.shape(mat)[0]
    inv_mat = np.copy(mat)
    max_iter = n * n
    
    test_product = np.round(mat @ inv_mat, 10)
    i = 0
    while not is_identity(test_product):
        test_product = np.round(mat @ inv_mat, 10)
        inv_mat = np.round(inv_mat @ mat, 10)
        i+=1
    return inv_mat

def get_matrix_order(mat, max_order=100):
    """
    Find the order of a matrix (smallest positive integer n where mat^n = I).
    """
    n = np.shape(mat)[0]
    current = np.eye(n)
    for i in range(1, max_order + 1):
        current = np.round(current @ mat, 10)
        if np.allclose(current, np.eye(n), rtol=1e-10, atol=1e-10):
            return i
    return None

def efficient_inverse(mat):
    """
    Efficiently compute inverse for matrices with finite order.
    For a matrix A of order n, A^(-1) = A^(n-1)
    """
    order = get_matrix_order(mat)
    if order is None:
        return None
    
    # For a matrix of order n, its inverse is itself raised to power (n-1)
    n = np.shape(mat)[0]
    result = mat
    for _ in range(order - 2):
        result = np.round(result @ mat, 10)
    return result

def is_commuting(a, b):
    return is_identity(a @ b @ hermitian(a) @ hermitian(b))

def is_anticommuting(self, A: np.ndarray, B: np.ndarray, rtol: float = 1e-10, atol: float = 1e-10) -> bool:
        """
        Check if two operators (numpy arrays) anti-commute.
        Two operators A and B anti-commute if AB + BA = 0.
        
        Parameters:
        -----------
        A : np.ndarray
            First operator (2D numpy array)
        B : np.ndarray
            Second operator (2D numpy array)
        rtol : float, optional
            Relative tolerance for comparison (default: 1e-10)
        atol : float, optional
            Absolute tolerance for comparison (default: 1e-10)
        
        Returns:
        --------
        bool
            True if operators anti-commute, False otherwise
        
        Raises:
        -------
        ValueError
            If operators are not square matrices or have incompatible dimensions
        """
        # Check if inputs are 2D arrays
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Operators must be 2D arrays")
        
        # Check if operators are square matrices
        if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            raise ValueError("Operators must be square matrices")
        
        # Check if dimensions match
        if A.shape != B.shape:
            raise ValueError("Operators must have the same dimensions")
        
        # Calculate AB + BA
        anticommutator = A @ B + B @ A
        
        # Check if result is zero (within numerical tolerance)
        return np.allclose(anticommutator, np.zeros_like(anticommutator), rtol=rtol, atol=atol)

def is_identity(test_mat):
        """Helper function to check if a matrix is the identity, handling complex numbers."""
        n = np.shape(test_mat)[0]
        identity = np.eye(n, dtype=complex)
        # Check real and imaginary parts separately with appropriate tolerance
        real_close = np.allclose(test_mat.real, identity.real, rtol=1e-5, atol=1e-5)
        imag_close = np.allclose(np.abs(test_mat.imag), np.zeros_like(identity), rtol=1e-5, atol=1e-5)
        return real_close and imag_close

def hermitian(mat):
    return np.conjugate(mat.T)

def check_pauli_relation(pvec0, pvec1, d: int, non_commute: int=None):
        """Check if two Pauli operators commute or anticommute.
        
        Args:
            pvec0: First Pauli vector
            pvec1: Second Pauli vector 
            d: The modulus for the symplectic product
            non_commute: If None, check for commutation, else check for this non-commutation value
        
        Returns:
            bool: True if operators have specified commutation relation, False otherwise
        """
        u, v = np.asarray(pvec0), np.asarray(pvec1)
        assert len(u)==len(v), 'Pauli vectors must be the same length.'
        n = len(u) // 2
        s = np.block([[np.zeros((n,n)), np.eye(n)],[-np.eye(n), np.zeros((n,n))]])
        symplectic = (u.T @ s @ v) % d
        return symplectic == non_commute if non_commute else symplectic == 0

def sympletic_product(s1, s2, d):
    if type(s1) == list or type(s1) == tuple:
        s1 = np.array(s1)
    if type(s2) == list or type(s2) == tuple:
        s2 = np.array(s2)
    n = len(s1)//2
    return (s1[:n] @ s2[n:] - s1[n:] @ s2[:n]) % d

def normalize_matrix_phase(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a unitary matrix by rotating its global phase so that the first
    non-zero element is real and positive.
    """
    # Find first non-zero element
    nonzero_idx = np.nonzero(matrix)
    if len(nonzero_idx[0]) == 0:
        return matrix
    
    first_element = matrix[nonzero_idx[0][0], nonzero_idx[1][0]]
    phase = first_element / np.abs(first_element)
    return matrix / phase

def get_matrix_fingerprint(matrix: np.ndarray, tol: float = 1e-10) -> tuple:
    """
    Create a phase-invariant fingerprint of a matrix using absolute values 
    and phase differences between elements.
    """
    # Get absolute values of all elements
    abs_values = np.abs(matrix)
    
    # Get phase differences between adjacent elements
    phases = np.angle(matrix)
    phase_diffs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs_values[i,j] > tol:  # Only consider non-zero elements
                for ii in range(i, matrix.shape[0]):
                    for jj in range(j+1 if ii==i else 0, matrix.shape[1]):
                        if abs_values[ii,jj] > tol:
                            diff = (phases[ii,jj] - phases[i,j]) % (2*np.pi)
                            phase_diffs.append(round(diff, 10))
    
    return (tuple(abs_values.flatten()), tuple(sorted(phase_diffs)))

def analyze_clifford_set(matrices: List[np.ndarray], tol: float = 1e-10) -> Tuple[List[np.ndarray], dict]:
    """
    Analyze the input set of matrices and group them by their phase-invariant properties.
    """
    groups = defaultdict(list)
    for idx, matrix in enumerate(matrices):
        fingerprint = get_matrix_fingerprint(matrix, tol)
        groups[fingerprint].append(idx)
    
    # Sort groups by size for analysis
    group_sizes = {k: len(v) for k, v in groups.items()}
    
    return groups, group_sizes

def verify_phase_relationship(mat1: np.ndarray, mat2: np.ndarray, tol: float = 1e-10) -> Tuple[bool, complex]:
    """
    Verify if two matrices are related by a global phase and return the phase if they are.
    """
    ratios = []
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            if abs(mat1[i,j]) > tol and abs(mat2[i,j]) > tol:
                ratio = mat2[i,j] / mat1[i,j]
                ratios.append(ratio)
    
    if not ratios:
        return False, None
    
    # Check if all ratios are the same (up to numerical precision)
    reference = ratios[0]
    is_phase_related = all(abs(r - reference) < tol for r in ratios)
    
    return is_phase_related, reference if is_phase_related else None

def unitary_to_special_unitary(U):
    """
    Convert a unitary matrix U ∈ U(N) to a special unitary matrix ∈ SU(N).
    
    Parameters:
    -----------
    U : numpy.ndarray
        A unitary matrix (U†U = UU† = I) of shape (N, N)
    dtype : type
        
    Returns:
    --------
    numpy.ndarray
        A special unitary matrix (determinant = 1) of the same shape
    """
    # Check if the input is approximately unitary
    N = U.shape[0]
    if not np.allclose(U @ U.conj().T, np.eye(N), atol=1e-10):
        raise ValueError("Input matrix is not unitary")
    
    # Calculate the determinant (which should have magnitude 1 for unitary matrices)
    det_U = np.linalg.det(np.array(U, dtype=np.complex128))
    
    # Calculate the phase of the determinant
    # For a unitary matrix, det(U) = e^(iθ) for some θ
    theta = np.angle(det_U)
    
    # Create the correction factor: e^(-iθ/N)
    correction = np.exp(-1j * theta / N, dtype=np.complex128)
    
    # Apply the correction to get a special unitary matrix
    U_special = correction * U
    
    # Verify the determinant is now ~1
    new_det = np.linalg.det(np.array(U_special, dtype=np.complex128))
    if not np.isclose(new_det, 1.0, atol=1e-10):
        raise RuntimeError(f"Failed to create special unitary matrix. New determinant: {new_det}")
    
    return U_special, correction
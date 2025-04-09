"""
Quantum circuit decomposition module for converting quantum circuits into elementary rotations.

This module provides functionality to decompose quantum circuits into a sequence of elementary
rotation gates (Rx, Ry, Rz) using SU(n) factorization techniques.
"""

import numpy as np
from scipy.stats import unitary_group
from cirq import Circuit, MatrixGate, LineQid, unitary
from qnlib.utils.sun_factorization import sun_factorization
from qnlib.utils.sun_reconstruction import sun_reconstruction
from qnlib.utils.math import unitary_to_special_unitary
from numpy.linalg import det

class CircuitDecomposer:
    """
    A class for decomposing quantum circuits into elementary rotation gates.

    This class takes a quantum circuit and decomposes it into elementary rotation gates
    using SU(n) matrix factorization. It supports conversion between unitary and special
    unitary matrices, factorization, and reconstruction of quantum circuits.

    Attributes:
        original_unitary (np.ndarray): The unitary matrix of the input circuit
        n (int): Dimension of the unitary matrix
        det (complex): Determinant of the original unitary
        special_unitary (np.ndarray): The SU(n) matrix derived from the original unitary
        correction (complex): The correction factor to convert between U(n) and SU(n)
        factors (list): Factorization results stored as (indices, values) pairs
    """

    @staticmethod
    def rx(a):
        """
        Generate a rotation matrix around the X axis.

        Args:
            a (float): Rotation angle in radians

        Returns:
            np.ndarray: 2x2 rotation matrix
        """
        return np.array([[np.exp(1j*a/2), 0], [0, np.exp(-1j*a/2)]])
    
    @staticmethod
    def ry(b):
        """
        Generate a rotation matrix around the Y axis.

        Args:
            b (float): Rotation angle in radians

        Returns:
            np.ndarray: 2x2 rotation matrix
        """
        return np.array([[np.cos(b/2), -np.sin(b/2)], [np.sin(b/2), np.cos(b/2)]])
    
    @staticmethod
    def rz(g):
        """
        Generate a rotation matrix around the Z axis.

        Args:
            g (float): Rotation angle in radians

        Returns:
            np.ndarray: 2x2 rotation matrix
        """
        return np.array([[np.exp(1j*g/2), 0], [0, np.exp(-1j*g/2)]])

    def __init__(self, circuit):
        """
        Initialize the CircuitDecomposer with a quantum circuit.

        Args:
            circuit (cirq.Circuit): Input quantum circuit to be decomposed

        The constructor converts the input circuit to its unitary matrix representation
        and prepares it for SU(n) factorization by computing necessary attributes.
        """
        # Convert circuit to unitary matrix representation
        circuit_unitary = np.array(unitary(circuit), dtype=np.complex128)
        # Store the original unitary matrix
        self.original_unitary = circuit_unitary
        # Get dimension of the unitary matrix 
        self.n = len(circuit_unitary)
        # Calculate determinant of the unitary matrix
        self.det = det(np.array(circuit_unitary, dtype=np.complex128))
        # Convert U(n) matrix to SU(n) matrix and get correction factor
        self.special_unitary, self.correction = unitary_to_special_unitary(circuit_unitary)
        # Initialize factors to None (will be populated by factor() method)
        self.factors = None

    def factor(self):
        """
        Factorize the special unitary matrix of the circuit.

        Returns:
            list: Factorization results as (indices, values) pairs

        Raises:
            AssertionError: If the matrix is not a valid special unitary
        """
        assert np.abs(det(self.special_unitary) - 1) < 1e-10, f"Matrix conversion to special unitary failed: {det(self.special_unitary)}"
        self.factors = sun_factorization(np.matrix(self.special_unitary))
        return self.factors

    def reconstruct(self, dim, num_qudits):
        """
        Reconstruct the original unitary matrix from its factors.

        Args:
            dim (int): Dimension of the qudits
            num_qudits (int): Number of qudits in the circuit

        Returns:
            np.ndarray: Reconstructed unitary matrix

        Raises:
            AssertionError: If reconstruction fails
        """
        if self.factors is None:
            self.factor()
        reconstructed = sun_reconstruction(dim**num_qudits, self.factors)
        # Restore the original determinant
        reconstructed_circuit = reconstructed * np.power(self.correction, -1)
        assert np.allclose(reconstructed_circuit, self.original_unitary), "Reconstruction failed"
        return reconstructed_circuit

    def decompose_circuit(self, num_qudits, dim):
        """
        Decompose the circuit into a sequence of elementary rotation gates.

        Args:
            num_qudits (int): Number of qudits in the circuit
            dim (int): Dimension of the qudits

        Returns:
            cirq.Circuit: Decomposed circuit
        """
        # Ensure the circuit has been factored
        if self.factors is None:
            self.factor()
            
        # Create an empty circuit and initialize qudits
        expanded_circuit = Circuit()
        qudits = [LineQid(i, dimension=dim) for i in range(num_qudits)]
        
        # Iterate through each factor to build the circuit
        for (indices, values) in self.factors:
            # Create identity matrix of appropriate size
            gate = np.eye(dim**num_qudits, dtype=complex)
            
            # Extract indices for gate placement
            i, _, j = tuple(indices)
            
            # Apply rotation gates (Rx, Ry, Rz) to the appropriate submatrix
            gate[int(i)-1:int(i)+1, int(i)-1:int(i)+1] = (
            self.rx(values[0]) @ self.ry(values[1]) @ self.rz(values[2])
            )
            
            # Add the gate to the circuit, converting matrix dimensions to match qudit shape
            expanded_circuit.append(
            MatrixGate(matrix=gate, qid_shape=(dim,)*int(np.emath.logn(dim, len(gate[0]))))(*qudits)
            )
        return expanded_circuit

    @staticmethod
    def random_special_unitary_matrix(n):
        """
        Generate a random special unitary matrix of size n x n.

        Args:
            n (int): Dimension of the matrix

        Returns:
            np.ndarray: Random special unitary matrix
        """
        unitary_matrix = unitary_group.rvs(n)
        det = det(unitary_matrix)
        return unitary_matrix / np.power(det, 1/n)

__all__ = ['CircuitDecomposer']
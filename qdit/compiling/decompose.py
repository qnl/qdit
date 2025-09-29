"""
Quantum circuit decomposition module for converting quantum circuits into elementary rotations.

This module provides functionality to decompose quantum circuits into a sequence of elementary
rotation gates (Rx, Ry, Rz) using SU(n) factorization techniques.
"""

import numpy as np
from cirq import Circuit, LineQid, unitary
from qdit.utils.sun_factorization import sun_factorization
from qdit.utils.sun_reconstruction import sun_reconstruction
from qdit.utils.math import unitary_to_special_unitary
from qdit.gates import H
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

    def __init__(self, circuit):
        """
        Initialize the CircuitDecomposer with a circuit.

        Args:
            circuit (cirq.Circuit): Input circuit to be decomposed

        The constructor converts the input circuit to its unitary matrix representation
        and prepares it for SU(n) factorization by computing necessary attributes.
        """
        self.circuit = circuit
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

    def decompose_circuit(self, num_qudits, dim, native_gates='RzRy'):
        """
        Decompose the circuit into a sequence of elementary rotation gates.

        Args:
            num_qudits (int): Number of qudits in the circuit
            dim (int): Dimension of the qudits
            native_gates (str): Types of native gates used for the rotations. 
                'RzRy': Use Rz and Ry rotations (default)
                'RzRx': Use Rz and Rx(±π/2) rotations (X90 gates)

        Returns:
            cirq.Circuit: Decomposed circuit
        """
        from qdit.gates.single_qudit import Rz, Ry, X90, Id
        
        # Ensure the circuit has been factored
        if self.factors is None:
            self.factor()
            
        # Create an empty circuit and initialize qudits
        expanded_circuit = Circuit()
        qudits = [LineQid(i, dimension=dim) for i in range(num_qudits)]
        
        # Iterate through each factor to build the circuit
        for (indices, values) in self.factors:

            # First Rz rotation
            if not np.isclose(values[0], 0, 1e-5):
                expanded_circuit.append(
                    Rz(values[0], dimension=dim, subspace=indices, n_qudits=num_qudits)(*qudits)
                )
            
            # Middle rotation (Ry or Rx-based)
            if not np.isclose(values[1], 0, 1e-5):
                if native_gates == 'RzRy':
                    expanded_circuit.append(
                    Ry(values[1], dimension=dim, subspace=indices, n_qudits=num_qudits)(*qudits)
                    )
                else:  # RzRx case
                    # Rx(π/2) repeated 4 times with Rz in middle
                    for _ in range(3):
                        expanded_circuit.append(
                            X90(dimension=dim, subspace=indices, n_qudits=num_qudits)(*qudits)
                        )
                    expanded_circuit.append(
                    Rz(values[1], dimension=dim, subspace=indices, n_qudits=num_qudits)(*qudits)
                    )
                    expanded_circuit.append(
                    X90(dimension=dim, subspace=indices, n_qudits=num_qudits)(*qudits)
                    )
            
            # Final Rz rotation
            if not np.isclose(values[2], 0, 1e-5):
                expanded_circuit.append(
                    Rz(values[2], dimension=dim, subspace=indices, n_qudits=num_qudits)(*qudits)
                )
        
        #If expanded circuit is empty, add identity gate:
        if len(expanded_circuit) == 0:
            expanded_circuit.append(Id(dimension=dim)(*qudits))
        return expanded_circuit

    def parse_and_decompose(self, num_qudits, dim, native_gates='RzRy'):
        """
        Parse the circuit, classify moments, expand hard gates, and decompose easy moments.

        Args:
            num_qudits (int): Number of qudits in the circuit
            dim (int): Dimension of the qudits
            native_gates (str): Native gate set

        Returns:
            cirq.Circuit: Fully decomposed circuit
        """
        new_circuit = Circuit()
        for moment in self.circuit:
            # Classify moment
            if all(len(op.qubits) == 1 for op in moment.operations):
                # Easy moment: decompose using existing logic
                decomposer = CircuitDecomposer(Circuit([moment]))
                decomposed = decomposer.decompose_circuit(num_qudits, dim, native_gates)
                new_circuit += decomposed
            else:
                # Hard moment: expand hard gates (example for CX to CZ)
                expanded_ops = []
                for op in moment.operations:
                    if (op.gate.__class__.__name__ == "CX" and "CZ" in native_gates) \
                        or (op.gate.__class__.__name__ == "CZ" and "CX" in native_gates):
                        control, target = op.qubits
                        expanded_ops.append(H(dim)(target))
                        expanded_ops.append(op)
                        expanded_ops.append(H(dim)(target))
                    else:
                        expanded_ops.append(op)
                expanded_moment = expanded_ops.Moment(expanded_ops)
                new_circuit += expanded_moment
        return new_circuit

__all__ = ['CircuitDecomposer']
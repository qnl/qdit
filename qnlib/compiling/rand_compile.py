from cirq import Circuit, MatrixGate, GateOperation, InsertStrategy, unitary, inverse
import numpy as np
from qnlib.gates.paulis.utils import rand_pauli

class RandomCompiling:
    def __init__(self, circuit: Circuit, d: int):
        self.n = circuit.all_qubits()
        self.input_circuit = circuit
        self.d = d
        self.easy_moment = Circuit()
        self.arranged_circuit = Circuit()
        
    def apply_twirl(self, circuit: Circuit, qubits, moment_index: int):
        """
        Apply random twirl gates to given qubits as a single combined operation.
        For qubits not in the input set, apply identity operations.
        Returns a dictionary mapping qubits to their twirl matrices.
        """
        twirl_dict = {}
        # Get all qubits in the circuit
        all_qubits = sorted(circuit.all_qubits())
        
        # Generate matrices for specified qubits, identity for others
        matrices = {}
        # for q in all_qubits:
        for q in qubits:
            twirl_matrix = rand_pauli(self.d)
            circuit.append(MatrixGate(matrix=twirl_matrix, name=f'T{moment_index},{str(q)[2]}', qid_shape=(self.d,)).on(q), strategy=InsertStrategy.INLINE)
            matrices[q] = twirl_matrix
            twirl_dict[q] = twirl_matrix
        return twirl_dict
    
    def handle_two_qubit_gate(self, op: GateOperation, moment_index: int):
        """Handle the compilation of a two-qubit gate operation."""
        # Append accumulated easy gates
        if len(self.easy_moment) > 0:
            self.arranged_circuit.append(self.easy_moment, strategy=InsertStrategy.EARLIEST)
        
        # Apply twirls and get the twirl matrices
        twirl_dict = self.apply_twirl(self.arranged_circuit, self.input_circuit.all_qubits(), moment_index)
        
        # Add operation and its inverse
        self.arranged_circuit.append(op, strategy=InsertStrategy.NEW)
        self.append_inverse_gate(op.with_tags(moment_index))
        
        # Apply inverse twirls
        self.append_inverse_twirls(twirl_dict, moment_index)
        
        # Reset easy moment and append final operation
        self.easy_moment = Circuit()
        self.arranged_circuit.append(op, strategy=InsertStrategy.NEW)

    def append_inverse_gate(self, op: GateOperation):
        """Append the inverse of a given gate operation."""
        inverse_gate = inverse(op.gate)
        self.arranged_circuit.append(inverse_gate(*op.qubits))

    def append_inverse_twirls(self, twirl_dict: dict, moment_index: int):
        """Append inverse twirl gates for all qubits."""
        all_qubits = sorted(self.arranged_circuit.all_qubits())
        for q in all_qubits:
            q_twirl_matrix = twirl_dict.get(q, np.eye(self.d))
            inverse_twirl = MatrixGate(
                matrix=np.conjugate(q_twirl_matrix.T),
                qid_shape=(self.d,),
                name=f'T{moment_index},{str(q)[2]}_inv'
            )
            self.arranged_circuit.append(inverse_twirl(q), strategy=InsertStrategy.INLINE)

    def compile(self):
        """Compile the input circuit using random compilation technique."""
        for moment_index, moment in enumerate(self.input_circuit.moments):
            for op in moment.operations:
                if op.gate.num_qubits() == 2:
                    self.handle_two_qubit_gate(op, moment_index)
                else:
                    self.easy_moment.append(op, strategy=InsertStrategy.EARLIEST)

        # Handle remaining operations
        if len(self.easy_moment) > 0:
            self.arranged_circuit.append(self.easy_moment)

        assert np.allclose(np.array(unitary(self.arranged_circuit)), np.array(unitary(self.input_circuit)), atol=1e-10)
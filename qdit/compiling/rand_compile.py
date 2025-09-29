from cirq import Circuit, MatrixGate, GateOperation, InsertStrategy, unitary, inverse, MeasurementGate
import numpy as np
from qdit.gates.single_qudit import WeylOperator
from qdit.gates.utils import rand_pauli  # assuming utils is in the new location
from typing import Dict, List, Tuple


class RandomCompiling:
    def __init__(self, circuit: Circuit, d: int, measurement_twirl: bool = False):
        self.n = circuit.all_qubits()
        self.input_circuit = circuit
        self.d = d
        self.easy_moment = Circuit()
        self.arranged_circuit = Circuit()
        self.measurement_twirl = measurement_twirl
        self.measurement_corrections: Dict[str, List[Tuple[int, int]]] = {}
        
    def apply_twirl(self, circuit: Circuit, qubits, moment_index: int):
        """
        Randomly twirl input qubits as a combined operation.
        For qubits not in the input set, apply identity operations.
        Returns a dictionary mapping qubits to their twirls.
        """
        twirl_dict = {}
        for q in qubits:
            twirl_matrix = rand_pauli(self.d)
            circuit.append(MatrixGate(matrix=twirl_matrix, name=f'T{moment_index},{str(q)[2]}', qid_shape=(self.d,)).on(q), strategy=InsertStrategy.INLINE)
            twirl_dict[q] = twirl_matrix
        return twirl_dict
    
    def handle_two_qubit_gate(self, op: GateOperation, moment_index: int):
        """Handle the compilation of a two-qubit gate operation."""
        # Append easy gates
        if len(self.easy_moment) > 0:
            self.arranged_circuit.append(self.easy_moment, strategy=InsertStrategy.EARLIEST)
        # Select twirl gates
        twirl_dict = self.apply_twirl(self.arranged_circuit, self.input_circuit.all_qubits(), moment_index)
        # Apply operation and its inverse
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

    def handle_measurement(self, op: GateOperation, moment_index: int):
        """Handle measurement gates with random Weyl operator twirling."""
        
        assert isinstance(op.gate ,MeasurementGate), "Only measurements should be passed to this function."
        
        if not self.measurement_twirl:
            self.easy_moment.append(op, strategy=InsertStrategy.EARLIEST)
            return
            
        # Get measurement key and qudits
        meas_gate = op.gate
        key = meas_gate.key
        qudits = op.qubits
        
        # Initialize correction tracking for this measurement
        if key not in self.measurement_corrections:
            self.measurement_corrections[key] = []
            
        # Generate random Weyl operators for each qudit
        for q in qudits:
            # Random powers for X and Z components
            a = np.random.randint(0, self.d)
            b = np.random.randint(0, self.d)
            
            # Create and append Weyl operator
            weyl = WeylOperator(a, b, dimension=self.d)
            self.easy_moment.append(weyl(q), strategy=InsertStrategy.NEW)
            
            # Track X component for measurement correction
            if a != 0:  # Only track if X component is non-zero
                qubit_idx = qudits.index(q)
                self.measurement_corrections[key].append((qubit_idx, a))
                
        # Append the measurement gate
        self.easy_moment.append(op, strategy=InsertStrategy.NEW)
        
    def compile(self) -> Circuit:
        """Compile the input circuit using random compilation technique."""
        for moment_index, moment in enumerate(self.input_circuit.moments):
            for op in moment.operations:
                if isinstance(op.gate, MeasurementGate):
                    self.handle_measurement(op, moment_index)
                elif op.gate.num_qubits() == 2:
                    self.handle_two_qubit_gate(op, moment_index)
                else:
                    self.easy_moment.append(op, strategy=InsertStrategy.EARLIEST)

        # Handle remaining operations
        if len(self.easy_moment) > 0:
            self.arranged_circuit.append(self.easy_moment)
            
        # Only check unitarity if no measurements are present
        if not any(isinstance(op.gate, MeasurementGate) 
                  for moment in self.input_circuit 
                  for op in moment.operations):
            assert np.allclose(np.array(unitary(self.arranged_circuit)), 
                             np.array(unitary(self.input_circuit)), 
                             atol=1e-10)
        
        return self.arranged_circuit
                             
    def get_measurement_corrections(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Returns the measurement corrections dictionary.
        
        Returns:
            Dict mapping measurement keys to lists of (qudit_index, X_power) tuples.
            These indicate which qudits need to be corrected and by how much.
        """
        return self.measurement_corrections
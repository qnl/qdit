from cirq import Circuit, MatrixGate, GateOperation, InsertStrategy, unitary, inverse, MeasurementGate, LineQid
import numpy as np
from typing import Dict, List, Tuple

from qdit.gates.single_qudit import WeylOperator, Rz, Ry, X90, Id, H, X, Y, Z, Sdg, S, P
from qdit.gates.multi_qudit import CX, CZ, iSWAP
from qdit.gates.utils import rand_pauli, quditX_mat, quditZ_mat
from qdit.utils.sun_factorization import sun_factorization, su2_parameters
from qdit.utils.sun_reconstruction import sun_reconstruction
from qdit.utils.math import unitary_to_special_unitary
from numpy.linalg import det
from scipy.linalg import logm, expm

MULTI_QUDIT_GATE_IDENTITIES = {
    "CXtoCZ": lambda d, control, target: [H(d)(target), CZ(d)(control, target), H(d)(target)**(((d-1)**2-1) if d > 2 else 1)],
    "CXtoiSWAP": lambda d, control, target: [Sdg(d)(target), Sdg(d)(control), H(d)(target), iSWAP()(control, target), Sdg(d)(control), H(d)(control), Sdg(d)(control), iSWAP()(control, target), S(d)(target)] if d==2 else NotImplementedError,
    # "CXtRzRxiSWAP": lambda d, control, target: [Rz(np.pi/2,d)(target), Rz(-np.pi/2,d)(control), X90(d)(target), iSWAP()(control, target), X90(d)(control), iSWAP()(control, target), S(d)(target)] if d==2 else NotImplementedError,
    # "CXtoRzRyiSWAP": lambda d, control, target: [Sdg(d)(target), Sdg(d)(control), H(d)(target), iSWAP()(control, target), Sdg(d)(control), H(d)(control), Sdg(d)(control), iSWAP()(control, target), S(d)(target)] if d==2 else NotImplementedError,
    "CZtoCX": lambda d, control, target: [H(d)(target)**(((d-1)**2-1) if d > 2 else 1), CX(d)(control, target), H(d)(target)],
    "CZtoiSWAP": lambda d, control, target:[H(d)(target)**(((d-1)**2-1) if d > 2 else 1), Sdg(d)(target), Sdg(d)(control), H(d)(target), iSWAP()(control, target), Sdg(d)(control), H(d)(control), Sdg(d)(control), iSWAP()(control, target), S(d)(target), H(d)(target)] ,
    "iSWAPtoCX": lambda d, control, target: [S(d)(control),S(d)(target),H(d)(control),CX(d)(control,target),CX(d)(target,control),H(d)(target)] if d==2 else NotImplementedError,
    "iSWAPtoCZ": lambda d, control, target: [S(d)(control),S(d)(target),H(d)(control),H(d)(target), CZ(d)(control, target), H(d)(target)**(((d-1)**2-1) if d > 2 else 1),H(d)(control), CZ(d)(target, control), H(d)(control)**(((d-1)**2-1) if d > 2 else 1),H(d)(control)] if d==2 else NotImplementedError
}

ISWAP_IDENTITIES = {
    "b2b": lambda d, control, target: [Z(d)(control), Z(d)(target)],
}

SINGLE_QUDIT_GATE_UNITARIES = {
    # "Id": lambda d: unitary(Id(d)),
    "X": lambda d: unitary(X(d)),
    "Y": lambda d: unitary(Y(d)),
    "Z": lambda d: unitary(Z(d)),
    "H": lambda d: unitary(H(d)),
    "P": lambda d: unitary(P(d)),
    "Sdg": lambda d: unitary(Sdg(d)),
    "X90": lambda d, subspace=None: unitary(X90(d, subspace=subspace)),
    "Rz": lambda d, theta=0, subspace=None: unitary(Rz(theta, d, subspace=subspace)),
    "Ry": lambda d, theta=0, subspace=None: unitary(Ry(theta, d, subspace=subspace)),
    "WeylOperator": lambda d, a=0, b=0: unitary(WeylOperator(a, b, d)),
}


def twirl_measure(circuit: Circuit, op: GateOperation, d: int, measurement_corrections: Dict):
    """Handle measurement gates with random Weyl operator twirling."""
    assert isinstance(op.gate, MeasurementGate), "Only measurements should be passed to this function."
    meas_gate = op.gate
    key = meas_gate.key
    qudits = op.qubits
    if key not in measurement_corrections:
        measurement_corrections[key] = []
    for q in qudits:
        a = np.random.randint(0, d)
        b = np.random.randint(0, d)
        weyl = WeylOperator(a, b, dimension=d)
        circuit.append(weyl(q), strategy=InsertStrategy.NEW)
        if a != 0:
            qubit_idx = qudits.index(q)
            measurement_corrections[key].append((qubit_idx, a))
    circuit.append(op, strategy=InsertStrategy.NEW)

def decomposed_easy_moment(easy_circuit: Circuit, d, native_gates): 
        qudit_unitaries = []
        measured_qudits = []
        factored_circuit = Circuit()
        for q in easy_circuit.all_qubits():
            u = np.eye(d)
            for m in easy_circuit.moments:
                if m.operates_on_single_qubit(q):
                    if isinstance(m.operation_at(q).gate, MeasurementGate):
                        measured_qudits.append(q)
                        break
                    else:
                        u = np.dot(unitary(m.operation_at(q)), u)
            qudit_unitaries.append(u)
        special_unitaries = [unitary_to_special_unitary(u)[0] for u in qudit_unitaries]
        factors = [
            sun_factorization(np.matrix(su, dtype=np.complex128))
            if np.shape(su) != (2, 2)
            else [('1,2', su2_parameters(np.matrix(su, dtype=np.complex128)))]
            for su in special_unitaries
        ]
        for qudit, factor in zip(easy_circuit.all_qubits(), factors):
            for (indices, values) in reversed(factor):
                if np.isclose(values[1], 0, 1e-5):
                    if np.isclose(values[2]+values[0], 0, 1e-5):
                        factored_circuit.append(
                            Id(dimension=d)(qudit)
                        )
                    else:
                        factored_circuit.append(
                            Rz(values[2]+values[0], dimension=d, subspace=indices)(qudit)
                        )
                else:
                    if 'Ry' in native_gates:
                        if not np.isclose(values[2], 0, 1e-5):
                            factored_circuit.append(
                                Rz(values[2], dimension=d, subspace=indices)(qudit)
                            )
                        factored_circuit.append(
                            Ry(values[1], dimension=d, subspace=indices)(qudit)
                        )
                    else:
                        if not np.isclose(values[2], -np.pi, 1e-5):
                            factored_circuit.append(
                                Rz(values[2]+np.pi, dimension=d, subspace=indices)(qudit)
                            )
                        factored_circuit.append(
                            X90(dimension=d, subspace=indices)(qudit)
                        )
                        factored_circuit.append(
                            Rz(values[1]+np.pi, dimension=d, subspace=indices)(qudit)
                        )
                        factored_circuit.append(
                            X90(dimension=d, subspace=indices)(qudit)
                        )
                        if d>2:
                            factored_circuit.append(
                                Rz(2 * np.pi, dimension=d, subspace=indices)(qudit)
                            )
                    if not np.isclose(values[0], 0, 1e-5):
                        factored_circuit.append(
                            Rz(values[0], dimension=d, subspace=indices)(qudit)
                        )
                    if np.allclose(values, [0, 0, 0], atol=1e-5):
                        factored_circuit.append(Id(dimension=d)(qudit))
        for qudit in measured_qudits:
            factored_circuit.append(MeasurementGate(qid_shape=(d,))(qudit))
        return factored_circuit

def decompose(circuit: Circuit, d: int, native_gates='RzRyCX') -> Circuit:
    """
    Decompose a circuit into a sequence of native gates.
    """
    decomposed_circuit = Circuit()
    easy_moment = Circuit()
    for moment in circuit:
        for op in moment: 
            if op.gate.num_qubits() == 2:
                native_gate = op.gate.__class__.__name__ in native_gates
                if not native_gate:
                    control, target = op.qubits
                    if "CZ" in native_gates:
                        ops = MULTI_QUDIT_GATE_IDENTITIES[str(op.gate.__class__.__name__)+"toCZ"](d, control, target)
                    elif "CX" in native_gates:
                        ops = MULTI_QUDIT_GATE_IDENTITIES[str(op.gate.__class__.__name__)+"toCX"](d, control, target)
                    elif "iSWAP" in native_gates:
                        ops = MULTI_QUDIT_GATE_IDENTITIES[str(op.gate.__class__.__name__)+"toiSWAP"](d, control, target)
                    else: return NotImplementedError()
                    for op in ops:
                        if op.gate.num_qubits()==2:
                            if len(easy_moment)>0:
                                decomposed_circuit.append(decomposed_easy_moment(easy_moment, d, native_gates))
                                easy_moment = Circuit()
                            decomposed_circuit.append(op, strategy=InsertStrategy.NEW)
                        else:
                            easy_moment.append(op)
                else: 
                    if len(easy_moment)>0:
                        decomposed_circuit.append(decomposed_easy_moment(easy_moment, d, native_gates))
                        easy_moment = Circuit()
                    decomposed_circuit.append(op)
            else: 
                easy_moment.append(op, strategy=InsertStrategy.NEW)
    if len(easy_moment)>0:
        decomposed_circuit.append(decomposed_easy_moment(easy_moment, d, native_gates))
    return decomposed_circuit

def reduce(circuit: Circuit, d: int) -> Circuit:
    prev_ops = []
    reduced_circuit = Circuit()
    for moment in circuit:
        for op in moment:
            if not prev_ops:
                prev_ops.append(op)
            # reduce rotations
            # if isinstance(op.gate, Rz) and isinstance(prev_ops[-1].gate, Rz):
            #     prev_Rz = prev_ops[-1].gate
            #     prev_Rz_theta = prev_Rz._rads
            #     curr_Rz_theta = op.gate._rads
            #     prev_ops[-1] = Rz(prev_Rz_theta+curr_Rz_theta, d)()

            

class Compiler:
    def __init__(self, circuit: Circuit, d: int, rc: bool = False, mrc: bool = False, native: bool = False, native_gates='RzRyCX'):
        self.input_circuit = circuit
        self.d = d
        self.rc = rc
        self.mrc = mrc
        self.native = native
        self.native_gates = native_gates
        self.measurement_corrections: Dict[str, List[Tuple[int, int]]] = {}

    def compile(self) -> Circuit:
        circuit = self.input_circuit
        d = self.d
        rc = self.rc
        mrc = self.mrc
        decompose_flag = self.native
        native_gates = self.native_gates

        def twirl(op: GateOperation):
            """Apply random twirl gates to given qubits as a single combined operation."""
            d=self.d
            twirl_circuit = Circuit()
            control_q = op.qubits[0]
            target_q = op.qubits[1]
            control_vec = [np.random.randint(0, d), np.random.randint(0, d)]
            target_vec = [np.random.randint(0, d), np.random.randint(0, d)]
            control_twirl = WeylOperator(a=control_vec[0], b=control_vec[1], dimension=d)
            target_twirl = WeylOperator(a=target_vec[0], b=target_vec[1], dimension=d)
            twirl_circuit.append(control_twirl(control_q), strategy=InsertStrategy.INLINE)
            twirl_circuit.append(target_twirl(target_q), strategy=InsertStrategy.INLINE)
            twirl_circuit.append(op)
            twirl_circuit.append(WeylOperator(a=-control_twirl.a, b=-control_twirl.b+target_twirl.b, dimension=d)(control_q))
            twirl_circuit.append(WeylOperator(a=-control_twirl.a-target_twirl.a, b=-target_twirl.b, dimension=d)(target_q))
            return twirl_circuit

        # Parsing
        sub_circuit = Circuit()
        compiled_circuit = Circuit()

        if rc or mrc:
            for moment_index, moment in enumerate(circuit.moments):
                for op in moment.operations:
                    if isinstance(op.gate, MeasurementGate):
                        twirl_measure(sub_circuit, op, d, self.measurement_corrections) if mrc else sub_circuit.append(op)
                        continue
                    elif op.gate.num_qubits() == 1:
                        sub_circuit.append(op)
                    else:
                        if len(sub_circuit) > 0:
                            compiled_circuit.append(sub_circuit)
                            sub_circuit = Circuit()
                        compiled_circuit.append(twirl(op)) if rc else compiled_circuit.append(op)
            if len(sub_circuit)>0:
                compiled_circuit.append(sub_circuit)
            if decompose_flag:
                compiled_circuit=decompose(compiled_circuit, d, native_gates)
            if mrc:
                return compiled_circuit, self.measurement_corrections
            else: return compiled_circuit

        elif decompose_flag:
            return decompose(circuit, d, native_gates)
        
        else: return circuit

    def get_measurement_corrections(self) -> Dict[str, List[Tuple[int, int]]]:
        return self.measurement_corrections

def compile_circuit(circuit: Circuit, d: int, rc: bool = False, mrc: bool = False, native: bool = False, native_gates='RzRyCZ'):
    """
    Main entry point for compiling a circuit with various options.
    Args:
        circuit: cirq.Circuit
        d: int, qudit dimension
        rc: bool, random compiling
        mrc: bool, measurement random compiling
        native: bool, decompose to elementary gates
        native_gates: str, native gate set for decomposition
    Returns:
        cirq.Circuit
    """
    compiler = Compiler(circuit, d, rc=rc, mrc=mrc, native=native, native_gates=native_gates)
    return compiler.compile()
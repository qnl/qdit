import pytest
from qdit.benchmarking import sample_clifford
import qdit.gates.utils as qcp
from qdit.utils.math import multiplicative_group_mod_d, unitary_equiv
from cirq import LineQid, Circuit, measure, Simulator
from qdit.compiling import Compiler
from qdit.gates import *
import numpy as np

PRIME_DIMS = [2, 3]#, 5]
NATIVE_SETS = ['RzRyCX', 'RzRxCX', 'RzRyCZ', 'RzRxCZ', 'RzRxiSWAP', 'RzRyiSWAP']
NUM_QUDITS = [2]#, 3]

SINGLE_QUDIT_GATES = [
    lambda d: Rx(np.pi/3, d),
    lambda d: Ry(np.pi/3, d),
    lambda d: Rz(np.pi/3, d),
    lambda d: X(d),
    lambda d: Y(d),
    lambda d: Z(d),
    lambda d: H(d),
    lambda d: S(d),
    lambda d: Sdg(d),
    lambda d: Id(d),
]

def all_Ma_gates(d):
    return [lambda d, a=a: Ma(d, a) for a in multiplicative_group_mod_d(d)]

for d in set(PRIME_DIMS):
    SINGLE_QUDIT_GATES.extend([lambda d, a=a: Ma(d, a) for a in multiplicative_group_mod_d(d)])

MULTI_QUDIT_GATES = [
    lambda d: CX(d),
    lambda d: CZ(d),
]

def test_qudit_pauli_group(nq=2, d=3, num_elements=10):
    try:
        pauli_group = list(qcp.qudit_pauli_group(nq=nq, d=d))[:num_elements]
        print(f"First {num_elements} elements of d={d} {nq}-qudit Pauli Group:")
        for i, element in enumerate(pauli_group):
            print(f"{i+1}: {element}")
        return True
    except Exception as e:
        print(f"Error generating Pauli group: {e}")
        return False

@pytest.mark.parametrize("d", PRIME_DIMS)
@pytest.mark.parametrize("native_gates", NATIVE_SETS)
def test_single_qudit_decomposition(d, native_gates):
    # Only test d=2 for native_gates containing 'iSWAP'
    if 'iSWAP' in native_gates and d != 2:
        pytest.skip(f"Skipping d={d} for native_gates={native_gates} (only d=2 supported for iSWAP)")
    q = LineQid(0, dimension=d)
    for gate_fn in SINGLE_QUDIT_GATES:
        if gate_fn is None:
            continue
        try:
            gate = gate_fn(d)
        except Exception:
            continue
        circ = Circuit()
        circ.append(gate.on(q))
        compiler = Compiler(circ, d=d, native=True, native_gates=native_gates)
        decomposed = compiler.compile()
        U_orig = circ.unitary()
        U_decomp = decomposed.unitary()
        assert unitary_equiv(U_orig, U_decomp), f"Failed for {gate} in d={d}, native={native_gates}"

@pytest.mark.parametrize("d", PRIME_DIMS)
@pytest.mark.parametrize("native_gates", NATIVE_SETS)
def test_multi_qudit_decomposition(d, native_gates):
    if 'iSWAP' in native_gates and d != 2:
        pytest.skip(f"Skipping d={d} for native_gates={native_gates} (only d=2 supported for iSWAP)")
    q0 = LineQid(0, dimension=d)
    q1 = LineQid(1, dimension=d)
    for gate_fn in MULTI_QUDIT_GATES:
        gate = gate_fn(d)
        circ = Circuit()
        circ.append(gate(q0, q1))
        compiler = Compiler(circ, d=d, native=True, native_gates=native_gates)
        decomposed = compiler.compile()
        U_orig = circ.unitary()
        U_decomp = decomposed.unitary()
        assert unitary_equiv(U_orig, U_decomp), f"Failed for {gate} in d={d}, native={native_gates}"

@pytest.mark.parametrize("d", PRIME_DIMS)
@pytest.mark.parametrize("native_gates", NATIVE_SETS)
def test_pair_single_qudit_decomposition(d, native_gates):
    if 'iSWAP' in native_gates and d != 2:
        pytest.skip(f"Skipping d={d} for native_gates={native_gates} (only d=2 supported for iSWAP)")
    q0 = LineQid(0, dimension=d)
    q1 = LineQid(1, dimension=d)
    gates = [g for g in SINGLE_QUDIT_GATES if g is not None]
    for g1 in gates:
        for g2 in gates:
            try:
                gate1 = g1(d)
                gate2 = g2(d)
            except Exception:
                continue
            circ = Circuit(gate1(q0), gate2(q1))
            compiler = Compiler(circ, d=d, native=True, native_gates=native_gates)
            decomposed = compiler.compile()
            U_orig = circ.unitary()
            U_decomp = decomposed.unitary()
            assert unitary_equiv(U_orig, U_decomp), f"Failed to decompose {gate2}*{gate1}. \nd: {d} \nGate set: {native_gates} \nCircuit: {circ}"

@pytest.mark.parametrize("d", PRIME_DIMS)
@pytest.mark.parametrize("native_gates", NATIVE_SETS)
@pytest.mark.parametrize("num_qudits", NUM_QUDITS)
def test_random_clifford_decomposition(d, native_gates, num_qudits, num_tests=100):
    if 'iSWAP' in native_gates and d != 2:
        pytest.skip(f"Skipping d={d} for native_gates={native_gates} (only d=2 supported for iSWAP)")
    for _ in range(num_tests):
        circ = sample_clifford(num_qudits=num_qudits, dimension=d)
        compiler = Compiler(circ, d=d, native=True, native_gates=native_gates)
        compiled = compiler.compile()
        U_orig = circ.unitary()
        U_comp = compiled.unitary()
        assert unitary_equiv(U_orig, U_comp), (
            f"Random Clifford circuit decomposition failed. "
            f"\nd: {d} \nGate set: {native_gates} \nQudits: {num_qudits} "
            f"\nCircuit: {circ} \nDecomposed: {compiled}"
        )

@pytest.mark.parametrize("d", PRIME_DIMS)
@pytest.mark.parametrize("num_qudits", NUM_QUDITS)
def test_rc_random_clifford_circuits(d, num_qudits, num_tests=100):
    for _ in range(num_tests):
        circ = sample_clifford(num_qudits=num_qudits, dimension=d)
        compiler = Compiler(circ, d=d, rc=True)
        compiled = compiler.compile()
        U_orig = circ.unitary()
        U_comp = compiled.unitary()
        assert unitary_equiv(U_orig, U_comp), (
            f"Random Clifford circuit RC failed to maintain circuit logic "
            f"(d={d}, qudits={num_qudits}). \nCircuit: {circ} \nTwirled: {compiled}"
        )

@pytest.mark.parametrize("d", PRIME_DIMS)
@pytest.mark.parametrize("native_gates", NATIVE_SETS)
@pytest.mark.parametrize("num_qudits", NUM_QUDITS)
def test_rc_decomp_random_clifford_circuits(d, native_gates, num_qudits, num_tests=100):
    if 'iSWAP' in native_gates and d != 2:
        pytest.skip(f"Skipping d={d} for native_gates={native_gates} (only d=2 supported for iSWAP)")
    for _ in range(num_tests):
        circ = sample_clifford(num_qudits=num_qudits, dimension=d)
        compiler = Compiler(
            circ, d=d, rc=True, mrc=False, native=True, native_gates=native_gates
        )
        compiled = compiler.compile()
        U_orig = circ.unitary()
        U_comp = compiled.unitary()
        assert unitary_equiv(U_orig, U_comp), (
            f"Random Clifford circuit with RC failed to decompose. "
            f"\nd: {d} \nGate set: {native_gates} \nQudits: {num_qudits} "
            f"\nCircuit: {circ} \nCompiled circuit: {compiled}"
        )

@pytest.mark.parametrize("d", PRIME_DIMS)
def test_mrc_measurement_circuit(d, num_tests=100):
    simulator = Simulator()
    circuits = []
    for _ in range(num_tests):
        measurement_circuit = Circuit()
        qudit = LineQid(0, dimension = d)
        measurement_circuit.append(measure(qudit, key="x"))
        compiler = Compiler(measurement_circuit, d=d, mrc=True)
        compiled, measurement_corrections = compiler.compile()
        circuits.append((compiled, measurement_corrections))
        for compiled_circuit, correction in circuits:
            results = simulator.run(compiled_circuit, repetitions=100)
            # Apply corrections
            for key, corrections in correction.items():
                # Get the measurement data as a numpy array
                data = results.measurements[key]
                # Apply corrections
                for qudit_idx, x_power in corrections:
                    data[:, qudit_idx] = (data[:, qudit_idx] - x_power) % d
        measurement_data = results.measurements['x']
        assert np.array_equal(measurement_data, np.zeros(shape=(len(measurement_data),1))), f"Measurement circuit failed to correct: {measurement_data}"


def test_native_gate_set_expansion():
    # If new native gates are added, test that the compiler still works
    d = 3
    q = LineQid(0, dimension=d)
    circ = Circuit(Rz(np.pi/4, d)(q), Ry(np.pi/4, d)(q))
    compiler = Compiler(circ, d=d, native=True, native_gates='RzRyCXNEWGATE')
    decomposed = compiler.compile()
    U_orig = circ.unitary()
    U_decomp = decomposed.unitary()
    assert unitary_equiv(U_orig, U_decomp), "Expansion test failed"

if __name__ == "__main__":
    pytest.main([__file__])
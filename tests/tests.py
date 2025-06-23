from qnlib.benchmarking import Tableau

def test_tableau(num_tests: int = 50, num_qudits: int = 2, dim: int = 3, iters = 1):
    for _ in range(num_tests):
        for i in range(iters):
            T = Tableau(num_qudits, dim, iters)
            T.populate_and_sweep(display=False)
            assert (T.xtab[2*i][i] == T.ztab[2*i+1][i] == 1 and not (any(T.xtab[2*i][i+1:] + T.ztab[2*i+1][i+1:]))), (f'Error at {_}', T.xtab[2*i], T.ztab[2*i+1])

### Generating Symbolic Pauli Groups ###
import qnlib.gates.single_qudit.utils as qcp

def test_qudit_pauli_group(nq=2, d=3, num_elements=10):
    """
    Test function to generate and print elements of a qudit Pauli group.
    
    Args:
        nq (int): Number of qudits
        d (int): Dimension of each qudit
        num_elements (int): Number of elements to print
    """
    try:
        pauli_group = list(qcp.qudit_pauli_group(nq=nq, d=d))[:num_elements]
        print(f"First {num_elements} elements of d={d} {nq}-qudit Pauli Group:")
        for i, element in enumerate(pauli_group):
            print(f"{i+1}: {element}")
        return True
    except Exception as e:
        print(f"Error generating Pauli group: {e}")
        return False
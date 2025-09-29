import pytest
from qdit.benchmarking import Tableau

PRIME_DIMS = [2, 3]
NATIVE_ENTANGLING_GATES = ['CX', 'iSWAP']


@pytest.mark.parametrize("d", PRIME_DIMS)
@pytest.mark.parametrize("native_gates", NATIVE_ENTANGLING_GATES)
def test_tableau_result(num_tests: int = 50, num_qudits: int = 2, dim: int = 3, iters = 1):
    
    for _ in range(num_tests):
        for i in range(iters):
            T = Tableau(num_qudits, dim, iters)
            T.populate_and_sweep(display=False)
            assert (T.xtab[2*i][i] == T.ztab[2*i+1][i] == 1 and not (any(T.xtab[2*i][i+1:] + T.ztab[2*i+1][i+1:]))), (f'Error at {_}', T.xtab[2*i], T.ztab[2*i+1])

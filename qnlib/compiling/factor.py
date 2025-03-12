import qnlib.benchmarking.benchmarking as qrb
from qnlib.utils.sun_factorization import *
from qnlib.utils.sun_reconstruction import *
import numpy as np

def factor(circuit):
    return sun_factorization(np.matrix(circuit, dtype=np.clongdouble))

def reconstruct(factors, dim, num_qudits):
    return sun_reconstruction(dim**num_qudits, factors)

#Reconstruct elementary gates:
def elementary_gates(factored_circuit, n ,d):
    gate_set = []
    for (indices, values) in factored_circuit:
        gate = np.eye(d**n, dtype=complex)
        i, _ , j = tuple(indices) # indices are strings of neighboring integers
        gate[int(i)-1:int(i)+1, int(i)-1:int(i)+1] = rx(values[0]) @ ry(values[1]) @ rz(values[2])
        gate_set.append(gate)
    return np.array(gate_set)

rx = lambda a: np.array([[np.exp(1j*a/2), 0], [0, np.exp(-1j*a/2)]])
ry = lambda b: np.array([[np.cos(b/2), -np.sin(b/2)], [np.sin(b/2), np.cos(b/2)]])
rz = lambda g: np.array([[np.exp(1j*g/2), 0], [0, np.exp(-1j*g/2)]])
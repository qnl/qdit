import numpy as np
import matplotlib.pyplot as plt
from cirq import protocols

def plot_gate_matrix(gate):
    """Visualize the real and imaginary parts of a gate's unitary matrix.
    
    Args:
        gate: A Cirq gate object that can be converted to a unitary matrix
    """
    matrix = protocols.unitary(gate)
    plt.figure(figsize=(10, 4))
    
    # Plot real part
    plt.subplot(121)
    plt.imshow(np.real(matrix), cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Real Part')
    
    # Plot imaginary part
    plt.subplot(122)
    plt.imshow(np.imag(matrix), cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Imaginary Part')
    
    plt.tight_layout()
    return plt.gcf()
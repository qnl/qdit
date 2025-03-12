import cirq
import numpy as np
from typing import Union, Optional

def matrix_to_cirq_gate(
    matrix: np.ndarray,
    dimension: Optional[int] = None,
    name: Optional[str] = None,
    num_controls: int = 0
) -> cirq.Gate:
    """Convert a unitary matrix to a Cirq gate.
    
    Args:
        matrix: Unitary matrix representing the gate
        dimension: Qudit dimension. If None, inferred from matrix size
        name: Optional name for the gate
        
    Returns:
        cirq.MatrixGate that implements the unitary
    """
    controlled = (num_controls != 0)

    if dimension is None:
        dimension = matrix.shape[0]
        
    if not controlled and matrix.shape != (dimension, dimension):
        raise ValueError(f"Matrix shape {matrix.shape} incompatible with dimension {dimension}")
    
    gate= cirq.MatrixGate(
        matrix,
        name=name,
        qid_shape=[dimension for _ in range(num_controls+1)] if controlled else (dimension,)
    )   

    gate._circuit_diagram_info_(name)

    return gate
# Qdit: QNL's qudit tool shed

Qdit is a Python package for working with qudit (d-dimensional quantum) systems. Built on top of Google's Cirq framework, it provides tools for quantum circuit generation, noise modeling, and randomized benchmarking.

## Installation

```bash
git clone https://github.com/qnl/qdit.git
cd qdit
pip install -e .
```

## Features

### Qudit Gates
- Complete set of qudit Clifford gates
- Pauli operators for d-dimensional systems
- Support for arbitrary unitary decomposition into SU(2) gates

### Noise Models
- Amplitude damping channel for qutrits
- Depolarizing noise
- Trit-flip channel
- Random compilation for noise tailoring

### Benchmarking Tools
- Randomized benchmarking for single and multi-qudit systems
- Efficient Clifford circuit sampling using tableau simulation
- Built-in visualization and analysis tools

## Quick Start

```python
from qdit.benchmarking import QuditBenchmarking
from cirq import Simulator

# Create a 2-qutrit benchmarking instance
rb = QuditBenchmarking(
    num_qudits=2,
    dimension=3,
    sampler=Simulator()
)

# Generate and run benchmark circuits
circuits = rb.generate_benchmark_circuits(
    depths=[5, 10, 15],
    num_circuits=10
)

# Run noisy simulation
results = rb.simulate_benchmarking(
    depths=[5, 10, 15],
    noise_levels=[0.01, 0.05]
)
```

## Example Use Cases

### Randomized Benchmarking
See `examples/qudit_rb.ipynb` for:
- Single qutrit benchmarking
- Multi-qudit systems
- Noise characterization

### Circuit Compilation
See `examples/sun_decomp.ipynb` for:
- SU(N) decomposition into SU(2) gates
- Random circuit compilation
- Hardware-efficient gate sequences

### Noise Studies
See `examples/random_compiling.ipynb` for:
- Noise tailoring techniques
- Random compilation methods
- Error mitigation strategies

### Basic Qutrit Operations
See `examples/qutrits_in_cirq.ipynb` for:
- Basic qutrit gate operations
- Circuit construction
- Measurement and simulation

## Core Components

### Tableau Simulation
```python
from qdit.benchmarking import Tableau

# Generate random 2-qutrit Clifford circuit
tableau = Tableau(num_qudits=2, dimension=3)
tableau.populate_and_sweep()
circuit = tableau.circuit
```

### Noise Channels
```python
from qdit.noise import AmplitudeDamping

# Create qutrit amplitude damping channel
noise = AmplitudeDamping(
    gamma_10=0.01,  # |1⟩ → |0⟩ probability
    gamma_20=0.01,  # |2⟩ → |0⟩ probability
    gamma_21=0.01   # |2⟩ → |1⟩ probability
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

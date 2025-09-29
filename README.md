# Qdit: QNL's qudit tool shed

Qdit is a Python package for working with qudit (d-dimensional quantum) systems. Built on top of Google's Cirq framework, it provides tools for quantum circuit generation, noise modeling, and randomized benchmarking.

## Installation

```bash
git clone https://github.com/qnl/qdit.git
cd qdit
pip install -e .
```

## Features (with examples linked)

### [Qudit Gates](./qdit/gates/)
- [Able to generate the complete set of n-qudit Clifford circuits](./examples/clifford_sampling.ipynb)
- [Arbitrary single-qudit unitary decomposition into subspace rotations](./examples/single_qudit_decomp.ipynb)
- Generalized Pauli operators, qudit rotation gates, and other multi-qudit gates

### [Benchmarking Tools](./qdit/benchmarking/)
- [Efficient n-qudit Clifford circuit generation](./examples/clifford_sampling.ipynb)
- [Randomized benchmarking for single and multi-qudit systems](./examples/randomized_benchmarking.ipynb)

### [Qudit Circuit Compilation](./qdit/compiling/)
- [Native gate decomposition](./examples/basic_compiling.ipynb)
- [Randomized compiling for noise tailoring](./examples/random_compiling.ipynb)
- [Measurement randomized compiling](./examples/measurement_random_compiling.ipynb)
- [Transpiling cirq to qcal](./examples/cirq_to_qcal_transpiling.ipynb)

### [Noise Models](./qdit/noise/) (work in progress)
- [Amplitude damping channel for qutrits](./examples/noise_channels.ipynb)
- Depolarizing noise
- Trit-flip channel

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

import numpy as np
from cirq import Sampler, Simulator, LineQid, Circuit, MatrixGate, unitary, measure, to_json
from typing import List, Tuple, Optional, Sequence
from dataclasses import dataclass
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt
from ..gates.cliffords.utils import single_qutrit_cliffords
from ..noise import *
from .tableau import Tableau
from ..gates.utils.convert import matrix_to_cirq_gate
from ..utils.sun_factorization import *
import os

class QuditBenchmarking:
    """Class for performing randomized benchmarking on qudit systems."""
    
    def __init__(
        self,
        num_qudits: int,
        dimension: int,
        sampler: Sampler = Simulator()
    ):
        """Initialize benchmarking parameters.
        
        Args:
            num_qudits: Number of qudits in system
            dimension: Dimension of qudits (prime)
            sampler: Cirq sampler for running circuits
        """
        self.num_qudits = num_qudits
        self.dimension = dimension
        self.sampler = sampler
        self.qudits = [LineQid(i, dimension=dimension) 
                      for i in range(num_qudits)]
        
        self.circuits = None
    
    def get_inverse(self, circuit: Circuit) -> Circuit:
        """Compute inverse of given circuit."""
        # Get total matrix
        circuit_matrix = unitary(circuit)
        # Get inverse
        inverse_matrix = circuit_matrix.conj().T
        # Convert to gate
        inverse_gate = matrix_to_cirq_gate(inverse_matrix, dimension=self.dimension, name='Inverse', num_controls=len(circuit.qid_shape())-1)
        inverse_circuit = Circuit()
        inverse_circuit.append(inverse_gate(*circuit.all_qubits()))
        return inverse_circuit
    
    def simulate(self, circuit: Circuit, repetitions: int = 100) -> float:
        """Run circuit simulation and calculate fidelity."""
        return simulate_circuit_fidelity(circuit, self.dimension, repetitions)
    
    def generate_benchmark_circuits(
            self,
            depths: List[int],
            num_circuits: int = 10,
            expanded = False,
            json = False
    ) -> List[Circuit]:
        """Generate random Clifford circuits for benchmarking.
        
        Args:
            depths: List of circuit depths to generate
            num_circuits: Number of random circuits per depth
            expanded: If True, appends individual gates; if False, uses combined matrix gates
            
        Returns:
            List of lists of circuits, organized by depth
        """
        
        circuits = [[] for _ in range(len(depths))]

        for i, depth in enumerate(depths):
            for j in range(num_circuits):
                circuit = Circuit()
                for _ in range(depth):
                    if expanded:
                        clifford = sample_clifford(self.num_qudits, self.dimension)
                        circuit.append(clifford)
                    else:
                        clifford_unitary = unitary(sample_clifford(self.num_qudits, self.dimension))
                        clifford = MatrixGate(matrix=clifford_unitary,
                                              qid_shape=(self.dimension, )*self.num_qudits,
                                              name=f'C{_}'
                                              )
                        circuit.append(clifford.on(*self.qudits))
                circuit.append(self.get_inverse(circuit))
                circuits[i].append(circuit)
                
        self.circuits = circuits

        if json:
            self.to_json(self.circuits, "rb_circuits.json")

        return circuits
    
    def to_json(self, circuits, filename):
        """Save circuits to JSON file."""
        # create data folder in parent directory if it doesn't exist
        if not os.path.exists('../data'):
            os.makedirs('../data')
        filename = os.path.join('../data', filename)
        to_json(circuits, filename)

    def simulate_benchmarking(
        self,
        depths: List[int],
        num_circuits: int = 10,
        noise_levels: Sequence[float] = (0.0, 0.01),
        repetitions: int = 100,
        plot: bool = True
    ) -> dict:
        """Run full benchmarking procedure with depolarizing noise.
        
        Args:
            depths: Circuit depths to test
            num_circuits: Number of random circuits per depth
            noise_levels: Depolarizing noise strengths to test
            repetitions: Measurement repetitions per circuit
            plot: Whether to show results plot for each noise level
            
        Returns:
            Dictionary mapping noise levels to results containing:
                - depths: Circuit depths tested
                - fidelities: Mean fidelities at each depth
                - errors: Standard errors of the means
        """
        results = {}
        
        for noise in noise_levels:
            fidelities = np.zeros((len(depths), num_circuits))
            
            for i, depth in enumerate(depths):
                circuit = Circuit()
                for j in range(num_circuits):
                    gate = sample_clifford(self.num_qudits, self.dimension)
                    if noise > 0:
                        # Add noise channels
                        noisy_gate = self.add_noise(gate, noise)
                    else:
                        noisy_gate = gate
                    circuit.append(noisy_gate)
                        
                fidelities[i,j] = self.simulate(circuit, repetitions)
                    
            mean_fids = np.mean(fidelities, axis=1)
            errors = np.std(fidelities, axis=1) / np.sqrt(num_circuits)
            
            results[noise] = {
                'depths': depths,
                'fidelities': mean_fids,
                'errors': errors
            }
            
            if plot:
                self.plot_results(depths, mean_fids, errors, 
                                title=f'Noise Level: {noise}')
                
        return results
    
    def add_noise(self, circuit: Circuit, strength: float) -> Circuit:
        """Add depolarizing noise after each gate."""
        noisy_circuit = Circuit()
        for op in circuit.all_operations():
            noisy_circuit.append(op)
            noisy_circuit.append(Depolarizing(strength).on(op.qubits[0]))
        return noisy_circuit
    
    def plot_results(
        self,
        depths: np.ndarray,
        fidelities: np.ndarray,
        errors: np.ndarray,
        title: str = 'Benchmarking Results'
    ) -> None:
        """Plot RB results with exponential fit."""
        
        # Fit decay curve
        def decay(x, a, b, p):
            return a * p**x + b
        
        popt, _ = curve_fit(
            decay, depths, fidelities,
            p0=[1.0, 0.0, 0.95],
            bounds=([0, -0.5, 0], [2, 0.5, 1])
        )
        
        # Create plot
        plt.figure(figsize=(10,6))
        plt.errorbar(depths, fidelities, yerr=errors, 
                    fmt='o', label='Data')
        
        # Plot fit
        x_fit = np.linspace(min(depths), max(depths), 100)
        plt.plot(x_fit, decay(x_fit, *popt), 'r-',
                label=f'Fit: p = {popt[2]:.3f}')
        
        plt.xlabel('Circuit Depth')
        plt.ylabel('Average Fidelity')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


def sample_clifford(num_qudits, dimension) -> Circuit:
    """Generate random Clifford circuit using Tableau simulation.
        
    Args:
        num_qudits: Number of qudits in the circuit
        dimension: Dimension of each qudit
                    
    Returns:
        Cirq Circuit implementing a random Clifford operation
    """
    tableau = Tableau(num_qudits, dimension)
    tableau.populate_and_sweep(display=False)
    return tableau.circuit


@dataclass
class RandomizedBenchmarkingResult:
    """Class to store benchmarking results."""
    data: List[Tuple[int, float]]  # List of (num_cliffords, probability)

    def plot(self, ax: Optional[plt.Axes] = None):
        """Plot the benchmarking results.
        
        Args:
            ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates new figure.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        x_data = [x for x, _ in self.data]
        y_data = [y for _, y in self.data]
        ax.plot(x_data, y_data, marker='o', label="Benchmarking Results")
        ax.set_xlabel("Number of Cliffords")
        ax.set_ylabel("Ground State Probability")
        ax.legend()
        return ax

def single_qutrit_randomized_benchmarking(
    sampler: Sampler,
    qutrit: LineQid,
    num_clifford_range: range,
    num_circuits: int,
    repetitions: int,
    noise_level: Optional[float] = None,
    progress_bar: Optional[tqdm] = None,
    noise = Depolarizing
) -> RandomizedBenchmarkingResult:
    """Perform single-qutrit randomized benchmarking with optional noise.
    
    Args:
        sampler: Cirq sampler for running circuits
        qutrit: The qutrit to benchmark
        num_clifford_range: Range of Clifford gate numbers to test
        num_circuits: Number of random circuits per length
        repetitions: Number of repetitions per circuit
        noise_level: Optional noise level to apply after each gate
        progress_bar: Optional progress bar for tracking

    Returns:
        RandomizedBenchmarkingResult containing the benchmarking data
    """
    clifford_gates = single_qutrit_cliffords()
    flattened_gates = [gate[0] for gate in clifford_gates.gates]
    results = []
    
    for num_cliffords in num_clifford_range:
        probabilities = []
        for _ in range(num_circuits):
            # Create circuit with Clifford gates
            clifford_circuit = Circuit()
            random_gates = np.random.choice(flattened_gates, size=num_cliffords, replace=True)
            
            for gate in random_gates:
                clifford_circuit.append(gate.on(qutrit))
            
            # Compute inverse
            forward_unitary = unitary(clifford_circuit)
            inverse_unitary = np.linalg.inv(forward_unitary)
            inversion_gate = MatrixGate(inverse_unitary, qid_shape=(3,))
            
            # Create full circuit with noise
            full_circuit = Circuit()
            
            for gate in random_gates:
                full_circuit.append(gate.on(qutrit))
                if noise_level is not None:
                    full_circuit.append(noise(noise_level).on(qutrit))
            
            full_circuit.append(inversion_gate.on(qutrit))
            if noise_level is not None:
                full_circuit.append(noise(noise_level).on(qutrit))
            
            full_circuit.append(measure(qutrit, key='m'))
            
            # Simulate
            result = sampler.run(full_circuit, repetitions=repetitions)
            counts = result.histogram(key='m')
            ground_state_prob = counts.get(0, 0) / repetitions
            probabilities.append(ground_state_prob)
            
            if progress_bar is not None:
                progress_bar.update(1)
        
        avg_probability = np.mean(probabilities)
        results.append((num_cliffords, avg_probability))
    
    return RandomizedBenchmarkingResult(data=results)

def simulate_circuit_fidelity(
    circuit: Circuit,
    dimension: int,
    repetitions: int = 100
) -> float:
    """Simulate circuit and calculate fidelity from measurements."""
    # Add measurements
    qudits = circuit.all_qubits()
    for q in qudits:
        circuit.append(measure(q, key=f"q{q.x}"))
    # Run simulation
    simulator = Simulator()
    result = simulator.run(circuit, repetitions=repetitions)
    # Calculate fidelity from |0> state populations
    fidelities = []
    for key in sorted(result.measurements.keys()):
        counts = np.bincount(result.measurements[key][1], minlength=dimension)
        fidelities.append(counts[0] / repetitions)
        
    return np.mean(fidelities)

def test_noisy_qutrit_benchmarking(
    qutrit: LineQid = LineQid(0, dimension=3),
    sampler: Sampler = Simulator(),
    num_clifford_range: range = range(10, 20, 5),
    num_circuits: int = 15,
    repetitions: int = 100,
    noise_levels: Optional[float]  = (None, 0.01, 0.05, 0.1),
    noise_type = Depolarizing,
    profile = False
) -> List[Tuple[Optional[float], 'RandomizedBenchmarkingResult']]:
    """Run a test of qutrit randomized benchmarking with different noise levels.
    
    Args:
        qutrit: The qutrit to benchmark. Defaults to LineQid(0, dimension=3).
        sampler: Cirq sampler for running circuits. Defaults to Simulator().
        num_clifford_range: Range of Clifford gate numbers to test. Defaults to range(10, 20, 5).
        num_circuits: Number of random circuits per length. Defaults to 15.
        repetitions: Number of repetitions per circuit. Defaults to 100.
        noise_levels: Sequence of noise levels to test. None represents no noise. 
                     Defaults to (None, 0.01, 0.05, 0.1).
    
    Returns:
        List of tuples containing (noise_level, benchmarking_result) pairs.
    
    Example:
        >>> results = test_noisy_qutrit_benchmarking()
        >>> for noise_level, result in results:
        ...     print(f"Noise level: {noise_level}")
        ...     print(f"Final probability: {result.data[-1][1]}")
    """
    results = []
    
    # Calculate total number of circuits for progress bar
    total_circuits = len(noise_levels) * len(num_clifford_range) * num_circuits
    
    # Create main progress bar
    with tqdm(total=len(noise_levels), desc="Noise Levels", position=0) as pbar_noise:
        # Create circuit progress bar
        with tqdm(total=total_circuits, desc="Circuits", position=1) as pbar_circuits:
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            start_time = time.time()
            
            for i, noise_level in enumerate(noise_levels):
                result=single_qutrit_randomized_benchmarking(
                    sampler=sampler,
                    qutrit=qutrit,
                    num_clifford_range=num_clifford_range,
                    num_circuits=num_circuits,
                    repetitions=repetitions,
                    noise_level=noise_level,
                    progress_bar=pbar_circuits,
                    noise=noise_type
                )
                
                x_data = [x for x, _ in result.data]
                y_data = [y for _, y in result.data]
                label = f"Noise level: {noise_level if noise_level else 'None'}"
                ax.plot(x_data, y_data, marker='o', label=label)
                
                results.append((noise_level, result))
                
                # Update noise level progress bar
                pbar_noise.update(1)
                
                # Calculate and display estimated time remaining
                elapsed_time = time.time() - start_time
                avg_time_per_noise_level = elapsed_time / (i + 1)
                remaining_noise_levels = len(noise_levels) - (i + 1)
                estimated_time_remaining = avg_time_per_noise_level * remaining_noise_levels
                
                pbar_noise.set_postfix({
                    'Elapsed': f'{elapsed_time:.1f}s',
                    'Remaining': f'{estimated_time_remaining:.1f}s'
                })

    ax.set_xlabel("Number of Cliffords")
    ax.set_ylabel("Ground State Probability")
    ax.set_title("Qutrit Randomized Benchmarking with Different Noise Levels")
    ax.grid(True)
    ax.legend()
    plt.show()

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print(f"Average time per noise level: {total_time/len(noise_levels):.1f} seconds")
    print(f"Average time per circuit: {total_time/total_circuits:.3f} seconds")

    return results
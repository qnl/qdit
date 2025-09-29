import numpy as np
import itertools
import cirq
from typing import Optional, List
from ..gates.utils import weyl
from ..utils.math import check_pauli_relation, sympletic_product
from ..gates.single_qudit import (
    X, 
    Y, 
    Z,
    Id,
    F,
    S, 
    Ma,
    C
)
from ..utils.math import check_pauli_relation, sympletic_product
from ..gates.multi_qudit import CX, iSWAP
class Tableau:
    """A class representing a quantum tableau for tracking Pauli operators and Clifford operations.
    
    The Tableau class implements functionality for working with quantum tableaus, particularly
    for tracking Pauli operators and Clifford operations in qudit systems. It supports arbitrary
    qudit dimensions and provides methods for applying various quantum operations.

    Attributes:
        nq (int): Number of qudits in the system
        d (int): Dimension of each qudit (default=2 for qubits)
        iterations (int): Number of RB iterations to perform
        remaining_iterations (int): Counter for remaining iterations
        reduction_qudit_idx (int): Index of the current qudit being reduced
        row1_idx (int): Index of the first row being processed
        row2_idx (int): Index of the second row being processed
        tableau (np.ndarray): 2D array storing the full tableau state
        xtab (np.ndarray): View of X components of the tableau
        ztab (np.ndarray): View of Z components of the tableau 
        sign_bits (np.ndarray): Array storing sign bits for each row
        qudits (List[cirq.LineQid]): List of qudit objects
        circuit (cirq.Circuit): Circuit storing the applied operations
    """

    def __init__(self, nq: int, d: int = 2, iterations: Optional[int] = None, entangling_gate: str = 'CX'):
        """Initialize a quantum tableau.
        
        Args:
            nq (int): Number of qudits in the system
            d (int): Dimension of each qudit (default=2 for qubits)
            iterations (Optional[int]): Number of RB iterations to perform. Defaults to nq if None.
            entangling_gate (str): Type of two-qudit gate used for entanglement. Options include 'CX' (default) and 'iSWAP.
        
        Raises:
            AssertionError: If iterations is greater than nq
        """
        self.nq = nq  # Number of qudits
        self.d = d    # Qudit dimension
        self.iterations = iterations if iterations else nq
        assert self.iterations <= nq, "Number of iterations must be less than or equal to the number of qudits."
        self.remaining_iterations = self.iterations
        self.reduction_qudit_idx = 0
        self.row1_idx = 0
        self.row2_idx = 1

        if d!=2 and entangling_gate!='CX':
            return NotImplementedError 
        else: self.entangling_gate = entangling_gate
        
        self.tableau = np.zeros((2*self.iterations, 2*nq), dtype=int)
        
        self.xtab = self.tableau[:, :nq]
        self.ztab = self.tableau[:, nq:]
        self.sign_bits = np.zeros(2*self.iterations, dtype=int)
        
        # Initialize empty circuit
        self.circuit = cirq.Circuit()
        self.qudits = [cirq.LineQid(i, dimension=d) for i in range(nq)]
        # Add empty qudits to circuit
        for q in self.qudits:
            self.circuit.append(Id(dimension=self.d).on(q))

    def reset(self):
        """Reset the tableau state and circuit."""
        self.tableau = np.zeros((2*self.iterations, 2*self.nq), dtype=int)
        self.xtab = self.tableau[:, :self.nq]
        self.ztab = self.tableau[:, self.nq:]
        self.sign_bits = np.zeros(2*self.iterations, dtype=int)
        self.circuit = cirq.Circuit()
        self.remaining_iterations = self.iterations
        self.reduction_qudit_idx = 0
        self.row1_idx = 0
        self.row2_idx = 1

    def enumerate_settings(self):
        """Enumerate all possible valid anticommuting Pauli settings for the tableau by pairs of rows."""
        settings = []
        for red_idx in range(self.iterations):
            pad_length = red_idx
            pauli_length = (self.nq-red_idx)
            current_settings = []
            row1_settings = itertools.product(range(self.d), repeat=2*pauli_length)
            for row1 in row1_settings:
                # Skip if row1 is all zeros
                if all(x == 0 for x in row1):
                    continue
                    
                row2_settings = itertools.product(range(self.d), repeat=2*pauli_length)
                for row2 in row2_settings:
                    # Skip if row2 is all zeros
                    if all(x == 0 for x in row2):
                        continue
                    
                    # Check if operators anticommute using check_pauli_relation
                    if sympletic_product(row1, row2, d=self.d) == 1:
                        for signs in itertools.product(range(self.d), repeat=2):
                            padded_row1 = [0] * pad_length + list(row1[pauli_length:]) + [0] * pad_length + list(row1[:pauli_length])
                            padded_row2 = [0] * pad_length + list(row2[pauli_length:]) + [0] * pad_length + list(row2[:pauli_length])
                            current_settings.append((np.vstack((padded_row1, padded_row2)), np.array(signs)))
            settings.append(current_settings)
            
        return settings

    def generate_clifford_set(self):
        """Generate all possible Clifford operations for the given tableau configuration.
        
        Yields:
            cirq.Circuit: A circuit representing each possible Clifford operation
        """
        settings = self.enumerate_settings()
        all_pairs = itertools.product(*settings)
        for setting_pairs in all_pairs:
            while self.remaining_iterations > 0:
                pair = setting_pairs[self.reduction_qudit_idx]
                self.tableau[self.row1_idx:self.row2_idx+1] = pair[0]
                self.sign_bits[self.row1_idx], self.sign_bits[self.row2_idx] = pair[1]
                
                self.sweep(display=False)
                self.decrement_iteration()
                
            yield self.circuit
            self.reset()

    def is_completed(self):
        """Check if the tableau has completed all iterations."""
        for i in range(self.iterations):
            if not (self.xtab[2*i][i] == self.ztab[2*i+1][i] == 1 and not (any(self.xtab[2*i][i+1:] + self.ztab[2*i+1][i+1:]))):
                return False
        return True
        
    def decrement_iteration(self):
            """Decrements remaining iterations and updates indices."""
            self.remaining_iterations -= 1
            self.reduction_qudit_idx = self.iterations - self.remaining_iterations
            self.row1_idx = 2 * self.reduction_qudit_idx
            self.row2_idx = self.row1_idx + 1

    def apply_S_operation(self, qudit_idx):
        """Apply S operation to a specific qubit."""
        for row in range(len(self.xtab)):
            x, z = self.xtab[row][qudit_idx], self.ztab[row][qudit_idx]
            self.xtab[row][qudit_idx], self.ztab[row][qudit_idx] = x, (z + x) % self.d
        
        self.circuit.append(S(self.d)(self.qudits[qudit_idx]))

    def apply_F_operation(self, qudit_idx):
        """Apply F (Fourier) operation to a specific qudit."""
        for row in range(len(self.xtab)):
            x, z = self.xtab[row][qudit_idx], self.ztab[row][qudit_idx]
            self.xtab[row][qudit_idx], self.ztab[row][qudit_idx] = -z % self.d, x % self.d
        
        self.circuit.append(F(self.d)(self.qudits[qudit_idx]))
    
    def apply_M_operation(self, qudit_idx, a):
        """Apply M (multiplication) operation with factor a."""
        assert a > 0, "Multiplication factor must be non-zero."
        for row in range(len(self.xtab)):
            self.xtab[row][qudit_idx] = (self.xtab[row][qudit_idx] * a) % self.d
            self.ztab[row][qudit_idx] = (self.ztab[row][qudit_idx]/a) % self.d

        self.circuit.append(Ma(self.d, a)(self.qudits[qudit_idx]))

    def apply_C_operation(self, qudit_idx):
        """Apply C-gate to a specified qudit"""
        
        for row in range(len(self.xtab)):
            x, z = self.xtab[row][qudit_idx], self.ztab[row][qudit_idx]
            self.xtab[row][qudit_idx], self.ztab[row][qudit_idx] = (x-z)%self.d, (x)%self.d
        
        self.circuit.append(C(self.d)(self.qudits[qudit_idx]))

    def apply_CNOT_operation(self, control_idx, target_idx):
        """Apply CNOT operation between two qudits.
        
        Args:
            control_idx (int): Index of the control qudit
            target_idx (int): Index of the target qudit
        """
        for row in range(len(self.xtab)):
            self.xtab[row][target_idx] = (self.xtab[row][target_idx] + self.xtab[row][control_idx]) % self.d
            
        for row in range(len(self.ztab)):
            self.ztab[row][control_idx] = (self.ztab[row][control_idx] - self.ztab[row][target_idx]) % self.d
            
        self.circuit.append(CX(self.d)(self.qudits[control_idx], self.qudits[target_idx]))

    def apply_iSWAP_operation(self, control_idx, target_idx):
        """Apply iSWAP operation between two qubits

        Args:
            control_idx (int): Index of the control qudit
            target_idx (int): Index of the target qudit
        """
        
        for row in range(len(self.xtab)):
            self.xtab[row][control_idx], self.xtab[row][target_idx] = (self.xtab[row][control_idx] - self.xtab[row][control_idx] - self.xtab[row][target_idx]) % self.d, (self.xtab[row][target_idx] - self.xtab[row][control_idx] - self.xtab[row][target_idx]) % self.d
            
        for row in range(len(self.ztab)):
            self.ztab[row][control_idx], self.ztab[row][target_idx] = (self.ztab[row][control_idx] - self.xtab[row][control_idx] - self.xtab[row][target_idx] + self.ztab[row][control_idx] + self.ztab[row][target_idx]) % self.d, (self.ztab[row][target_idx] - self.xtab[row][control_idx] - self.xtab[row][target_idx] + self.ztab[row][control_idx] + self.ztab[row][target_idx]) % self.d

        self.circuit.append(iSWAP(2)(self.qudits[control_idx], self.qudits[target_idx]))

    def print(self):
        """Print the current circuit and tableau state.

        Displays the sequence of operations for each qudit and the current X/Z values.
        """
        print(self.circuit)
        for iter in range(self.iterations):
            for i in range(2):
                x_str = ''
                z_str = ''
                for j in range(self.nq):
                    x_str+=str((self.xtab[2*iter+i][j]))
                    z_str+=str((self.ztab[2*iter+i][j]))
                    if j < self.nq-1:
                        x_str+='|'
                        z_str+='|'
                print(x_str,'\t',z_str,'\t',self.sign_bits[2*iter+i])
            print('\n')

    def populate_and_sweep(self, display: bool = False):
        """Iteratively populates and reduces rows in the tableau until all iterations are complete.
        
        Args:
            display (bool): If True, displays intermediate steps during the sweep operation.
                Defaults to False.
        """
        while self.remaining_iterations>0:
            
            # Sample two rows of the tableau
            self.tableau[self.row1_idx:self.row2_idx+1] = self.sample()
            
            # Sweep the sampled rows to reduce them, passing the row indices
            self.sweep(display)
            
            self.decrement_iteration()

        return

    def sample(self):
        """Sample two rows of anticommuting Pauli operators.
        
        Returns:
            np.ndarray: Array with shape (2, 2*nq) containing the sampled rows
        """
        self.sign_bits[self.row1_idx], self.sign_bits[self.row2_idx] = np.random.randint(self.d), np.random.randint(self.d)
        table_rows = self.tableau[self.row1_idx:self.row2_idx+1]
        remaining_qubits = self.nq-self.reduction_qudit_idx
        while True:  # Keep sampling until we get a non-identity first operator
            # Initialize first Pauli operator
            pauli0 = np.eye(self.d)
            x0_list = [0 for _ in range(self.reduction_qudit_idx)]
            z0_list = [0 for _ in range(self.reduction_qudit_idx)]
            is_identity = True
            
            # Generate first row of random Pauli operators
            for n in range(remaining_qubits):
                x0, z0 = np.round((self.d-1)*np.random.rand(2))
                if x0 != 0 or z0 != 0:  # Check if this component is non-identity
                    is_identity = False
                w0 = weyl(x0, z0, d=self.d)
                pauli0 = np.kron(pauli0, w0)
                x0_list.append(x0)
                z0_list.append(z0)
            
            # If we got a non-identity operator, break the sampling loop
            if not is_identity:
                # Populate first row of table
                for n in range(self.nq):
                    table_rows[0][n] = x0_list[n]
                    table_rows[0][self.nq+n] = z0_list[n]
                break
        
        # Keep sampling second row until we find anticommuting operator
        while True:
            pauli1 = np.eye(self.d)  # Initialize second Pauli operator
            x1_list = [0 for _ in range(self.reduction_qudit_idx)]
            z1_list = [0 for _ in range(self.reduction_qudit_idx)]
            
            # Generate second row of random Pauli operators
            for n in range(remaining_qubits):
                x1, z1 = np.round((self.d-1)*np.random.rand(2))
                w1 = weyl(x1, z1, d=self.d)
                pauli1 = np.kron(pauli1, w1)
                x1_list.append(x1)
                z1_list.append(z1)
            
            # Check if operators anticommute
            if check_pauli_relation(x0_list+z0_list, x1_list+z1_list, d=self.d, non_commute=1):
                # Populate second row of table
                for n in range(self.nq):
                    table_rows[1][n] = x1_list[n]
                    table_rows[1][self.nq+n] = z1_list[n]
                break
        
        return table_rows
    
    def clear_row(self, row):
        """Clear specified row of Z"""
        # Iterate through qudits starting from the reduction index
        for i in range(self.nq-self.reduction_qudit_idx):
            # Get X and Z components for current qudit
            x,z=self.xtab[row][i+self.reduction_qudit_idx],self.ztab[row][i+self.reduction_qudit_idx]
            # If X and Z components are 0, and qudit is not reduction qudit, do nothing
            if x==0 and z==0 and i!=self.reduction_qudit_idx:
                continue
            # If X component is 0, apply Fourier transform to swap X and Z
            if x==0:
                self.apply_F_operation(i+self.reduction_qudit_idx)
            else:
            # While Z component is non-zero, apply phase gates
            # This adds X to Z component until Z becomes 0
                while z!=0:
                    z=(z+x)%self.d
                    self.apply_S_operation(i+self.reduction_qudit_idx)
        return

    def reduce_x_entries(self, row):
        """Reduce X entries to a single non-zero entry"""
        # Find indices where X entries are non-zero
        J = [j for j, x in enumerate(self.xtab[row]) if x!=0]

        # Continue reducing until only one non-zero X entry remains
        while len(J)>1:
            # Process pairs of non-zero entries
            for i in range(len(J)):
                # For even indices that have a following entry
                if i%2==0 and i+1<len(J):


                    if self.entangling_gate == 'CX':
                        # Get the X value at the target qudit
                        target_value = self.xtab[row][J[i+1]]
                        
                        # Keep applying CNOT until target X entry becomes 0
                        while target_value!=0:
                            target_value = (target_value+self.xtab[row][J[i]])%self.d
                            self.apply_CNOT_operation(J[i], J[i+1])


                    elif self.entangling_gate == 'iSWAP':
                        target_x = self.xtab[row][J[i+1]]
                        target_z = self.ztab[row][J[i+1]]
                        control_x = self.xtab[row][J[i]]
                        control_z = self.ztab[row][J[i]]
                        
                        while control_x != 0 or control_z != 1:
                            x, z = control_x, control_z
                            control_x, control_z = (x-z)%self.d, (x)%self.d
                            self.apply_C_operation(J[i])

                        while target_x != 1 or target_z != 1:
                            x, z = target_x, target_z
                            target_x, target_z = (x-z)%self.d, (x)%self.d
                            self.apply_C_operation(J[i+1])

                        self.apply_iSWAP_operation(J[i],J[i+1])

                    else: return NotImplementedError
                    
                    # Remove the cleared entry from our list
                    J.remove(J[i+1])
        
        # Return list containing index of remaining non-zero X entry
        return J

    def iswap_step_four(self, row):
        # Find indices where X entries are non-zero
        J = [j for j, x in enumerate(self.xtab[row]) if x!=0]
        self.apply_F_operation(self.reduction_qudit_idx)
        # Continue reducing until only one non-zero X entry remains
        while len(J)>1:
            # Process pairs of non-zero entries
            for i in range(len(J)):
                if i%2==0 and i+1<len(J):
                    target_x = self.xtab[row][J[i+1]]
                    target_z = self.ztab[row][J[i+1]]
                    control_x = self.xtab[row][J[i]]
                    control_z = self.ztab[row][J[i]]
                    
                    # if control_x == 0 and control_z == 1:
                    #     control_z = (control_z+control_x)%self.d
                    #     self.apply_S_operation(self.reduction_qudit_idx)

                    # while control_x != 1 or control_z != 1:
                    #     x, z = control_x, control_z
                    #     control_x, control_z = (x-z)%self.d, (x)%self.d
                    #     self.apply_C_operation(self.reduction_qudit_idx)

                    # while target_x != 0 or target_z != 1:
                    #     x, z = target_x, target_z
                    #     target_x, target_z = (x-z)%self.d, (x)%self.d
                    #     self.apply_C_operation(J[i])
                    
                    self.apply_S_operation(J[i])
                    self.apply_C_operation(J[i])
                    self.apply_C_operation(J[i])

                    self.apply_C_operation(J[i+1])
                    self.apply_C_operation(J[i+1])

                    self.apply_iSWAP_operation(J[i], J[i+1])

                    self.apply_C_operation(J[i+1])
                    self.apply_C_operation(J[i+1])

                    self.apply_S_operation(J[i+1])

                    self.apply_iSWAP_operation(J[i],J[i+1])
                
                    # Remove the cleared entry from our list
                    J.remove(J[i+1])
        # control_x = self.xtab[row][J[i]]
        # control_z = self.ztab[row][J[i]]
        # while control_x != 0 or control_z != 1:
        #     x, z = control_x, control_z
        #     control_x, control_z = (x-z)%self.d, (x)%self.d
        #     self.apply_C_operation(J[i])
        # Return list containing index of remaining non-zero X entry
        return J
            
    
    def swap(self, i: int, j: int):
        """Swap two columns in the tableau.
        
        Args:
            i (int): First qudit index
            j (int): Second qudit index
        
        Performs an in-place swap of columns i and j in both X and Z components.
        """
        if self.entangling_gate=='CX':
            self.apply_CNOT_operation(i, j)
            for _ in range(self.d-1):
                self.apply_CNOT_operation(j, i)
            self.apply_CNOT_operation(i, j)
            self.apply_F_operation(i)
            self.apply_F_operation(i)
        else:
            return NotImplementedError
    
    def sweep(self, display=True):
        """
        Performs a sweeping operation on the tableau.

        Steps:
        1. Clears the first row of entries
        2. Reduces X entries in the first row
        3. If needed, swaps qubits to place the target qudit in the reduction position
        4. Processes the second row:
            - If second Pauli is not Z at reduction qubit site:
              * Applies F operation (Fourier transform)
              * Clears the second row
              * Reduces X entries
              * Applies inverse F operation
        5. If necessary, applies M operations to normalize the Pauli operators
        6. Clears sign bits

        Parameters:
        ----------
        display : bool, optional
             If True, prints the tableau state after each operation (default is True)

        Returns:
        -------
        None"""

        if display: 
            self.print()
            print('Sweeping...')
            print('Step 1:')

        # Step 1: Clear Z entries of first row
        self.clear_row(self.row1_idx)
        if display: 
            self.print()
            print('Step 2:')

        # Step 2: Reduce X entries in first row
        J = self.reduce_x_entries(self.row1_idx)
        if display: self.print()

        # Step 3: Swap qubits if needed
        if J[0]!=self.reduction_qudit_idx:
            if display: print(f'Step 3: Swapping qubit {J[0]} with qubit {self.reduction_qudit_idx}')
            if self.entangling_gate=='CX': 
                self.swap(self.reduction_qudit_idx, J[0])
            else: 
                if self.xtab[self.row1_idx][self.reduction_qudit_idx]==0 and self.ztab[self.row1_idx][self.reduction_qudit_idx] == 0:
                    self.apply_iSWAP_operation(self.reduction_qudit_idx, J[0])
                
                while self.xtab[self.row1_idx][self.reduction_qudit_idx]!=0 or self.ztab[self.row1_idx][self.reduction_qudit_idx]!=1:
                    self.apply_C_operation(self.reduction_qudit_idx)

                while self.xtab[self.row1_idx][J[0]]!=1 or self.ztab[self.row1_idx][J[0]]!=1:
                    self.apply_C_operation(J[0])

                self.apply_iSWAP_operation(self.reduction_qudit_idx, J[0])
            if display: self.print()
        else: 
            if display: print(f'Step 3: skipped')

        # Step 4: Process second row
        if display: print('Step 4: Repeat for second row')
        # Check if second pauli is not Z at reduction qubit site
        if np.any(self.xtab[self.row2_idx]) or \
           np.any(self.ztab[self.row2_idx][:self.reduction_qudit_idx]) or \
           np.any(self.ztab[self.row2_idx][self.reduction_qudit_idx+1:]) or \
           self.ztab[self.row2_idx][self.reduction_qudit_idx] == 0:
            if display: self.print()
            
            # Apply Fourier transform
            self.apply_F_operation(self.reduction_qudit_idx)
            if display: self.print()
            # Clear second row
            self.clear_row(self.row2_idx)
            if display: self.print()
            if self.entangling_gate == 'CX':
                # Reduce X entries
                self.reduce_x_entries(self.row2_idx)
                self.apply_F_operation(self.reduction_qudit_idx)
                if display: self.print()
            else:
                self.iswap_step_four(self.row2_idx)
                if display: self.print()
            # Apply inverse Fourier transform
            if display: self.print()
        else: 
            if display: print('Step 4: skipped')

        # Step 5: Normalize Pauli operators if needed
        if self.xtab[self.row1_idx][self.reduction_qudit_idx]==1 and self.ztab[self.row2_idx][self.reduction_qudit_idx]==1:
            if display: print('Step 5: skipped')
        while self.xtab[self.row1_idx][self.reduction_qudit_idx]!=1 and self.ztab[self.row2_idx][self.reduction_qudit_idx]!=1:
            if display: print(f'Step 5a: Reducing qudit X pauli from {self.xtab[self.row1_idx][self.reduction_qudit_idx]} --> 1')
            if display: print(f'Step 5a: Reducing qudit Z pauli from {self.ztab[self.row2_idx][self.reduction_qudit_idx]} --> 1')
            self.apply_M_operation(self.reduction_qudit_idx, self.ztab[self.row2_idx][self.reduction_qudit_idx])
            if display: self.print()

        # Step 6: Clear sign bits
        if display: print('Clearing Signs...')
        self.clear_signs()
        return
    
    def clear_signs(self):
        """Clear sign bits in the tableau by applying appropriate Pauli operations.
        
        Applies X, Y, or Z operations to clear the sign bits of the current rows
        being processed (row1_idx and row2_idx).
        """
        while self.sign_bits[self.row1_idx] != 0 or self.sign_bits[self.row2_idx] != 0:
            if self.sign_bits[self.row1_idx] and self.sign_bits[self.row2_idx]:
                # Apply Y operation
                self.circuit.append(Y(self.d)(self.qudits[self.reduction_qudit_idx]))
                self.sign_bits[self.row1_idx] -= 1
                self.sign_bits[self.row2_idx] -= 1
            elif self.sign_bits[self.row1_idx]:
                # Apply Z operation
                self.circuit.append(Z(self.d)(self.qudits[self.reduction_qudit_idx]))
                self.sign_bits[self.row1_idx] -= 1
            elif self.sign_bits[self.row2_idx]:
                # Apply X operation
                self.circuit.append(X(self.d)(self.qudits[self.reduction_qudit_idx]))
                self.sign_bits[self.row2_idx] -= 1
        return
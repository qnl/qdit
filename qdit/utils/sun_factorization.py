#!/usr/bin/python                                                                  
# -*- coding: utf-8 -*-                                                            
#                                                                                  
# sun_factorization.py: Functions to perform factorization of SU(n)
#                       transformations into SU(2) transformations using
#                       the method of arXiv:1708.00735.
#                                                                                  
# © 2017 Olivia Di Matteo (odimatte@uwaterloo.ca)                                  
#                                                                                  
# This file is part of the project Caspar.
# Licensed under BSD-3-Clause                                                      
#  

import numpy as np
from numpy.linalg import det
import math

def su2_parameters(U):
    """ Given a matrix in SU(2), parametrized as 
            U(a, b, g) = [ e^(i(a+g)/2) cos(b/2)  -e^(i(a-g)/2) sin(b/2)
                           e^(-i(a-g)/2) sin(b/2)  e^(-i(a+g)/2) cos(b/2) ]
        compute and return the parameters [a, b, g].
    """
    if U.shape != (2, 2):
        print("Error, matrix dimensions of su2_parameters must be 2x2.")
        return
    assert np.isclose(det(U), 1, atol=1e-3), \
        f"Error, matrix must have det 1 to be decomposed into SU(2) parameters but det({U}) = {det(U)}."

    # Sometimes the absolute value of the matrix entry is very, very close to
    # 1 and slightly above, when it should be 1 exactly. Isolate these cases
    # to prevent us from getting NaN.
    b = None
    if np.isclose(np.absolute(U[0, 1]), 1):
        b = 2 * np.arcsin(1, dtype=np.complex128)
    else:
        b = 2 * np.arcsin(np.absolute(U[0, 1]), dtype=np.complex128)

    arg_pos = np.angle(U[0, 0]) #(a + g)/2
    arg_neg = -np.angle(U[1, 0]) #(a - g)/2
    a, g = math.fsum([arg_pos, arg_neg]), math.fsum([arg_pos, -arg_neg])
    return [a, b, g]


def su3_parameters(U):
    """ Uses the factorization of 
            Rowe et. al (1999), "Representations of the Weyl group and Wigner
            functions for SU(3)", J. Math. Phys. 40 (7), 3604.
        to factorize an SU(3) transformation into 3 SU(2) transformations.

        Parameters for each SU(2) transformation are returned as a list
        [a, b, g] (three-parameter transformation) or [a, b, a] (two-parameter
        transformation) where the matrices are to be parametrized as 
            SU_ij(a, b, g) = [ e^(i(a+g)/2) cos(b/2)  -e^(i(a-g)/2) sin(b/2)
                               e^(-i(a-g)/2) sin(b/2)  e^(-i(a+g)/2) cos(b/2) ]

        The ij subscript indicates that the matrix should be embedded into
        modes i and j of the full n-dimensional transformation.

        The resultant matrix is expressed as
            U = SU_23(a1, b1, g1) SU_12(a2, b2, a2) SU_23(a3, b3, g3).
    """
    if U.shape != (3, 3):                                                       
        print("Error, matrix dimensions of su3_parameters must be 3x3.")        
        return
    assert np.isclose(det(U), 1), f"Error, matrix must have det 1 to be decomposed into SU(2) parameters but det({U}) = {det(U)}."

    # Grab the entries of the first row
    x, y, z = U[0,0], U[1,0], U[2,0]

    # Special case: if the top left element is 1, then we essentially
    # already have an SU(2) transformation embedded in an SU(3) transform,
    # so all we need to do is get the parameters of that SU(2) transform.
    if np.isclose(x, 1):
        return [[0., 0., 0.], [0., 0., 0.], su2_parameters(U[1:,1:])]
    # Another special case: the modulus of the top left element is 1. 
    # Then we need to do a transformation on modes 1 and 2 to make the top 
    # entry 1, then an SU(2) transformation on modes 2 and 3 with what's left.
    elif np.isclose(np.abs(x), 1):
        # Compute the required phase matrix and embed into SU(3)
        phase_su2 = np.matrix([[np.conj(x), 0], [0, x]])

        full_phase_su2 = np.asmatrix(np.identity(3)) + 0j
        full_phase_su2[0:2, 0:2] = phase_su2

        # Compute what's left of the product, and the parameters
        running_product = accurate_mat_mul(full_phase_su2, U)
        
        remainder_su2 = running_product[1:, 1:]

        return [[0., 0., 0.], su2_parameters(phase_su2.getH()), su2_parameters(remainder_su2)]

    # Typical case
    cf = np.sqrt(1 - pow(np.absolute(x), 2), dtype=np.complex128)
    capY, capZ = y / cf, z / cf 

    # Build the SU(2) transformation matrices
    # SU_23(3) - three parameters
    left = np.matrix([[1, 0, 0], 
                      [0, capY, -np.conj(capZ)],
                      [0, capZ, np.conj(capY)]])
    left_params = su2_parameters(left[1:, 1:])

    # SU_12(2) - only two parameters
    middle = np.matrix([[x, -cf, 0],
                        [cf, np.conj(x), 0],
                        [0, 0, 1]], dtype=np.complex128)
    middle_params = su2_parameters(middle[0:2, 0:2])

    # SU_23(3) - again three parameters
    right = middle.getH() * left.getH() * U
    right_params = su2_parameters(right[1:, 1:]) 

    return [left_params, middle_params, right_params]

def accurate_mat_mul(A, B):
    prod = np.array([[complex(math.fsum([np.real(A[i,k] * B[k,j]) for k in range(A.shape[1])]), 
                        math.fsum([np.imag(A[i,k] * B[k,j]) for k in range(A.shape[1])]))
                for j in range(B.shape[1])]
                for i in range(A.shape[0])], dtype=np.complex128)
    return np.matrix(prod)

def build_staircase(U):
    """ Take a matrix in SU(n) and find the staircase of SU(2)
        transformations which turns it into an SU(n-1) transformation
        on all but the first mode.

        Returns the list of parameters in the order in which they appear
        graphically, e.g. for SU(5) will return parameters for a staircase
        order as transformations on modes 45, 34, 23, and finally 12.
    """
    n = U.shape[0]

    # We need to do n - 1 transformations, starting from the bottom up.
    transformations = []
    running_prod = U

    # There are a number of special cases to consider which occur when the
    # left-most column contains all 0s except for one entry.
    moduli = [np.abs(U[x, 0]) for x in range(n)]
    if np.allclose(sorted(moduli), [0.] * (n - 1) + [1]): 
        # In the special case where the top-most entry is a 1, or within some
        # small tolerance of it, we basically already have an SU(n-1) transformation
        # in there so just fill with empty parameters 
        if np.isclose(running_prod[0, 0], 1):
            transformations = [[0., 0., 0.]] * (n - 1)
        # Another special case is when the top left entry has modulus 1 (or close
        # to it). Now we need to add a separate phase shift as well. 
        elif np.isclose(np.abs(running_prod[0, 0]), 1):
            # "Phase shift" by applying an SU(2) transformation to cancel out the
            # top-most phase. Do nothing to everything else.
            phase_su2 = np.matrix([[np.conj(running_prod[0, 0]), 0], [0, running_prod[0, 0]]])
            transformations = [[0., 0., 0.]] * (n - 2) + [su2_parameters(phase_su2.getH())]

            full_phase_su2 = np.asmatrix(np.identity(n)) + 0j
            full_phase_su2[0:2, 0:2] = phase_su2
            running_prod = accurate_mat_mul(full_phase_su2,running_prod)
            
        else:
            # If the non-zero entry is lower down, permute until it
            # reaches the top and then apply a phase transformation.
            for rot_idx in range(n - 1, 0, -1):
                if np.round(running_prod[rot_idx, 0], 10) != 0:
                    permmat = np.matrix([[0, -1], [1, 0]])
                
                    full_permmat = np.asmatrix(np.identity(n)) + 0j
                    full_permmat[rot_idx-1:rot_idx+1, rot_idx-1:rot_idx+1] = permmat
                    temp_product = full_permmat * running_prod

                    if rot_idx == 1: # If we're at the top, add the phase too
                        phase_su2 = np.matrix([[np.conj(temp_product[0, 0]), 0], [0, temp_product[0, 0]]])
                        permmat = phase_su2 * permmat

                    transformations.append(su2_parameters(permmat.getH()))

                    full_trans = np.asmatrix(np.identity(n)) + 0j
                    full_trans[rot_idx-1:rot_idx+1, rot_idx-1:rot_idx+1] = permmat 
                        
                    running_prod = accurate_mat_mul(full_trans, running_prod)

                else: # Otherwise do nothing between these modes
                    transformations.append([0, 0, 0])
    else:
        for rot_idx in range(n - 1):
            # Start at the bottom
            i, j = n - rot_idx - 2, n - rot_idx - 1

            # Initially we work with the inverses in order to "0 out" entries
            # from the left; later we'll get the parameters from the "true" matrices.
            Rij_inv = np.asmatrix(np.identity(2)) + 0j
            full_Rij_inv = np.asmatrix(np.identity(n)) + 0j

            # Skip if entry is already 0 (within numerical precision)
            if np.isclose(running_prod[j, 0], 0):
                transformations.append([0., 0., 0.])
                continue

            if rot_idx != n - 2:
                
                # The denominator of the transformation is the difference of
                # absolute values of all columns *up* to this point.
                sum_of_column = 0
                for k in range(i):
                    sum_of_column += pow(np.absolute(running_prod[k,0]), 2)

                cf = np.sqrt(1 - sum_of_column)
                y, z = running_prod[i, 0], running_prod[j, 0] 
                capY, capZ  = y / cf, z / cf 

                # Build the SU(2) transformation and embed it into the larger matrix
                Rij_inv = np.matrix([[np.conj(capY), np.conj(capZ)], 
                                        [-capZ, capY]])
            else: 
                # The last transformation, R12 is special and the rotation has
                # a different form
                x = running_prod[0,0]
                y = running_prod[1,0]
                Rij_inv = np.matrix([[np.conj(x), np.conj(y)], 
                                    [-y, x]])

            # Add the transformation to the sequence and update the product
            Rij = Rij_inv.getH()
            # Embed into larger space
            full_Rij_inv[i:j+1, i:j+1] = Rij_inv
            # Use math.fsum for more accurate matrix multiplication
            running_prod = accurate_mat_mul(full_Rij_inv, running_prod)
            transformations.append(su2_parameters(Rij))

    return transformations, running_prod


def sun_parameters(U):
    """ Compute the set of parameters of the SU(2) transforms in the
        factorization scheme.
        
        This is a recursive process. The first step is to produce a 
        "staircase" of transformations on adjacent modes (d-1, d), (d-2, d-1),...
        so that what's left is an SU(n-1) transformation embedded in the lower 
        portion of the original system. This is performed recursively down to
        the case of SU(3) where we use the Rowe et al algorithm to get the
        rest of the transformation.
    """
    if U.shape == (3, 3):
        # Base case of recursion is SU(3).
        return su3_parameters(U)
    else:
        # All other cases we must build the staircase and repeat the
        # process on the resultant SU(n-1) transformation
        staircase_transformation, new_U = build_staircase(U)
        assert np.isclose(new_U[0,0],1,rtol=1e-10), f"Error, matrix must have 1 in the top left corner to be decomposed into SU(2) parameters but U[0,0] = {U[0,0]}."
        Unm1 = new_U[1:,1:] # Grab the lower chunk of the matrix
        return staircase_transformation + sun_parameters(Unm1) 


def sun_factorization(U):
    """ Decompose an arbitrary element in SU(n) as a sequence of 
        SU(2) transformations as per our method arXiv:1708.00735. 

        Takes as input a numpy matrix U.
        Returns a list of operations in the form "i,i+1", [a, b, g] where
        the i indicate the modes of an SU(2) transformation and [a, b, g]
        are its parameters.
    """
    # Check that the matrix has the proper form, i.e. it is unitary and
    # it has det 1.
    n = U.shape[0]

    if n == 2: 
        print("Matrix is already decomposed")
        return U
    
    assert n >= 3, f"Error, matrix for decomposition must be at least 3x3 but has dim {U.shape}."

    assert np.isclose(det(U), 1), f"Error, matrix must have det 1 to be decomposed into SU(2) parameters. det: {det(U)}"
    assert np.allclose(U * U.getH(), np.eye(n)), "Error, matrix must be unitary."

    # Decompose the matrix
    parameters_no_modes = sun_parameters(U)

    # Add the info about which modes each transformation is on
    parameters = []
    param_idx = 0                                                                   

    for md2 in range(2, n + 1):                                                 
        for md1 in range(n - 1, md2 - 2, -1):                               
            parameters.append((str(md1) + "," + str(md1+1), parameters_no_modes[param_idx]))
            param_idx += 1  

    return parameters
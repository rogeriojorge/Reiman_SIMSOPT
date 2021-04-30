#!/usr/bin/env python3

"""
This module provides a function for finding periodic field lines and the magnetic axis.
"""

import numpy as np
import scipy.optimize
from scipy.integrate import solve_ivp
from spectral_diff_matrix import spectral_diff_matrix
from fourier_interpolation import fourier_interpolation

class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """
    pass

def func(x, n, D, phi, field):
    """
    This is the vector-valued function that returns the residual.
    """
    R = x[0:n]
    Z = x[n:2 * n]
    #print('func eval ')
    #print('R=',R)
    #print('Z=',Z)
    BR = np.zeros(n)
    Bphi = np.zeros(n)
    BZ = np.zeros(n)
    for j in range(n):
        BR[j], Bphi[j], BZ[j] = field.BR_Bphi_BZ(R[j], phi[j], Z[j])
    R_residual = R * BR / Bphi - np.matmul(D, R)
    Z_residual = R * BZ / Bphi - np.matmul(D, Z)
    #R_residual = 0 * R * BR / Bphi - np.matmul(D, R)
    #Z_residual = 0 * R * BZ / Bphi - np.matmul(D, Z)
    return np.concatenate((R_residual, Z_residual))

def jacobian(x, n, D, phi, field):
    """
    This function returns the derivative of "func"
    """
    R = x[0:n]
    Z = x[n:2 * n]
    print('jacobian eval ')
    #print('R=',R)
    #print('Z=',Z)
    jac = np.zeros((2 * n, 2 * n))
    jac[0:n, 0:n] = -D
    jac[n:, n:] = -D
    for j in range(n):
        BR, Bphi, BZ = field.BR_Bphi_BZ(R[j], phi[j], Z[j])
        grad_B = field.grad_B(R[j], phi[j], Z[j])
        # Top left quadrant: d (R residual) / d R
        jac[j, j] += BR / Bphi + R[j] * grad_B[0, 0] / Bphi - R[j] * BR / (Bphi * Bphi) * grad_B[1, 0]
        # Top right quadrant: d (R residual) / d Z
        jac[j, j + n] = R[j] * grad_B[0, 2] / Bphi - R[j] * BR / (Bphi * Bphi) * grad_B[1, 2]
        # Bottom left quadrant: d (Z residual) / d R
        jac[j + n, j] = BZ / Bphi + R[j] * grad_B[2, 0] / Bphi - R[j] * BZ / (Bphi * Bphi) * grad_B[1, 0]
        # Bottom right quadrant: d (Z residual) / d Z
        jac[j + n, j + n] += R[j] * grad_B[2, 2] / Bphi - R[j] * BZ / (Bphi * Bphi) * grad_B[1, 2]
    #R_residual = R * BR / Bphi - np.matmul(D, R)
    #Z_residual = R * BZ / Bphi - np.matmul(D, Z)
    return jac


def periodic_field_line(field, n, periods=1, R0=None, Z0=None, tol=1e-13):
    """
    Solves for a periodic field line.

    field: An object with a BR_Bphi_BZ method and nfp attribute..
    periods: The number of field periods over which the field line will be periodic.
    n: The number of grid points to use for the solve. If n is even, 1 will be added
       so n always ends up odd.
    R0: An initial guess for R. Either a float or a numpy array of shape (n,). 
       If a single float, B will be roughly integrated to generate an array.
    Z0: An initial guess for Z. Either a float or a numpy array of shape (n,). 
       If a single float, B will be roughly integrated to generate an array

    R0 and Z0 must either be both numpy ndarrays, or else both floats.
    """
    
    # Ensure n is odd:
    if np.mod(n, 2) == 0:
        n += 1
    
    if R0 is None:
        R0 = 1.0
    if Z0 is None:
        Z0 = 0.0

    phimax = periods * 2 * np.pi / field.nfp
    dphi = phimax / n
    phi = np.linspace(0, phimax, n, endpoint=False)
    assert np.abs(phi[1] - phi[0] - dphi) < 1.0e-13

    array_input = isinstance(R0, np.ndarray)
    if array_input:
        assert isinstance(Z0, np.ndarray)
        assert R0.shape == Z0.shape

    if not array_input:
        # Use Runge-Kutta to initialize our guess for the field line
        t_span = (0, phimax)
        phi_plus1 = np.linspace(0, phimax, n + 1, endpoint=True)
        sol = solve_ivp(field.d_RZ_d_phi, t_span, [R0, Z0], t_eval = phi_plus1, atol=1e-8, rtol=1e-11)
        # Shift the trajectory by a linear function to make it periodic
        R0 = sol.y[0, :-1] - np.linspace(0, sol.y[0, -1] - sol.y[0, 0], n, endpoint=True)
        Z0 = sol.y[1, :-1] - np.linspace(0, sol.y[1, -1] - sol.y[1, 0], n, endpoint=True)
        assert R0.shape == phi.shape
        ## Use a crude 1st-order Euler step to generate the initial guess.
        #R0 = np.full(n, R0)
        #Z0 = np.full(n, Z0)
        #for j in range(n - 1):
        #    BR, Bphi, BZ = field.BR_Bphi_BZ(R0[j], dphi * j, Z0[j])
        #    R0[j + 1] = R0[j] + dphi * R0[j] * BR / Bphi
        #    Z0[j + 1] = Z0[j] + dphi * R0[j] * BZ / Bphi
            
        print('Generated initial condition:')
        print('R0: ', R0)
        print('Z0: ', Z0)

    D = spectral_diff_matrix(n, xmin=0, xmax=phimax)
        
    state = np.concatenate((R0, Z0))
    #root, infodict, ier, mesg = fsolve(func, state)
    #root = fsolve(func, state, xtol=1e-13, args=(n, D, phi, field))
    soln = scipy.optimize.root(func, state, tol=tol, args=(n, D, phi, field), jac=jacobian, method='lm', options={'maxiter':100})
    root = soln.x
    R = root[0:n]
    Z = root[n:2 * n]

    #residual = func(root, n, D, phi, field)
    residual = soln.fun
    print('Residual: ', np.max(np.abs(residual)))

    # Find the intersections of the periodic field line with other symmetry planes:
    # Note that the Fourier interpolation routine assumes periodicity in [0, 2pi), not [0, 2pi*periods/nfp)
    phi_k = np.linspace(0, 2 * np.pi, periods, endpoint=False)
    R_k = fourier_interpolation(R, phi_k)
    Z_k = fourier_interpolation(Z, phi_k)

    # Pack results for return
    results = Struct()
    
    results.R = R
    results.phi = phi
    results.Z = Z
    results.periods = periods
    results.nfp = field.nfp
    results.R_k = R_k
    results.Z_k = Z_k
    results.residual = np.max(np.abs(residual))

    return results


def circumference(pfl, R0, Z0):
    """
    pfl: Data structure returned by periodic_field_lin. 
    R0, Z0: Point in the middle of the periodic field line, about which a poloidal
    angle will be calculated for ordering. Typically this point is the
    magnetic axis.

    Attributes are added to pfl and the resulting object is returned.
    """

    theta = np.arctan2(pfl.Z_k - Z0, pfl.R_k - R0)

    # Sort the points by increasing theta
    rho = np.argsort(theta)
    # Shift the ordering so index 0 is first:
    shift = list(rho).index(0)
    rho = np.roll(rho, -shift)

    Rs = pfl.R_k[rho]
    Zs = pfl.Z_k[rho]
    circumf = np.sqrt((Rs[0] - Rs[-1]) ** 2 \
                      + (Zs[0] - Zs[-1]) ** 2)
    for j in range(1, pfl.periods):
        circumf += np.sqrt((Rs[j] - Rs[j - 1]) ** 2 \
                           + (Zs[j] - Zs[j - 1]) ** 2)

    pfl.circumference = circumf
    pfl.rho = rho
    pfl.rho_inv = np.argsort(rho)
    pfl.Rs = Rs
    pfl.Zs = Zs

    return pfl

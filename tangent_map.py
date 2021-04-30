#!/usr/bin/env python3

"""
This module provides a function for finding properties derived from the full-orbit tangent map
of a periodic field line, including the rotational transform, Greene's residue, and the {e||, eperp} vectors.
"""

import numpy as np
from scipy.integrate import solve_ivp

def tangent_map_integrand(phi, RZU, field):
    """
    This is the function used for integrating the field line ODE.
    See the following Word doc for details:
    20200923-01 Equation for tangent map.docx
    """
    R = RZU[0]
    Z = RZU[1]
    u00 = RZU[2]
    u01 = RZU[3]
    u10 = RZU[4]
    u11 = RZU[5]
    #print('tangent_map_integrand called at R={}, phi={}, Z={}'.format(R, phi, Z))
    BR, Bphi, BZ = field.BR_Bphi_BZ(R, phi, Z)
    grad_B = field.grad_B(R, phi, Z)
    # VR = R * BR / Bphi
    # VZ = R * BZ / Bphi
    d_VR_d_R = BR / Bphi + R * grad_B[0, 0] / Bphi - R * BR / (Bphi * Bphi) * grad_B[1, 0]
    d_VR_d_Z = R * grad_B[0, 2] / Bphi - R * BR / (Bphi * Bphi) * grad_B[1, 2]
    d_VZ_d_R = BZ / Bphi + R * grad_B[2, 0] / Bphi - R * BZ / (Bphi * Bphi) * grad_B[1, 0]
    d_VZ_d_Z = R * grad_B[2, 2] / Bphi - R * BZ / (Bphi * Bphi) * grad_B[1, 2]
    
    dydt = [R * BR / Bphi,
            R * BZ / Bphi,
            d_VR_d_R * u00 + d_VR_d_Z * u10,
            d_VR_d_R * u01 + d_VR_d_Z * u11,
            d_VZ_d_R * u00 + d_VZ_d_Z * u10,
            d_VZ_d_R * u01 + d_VZ_d_Z * u11]
    return dydt
    
class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """
    pass

def tangent_map(field, pfl, rtol=1e-6, atol=1e-9):
    """
    Computes the full-orbit tangent map for a periodic fieldline, as well
    as quantities derived from the map like the rotational transform and Greene's residue.

    field: An object with BR_Bphi_BZ and grad_B methods and nfp attribute.
    periods: The number of field periods over which the field line will be periodic.
    R0: Location of the periodic field line at phi=0.
    Z0: Location of the periodic field line at phi=0.
    rtol: relative tolerance for integration along the field line.
    atol: absolute tolerance for integration along the field line.

    R0 and Z0 should each be a float, previously computed using periodic_field_line().
    """

    periods = pfl.periods # shorthand
    
    def tangent_map_integrand_wrapper(t, y):
        return tangent_map_integrand(t, y, field)

    single_period_tangent_maps = []
    R_end = np.roll(pfl.R_k, -1)
    Z_end = np.roll(pfl.Z_k, -1)
    print('R_k:  ', pfl.R_k)
    print('R_end:', R_end)
    print('Z_k:  ', pfl.Z_k)
    print('Z_end:', Z_end)
    for j in range(periods):
        print('---- Computing tangent map for period {} ----'.format(j))
        phimax = 2 * np.pi / field.nfp
        t_span = (0, phimax)
        # Initialize using (R,Z) from the spectral approach, since it
        # may be more accurate than the initial-value calculation
        # here.
        R0 = pfl.R_k[j]
        Z0 = pfl.Z_k[j]
        
        # The state vector has 6 unknowns: R, Z, and the 4 elements of the U matrix.
        x0 = [R0, Z0, 1, 0, 0, 1]

        soln = solve_ivp(tangent_map_integrand_wrapper, t_span, x0, rtol=rtol, atol=atol)

        print('# of function evals: ', soln.nfev)

        # Make sure we got to the end:
        assert np.abs(soln.t[-1] - phimax) < 1e-13

        R = soln.y[0, :]
        Z = soln.y[1, :]
        # Make sure field line is close to the result from periodic_field_line:
        print('R(end) - R0(k+1): ', R[-1] - R_end[j])
        print('Z(end) - Z0(k+1): ', Z[-1] - Z_end[j])

        tol = 2e-3
        if np.abs(R[-1] - R_end[j]) > tol or np.abs(Z[-1] - Z_end[j]) > tol:
            raise RuntimeError('Field line is not closed. Values of R0 and Z0 provided must have been incorrect')

        # Form the single-period tangent map:
        S = np.array([[soln.y[2, -1], soln.y[3, -1]],
                      [soln.y[4, -1], soln.y[5, -1]]])
        
        print('S: ', S)
        det = np.linalg.det(S)
        print('determinant of S: ', det)

        single_period_tangent_maps.append(S)

    full_orbit_tangent_maps = []
    eigvals = []
    eigvects = []
    iotas = []
    iotas_per_period = []
    residues = []
    W_eigvals = []
    W_eigvects = []
    epars = []
    eperps = []
    sigma = np.array([[0, 1], [-1, 0]])
    for j in range(periods):
        print('---- Period {} ----'.format(j))
        # Multiple all the single-period tangent maps together to get the full-orbit tangent map:
        M = single_period_tangent_maps[j]
        for k in range(1, periods):
            M = np.matmul(single_period_tangent_maps[np.mod(j + k, periods)], M)
            #M = np.matmul(M, single_period_tangent_maps[np.mod(j + k, periods)])

        print("M:", M)
        det = np.linalg.det(M)
        print('determinant of M: ', det)
        if np.abs(det - 1) > 1e-2:
            raise RuntimeError('Determinant of tangent map is not close to 1!')

        eigvals_j, eigvects_j = np.linalg.eig(M)
        print('eigvals: ', eigvals_j)
        print('eigvects: ', eigvects_j)

        iota_per_period = np.angle(eigvals_j[0]) / (2 * np.pi)
        iota = iota_per_period * field.nfp
        residue = 0.25 * (2 - np.trace(M))
        print('iota per period: {},  total iota: {},  residue: {}'.format(iota_per_period, iota, residue))

        tempmat = np.matmul(sigma, M)
        W = 0.5 * (tempmat + tempmat.transpose())
        W_eigvals_j, W_eigvects_j = np.linalg.eig(W)
        print('W_eigvals: ', W_eigvals_j)
        print('W_eigvects: ', W_eigvects_j)
        eig_ratio = np.max(np.abs(W_eigvals_j)) / np.min(np.abs(W_eigvals_j))
        print('Ratio of W eigvals: ', eig_ratio)
        # The larger eigenvalue corresponds to eperp
        if np.abs(W_eigvals_j[0]) > np.abs(W_eigvals_j[1]):
            eperp = W_eigvects_j[:,0]
            epar = W_eigvects_j[:,1]
        else:
            eperp = W_eigvects_j[:,1]
            epar = W_eigvects_j[:,0]

        """
        if eig_ratio < 5:
            print('W eigenvalue ratio is close to 1, so using alternative method to pick epar and eperp.')
            Ravg = np.mean(pfl.R_k)
            Zavg = np.mean(pfl.Z_k)
            print('Ravg: {},  Zavg: {}'.format(Ravg, Zavg))
            dR = pfl.R_k[j] - Ravg
            dZ = pfl.Z_k[j] - Zavg
            prod0 = np.abs(dR * W_eigvects_j[0, 0] + dZ * W_eigvects_j[1, 0])
            prod1 = np.abs(dR * W_eigvects_j[0, 1] + dZ * W_eigvects_j[1, 1])
            if prod0 > prod1:
                eperp = W_eigvects_j[:,0]
                epar = W_eigvects_j[:,1]
            else:
                eperp = W_eigvects_j[:,1]
                epar = W_eigvects_j[:,0]
        """
        
        # Alessandro uses the convention that epar * M * eperp is >0.
        sign_fac = np.dot(epar, np.dot(M, eperp))
        print('sign_fac: ', sign_fac)
        if sign_fac < 0:
            epar = -epar

        # Alessandro's second sign convention: epar(q) * S^q * eperp(0) is > 0
        # First form S^q
        if j == 1:
            Sq = single_period_tangent_maps[0]
        elif j > 1:
            Sq = np.matmul(single_period_tangent_maps[j - 1], Sq)
        # Now that we have Sq, flip the signs if needed:
        if j > 0 and np.dot(epar, np.dot(Sq, eperps[0])) < 0:
            epar = -epar
            eperp = -eperp
            
        full_orbit_tangent_maps.append(M)
        eigvals.append(eigvals_j)
        eigvects.append(eigvects_j)
        iotas_per_period.append(iota_per_period)
        iotas.append(iota)
        residues.append(residue)
        W_eigvals.append(W_eigvals_j)
        W_eigvects.append(W_eigvects_j)
        epars.append(epar)
        eperps.append(eperp)
    
    results = Struct()
    results.single_period_tangent_maps = single_period_tangent_maps
    results.full_orbit_tangent_maps = full_orbit_tangent_maps
    results.eigvals = eigvals
    results.eigvects = eigvects
    results.iota_per_period = iota_per_period
    results.iota = iota
    results.residue = residue
    results.epars = epars
    results.eperps = eperps
    
    return results

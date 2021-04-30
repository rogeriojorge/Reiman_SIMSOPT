#!/usr/bin/env python3
######################################################################
######## Reiman Model Optimization with Matt's Residue Script ########
################## Rogerio Jorge, April 30, 2021 #####################
######################################################################
from simsopt import LeastSquaresProblem,least_squares_serial_solve
from simsopt.geo.magneticfieldclasses import Reiman
from pyoculus.solvers  import FixedPoint, PoincarePlot
from simsopt._core.optimizable import Optimizable
from periodic_field_line import periodic_field_line
from tangent_map import tangent_map
from pyoculus.problems import CartesianBfield
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
# ###########################
# BioSavart class to pyOculus
class SimsgeoBiotSavart(CartesianBfield):
    def __init__(self, bs, R0, Z0, Nfp=1):
        super().__init__(R0, Z0, Nfp)
        self._bs = bs
    def B(self, xyz, args=None):
        point = np.array([xyz])
        self._bs.set_points(point)
        Bfield=self._bs.B()
        return Bfield[0]
    def dBdX(self, xyz, args=None):
        point = np.array([xyz])
        self._bs.set_points(point)
        dB=self._bs.dB_by_dX()
        return dB[0]
###########################################
# Field class for the Matt Landreman's residue script
class Field():
    def __init__(self, Bfield, NFP):
        self.nfp = NFP
        self.Bfield=Bfield
    def BR_Bphi_BZ(self,R,phi,Z):
        self.Bfield.set_points([[R*np.cos(phi),R*np.sin(phi),Z]])
        Bxyz  = self.Bfield.B()
        B_phi = cos(phi)*Bxyz[0,1] - sin(phi)*Bxyz[0,0]
        B_r   = cos(phi)*Bxyz[0,0] + sin(phi)*Bxyz[0,1]
        B_z   = Bxyz[0, 2]
        return B_r, B_phi, B_z
    def grad_B(self,R,phi,Z):
        self.Bfield.set_points([[R*np.cos(phi),R*np.sin(phi),Z]])
        dB       = self.Bfield.dB_by_dX()[0]
        drBr     = cos(phi)*( cos(phi)*dB[0,0]+sin(phi)*dB[0,1])+sin(phi)*( cos(phi)*dB[1,0]+sin(phi)*dB[1,1])
        dphiBr   = cos(phi)*(-sin(phi)*dB[0,0]+cos(phi)*dB[0,1])+sin(phi)*(-sin(phi)*dB[1,0]+cos(phi)*dB[1,1])
        dzBr     = cos(phi)*dB[0,2]+sin(phi)*dB[1,2]
        drBphi   = cos(phi)*( cos(phi)*dB[1,0]+sin(phi)*dB[1,1])-sin(phi)*( cos(phi)*dB[0,0]+sin(phi)*dB[0,1])
        dphiBphi = cos(phi)*(-sin(phi)*dB[1,0]+cos(phi)*dB[1,1])-sin(phi)*(-sin(phi)*dB[0,0]+cos(phi)*dB[0,1])
        dzBphi   = cos(phi)*dB[1,2]-sin(phi)*dB[0,2]
        drBz     = cos(phi)*dB[2,0]+sin(phi)*dB[2,1]
        dphiBz   =-sin(phi)*dB[2,0]+cos(phi)*dB[2,1]
        dzBz     = dB[2,2]
        return np.array([[drBr,dphiBr,dzBr],[drBphi,dphiBphi,dzBphi],[drBz,dphiBz,dzBz]])
    def d_RZ_d_phi(self, phi, RZ):
        R = RZ[0]
        Z = RZ[1]
        BR, Bphi, BZ = self.BR_Bphi_BZ(R, phi, Z)
        return [R * BR / Bphi, R * BZ / Bphi]
######################################################
# Optimizable class specific to the CaryHanson problem
class objBfieldResidue(Optimizable):
    def __init__(self):
        self.epsilon = 0.01
        self.NFP     = 1
        self.magnetic_axis_radius = 1.0
        self.Bfield  = Reiman(epsilonk=self.epsilon)
        self.sbsp    = SimsgeoBiotSavart(self.Bfield, self.magnetic_axis_radius, Z0=0, Nfp=self.NFP)
        self._set_names()
    def _set_names(self):
        self.names = ['epsilon']
    def get_dofs(self):
        return np.array([self.epsilon])
    def set_dofs(self, dofs):
        self.epsilon = dofs[0]
        self.Bfield  = Reiman(epsilonk=self.epsilon)
        self.sbsp    = SimsgeoBiotSavart(self.Bfield, self.magnetic_axis_radius, Z0=0, Nfp=self.NFP)
    def residue(self, R0=1.2088, periods=6, points=100):
        R0 = R0 - self.epsilon/500
        field = Field(self.Bfield,self.NFP)
        pfl   = periodic_field_line(field, points, R0=R0, periods=periods)
        R_k   = pfl.R_k
        Z_k   = pfl.Z_k
        tm    = tangent_map(field, pfl)
        return [tm.residue, R_k, Z_k, periods, R0]
    def residuePy(self, guess=1.2088, qq=6, sbegin=1.15, send=1.25):
        fp       = FixedPoint(self.sbsp, {"Z":-2.8e-6, "niter":10000, "nrestart":3})
        output   = fp.compute(guess=guess, pp=self.NFP, qq=qq, sbegin=sbegin, send=send)
        residueN = output.GreenesResidue
        R_k      = output.x
        Z_k      = output.z
        return [residueN, R_k, Z_k, qq, guess]
    def residue1(self):   return self.residue()[0]
    def residue1Py(self): return self.residuePy()[0]
    def poincare(self, Rbegin=1.19, Rend=1.23, nPpts=350, nPtrj=8):
        params = dict()
        params["Rbegin"]     = Rbegin
        params["Rend"]       = Rend
        params["nPpts"]      = nPpts
        params["nPtrj"]      = nPtrj
        self.p               = PoincarePlot(self.sbsp, params)
        self.poincare_output = self.p.compute()
        self.iota            = self.p.compute_iota()
        return self.p
# ############################################
if __name__ == "__main__":
    ## Start optimizable class
    obj             = objBfieldResidue()
    initialDofs     = obj.get_dofs()
    initialPoincare = 1
    ## Create initial Poincare Plot
    solInitial = [obj.residue()]
    if initialPoincare == 1:
        p = obj.poincare(nPpts=350,nPtrj=8,Rbegin=1.19,Rend=1.23); p.plot(s=1.5)
        [plt.scatter(fSol[1], fSol[2], s=35, marker="x", label=f"Periods = {fSol[3]:.0f}, Residue = {fSol[0]:.4f}") for fSol in solInitial]
        plt.xlim([0.7 , 1.3]); plt.ylim([-0.25, 0.25]); plt.legend(); plt.tight_layout()
        plt.savefig('Results/ReimanInitialPoincare.png', dpi=500)
        plt.savefig('Results/ReimanInitialPoincare.pdf')
        p.plot_iota(); plt.tight_layout()
        plt.savefig('Results/ReimanInitialIota.png')
    # exit()
    ## Optimization
    prob = LeastSquaresProblem([(obj.residue1,0,1)])
    # Set degrees of freedom for the optimization
    obj.all_fixed()
    obj.set_fixed('epsilon', False)
    # Run optimization problem
    # nIterations = 10
    print('Starting optimization...')
    least_squares_serial_solve(prob, bounds=[-0.037,0.021], xtol=1e-12, ftol=1e-12, gtol=1e-12)#, method='lm', max_nfev=nIterations)
    print('Optimization finished...')

    ## Create final Poincare Plot
    solFinal = [obj.residue()]
    p = obj.poincare(nPpts=350,nPtrj=8,Rbegin=1.19,Rend=1.23); p.plot(s=1.5)
    [plt.scatter(fSol[1], fSol[2], s=35, marker="x", label=f"Periods = {fSol[3]:.0f}, Residue = {fSol[0]:.4f}") for fSol in solFinal]
    plt.xlim([0.7 , 1.3]); plt.ylim([-0.25, 0.25]); plt.legend(); plt.tight_layout()
    plt.savefig('Results/ReimanFinalPoincare.png', dpi=500)
    plt.savefig('Results/ReimanFinalPoincare.pdf')
    p.plot_iota(); plt.tight_layout()
    plt.savefig('Results/ReimanFinalIota.png')

    ## Print final results
    print('Initial degrees of freedom =',initialDofs)
    print('Final   degrees of freedom =',obj.get_dofs())
    print('Initial Residue = ',solInitial[0][0])
    print('Final   Residue = ',solFinal[0][0])

    # Show plots
    # plt.show()
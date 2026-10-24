#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Egor Trushin
#

'''
Exact-exchange optimized effective potential (EXX-OEP) method for
closed-shell systems.

The local exchange potential is expanded in an auxiliary Gaussian basis
(the OEP basis), v_x(r) = sum_P v_P int chi_P(r') / |r - r'| dr', and the
OEP equation is solved in a preprocessed OEP basis that removes the parts
of the basis the orbital basis cannot represent.

Section numbers in the docstrings below refer to

    E. Trushin, A. Görling, J. Chem. Phys. 155, 054109 (2021)
    https://doi.org/10.1063/5.0056431
'''

from functools import reduce

import numpy
import scipy.linalg

from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import df
from pyscf.dft import gen_grid
from pyscf.scf import hf
from pyscf.scf import diis as scf_diis


def kernel(oep, mo_energy=None, mo_coeff=None, verbose=None):
    '''Self-consistent OEP iterations.

    The local potential of each cycle is provided by
    oep.get_local_potential, so the same driver serves the restricted and
    unrestricted EXX-OEP and RPA-OEP classes. As in the SCF kernel of PySCF,
    the iterations are converged when both the change of the total energy is
    below oep.conv_tol and the norm of the orbital gradient is below
    oep.conv_tol_grad (default sqrt(conv_tol)).

    Returns:
        converged, e_tot, mo_energy, mo_coeff, pot
        where pot is the dict of potentials (OEP basis coefficients) of the
        last cycle, see get_local_potential
    '''
    log = logger.new_logger(oep, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if mo_energy is None:
        mo_energy = oep.mo_energy
    if mo_coeff is None:
        mo_coeff = oep.mo_coeff

    conv_tol_grad = oep.conv_tol_grad
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(oep.conv_tol)

    h1e = oep._scf.get_hcore()
    s1e = oep._scf.get_ovlp()
    e_tot, vj, vxnl = oep.get_energies_and_potentials(mo_coeff)
    mix_fock = oep.get_fock_mixer()

    converged = False
    e_last = None
    pot = {}
    for cycle in range(oep.max_cycle):
        vj, v_ao, pot = oep.get_local_potential(mo_energy, mo_coeff, vj, vxnl)
        fock = h1e + vj + v_ao
        # Orbital gradient of the orbitals which entered this cycle
        norm_gorb = numpy.linalg.norm(oep.get_grad(mo_coeff, oep.mo_occ, fock))
        fock = mix_fock(fock, s1e, mo_coeff)
        mo_energy, mo_coeff = oep.eig(fock, s1e)
        e_tot, vj, vxnl = oep.get_energies_and_potentials(mo_coeff)
        # Correlation energy of the orbitals which entered this cycle
        e_tot += pot.get('e_corr', 0)

        if e_last is None:
            log.info('cycle= %d E= %.15g  |g|= %4.3g', cycle + 1, e_tot, norm_gorb)
        else:
            de = e_tot - e_last
            log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g', cycle + 1, e_tot, de, norm_gorb)
            if abs(de) < oep.conv_tol and norm_gorb < conv_tol_grad:
                converged = True
                break
        e_last = e_tot

    log.timer(oep.__class__.__name__, *cput0)
    return converged, e_tot, mo_energy, mo_coeff, pot


def eval_potential(auxmol, coords, v_oep, blksize=2000):
    r'''Evaluates a potential given by OEP basis coefficients on a grid,

        v(r) = \sum_P v_P \int \chi_P(r') / |r - r'| dr'

    Args:
        auxmol : Mole object of the OEP basis
        coords : (ngrids, 3) ndarray of coordinates in Bohr
        v_oep : (naux,) or (nset, naux) ndarray of expansion coefficients

    Returns:
        ndarray of shape (ngrids,) or (nset, ngrids)
    '''
    coords = numpy.asarray(coords, dtype=float).reshape(-1, 3)
    v_oep = numpy.asarray(v_oep)
    out = numpy.empty(v_oep.shape[:-1] + (len(coords),))
    for p0, p1 in lib.prange(0, len(coords), blksize):
        # Point charges of unit charge, see gto.fakemol_for_charges
        fakemol = gto.fakemol_for_charges(coords[p0:p1], expnt=1e16)
        vcoul = gto.mole.intor_cross('int2c2e', fakemol, auxmol)
        out[..., p0:p1] = numpy.dot(v_oep, vcoul.T)
    return out


class REXXOEP(lib.StreamObject):
    '''Exact-exchange optimized effective potential method for closed-shell
    systems.

    Args:
        mf : RHF or RKS object
            Converged mean-field calculation which provides the initial
            orbitals and the Coulomb and exchange matrices (including
            density fitting, if mf uses it).
        oep_basis : str or dict
            Auxiliary basis in which the local exchange potential is expanded.

    Attributes:
        use_homo_condition : bool
            Whether to enforce the HOMO condition (Appendix C). Default is False.
        vh_via_oep : bool
            Whether to build the Hartree potential via the OEP basis.
            Default is False.
        space_sym : bool
            Whether to keep only the totally symmetric part of the OEP basis
            (for atoms). Default is False.
        max_cycle : int
            Maximum number of self-consistent iterations. Default is 50.
        conv_tol : float
            Convergence threshold on the total energy. Default is 1e-8.
        conv_tol_grad : float
            Convergence threshold on the norm of the orbital gradient.
            Default is None, which means sqrt(conv_tol).
        thr_fai_oep : float
            Threshold T_ai for the occupied-virtual coupling filter of the
            OEP basis (Section IIA5). Default is 5e-2.
        linear_mixing : float
            If > 0, use linear Fock-matrix mixing with this coefficient
            instead of DIIS. Default is -1.
        grids : Grids object
            Integration grid built on the OEP basis molecule, used for the
            charge condition of the potential.

    Saved results:
        converged : bool
        e_tot : float
            Total energy.
        mo_energy, mo_coeff, mo_occ : ndarray
            Kohn-Sham orbitals of the local exchange potential.
        vref_oep : ndarray
            OEP basis coefficients of the reference (Fermi-Amaldi type)
            exchange potential.
        vrest_oep : ndarray
            OEP basis coefficients of the remaining exchange potential.
            The exchange potential is vref_oep + vrest_oep, see eval_vx.
        scf_summary : dict
            Energy components.

    Examples:

    >>> mol = gto.M(atom='Ne', basis='cc-pvtz')
    >>> mf = scf.RHF(mol).density_fit().run()
    >>> mf_oep = oep.EXXOEP(mf, 'aug-cc-pvdz-ri').run()
    '''

    max_cycle = 50
    conv_tol = 1e-8
    conv_tol_grad = None
    thr_fai_oep = 5e-2
    linear_mixing = -1.0

    _keys = {
        'mol', 'oep_basis', 'use_homo_condition', 'vh_via_oep', 'space_sym',
        'max_cycle', 'conv_tol', 'conv_tol_grad', 'thr_fai_oep', 'linear_mixing',
        'auxmol', 'grids', 'y', 'WII', 'WII_inv', 'yII', 'charge_norm',
        'converged', 'e_tot', 'mo_energy', 'mo_coeff', 'mo_occ',
        'vref_oep', 'vrest_oep', 'scf_summary',
    }

    def __init__(self, mf, oep_basis, use_homo_condition=False,
                 vh_via_oep=False, space_sym=False):
        if mf.mo_coeff is None:
            raise RuntimeError('The mean-field object must be run first')
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory

        self.oep_basis = oep_basis
        self.use_homo_condition = use_homo_condition
        self.vh_via_oep = vh_via_oep
        self.space_sym = space_sym

        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self._check_mo_occ()

        self.auxmol = None
        self.grids = None
        self.y = None
        self.WII = None
        self.WII_inv = None
        self.yII = None
        self.charge_norm = None
        self._int3c = None
        self._x_lindep = None

        self.converged = False
        self.e_tot = None
        self.vref_oep = None
        self.vrest_oep = None
        self.scf_summary = {}

    def _check_mo_occ(self):
        mo_occ = numpy.asarray(self.mo_occ)
        if mo_occ.ndim != 1:
            raise TypeError('%s requires RHF or RKS orbitals' % self.__class__.__name__)
        nocc = self.nocc
        if not (numpy.all(mo_occ[:nocc] == 2) and numpy.all(mo_occ[nocc:] == 0)):
            raise NotImplementedError('Only aufbau occupations 2,...,2,0,...,0 are supported')

    @property
    def nocc(self):
        return int(numpy.count_nonzero(self.mo_occ > 0))

    @property
    def nmo(self):
        return self.mo_coeff.shape[-1]

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('OEP basis = %s', self.oep_basis)
        log.info('nocc = %s  nmo = %d', self.nocc, self.nmo)
        log.info('use_homo_condition = %s', self.use_homo_condition)
        log.info('vh_via_oep = %s', self.vh_via_oep)
        log.info('space_sym = %s', self.space_sym)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('thr_fai_oep = %g', self.thr_fai_oep)
        if self.linear_mixing > 0:
            log.info('linear mixing = %g', self.linear_mixing)
        else:
            log.info('DIIS')
        return self

    def build(self):
        '''Builds the OEP basis, three-center integrals, integration grid and
        the charge-condition preprocessing of the OEP basis.'''
        mol = self.mol
        self.auxmol = df.addons.make_auxmol(mol, self.oep_basis)
        # (nao, nao, naux)
        self._int3c = df.incore.aux_e2(mol, self.auxmol, intor='int3c2e')

        # PySCF may drop linearly dependent AOs during the orthogonalization.
        # Keep the Fock diagonalization in the same nmo-dimensional subspace.
        s = self._scf.get_ovlp()
        e, v = scipy.linalg.eigh(s)
        nmo = self.nmo
        self._x_lindep = v[:, -nmo:] / numpy.sqrt(e[-nmo:])

        if self.grids is None:
            self.grids = gen_grid.Grids(self.auxmol)
        if self.grids.coords is None:
            self.grids.build()
        ao = self.auxmol.eval_gto('GTOval', self.grids.coords)
        self.y = ao.T.dot(self.grids.weights)
        self.WII, self.WII_inv = self.get_WII()
        self.yII = self.WII.T.dot(self.y)
        self.charge_norm = numpy.dot(self.yII, self.yII)
        return self

    def eig(self, h, s):
        '''Diagonalizes the Fock matrix in the linearly independent subspace
        spanned by the initial orbitals.'''
        x = self._x_lindep
        e, c = scipy.linalg.eigh(reduce(numpy.dot, (x.T, h, x)))
        return e, x.dot(c)

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return hf.make_rdm1(mo_coeff, mo_occ)

    def get_grad(self, mo_coeff, mo_occ, fock):
        '''Orbital gradient, the occupied-virtual block of the Fock matrix in
        the basis of mo_coeff, as pyscf.scf.hf.get_grad.'''
        return hf.get_grad(mo_coeff, mo_occ, fock)

    def get_energies_and_potentials(self, mo_coeff):
        '''Hartree-Fock energy functional of the given orbitals.

        Returns:
            e_tot, Coulomb matrix, non-local exchange potential matrix
        '''
        mf = self._scf
        dm = self.make_rdm1(mo_coeff)
        h1e = mf.get_hcore()
        vj, vk = mf.get_jk(self.mol, dm)
        e1 = numpy.einsum('ij,ji->', h1e, dm)
        e_coul = numpy.einsum('ij,ji->', vj, dm) * .5
        e_x = numpy.einsum('ij,ji->', vk, dm) * -.25
        e_nuc = mf.energy_nuc()
        self.scf_summary = {'e1': e1, 'coul': e_coul, 'exc': e_x, 'nuc': e_nuc}
        return e1 + e_coul + e_x + e_nuc, vj, vk * -.5

    def ao2mo(self, mo_coeff, nocc, int3c=None):
        '''Three-center integrals (P|pi) with all MOs p and the first nocc
        MOs i, in the OEP basis or in the auxiliary basis of int3c.

        Returns:
            ndarray of shape (naux, nmo, nocc)
        '''
        if int3c is None:
            int3c = self._int3c
        tmp = lib.einsum('uvP,vi->Pui', int3c, mo_coeff[:, :nocc])
        return lib.einsum('Pui,up->Ppi', tmp, mo_coeff)

    def oep2ao(self, v_oep):
        '''AO matrix of a potential given by OEP basis coefficients.'''
        return numpy.dot(self._int3c, v_oep)

    def get_u(self, ints_3c, nocc):
        '''u_P = sum_i (P|ii) over occupied orbitals of one spin.'''
        return numpy.einsum('Pii->P', ints_3c[:, :nocc, :nocc])

    def get_WII(self):
        '''First two steps of the OEP basis preprocessing (Sections IIB1, IIB2).

        Returns:
            WII and its left inverse WII_inv (WII_inv.dot(WII) = 1, Section IIC2)
        '''
        smat = self.auxmol.intor('int2c2e')
        diag = 1. / numpy.sqrt(numpy.diag(smat))
        smat = diag[:, None] * smat * diag
        eigs, evecs = scipy.linalg.eigh(smat)
        if self.space_sym:
            # Keep only the eigenvectors which overlap with the basis
            # functions that carry charge (totally symmetric for atoms)
            symvec = self.y.copy()
            symvec[abs(symvec) > 1e-10] = 1.
            eigs[abs(evecs.T.dot(symvec)) < 1e-10] = 0.
            mask = abs(eigs) > 1e-10
            eigs, evecs = eigs[mask], evecs[:, mask]
        smh = evecs / numpy.sqrt(eigs)
        sh = evecs.T * numpy.sqrt(eigs)[:, None]
        return diag[:, None] * smh, sh / diag

    def get_W3_charge(self):
        '''Projection on the charge-free subspace (Section IIB3).'''
        return scipy.linalg.null_space(self.yII[None, :])

    def get_W3_charge_and_homo(self, zII):
        '''Projection on the subspace free of charge and of HOMO
        contribution (Appendix C).'''
        return scipy.linalg.null_space(numpy.vstack((self.yII, zII)))

    def get_W(self, W3, ints_3c, nocc, thr_fai=None):
        '''Last step of the OEP basis preprocessing: removes the combinations
        of OEP basis functions which do not couple occupied and virtual
        orbitals (Section IIB5).

        Returns:
            Final transformation matrix of shape (naux, nW)
        '''
        if thr_fai is None:
            thr_fai = self.thr_fai_oep
        trans = self.WII.dot(W3)
        amat = self._get_amat(trans, ints_3c, nocc)
        return trans.dot(self._coupling_vectors(amat, thr_fai))

    def _get_amat(self, trans, ints_3c, nocc):
        naux = ints_3c.shape[0]
        dmat = trans.T.dot(ints_3c[:, nocc:].reshape(naux, -1))
        return dmat.dot(dmat.T)

    def _coupling_vectors(self, amat, thr_fai):
        '''Eigenvectors of the coupling matrix with eigenvalues >= thr_fai.'''
        eigs, evecs = scipy.linalg.eigh(amat)
        nsing = numpy.count_nonzero(eigs < thr_fai)
        return evecs[:, nsing:]

    def get_v_ref(self, u):
        '''Fermi-Amaldi reference potential (Section IIB4), normalized to
        a charge of -1.'''
        vc = self.WII.dot(self.WII.T.dot(u))
        return -vc / numpy.dot(self.y, vc)

    def get_z_and_zII(self, ints_3c, mo_energy, nocc):
        '''HOMO density coefficients, averaged over degenerate HOMOs
        (Appendix C).'''
        ndeg = 1
        for i in range(1, nocc - 1):
            if abs(mo_energy[nocc - 1 - i] - mo_energy[nocc - 1]) < 1e-12:
                ndeg += 1
            else:
                break
        idx = numpy.arange(nocc - ndeg, nocc)
        z = ints_3c[:, idx, idx].sum(axis=1) / ndeg
        return z, self.WII.T.dot(z)

    def get_homo_target(self, mo_coeff, nocc, vxnl, vj=None, spin=None):
        '''The value <HOMO|v_ref|HOMO> required by the HOMO condition.

        For EXX-OEP the expectation value of the non-local exchange potential
        vxnl. vj and the spin index (unrestricted classes) are available for
        other conditions in subclasses.
        '''
        c_homo = mo_coeff[:, nocc - 1]
        return reduce(numpy.dot, (c_homo, vxnl, c_homo))

    def get_v_ref_w_homo(self, u, zII, homo_target):
        '''Reference potential with HOMO condition (Appendix C): normalized to
        a charge of -1 and with <HOMO|v_ref|HOMO> = homo_target.'''
        vcII = self.WII.T.dot(u)
        aux_x = numpy.array([[self.charge_norm, numpy.dot(self.yII, vcII)],
                             [numpy.dot(zII, self.yII), numpy.dot(zII, vcII)]])
        aux_y = numpy.array([-1., homo_target])
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        return self.WII.dot(aux_sol[0] * self.yII + aux_sol[1] * vcII)

    def get_vh_via_oep(self, u, nelectron):
        '''Hartree potential matrix from the density fitted in the OEP basis,
        normalized to nelectron electrons.'''
        vc = self.WII.dot(self.WII.T.dot(u))
        vc *= nelectron / numpy.dot(self.y, vc)
        return self.oep2ao(vc)

    def get_e_ov(self, mo_energy, nocc):
        '''1 / (e_i - e_a) of shape (nvir, nocc), with near-zero energy
        differences clamped to 1e-6.'''
        e_ov = mo_energy[None, :nocc] - mo_energy[nocc:, None]
        e_ov = numpy.where(abs(e_ov) > 1e-6, e_ov, numpy.sign(e_ov + 1e-12) * 1e-6)
        return 1. / e_ov

    def get_X0(self, ints_3c, mo_energy, nocc, W):
        '''Kohn-Sham response matrix in the transformed OEP basis.'''
        naux = ints_3c.shape[0]
        e_ov = self.get_e_ov(mo_energy, nocc)
        b = W.T.dot(ints_3c[:, nocc:].reshape(naux, -1))
        return (b * e_ov.ravel()).dot(b.T)

    def get_rhs(self, vref_ao, vxnl, ints_3c, mo_coeff, mo_energy, nocc, W):
        '''Right-hand side of the EXX-OEP equation in the transformed OEP
        basis.'''
        naux = ints_3c.shape[0]
        vxmo = reduce(numpy.dot, (mo_coeff[:, nocc:].T, vxnl - vref_ao, mo_coeff[:, :nocc]))
        vxmo *= self.get_e_ov(mo_energy, nocc)
        return W.T.dot(ints_3c[:, nocc:].reshape(naux, -1).dot(vxmo.ravel()))

    def check_potentials(self, vrest_oep, vref_oep, vrest_ao, vref_ao,
                         mo_coeff, nocc, z=None, homo_target=None, tol=1e-12):
        '''Checks the charge and HOMO conditions of the potentials.'''
        log = logger.new_logger(self)
        if abs(numpy.dot(self.y, vrest_oep)) > tol:
            log.warn('y*vrest_oep = %g', numpy.dot(self.y, vrest_oep))
        if abs(numpy.dot(self.y, vref_oep) + 1.) > tol:
            log.warn('y*vref_oep = %g', numpy.dot(self.y, vref_oep))
        if z is not None:
            c_homo = mo_coeff[:, nocc - 1]
            if abs(numpy.dot(z, vrest_oep)) > tol:
                log.warn('z*vrest_oep = %g', numpy.dot(z, vrest_oep))
            vrest_homo = reduce(numpy.dot, (c_homo, vrest_ao, c_homo))
            if abs(vrest_homo) > tol:
                log.warn('<HOMO|vrest|HOMO> = %g', vrest_homo)
            vref_homo = reduce(numpy.dot, (c_homo, vref_ao, c_homo))
            if abs(vref_homo - homo_target) > tol:
                log.warn('<HOMO|vref|HOMO> = %.12g, required %.12g',
                         vref_homo, homo_target)

    def get_local_potential(self, mo_energy, mo_coeff, vj, vxnl):
        '''Solves the EXX-OEP equation for the given orbitals.

        Args:
            vj : Coulomb matrix of the orbitals
            vxnl : non-local exchange potential matrix of the orbitals

        Returns:
            Hartree potential matrix, local potential matrix (AO basis) and
            a dict of the OEP basis coefficients of the potentials
        '''
        nocc = self.nocc
        ints_3c = self.ao2mo(mo_coeff, nocc)
        u = 2 * self.get_u(ints_3c, nocc)

        if self.vh_via_oep or self.space_sym:
            vj = self.get_vh_via_oep(u, 2 * nocc)

        z = homo_target = None
        if self.use_homo_condition:
            z, zII = self.get_z_and_zII(ints_3c, mo_energy, nocc)
            homo_target = self.get_homo_target(mo_coeff, nocc, vxnl, vj)
            vref_oep = self.get_v_ref_w_homo(u, zII, homo_target)
            W3 = self.get_W3_charge_and_homo(zII)
        else:
            vref_oep = self.get_v_ref(u)
            W3 = self.get_W3_charge()

        W = self.get_W(W3, ints_3c, nocc)
        X0 = self.get_X0(ints_3c, mo_energy, nocc, W)
        vref_ao = self.oep2ao(vref_oep)
        rhs = self.get_rhs(vref_ao, vxnl, ints_3c, mo_coeff, mo_energy, nocc, W)
        vrest_oep = W.dot(scipy.linalg.solve(X0, rhs))
        vrest_ao = self.oep2ao(vrest_oep)

        self.check_potentials(vrest_oep, vref_oep, vrest_ao, vref_ao,
                              mo_coeff, nocc, z, homo_target)
        return vj, vref_ao + vrest_ao, {'vref_oep': vref_oep, 'vrest_oep': vrest_oep}

    def get_fock_mixer(self):
        '''Returns a function mix(fock, s1e, mo_coeff) which applies linear
        mixing (linear_mixing > 0) or DIIS to the Fock matrix of each cycle.'''
        if self.linear_mixing > 0:
            alpha = self.linear_mixing
            fock_last = []

            def mix(fock, s1e, mo_coeff):
                if fock_last:
                    fock = (1 - alpha) * fock + alpha * fock_last[0]
                fock_last[:] = [fock]
                return fock
            return mix
        return self._diis_mixer()

    def _diis_mixer(self):
        # CDIIS of PySCF: commutator SDF - FDS as error vector, in the
        # orthonormal basis in which the Fock matrix is diagonalized
        adiis = scf_diis.CDIIS(self, Corth=self._x_lindep)
        return lambda fock, s1e, mo_coeff: adiis.update(s1e, self.make_rdm1(mo_coeff), fock)

    def kernel(self, mo_energy=None, mo_coeff=None):
        '''Runs the self-consistent OEP calculation, starting from the
        orbitals of the mean-field object (or of a previous run).

        Returns:
            Total energy
        '''
        self.check_sanity()
        self.dump_flags()
        self.build()
        (self.converged, self.e_tot, self.mo_energy, self.mo_coeff,
         pot) = kernel(self, mo_energy, mo_coeff)
        self._save_potentials(pot)
        self._finalize()
        return self.e_tot

    def _save_potentials(self, pot):
        self.vref_oep = pot.get('vref_oep')
        self.vrest_oep = pot.get('vrest_oep')

    def _finalize(self):
        log = logger.new_logger(self)
        name = self.__class__.__name__
        if self.converged:
            log.note('converged %s energy = %.15g', name, self.e_tot)
        else:
            log.note('%s not converged, energy = %.15g', name, self.e_tot)
        self._log_homo_lumo(log)

    def _log_homo_lumo(self, log):
        nocc = self.nocc
        homo = self.mo_energy[nocc - 1]
        if nocc < self.nmo:
            lumo = self.mo_energy[nocc]
            log.info('HOMO = %.15g  LUMO = %.15g  gap = %.15g', homo, lumo, lumo - homo)
        else:
            log.info('HOMO = %.15g', homo)

    def eval_potential(self, coords, v_oep):
        '''Potential given by OEP basis coefficients on the grid coords (Bohr).'''
        return eval_potential(self.auxmol, coords, v_oep)

    def eval_vx(self, coords):
        '''Local exchange potential vref + vrest on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vref_oep + self.vrest_oep)

EXXOEP = REXXOEP

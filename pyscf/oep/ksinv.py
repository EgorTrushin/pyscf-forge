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
Response-function Kohn-Sham inversion for closed-shell systems.

Determines the local Kohn-Sham potential which reproduces a given reference
density, e.g. from a correlated calculation. The potential is expanded in the
OEP basis and preprocessed as in the EXX-OEP method (pyscf.oep.exxoep).

References:

    J. Erhard, E. Trushin, A. Görling, J. Chem. Phys. 156, 204124 (2022)
    https://doi.org/10.1063/5.0087356

    J. Erhard, E. Trushin, A. Görling, J. Chem. Phys. 162, 034116 (2025)
    https://doi.org/10.1063/5.0239422
'''

import numpy
import scipy.linalg

from pyscf.lib import logger
from pyscf.dft import gen_grid, numint
from pyscf.oep import exxoep


def get_mixing(cycle, delta_rhs, mixing_a, mixing_i, delta_rhs_i):
    '''Mixing parameter for the update of the potential
    (Appendix A of J. Chem. Phys. 156, 204124 (2022)).

    Constant mixing_a in the first mixing_i cycles, afterwards increasing to 1
    as the norm of the right-hand side delta_rhs decreases relative to its
    value delta_rhs_i at cycle mixing_i.

    Returns:
        mixing, delta_rhs_i
    '''
    if cycle < mixing_i:
        return mixing_a, delta_rhs_i
    if delta_rhs_i is None:
        delta_rhs_i = delta_rhs
    mixing = 2. / (1. + (2. / mixing_a - 1.) ** (delta_rhs / delta_rhs_i))
    return mixing, delta_rhs_i


def kernel(inv, verbose=None):
    '''Kohn-Sham inversion iterations for closed-shell systems.

    Returns:
        converged, mo_energy, mo_coeff, vref_oep, vrest_oep, vx_oep, summary
    '''
    log = logger.new_logger(inv, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    mf = inv._scf
    mol = inv.mol
    nocc = inv.nocc
    dm_ref = inv.dm_ref

    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()
    vj_ref = vj_ref_exact = mf.get_j(mol, dm_ref)
    u_ref = inv.get_u_ao(dm_ref)

    # Initial guess: Hartree potential of the reference density plus the
    # Fermi-Amaldi reference potential
    vref_oep = inv.get_v_ref(u_ref)
    vref_ao = inv.oep2ao(vref_oep)
    mo_energy, mo_coeff = inv.eig(h1e + vj_ref + vref_ao, s1e)
    if inv.vh_via_oep or inv.space_sym:
        vj_ref = inv.get_vh_via_oep(u_ref, 2 * nocc)

    converged = False
    vrest_oep = None
    delta_rhs_i = None
    for cycle in range(inv.max_cycle):
        dm = inv.make_rdm1(mo_coeff)
        ints_3c = inv.ao2mo(mo_coeff, nocc)

        z = homo_target = None
        if inv.ip is not None:
            z, zII = inv.get_z_and_zII(ints_3c, mo_energy, nocc)
            homo_target = inv.get_homo_target_ip(mo_coeff[:, nocc - 1], h1e + vj_ref, inv.ip)
            vref_oep = inv.get_v_ref_w_homo(u_ref, zII, homo_target)
            vref_ao = inv.oep2ao(vref_oep)
            W3 = inv.get_W3_charge_and_homo(zII)
        else:
            W3 = inv.get_W3_charge()
        W, W_inv = inv.get_W_and_inv(W3, ints_3c, nocc)

        X0x = inv.get_X0(ints_3c, mo_energy, nocc, W)
        rhs = W.T.dot(u_ref - inv.get_u_ao(dm))
        delta_rhs = numpy.linalg.norm(rhs)
        converged = delta_rhs < inv.conv_tol

        mixing, delta_rhs_i = get_mixing(cycle, delta_rhs, inv.mixing_a, inv.mixing_i,
                                         delta_rhs_i)
        dv = mixing * W.dot(scipy.linalg.solve(4 * X0x, rhs))
        vrest_oep = dv if vrest_oep is None else vrest_oep + dv
        # Remove components outside the current transformed OEP basis
        vrest_oep = W.dot(W_inv.dot(vrest_oep))
        vrest_ao = inv.oep2ao(vrest_oep)

        inv.check_potentials(vrest_oep, vref_oep, vrest_ao, vref_ao,
                             mo_coeff, nocc, z, homo_target)

        if converged or cycle == inv.max_cycle - 1:
            # Exact-exchange potential for the current orbitals
            vk = mf.get_k(mol, dm)
            rhs_x = inv.get_rhs(vref_ao, vk * -.5, ints_3c, mo_coeff, mo_energy, nocc, W)
            vx_oep = W.dot(scipy.linalg.solve(X0x, rhs_x)) + vref_oep

        mo_energy, mo_coeff = inv.eig(h1e + vj_ref + vref_ao + vrest_ao, s1e)
        log.info('cycle= %d  |rhs|= %.3e  mixing= %.3e', cycle + 1, delta_rhs, mixing)
        if converged:
            break

    dm = inv.make_rdm1(mo_coeff)
    vj, vk = mf.get_jk(mol, dm)
    e_x = numpy.einsum('ij,ji->', vk, dm) * -.25
    sum_eps = 2 * mo_energy[:nocc].sum()
    ref_vxc = numpy.einsum('ij,ji->', dm_ref, vref_ao + vrest_ao)
    summary = inv.energy_summary(dm, dm_ref, e_x, vj, vj_ref_exact, vj_ref, sum_eps, ref_vxc)

    log.timer('KS inversion', *cput0)
    return converged, mo_energy, mo_coeff, vref_oep, vrest_oep, vx_oep, summary


class KSINVBase:
    '''Settings and analysis shared by the restricted and unrestricted
    Kohn-Sham inversion classes.'''

    max_cycle = 100
    conv_tol = 1e-10
    mixing_a = 0.1
    mixing_i = 3

    _keys = {
        'dm_ref', 'e_ref', 'ip', 'mixing_a', 'mixing_i',
        'vx_oep', 'vc_oep', 'e_c', 'density_error',
    }

    def _init_ksinv(self, dm_ref, e_ref, ip):
        self.dm_ref = numpy.asarray(dm_ref)
        self.e_ref = e_ref
        self.ip = ip
        self.vx_oep = None
        self.vc_oep = None
        self.e_c = None
        self.density_error = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('OEP basis = %s', self.oep_basis)
        log.info('nocc = %s  nmo = %d', self.nocc, self.nmo)
        log.info('HOMO condition with ip = %s', self.ip)
        log.info('vh_via_oep = %s', self.vh_via_oep)
        log.info('space_sym = %s', self.space_sym)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('thr_fai_oep = %g', self.thr_fai_oep)
        log.info('mixing_a = %g  mixing_i = %d', self.mixing_a, self.mixing_i)
        return self

    def get_u_ao(self, dm):
        '''u_P = sum_uv D_uv (P|uv) for the AO density matrix dm.'''
        naux = self._int3c.shape[-1]
        return numpy.dot(numpy.asarray(dm).ravel(), self._int3c.reshape(-1, naux))

    def get_W_and_inv(self, W3, ints_3c, nocc, thr_fai=None):
        '''Final transformation matrix W of the OEP basis and its left inverse
        (Section IIC2 of J. Chem. Phys. 155, 054109 (2021)).

        For spin-symmetrized potentials, ints_3c and nocc are pairs (alpha,
        beta) and the coupling matrix is averaged over spins.
        '''
        if thr_fai is None:
            thr_fai = self.thr_fai_oep
        trans = self.WII.dot(W3)
        if isinstance(nocc, (int, numpy.integer)):
            amat = self._get_amat(trans, ints_3c, nocc)
        else:
            amat = .5 * (self._get_amat(trans, ints_3c[0], nocc[0]) +
                         self._get_amat(trans, ints_3c[1], nocc[1]))
        e = self._coupling_vectors(amat, thr_fai)
        return trans.dot(e), e.T.dot(W3.T.dot(self.WII_inv))

    def get_homo_target_ip(self, c_homo, h1e_vj, ip):
        '''<HOMO|v_ref|HOMO> for which the HOMO energy equals -ip.'''
        return -ip - c_homo.dot(h1e_vj).dot(c_homo)

    def energy_summary(self, dm, dm_ref, e_x, vj, vj_ref_exact, vj_ref, sum_eps, ref_vxc):
        '''Energy contributions of the Kohn-Sham and reference densities.

        Args:
            dm, dm_ref : Kohn-Sham and reference density matrices
                (total, or alpha and beta)
            e_x : exchange energy of the Kohn-Sham orbitals
            vj : Coulomb matrix of the Kohn-Sham density
            vj_ref_exact : Coulomb matrix of the reference density
            vj_ref : Hartree potential matrix used in the inversion
            sum_eps : sum of occupied orbital energies
            ref_vxc : int rho_ref v_xc

        If e_ref is given, also the correlation energy e_c = t_c + v_c with
        its kinetic and potential parts, and the total energy.
        '''
        mf = self._scf
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
            dm_ref = dm_ref[0] + dm_ref[1]
        t = self.mol.intor_symmetric('int1e_kin')
        vext = mf.get_hcore() - t
        e_kin = numpy.einsum('ij,ji->', t, dm)
        e_kin_ref = numpy.einsum('ij,ji->', t, dm_ref)
        e_ext = numpy.einsum('ij,ji->', vext, dm)
        e_ext_ref = numpy.einsum('ij,ji->', vext, dm_ref)
        e_coul = numpy.einsum('ij,ji->', vj, dm) * .5
        e_coul_ref = numpy.einsum('ij,ji->', vj_ref_exact, dm_ref) * .5
        e_nuc = mf.energy_nuc()
        # Lieb functional of the reference density
        lieb = sum_eps - numpy.einsum('ij,ji->', vext + vj_ref, dm_ref) - ref_vxc
        summary = {
            'e1': e_kin + e_ext, 'kin': e_kin, 'ext': e_ext, 'coul': e_coul,
            'exc': e_x, 'nuc': e_nuc, 'kin_ref': e_kin_ref, 'ext_ref': e_ext_ref,
            'coul_ref': e_coul_ref, 'lieb': lieb, 'lieb_error': lieb - e_kin,
        }
        if self.e_ref is not None:
            e_ee_ref = self.e_ref - e_kin_ref - e_ext_ref - e_nuc
            t_c = e_kin_ref - e_kin
            v_c = e_ee_ref - e_coul - e_x
            summary.update({
                't_c': t_c, 'v_c': v_c, 'e_c': t_c + v_c,
                'e_tot': e_kin + e_ext + e_coul + e_x + e_nuc + t_c + v_c,
            })
        return summary

    def get_density_error(self, dm=None, grids=None):
        '''int |rho_ref(r) - rho_KS(r)| dr on a molecular grid.'''
        if dm is None:
            dm = self.make_rdm1()
        dm = numpy.asarray(dm)
        dm_ref = self.dm_ref
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
            dm_ref = dm_ref[0] + dm_ref[1]
        if grids is None:
            grids = gen_grid.Grids(self.mol).build(with_non0tab=True)
        ni = numint.NumInt()
        err = 0
        for ao, mask, weight, coords in ni.block_loop(self.mol, grids, self.mol.nao):
            rho = numint.eval_rho(self.mol, ao, dm_ref - dm, mask)
            err += numpy.dot(abs(rho), weight)
        return err

    def _finalize(self):
        log = logger.new_logger(self)
        name = self.__class__.__name__
        if self.converged:
            log.note('%s converged', name)
        else:
            log.note('%s not converged', name)
        s = self.scf_summary
        if self.e_ref is not None:
            log.note('E_c = %.15g  T_c = %.15g  V_c = %.15g', s['e_c'], s['t_c'], s['v_c'])
        if self.verbose >= logger.INFO:
            log.info('Kinetic energy           %.15g', s['kin'])
            log.info('Coulomb energy           %.15g  error %.6g', s['coul'],
                     s['coul_ref'] - s['coul'])
            log.info('External energy          %.15g  error %.6g', s['ext'],
                     s['ext_ref'] - s['ext'])
            log.info('Exchange energy          %.15g', s['exc'])
            log.info('Lieb functional          %.15g  error %.6g', s['lieb'], s['lieb_error'])
            self.density_error = self.get_density_error()
            log.info('Density error            %.6g', self.density_error)
            self._log_homo_lumo(log)

    def kernel(self):
        '''Runs the Kohn-Sham inversion.

        Returns:
            Total energy if e_ref is given (e_ref plus the error of the
            external energy), otherwise None
        '''
        self.check_sanity()
        self.dump_flags()
        self.build()
        (self.converged, self.mo_energy, self.mo_coeff, self.vref_oep,
         self.vrest_oep, self.vx_oep, self.scf_summary) = self._kernel_fn(self)
        self.vc_oep = self.vref_oep + self.vrest_oep - self.vx_oep
        self.e_tot = self.scf_summary.get('e_tot')
        self.e_c = self.scf_summary.get('e_c')
        self._finalize()
        return self.e_tot

    def eval_vxc(self, coords):
        '''Exchange-correlation potential vref + vrest on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vref_oep + self.vrest_oep)

    def eval_vx(self, coords):
        '''Exact-exchange potential of the Kohn-Sham orbitals on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vx_oep)

    def eval_vc(self, coords):
        '''Correlation potential v_xc - v_x on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vc_oep)


class RKSINV(KSINVBase, exxoep.REXXOEP):
    '''Response-function Kohn-Sham inversion for closed-shell systems.

    Args:
        mf : RHF or RKS object
            Mean-field object which provides the Coulomb and exchange matrices
            (including density fitting, if mf uses it) and the occupation
            numbers. Its orbitals are not used: the inversion starts from the
            Hartree and Fermi-Amaldi potentials of the reference density.
        oep_basis : str or dict
            Auxiliary basis in which the potential is expanded.
        dm_ref : ndarray
            Reference (target) density matrix in AO basis.
        e_ref : float
            Total energy of the calculation which provided dm_ref. Only
            needed for the correlation energy. Default is None.
        ip : float
            Ionization potential. If given, the HOMO energy is fixed to -ip
            (HOMO condition). Default is None.

    Attributes:
        max_cycle : int
            Maximum number of iterations. Default is 100.
        conv_tol : float
            Convergence threshold on the norm of the density difference
            (right-hand side). Default is 1e-10.
        mixing_a, mixing_i : float, int
            Parameters of the mixing scheme, see get_mixing.
            Default is 0.1 and 3.

        vh_via_oep, space_sym, thr_fai_oep and grids as in REXXOEP.

    Saved results:
        converged : bool
        mo_energy, mo_coeff, mo_occ : ndarray
            Kohn-Sham orbitals.
        vref_oep, vrest_oep : ndarray
            OEP basis coefficients of the exchange-correlation potential
            vref + vrest, see eval_vxc.
        vx_oep : ndarray
            OEP basis coefficients of the exact-exchange potential of the
            Kohn-Sham orbitals, see eval_vx.
        vc_oep : ndarray
            Correlation potential vref + vrest - vx, see eval_vc.
        e_c : float
            Correlation energy (if e_ref is given).
        e_tot : float
            Total energy (if e_ref is given).
        scf_summary : dict
            Energy contributions, see energy_summary.

    Examples:

    >>> mf = scf.RHF(mol).density_fit().run()
    >>> mycc = cc.CCSD(mf).run()
    >>> dm = mycc.make_rdm1(ao_repr=True)
    >>> inv = oep.KSINV(mf, 'aug-cc-pvdz-ri', dm, mycc.e_tot).run()
    '''

    _kernel_fn = staticmethod(kernel)

    def __init__(self, mf, oep_basis, dm_ref, e_ref=None, ip=None,
                 vh_via_oep=False, space_sym=False):
        exxoep.REXXOEP.__init__(self, mf, oep_basis, vh_via_oep=vh_via_oep,
                                space_sym=space_sym)
        self._init_ksinv(dm_ref, e_ref, ip)

KSINV = RKSINV

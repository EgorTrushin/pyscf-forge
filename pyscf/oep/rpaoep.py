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
Self-consistent random phase approximation optimized effective potential
(RPA-OEP) method for closed-shell systems.

The exchange potential is the EXX-OEP potential (pyscf.oep.exxoep). The RPA
correlation potential is expanded in the same OEP basis, while the RPA
response function is represented in a separate RI basis. The frequency
integration uses Gauss-Legendre quadrature.

References:

    P. Bleiziffer, A. Heßelmann, A. Görling, J. Chem. Phys. 139, 084113 (2013)
    https://doi.org/10.1063/1.4818984

    E. Trushin, S. Fauser, A. Mölkner, J. Erhard, A. Görling,
    Phys. Rev. Lett. 134, 016402 (2025)
    https://doi.org/10.1103/PhysRevLett.134.016402
'''

import numpy
import scipy.linalg

from pyscf.lib import logger
from pyscf import df
from pyscf.dft import gen_grid
from pyscf.gw.rpa import _get_scaled_legendre_roots
from pyscf.scf import diis as scf_diis
from pyscf.oep import exxoep


def _clamp(eps_ai, thr):
    return numpy.where(abs(eps_ai) > thr, eps_ai, numpy.sign(eps_ai + 1e-300) * thr)

def get_lambda_rpa(mo_energy, nocc, omega, thr=1e-6):
    '''4 e_ai / (e_ai^2 + omega^2) of shape (nvir, nocc), e_ai = e_a - e_i.'''
    eps_ai = _clamp(mo_energy[nocc:, None] - mo_energy[None, :nocc], thr)
    return 4. * eps_ai / (eps_ai**2 + omega**2)

def get_gamma_rpa(mo_energy, nocc, omega, thr=1e-6):
    '''-d lambda_ai / d e_ai, see get_lambda_rpa.'''
    eps_ai = _clamp(mo_energy[nocc:, None] - mo_energy[None, :nocc], thr)
    denom = eps_ai**2 + omega**2
    return -4. / denom + 8. * eps_ai**2 / denom**2

def _inv_denom(mo_energy, e_sub):
    d = e_sub[None, :] - mo_energy[:, None]
    mask = abs(d) > 1e-6
    return numpy.where(mask, 1. / numpy.where(mask, d, 1.), 0.)

def get_rpa_rhs(ints_3c_oep, ints_3c_ri, mo_energy, nocc, nw=20, x0=2.5):
    '''Right-hand side of the RPA-OEP equation and the RPA correlation energy.

    At each frequency the response matrix X0(omega) in the RI basis is built
    and diagonalized. The right-hand side receives the occupied (Rii) and
    virtual (Ria) orbital couplings and a diagonal term (Yia) from the
    orbital-energy dependence of X0.

    All arguments are sequences over spins: one entry for closed-shell
    systems, alpha and beta for open-shell systems, where X0(omega) is
    averaged over the spins.

    Args:
        ints_3c_oep : (naux_c, nmo, nmo) integrals in the transformed OEP
            basis for each spin. None skips the right-hand side of a spin.
            The right-hand side of a spin without electrons is zero.
        ints_3c_ri : (naux_ri_c, nmo, nmo) integrals in the transformed RI
            basis for each spin.
        mo_energy : orbital energies for each spin
        nocc : number of occupied orbitals for each spin
        nw : number of Gauss-Legendre frequency points
        x0 : frequency scaling parameter of the quadrature

    Returns:
        list of right-hand sides (naux_c,) for each spin (None if skipped),
        RPA correlation energy
    '''
    nspin = len(nocc)
    naux_ri = ints_3c_ri[0].shape[0]
    freqs, wts = _get_scaled_legendre_roots(nw, x0)

    fai = []
    pre = []
    for s in range(nspin):
        no = nocc[s]
        e = mo_energy[s]
        nmo = len(e)
        nv = nmo - no
        ri = ints_3c_ri[s]
        fai.append(ri[:, no:, :no].reshape(naux_ri, nv * no))
        oep = ints_3c_oep[s]
        if oep is None:
            pre.append(None)
            continue
        naux_c = oep.shape[0]
        pre.append({
            'active': no > 0,
            'm_virt_t': ri[:, no:, :].reshape(naux_ri * nv, nmo).T,
            'm_occ_t': ri[:, :no, :].reshape(naux_ri * no, nmo).T,
            'oep_occ': oep[:, :no, :].transpose(0, 2, 1).reshape(naux_c, -1),
            'oep_virt': oep[:, no:, :].transpose(0, 2, 1).reshape(naux_c, -1),
            'n_diag': numpy.diagonal(oep, axis1=1, axis2=2),
            'inv_denom_i': _inv_denom(e, e[:no]),
            'inv_denom_a': _inv_denom(e, e[no:]),
        })

    e_corr = 0.
    rhs = [None if p is None else numpy.zeros(p['oep_occ'].shape[0]) for p in pre]
    for omega, wt in zip(freqs, wts):
        lam = [get_lambda_rpa(mo_energy[s], nocc[s], omega) for s in range(nspin)]
        X0_ri = fai[0].dot((fai[0] * lam[0].reshape(1, -1)).T)
        for s in range(1, nspin):
            X0_ri += fai[s].dot((fai[s] * lam[s].reshape(1, -1)).T)
        X0_ri /= nspin
        eval_n, evec = scipy.linalg.eigh(X0_ri)

        e_corr += wt * numpy.sum(numpy.log(1. + eval_n) - eval_n) / (2. * numpy.pi)

        # Non-linear screening factor
        k_n = -eval_n / (eval_n + 1.)
        kmat = (evec * k_n).dot(evec.T)

        for s, p in enumerate(pre):
            if p is None or not p['active']:
                continue
            no = nocc[s]
            nv = len(mo_energy[s]) - no
            f_ai = evec.T.dot(fai[s]).reshape(naux_ri, nv, no)
            gamma = get_gamma_rpa(mo_energy[s], no, omega)
            r = kmat.dot((-lam[s].ravel()) * fai[s]).reshape(naux_ri, nv, no)

            # Rii: coupling through occupied orbital pairs
            t_si = p['m_virt_t'].dot(r.reshape(naux_ri * nv, no)) * p['inv_denom_i']
            rhs_om = 2. * p['oep_occ'].dot(t_si.ravel())
            # Ria: coupling through virtual orbital pairs
            t_sa = (p['m_occ_t'].dot(r.transpose(0, 2, 1).reshape(naux_ri * no, nv))
                    * p['inv_denom_a'])
            rhs_om += 2. * p['oep_virt'].dot(t_sa.ravel())
            # Yia: orbital energy derivative of X0
            y_ai = gamma * k_n.dot((f_ai**2).reshape(naux_ri, -1)).reshape(nv, no)
            n_diag = p['n_diag']
            rhs_om += n_diag[:, no:].dot(y_ai.sum(axis=1)) - n_diag[:, :no].dot(y_ai.sum(axis=0))

            rhs[s] += rhs_om * wt / (2. * numpy.pi)

    return rhs, e_corr


def _project(W, ints_3c):
    '''Integrals in the transformed basis W.T.dot(ints_3c).'''
    naux, nmo = ints_3c.shape[:2]
    return W.T.dot(ints_3c.reshape(naux, -1)).reshape(-1, nmo, nmo)


class RPABase:
    '''RI basis and settings shared by the restricted and unrestricted RPA-OEP
    classes.'''

    thr_fai_ri = 1e-8
    nw = 20
    x0 = 2.5

    _keys = {
        'ri_basis', 'thr_fai_ri', 'nw', 'x0',
        'auxmol_ri', 'WII_ri', 'W3_ri', 'vc_oep', 'e_corr',
    }

    def _init_rpa(self, ri_basis):
        self.ri_basis = ri_basis
        self.auxmol_ri = None
        self.WII_ri = None
        self.W3_ri = None
        self._int3c_ri = None
        self.vc_oep = None
        self.e_corr = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('RI basis = %s', self.ri_basis)
        log.info('thr_fai_ri = %g', self.thr_fai_ri)
        log.info('frequency quadrature nw = %d  x0 = %g', self.nw, self.x0)
        return self

    def build(self):
        '''Builds the OEP basis (see REXXOEP.build) and the RI basis with its
        overlap and charge preprocessing.'''
        super().build()
        mol = self.mol
        self.auxmol_ri = df.addons.make_auxmol(mol, self.ri_basis)
        self._int3c_ri = df.incore.aux_e2(mol, self.auxmol_ri, intor='int3c2e')

        smat = self.auxmol_ri.intor('int2c2e')
        diag = 1. / numpy.sqrt(numpy.diag(smat))
        eigs, evecs = scipy.linalg.eigh(diag[:, None] * smat * diag)
        self.WII_ri = diag[:, None] * evecs / numpy.sqrt(eigs)

        grids = gen_grid.Grids(self.auxmol_ri).build()
        y_ri = self.auxmol_ri.eval_gto('GTOval', grids.coords).T.dot(grids.weights)
        yII_ri = self.WII_ri.T.dot(y_ri)
        self.W3_ri = scipy.linalg.null_space(yII_ri[None, :])
        return self

    def get_W_ri(self, ints_3c_ri, nocc):
        '''Transformation matrix of the RI basis: charge-free combinations
        which couple occupied and virtual orbitals of any spin (eigenvalues
        of the summed coupling matrix >= thr_fai_ri).

        Args:
            ints_3c_ri : RI integrals (naux_ri, nmo, nmo) for each spin
            nocc : number of occupied orbitals for each spin
        '''
        trans = self.WII_ri.dot(self.W3_ri)
        amat = 0
        for ints, no in zip(ints_3c_ri, nocc):
            if no > 0:
                amat = amat + self._get_amat(trans, ints[:, :, :no], no)
        return trans.dot(self._coupling_vectors(amat, self.thr_fai_ri))

    def _save_potentials(self, pot):
        super()._save_potentials(pot)
        self.vc_oep = pot.get('vc_oep')
        self.e_corr = pot.get('e_corr')

    def _finalize(self):
        super()._finalize()
        logger.note(self, 'E_corr(RPA) = %.15g', self.e_corr)

    def eval_vc(self, coords):
        '''RPA correlation potential on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vc_oep)

    def eval_vxc(self, coords):
        '''Exchange-correlation potential vref + vrest + vc on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vref_oep + self.vrest_oep + self.vc_oep)


class RRPAOEP(RPABase, exxoep.REXXOEP):
    '''Self-consistent RPA optimized effective potential method for
    closed-shell systems.

    Args:
        mf : RHF or RKS object
        oep_basis : str or dict
            Auxiliary basis in which the exchange and correlation potentials
            are expanded.
        ri_basis : str or dict
            Auxiliary basis for the RPA response function.

    Attributes:
        use_homo_condition : bool
            Whether to impose the HOMO condition on the exchange potential.
            The correlation potential is always only charge-free.
        thr_fai_ri : float
            Threshold for the occupied-virtual coupling filter of the RI
            basis. Default is 1e-8.
        nw : int
            Number of frequency quadrature points. Default is 20.
        x0 : float
            Scaling parameter of the frequency quadrature. Default is 2.5.

        See REXXOEP for the other attributes.

    Saved results:
        e_tot : float
            Total energy, the EXX energy functional plus e_corr.
        e_corr : float
            RPA correlation energy.
        vref_oep, vrest_oep : ndarray
            OEP basis coefficients of the exchange potential, see eval_vx.
        vc_oep : ndarray
            OEP basis coefficients of the RPA correlation potential, see
            eval_vc and eval_vxc.

        See REXXOEP for the other results.

    Examples:

    >>> mol = gto.M(atom='Ne', basis='cc-pvtz')
    >>> mf = scf.RHF(mol).density_fit().run()
    >>> mf_rpa = oep.RPAOEP(mf, 'aug-cc-pvdz-ri', 'cc-pvqz-ri').run()
    '''

    def __init__(self, mf, oep_basis, ri_basis, use_homo_condition=False,
                 vh_via_oep=False, space_sym=False):
        exxoep.REXXOEP.__init__(self, mf, oep_basis, use_homo_condition,
                                vh_via_oep, space_sym)
        self._init_rpa(ri_basis)

    def _diis_mixer(self):
        adiis = scf_diis.CDIIS(self)
        nocc = self.nocc

        def mix(fock, s1e, mo_coeff):
            c_occ = mo_coeff[:, :nocc]
            return adiis.update(s1e, 2 * c_occ.dot(c_occ.T), fock)
        return mix

    def get_local_potential(self, mo_energy, mo_coeff, vj, vxnl):
        '''EXX-OEP exchange potential plus RPA correlation potential.

        Returns:
            Hartree potential matrix, local potential matrix (AO basis) and a
            dict of the OEP basis coefficients of the potentials and e_corr
        '''
        vj, v_ao, pot = exxoep.REXXOEP.get_local_potential(self, mo_energy, mo_coeff, vj, vxnl)

        nocc, nmo = self.nocc, self.nmo
        ints_3c = self.ao2mo(mo_coeff, nmo)
        ints_ri = self.ao2mo(mo_coeff, nmo, self._int3c_ri)
        ints_occ = ints_3c[:, :, :nocc]

        # The correlation potential is only charge-free, also with the HOMO
        # condition for the exchange potential
        W_c = self.get_W(self.get_W3_charge(), ints_occ, nocc)
        W_ri = self.get_W_ri([ints_ri], [nocc])
        (rhs,), e_corr = get_rpa_rhs([_project(W_c, ints_3c)], [_project(W_ri, ints_ri)],
                                     [mo_energy], [nocc], self.nw, self.x0)

        # get_X0 is negative definite (1/(e_i - e_a) < 0) while get_rpa_rhs
        # contains no such factor, hence -rhs
        X0 = 4. * self.get_X0(ints_occ, mo_energy, nocc, W_c)
        vc_oep = W_c.dot(scipy.linalg.solve(X0, -rhs))

        pot['vc_oep'] = vc_oep
        pot['e_corr'] = e_corr
        return vj, v_ao + self.oep2ao(vc_oep), pot

RPAOEP = RRPAOEP

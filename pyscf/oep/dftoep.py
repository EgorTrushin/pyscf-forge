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
Density functional theory optimized effective potential (DFT-OEP) method
for closed-shell systems.

The orbital-dependent exchange-correlation operator of a density functional
(the semilocal xc potential plus the non-local exact-exchange part of hybrid
functionals) is replaced by the local potential which minimizes the energy
functional, obtained from the OEP equation as in EXX-OEP
(pyscf.oep.exxoep).

References:

    E. Trushin, A. Görling, J. Chem. Theory Comput. 21, 1667 (2025)
    https://doi.org/10.1021/acs.jctc.4c01477

    E. Trushin, A. Görling, J. Chem. Phys. 155, 054109 (2021)
    https://doi.org/10.1063/5.0056431
'''

import numpy

from pyscf import dft
from pyscf.lib import logger
from pyscf.oep import exxoep


def energy_tot(mf, dm):
    '''Total energy of the Kohn-Sham functional of mf for the density matrix
    dm, without modifying mf.

    Returns:
        e_tot, effective potential matrix (tagged with vj, ecoul and exc),
        dict of the energy contributions
    '''
    veff = mf.get_veff(mf.mol, dm)
    dm = numpy.asarray(dm)
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    e1 = numpy.einsum('ij,ji->', mf.get_hcore(), dm_tot).real
    e_nuc = mf.energy_nuc()
    summary = {'e1': e1, 'coul': veff.ecoul.real, 'exc': veff.exc.real, 'nuc': e_nuc}
    return e1 + veff.ecoul.real + veff.exc.real + e_nuc, veff, summary


class DFTOEPBase:
    '''Settings and energy functional shared by the restricted and
    unrestricted DFT-OEP classes.'''

    _keys = {'xc_homo', 'ip_first_iter', 'ip'}

    def _init_dft(self, xc_homo, ip_first_iter):
        if not isinstance(self._scf, dft.KohnShamDFT):
            raise TypeError('%s requires a Kohn-Sham object' % self.__class__.__name__)
        self.xc_homo = xc_homo
        self.ip_first_iter = ip_first_iter
        self.ip = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('xc = %s', self._scf.xc)
        if self.use_homo_condition:
            log.info('xc_homo = %s  ip_first_iter = %s', self.xc_homo, self.ip_first_iter)
        return self

    def build(self):
        if self.use_homo_condition and self.xc_homo is None:
            raise ValueError('xc_homo, the functional for the ionization potential of '
                             'the HOMO condition, is not set')
        super().build()
        self.ip = None
        return self

    def get_energies_and_potentials(self, mo_coeff):
        '''Kohn-Sham energy functional of the given orbitals.

        If use_homo_condition, also updates the ionization potential self.ip,
        see get_ip.

        Returns:
            e_tot, Coulomb matrix, exchange-correlation operator matrix
            (the target of the local potential)
        '''
        dm = self.make_rdm1(mo_coeff)
        e_tot, veff, self.scf_summary = energy_tot(self._scf, dm)
        vj = veff.vj
        if vj.ndim == 3:
            vj = vj[0] + vj[1]
        if self.use_homo_condition and not (self.ip_first_iter and self.ip is not None):
            self.ip = self.get_ip(mo_coeff)
        return e_tot, vj, numpy.asarray(veff) - vj

    def _xc_homo_mf(self, mol, ks_class):
        '''Kohn-Sham object (ks_class = dft.RKS or dft.UKS) with the functional
        xc_homo, which shares the integration grids and density fitting of the
        mean-field object, so that their errors cancel in the ionization
        potential. The neutral system and the ion are set up identically.'''
        mf = self._scf
        mf_x = ks_class(mol, xc=self.xc_homo)
        mf_x.grids = mf.grids
        mf_x.nlcgrids = mf.nlcgrids
        if getattr(mf, 'with_df', None) is not None:
            mf_x = mf_x.density_fit(with_df=mf.with_df, only_dfj=getattr(mf, 'only_dfj', False))
        mf_x.verbose = 0
        return mf_x

    def _ion_energy(self, mo_coeff_ab, mo_occ_ab, charge, spin):
        '''Frozen-orbital energy of the ion with functional xc_homo.'''
        mol = self.mol.copy()
        mol.charge = charge
        mol.spin = spin
        mol.build(False, False)
        mf_ion = self._xc_homo_mf(mol, dft.UKS)
        dm = mf_ion.make_rdm1(mo_coeff_ab, mo_occ_ab)
        return energy_tot(mf_ion, dm)[0]

    def _neutral_energy(self, mo_coeff):
        '''Energy of the neutral system with functional xc_homo.'''
        dm = self.make_rdm1(mo_coeff)
        ks_class = dft.RKS if dm.ndim == 2 else dft.UKS
        return energy_tot(self._xc_homo_mf(self.mol, ks_class), dm)[0]


class RDFTOEP(DFTOEPBase, exxoep.REXXOEP):
    '''Density functional theory optimized effective potential method for
    closed-shell systems.

    Args:
        mf : RKS object
            Converged Kohn-Sham calculation. Its functional (mf.xc), grids
            and density fitting define the energy functional.
        oep_basis : str or dict
            Auxiliary basis in which the local potential is expanded.

    Attributes:
        use_homo_condition : bool
            Whether to fix the HOMO energy to minus the ionization potential
            from frozen-orbital energies of the N and N-1 electron systems
            with the functional xc_homo, see get_ip. Default is False.
        xc_homo : str
            Functional for the ionization potential of the HOMO condition,
            e.g. 'PBE,' (exchange only). Required with use_homo_condition.
        ip_first_iter : bool
            Whether to compute the ionization potential only for the initial
            orbitals. Default is False.

        See REXXOEP for the other attributes.

    Saved results:
        e_tot : float
            Total energy of the Kohn-Sham functional.
        vref_oep, vrest_oep : ndarray
            OEP basis coefficients of the local exchange-correlation
            potential vref + vrest, see eval_vxc.
        ip : float
            Ionization potential of the HOMO condition.

        See REXXOEP for the other results.

    Examples:

    >>> mf = dft.RKS(mol, xc='pbe0').density_fit().run()
    >>> mf_oep = oep.DFTOEP(mf, 'aug-cc-pvdz-ri').run()
    '''

    def __init__(self, mf, oep_basis, use_homo_condition=False, vh_via_oep=False,
                 space_sym=False, xc_homo=None, ip_first_iter=False):
        exxoep.REXXOEP.__init__(self, mf, oep_basis, use_homo_condition,
                                vh_via_oep, space_sym)
        self._init_dft(xc_homo, ip_first_iter)

    def get_homo_target(self, mo_coeff, nocc, vxnl, vj=None, spin=None):
        '''<HOMO|v_ref|HOMO> for which the HOMO energy equals -ip.'''
        c_homo = mo_coeff[:, nocc - 1]
        return -self.ip - c_homo.dot(self._scf.get_hcore() + vj).dot(c_homo)

    def get_ip(self, mo_coeff):
        '''Ionization potential E(N-1) - E(N) with the functional xc_homo for
        the frozen orbitals mo_coeff, removing one electron from the HOMO.'''
        nocc = self.nocc
        mo_occ = numpy.array((self.mo_occ, self.mo_occ)) * .5
        mo_occ[1, nocc - 1] = 0
        e_ion = self._ion_energy((mo_coeff, mo_coeff), mo_occ, self.mol.charge + 1, 1)
        return e_ion - self._neutral_energy(mo_coeff)

    def eval_vxc(self, coords):
        '''Local exchange-correlation potential vref + vrest on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vref_oep + self.vrest_oep)

DFTOEP = RDFTOEP

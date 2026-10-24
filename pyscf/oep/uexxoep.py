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
open-shell systems.

References:

    E. Trushin, A. Görling, J. Chem. Phys. 155, 054109 (2021)
    https://doi.org/10.1063/5.0056431

    E. Trushin, A. Görling, J. Chem. Phys. 159, 244109 (2023)
    https://doi.org/10.1063/5.0171546
'''

import numpy
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf
from pyscf.oep import exxoep


def kernel(oep, mo_energy=None, mo_coeff=None, verbose=None):
    '''Self-consistent EXX-OEP iterations for open-shell systems.

    Returns:
        converged, e_tot, mo_energy, mo_coeff, vref_oep, vrest_oep
        where vref_oep and vrest_oep have shape (2, naux)
    '''
    log = logger.new_logger(oep, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if mo_energy is None:
        mo_energy = oep.mo_energy
    if mo_coeff is None:
        mo_coeff = oep.mo_coeff
    mo_energy = numpy.array(mo_energy, copy=True)
    mo_coeff = numpy.array(mo_coeff, copy=True)
    nocc = oep.nocc
    spin_sym = oep.spin_sym

    h1e = oep._scf.get_hcore()
    s1e = oep._scf.get_ovlp()
    e_tot, vj, vxnl = oep.get_energies_and_potentials(mo_coeff)

    if oep.linear_mixing > 0:
        fock_last = None
    else:
        adiis = lib.diis.DIIS(oep)

    converged = False
    e_last = None
    for cycle in range(oep.max_cycle):
        ints_3c = [oep.ao2mo(mo_coeff[s], nocc[s]) for s in (0, 1)]
        u = [oep.get_u(ints_3c[s], nocc[s]) for s in (0, 1)]

        if oep.vh_via_oep or oep.space_sym:
            vj = oep.get_vh_via_oep(u[0] + u[1], nocc[0] + nocc[1])

        z = [None, None]
        homo_target = [None, None]
        if spin_sym:
            # One common potential for both spins, with the HOMO condition
            # (if any) imposed on the alpha HOMO
            u_avg = .5 * (u[0] + u[1])
            if oep.use_homo_condition:
                z[0], zII = oep.get_z_and_zII(ints_3c[0], mo_energy[0], nocc[0])
                homo_target[0] = oep.get_homo_target(mo_coeff[0], nocc[0], vxnl[0])
                vref = oep.get_v_ref_w_homo(u_avg, zII, homo_target[0])
                W3 = oep.get_W3_charge_and_homo(zII)
            else:
                vref = oep.get_v_ref(u_avg)
                W3 = oep.get_W3_charge()
            W = oep.get_W_spin_sym(W3, ints_3c, nocc)
            vref_oep = [vref, vref]
            W = [W, W]
        else:
            vref_oep = [None, None]
            W = [None, None]
            if not oep.use_homo_condition:
                W3 = oep.get_W3_charge()
            for s in (0, 1):
                if oep.use_homo_condition:
                    z[s], zII = oep.get_z_and_zII(ints_3c[s], mo_energy[s], nocc[s])
                    homo_target[s] = oep.get_homo_target(mo_coeff[s], nocc[s], vxnl[s])
                    vref_oep[s] = oep.get_v_ref_w_homo(u[s], zII, homo_target[s])
                    W3 = oep.get_W3_charge_and_homo(zII)
                else:
                    vref_oep[s] = oep.get_v_ref(u[s])
                W[s] = oep.get_W(W3, ints_3c[s], nocc[s])

        X0 = [oep.get_X0(ints_3c[s], mo_energy[s], nocc[s], W[s]) for s in (0, 1)]
        vref_ao = [oep.oep2ao(vref_oep[s]) for s in (0, 1)]
        rhs = [oep.get_rhs(vref_ao[s], vxnl[s], ints_3c[s], mo_coeff[s],
                           mo_energy[s], nocc[s], W[s]) for s in (0, 1)]

        if spin_sym:
            vrest = W[0].dot(scipy.linalg.solve(X0[0] + X0[1], rhs[0] + rhs[1]))
            vrest_oep = [vrest, vrest]
        else:
            vrest_oep = [W[s].dot(scipy.linalg.solve(X0[s], rhs[s])) for s in (0, 1)]
        vrest_ao = [oep.oep2ao(vrest_oep[s]) for s in (0, 1)]

        for s in ((0,) if spin_sym else (0, 1)):
            oep.check_potentials(vrest_oep[s], vref_oep[s], vrest_ao[s], vref_ao[s],
                                 mo_coeff[s], nocc[s], z[s], homo_target[s])

        fock = numpy.array([h1e + vj + vref_ao[s] + vrest_ao[s] for s in (0, 1)])
        if oep.linear_mixing > 0:
            if fock_last is not None:
                fock = (1 - oep.linear_mixing) * fock + oep.linear_mixing * fock_last
            fock_last = fock
        else:
            dm = [mo_coeff[s][:, :nocc[s]].dot(mo_coeff[s][:, :nocc[s]].T) for s in (0, 1)]
            err = numpy.array([fock[s].dot(dm[s]).dot(s1e) - s1e.dot(dm[s]).dot(fock[s])
                               for s in (0, 1)])
            fock = adiis.update(fock, xerr=err)

        for s in (0, 1):
            mo_energy[s], mo_coeff[s] = oep.eig(fock[s], s1e)
        e_tot, vj, vxnl = oep.get_energies_and_potentials(mo_coeff)

        if e_last is None:
            log.info('cycle= %d E= %.15g', cycle + 1, e_tot)
        else:
            de = e_tot - e_last
            log.info('cycle= %d E= %.15g  delta_E= %4.3g', cycle + 1, e_tot, de)
            if abs(de) < oep.conv_tol:
                converged = True
                break
        e_last = e_tot

    log.timer('UEXX-OEP', *cput0)
    return (converged, e_tot, mo_energy, mo_coeff,
            numpy.array(vref_oep), numpy.array(vrest_oep))


class UEXXOEP(exxoep.REXXOEP):
    '''Exact-exchange optimized effective potential method for open-shell
    systems.

    Args:
        mf : UHF or UKS object
        oep_basis : str or dict
            Auxiliary basis in which the local exchange potential is expanded.

    Attributes:
        spin_sym : bool
            Whether to use one common exchange potential for both spins
            (spin-symmetrized EXX-OEP, J. Chem. Phys. 159, 244109 (2023)).
            With use_homo_condition, the HOMO condition is imposed on the
            alpha HOMO. Default is False.

        See REXXOEP for the other attributes.

    Saved results:
        vref_oep, vrest_oep : ndarray of shape (2, naux)
            OEP basis coefficients of the alpha and beta potentials.

        See REXXOEP for the other results.

    Examples:

    >>> mol = gto.M(atom='N', basis='cc-pvtz', spin=3)
    >>> mf = scf.UHF(mol).density_fit().run()
    >>> mf_oep = oep.EXXOEP(mf, 'aug-cc-pvdz-ri', spin_sym=True).run()
    '''

    _keys = {'spin_sym'}

    def __init__(self, mf, oep_basis, use_homo_condition=False,
                 vh_via_oep=False, space_sym=False, spin_sym=False):
        exxoep.REXXOEP.__init__(self, mf, oep_basis, use_homo_condition,
                                vh_via_oep, space_sym)
        self.spin_sym = spin_sym

    def _check_mo_occ(self):
        mo_occ = numpy.asarray(self.mo_occ)
        if mo_occ.ndim != 2:
            raise TypeError('%s requires UHF or UKS orbitals' % self.__class__.__name__)
        for s, nocc in enumerate(self.nocc):
            if not (numpy.all(mo_occ[s, :nocc] == 1) and numpy.all(mo_occ[s, nocc:] == 0)):
                raise NotImplementedError('Only aufbau occupations 1,...,1,0,...,0 are supported')

    @property
    def nocc(self):
        return (int(numpy.count_nonzero(self.mo_occ[0] > 0)),
                int(numpy.count_nonzero(self.mo_occ[1] > 0)))

    def dump_flags(self, verbose=None):
        exxoep.REXXOEP.dump_flags(self, verbose)
        logger.new_logger(self, verbose).info('spin_sym = %s', self.spin_sym)
        return self

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return uhf.make_rdm1(mo_coeff, mo_occ)

    def get_energies_and_potentials(self, mo_coeff):
        '''Hartree-Fock energy functional of the given orbitals.

        Returns:
            e_tot, Coulomb matrix (total density),
            non-local exchange potential matrices of shape (2, nao, nao)
        '''
        mf = self._scf
        dm = self.make_rdm1(mo_coeff)
        h1e = mf.get_hcore()
        vj, vk = mf.get_jk(self.mol, dm)
        vj = vj[0] + vj[1]
        dm_tot = dm[0] + dm[1]
        e1 = numpy.einsum('ij,ji->', h1e, dm_tot)
        e_coul = numpy.einsum('ij,ji->', vj, dm_tot) * .5
        e_x = numpy.einsum('sij,sji->', vk, dm) * -.5
        e_nuc = mf.energy_nuc()
        self.scf_summary = {'e1': e1, 'coul': e_coul, 'exc': e_x, 'nuc': e_nuc}
        return e1 + e_coul + e_x + e_nuc, vj, -vk

    def get_W_spin_sym(self, W3, ints_3c, nocc, thr_fai=None):
        '''Common transformation matrix for both spins, from the occupied-virtual
        coupling matrix averaged over alpha and beta (J. Chem. Phys. 159,
        244109 (2023), Section IIB).'''
        if thr_fai is None:
            thr_fai = self.thr_fai_oep
        trans = self.WII.dot(W3)
        amat = .5 * (self._get_amat(trans, ints_3c[0], nocc[0]) +
                     self._get_amat(trans, ints_3c[1], nocc[1]))
        return trans.dot(self._coupling_vectors(amat, thr_fai))

    def kernel(self, mo_energy=None, mo_coeff=None):
        '''Runs the self-consistent OEP calculation.

        Returns:
            Total energy
        '''
        self.check_sanity()
        self.dump_flags()
        self.build()
        (self.converged, self.e_tot, self.mo_energy, self.mo_coeff,
         self.vref_oep, self.vrest_oep) = kernel(self, mo_energy, mo_coeff)
        self._finalize()
        return self.e_tot

    def _log_homo_lumo(self, log):
        for s, label in enumerate(('alpha', 'beta')):
            nocc = self.nocc[s]
            mo_energy = self.mo_energy[s]
            if nocc == 0:
                continue
            if nocc < self.nmo:
                log.info('%s HOMO = %.15g  LUMO = %.15g', label,
                         mo_energy[nocc - 1], mo_energy[nocc])
            else:
                log.info('%s HOMO = %.15g', label, mo_energy[nocc - 1])

EXXOEP = UEXXOEP

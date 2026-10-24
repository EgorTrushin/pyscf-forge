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
Response-function Kohn-Sham inversion for open-shell systems.

References:

    J. Erhard, E. Trushin, A. Görling, J. Chem. Phys. 156, 204124 (2022)
    https://doi.org/10.1063/5.0087356

    J. Erhard, E. Trushin, A. Görling, J. Chem. Phys. 162, 034116 (2025)
    https://doi.org/10.1063/5.0239422
'''

import numpy
import scipy.linalg

from pyscf.lib import logger
from pyscf.oep import uexxoep
from pyscf.oep.ksinv import KSINVBase, get_mixing


def kernel(inv, verbose=None):
    '''Kohn-Sham inversion iterations for open-shell systems.

    Returns:
        converged, mo_energy, mo_coeff, vref_oep, vrest_oep, vx_oep, summary
        where the potentials have shape (2, naux)
    '''
    log = logger.new_logger(inv, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    mf = inv._scf
    mol = inv.mol
    nocc = inv.nocc
    dm_ref = inv.dm_ref
    spin_sym = inv.spin_sym
    ip = inv.ip
    spins = (0,) if spin_sym else (0, 1)

    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()
    vj_ref = vj_ref_exact = mf.get_j(mol, dm_ref).sum(axis=0)
    u_ref = [inv.get_u_ao(dm_ref[0]), inv.get_u_ao(dm_ref[1])]
    u_ref_avg = .5 * (u_ref[0] + u_ref[1])

    # Initial guess: Hartree potential of the reference density plus the
    # Fermi-Amaldi reference potentials
    if spin_sym:
        vref = inv.get_v_ref(u_ref_avg)
        vref_oep = [vref, vref]
    else:
        vref_oep = [inv.get_v_ref(u_ref[s]) for s in (0, 1)]
    vref_ao = [inv.oep2ao(vref_oep[s]) for s in (0, 1)]
    mo_energy = numpy.empty((2, inv.nmo))
    mo_coeff = numpy.empty((2,) + inv.mo_coeff[0].shape)
    for s in (0, 1):
        mo_energy[s], mo_coeff[s] = inv.eig(h1e + vj_ref + vref_ao[s], s1e)
    if inv.vh_via_oep or inv.space_sym:
        vj_ref = inv.get_vh_via_oep(u_ref[0] + u_ref[1], nocc[0] + nocc[1])

    converged = False
    vrest_oep = [None, None]
    delta_rhs_i = [None, None]
    mixing = [None, None]
    delta_rhs = [None, None]
    for cycle in range(inv.max_cycle):
        dm = inv.make_rdm1(mo_coeff)
        ints_3c = [inv.ao2mo(mo_coeff[s], nocc[s]) for s in (0, 1)]

        z = [None, None]
        homo_target = [None, None]
        if ip is not None:
            W, W_inv = [None, None], [None, None]
            for s in spins:
                z[s], zII = inv.get_z_and_zII(ints_3c[s], mo_energy[s], nocc[s])
                homo_target[s] = inv.get_homo_target_ip(mo_coeff[s][:, nocc[s] - 1],
                                                        h1e + vj_ref, ip[s])
                W3 = inv.get_W3_charge_and_homo(zII)
                if spin_sym:
                    # Common potential with the HOMO condition on the alpha HOMO
                    vref = inv.get_v_ref_w_homo(u_ref_avg, zII, homo_target[0])
                    vref_oep = [vref, vref]
                    w, w_inv = inv.get_W_and_inv(W3, ints_3c, nocc)
                    W, W_inv = [w, w], [w_inv, w_inv]
                else:
                    vref_oep[s] = inv.get_v_ref_w_homo(u_ref[s], zII, homo_target[s])
                    W[s], W_inv[s] = inv.get_W_and_inv(W3, ints_3c[s], nocc[s])
            vref_ao = [inv.oep2ao(vref_oep[s]) for s in (0, 1)]
        else:
            W3 = inv.get_W3_charge()
            if spin_sym:
                w, w_inv = inv.get_W_and_inv(W3, ints_3c, nocc)
                W, W_inv = [w, w], [w_inv, w_inv]
            else:
                W, W_inv = zip(*[inv.get_W_and_inv(W3, ints_3c[s], nocc[s]) for s in (0, 1)])

        X0x = [inv.get_X0(ints_3c[s], mo_energy[s], nocc[s], W[s]) for s in (0, 1)]
        rhs = [W[s].T.dot(u_ref[s] - inv.get_u_ao(dm[s])) for s in (0, 1)]

        if spin_sym:
            rhs = rhs[0] + rhs[1]
            delta_rhs[0] = numpy.linalg.norm(rhs)
            mixing[0], delta_rhs_i[0] = get_mixing(cycle, delta_rhs[0], inv.mixing_a,
                                                   inv.mixing_i, delta_rhs_i[0])
            dv = mixing[0] * W[0].dot(scipy.linalg.solve(2 * X0x[0] + 2 * X0x[1], rhs))
            v = dv if vrest_oep[0] is None else vrest_oep[0] + dv
            v = W[0].dot(W_inv[0].dot(v))
            vrest_oep = [v, v]
        else:
            for s in (0, 1):
                delta_rhs[s] = numpy.linalg.norm(rhs[s])
                mixing[s], delta_rhs_i[s] = get_mixing(cycle, delta_rhs[s], inv.mixing_a,
                                                       inv.mixing_i, delta_rhs_i[s])
                dv = mixing[s] * W[s].dot(scipy.linalg.solve(2 * X0x[s], rhs[s]))
                v = dv if vrest_oep[s] is None else vrest_oep[s] + dv
                vrest_oep[s] = W[s].dot(W_inv[s].dot(v))
        vrest_ao = [inv.oep2ao(vrest_oep[s]) for s in (0, 1)]
        converged = all(delta_rhs[s] < inv.conv_tol for s in spins)

        for s in spins:
            inv.check_potentials(vrest_oep[s], vref_oep[s], vrest_ao[s], vref_ao[s],
                                 mo_coeff[s], nocc[s], z[s], homo_target[s])

        if converged or cycle == inv.max_cycle - 1:
            # Exact-exchange potentials for the current orbitals
            vk = mf.get_k(mol, dm)
            rhs_x = [inv.get_rhs(vref_ao[s], -vk[s], ints_3c[s], mo_coeff[s],
                                 mo_energy[s], nocc[s], W[s]) for s in (0, 1)]
            if spin_sym:
                vx = W[0].dot(scipy.linalg.solve(X0x[0] + X0x[1], rhs_x[0] + rhs_x[1]))
                vx_oep = [vx + vref_oep[0]] * 2
            else:
                vx_oep = [W[s].dot(scipy.linalg.solve(X0x[s], rhs_x[s])) + vref_oep[s]
                          for s in (0, 1)]

        for s in (0, 1):
            mo_energy[s], mo_coeff[s] = inv.eig(h1e + vj_ref + vref_ao[s] + vrest_ao[s], s1e)

        if spin_sym:
            log.info('cycle= %d  |rhs|= %.3e  mixing= %.3e', cycle + 1, delta_rhs[0], mixing[0])
        else:
            log.info('cycle= %d  |rhs|= %.3e %.3e  mixing= %.3e %.3e', cycle + 1,
                     delta_rhs[0], delta_rhs[1], mixing[0], mixing[1])
        if converged:
            break

    dm = inv.make_rdm1(mo_coeff)
    vj, vk = mf.get_jk(mol, dm)
    e_x = numpy.einsum('sij,sji->', vk, dm) * -.5
    sum_eps = mo_energy[0, :nocc[0]].sum() + mo_energy[1, :nocc[1]].sum()
    ref_vxc = sum(numpy.einsum('ij,ji->', dm_ref[s], vref_ao[s] + vrest_ao[s]) for s in (0, 1))
    summary = inv.energy_summary(dm, dm_ref, e_x, vj[0] + vj[1], vj_ref_exact, vj_ref,
                                 sum_eps, ref_vxc)

    log.timer('UKS inversion', *cput0)
    return (converged, mo_energy, mo_coeff, numpy.array(vref_oep),
            numpy.array(vrest_oep), numpy.array(vx_oep), summary)


class UKSINV(KSINVBase, uexxoep.UEXXOEP):
    '''Response-function Kohn-Sham inversion for open-shell systems.

    Args:
        mf : UHF or UKS object
        oep_basis : str or dict
            Auxiliary basis in which the potential is expanded.
        dm_ref : ndarray of shape (2, nao, nao)
            Alpha and beta reference (target) density matrices in AO basis.
        e_ref : float
            Total energy of the calculation which provided dm_ref. Only
            needed for the correlation energy. Default is None.
        ip : pair of floats
            Alpha and beta ionization potentials. If given, the HOMO energies
            are fixed to -ip (HOMO condition). With spin_sym, only the alpha
            HOMO condition is imposed. Default is None.

    Attributes:
        spin_sym : bool
            Whether to use one common potential for both spins.
            Default is False.

        See RKSINV for the other attributes and the saved results. The
        potentials vref_oep, vrest_oep, vx_oep and vc_oep have the shape
        (2, naux).
    '''

    _kernel_fn = staticmethod(kernel)

    def __init__(self, mf, oep_basis, dm_ref, e_ref=None, ip=None,
                 vh_via_oep=False, space_sym=False, spin_sym=False):
        uexxoep.UEXXOEP.__init__(self, mf, oep_basis, vh_via_oep=vh_via_oep,
                                 space_sym=space_sym, spin_sym=spin_sym)
        self._init_ksinv(dm_ref, e_ref, ip)
        if self.dm_ref.ndim != 3:
            raise ValueError('dm_ref must contain alpha and beta density matrices')

    def dump_flags(self, verbose=None):
        KSINVBase.dump_flags(self, verbose)
        logger.new_logger(self, verbose).info('spin_sym = %s', self.spin_sym)
        return self

KSINV = UKSINV

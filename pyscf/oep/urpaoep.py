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
Self-consistent RPA optimized effective potential (RPA-OEP) method for
open-shell systems.

References:

    P. Bleiziffer, A. Heßelmann, A. Görling, J. Chem. Phys. 139, 084113 (2013)
    https://doi.org/10.1063/1.4818984

    E. Trushin, S. Fauser, A. Mölkner, J. Erhard, A. Görling,
    Phys. Rev. Lett. 134, 016402 (2025)
    https://doi.org/10.1103/PhysRevLett.134.016402
'''

import numpy
import scipy.linalg

from pyscf.oep import uexxoep
from pyscf.oep.rpaoep import RPABase, get_rpa_rhs, _project


class URPAOEP(RPABase, uexxoep.UEXXOEP):
    '''Self-consistent RPA optimized effective potential method for
    open-shell systems.

    Args:
        mf : UHF or UKS object
        oep_basis : str or dict
            Auxiliary basis in which the exchange and correlation potentials
            are expanded.
        ri_basis : str or dict
            Auxiliary basis for the RPA response function.

    Attributes:
        spin_sym : bool
            Whether to use common exchange and correlation potentials for
            both spins. Default is False.

        See RRPAOEP and UEXXOEP for the other attributes.

    Saved results:
        vref_oep, vrest_oep, vc_oep : ndarray of shape (2, naux)
            OEP basis coefficients of the alpha and beta potentials.

        See RRPAOEP for the other results.
    '''

    def __init__(self, mf, oep_basis, ri_basis, use_homo_condition=False,
                 vh_via_oep=False, space_sym=False, spin_sym=False):
        uexxoep.UEXXOEP.__init__(self, mf, oep_basis, use_homo_condition,
                                 vh_via_oep, space_sym, spin_sym)
        self._init_rpa(ri_basis)

    def get_local_potential(self, mo_energy, mo_coeff, vj, vxnl):
        '''EXX-OEP exchange potentials plus RPA correlation potentials.

        Returns:
            Hartree potential matrix, alpha and beta local potential matrices
            (AO basis) and a dict of the OEP basis coefficients of the
            potentials, of shape (2, naux), and e_corr
        '''
        vj, v_ao, pot = uexxoep.UEXXOEP.get_local_potential(self, mo_energy, mo_coeff,
                                                           vj, vxnl)
        nocc, nmo = self.nocc, self.nmo
        ints_3c = [self.ao2mo(mo_coeff[s], nmo) for s in (0, 1)]
        ints_ri = [self.ao2mo(mo_coeff[s], nmo, self._int3c_ri) for s in (0, 1)]
        ints_occ = [ints_3c[s][:, :, :nocc[s]] for s in (0, 1)]

        W_ri = self.get_W_ri(ints_ri, nocc)
        ints_ri_W = [_project(W_ri, ints_ri[s]) for s in (0, 1)]
        # The correlation potentials are only charge-free, also with the
        # HOMO condition for the exchange potentials
        W3 = self.get_W3_charge()
        if self.spin_sym:
            W_c = self.get_W_spin_sym(W3, ints_occ, nocc)
            rhs, e_corr = get_rpa_rhs([_project(W_c, ints_3c[s]) for s in (0, 1)],
                                      ints_ri_W, mo_energy, nocc, self.nw, self.x0)
            X0 = [4. * self.get_X0(ints_occ[s], mo_energy[s], nocc[s], W_c) for s in (0, 1)]
            vc = W_c.dot(scipy.linalg.solve(X0[0] + X0[1], -(rhs[0] + rhs[1])))
            vc_oep = numpy.array([vc, vc])
        else:
            W_c = [self.get_W(W3, ints_occ[s], nocc[s]) if nocc[s] > 0 else None
                   for s in (0, 1)]
            rhs, e_corr = get_rpa_rhs(
                [None if W_c[s] is None else _project(W_c[s], ints_3c[s]) for s in (0, 1)],
                ints_ri_W, mo_energy, nocc, self.nw, self.x0)
            vc_oep = numpy.zeros((2, self.auxmol.nao))
            for s in (0, 1):
                if W_c[s] is not None:
                    X0 = 4. * self.get_X0(ints_occ[s], mo_energy[s], nocc[s], W_c[s])
                    vc_oep[s] = W_c[s].dot(scipy.linalg.solve(X0, -rhs[s]))

        pot['vc_oep'] = vc_oep
        pot['e_corr'] = e_corr
        return vj, v_ao + numpy.array([self.oep2ao(vc_oep[s]) for s in (0, 1)]), pot

RPAOEP = URPAOEP

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
for open-shell systems.

References:

    E. Trushin, A. Görling, J. Chem. Theory Comput. 21, 1667 (2025)
    https://doi.org/10.1021/acs.jctc.4c01477

    E. Trushin, A. Görling, J. Chem. Phys. 159, 244109 (2023)
    https://doi.org/10.1063/5.0171546
'''

import numpy

from pyscf.oep import uexxoep
from pyscf.oep.dftoep import DFTOEPBase


class UDFTOEP(DFTOEPBase, uexxoep.UEXXOEP):
    '''Density functional theory optimized effective potential method for
    open-shell systems.

    Args:
        mf : UKS object
        oep_basis : str or dict
            Auxiliary basis in which the local potentials are expanded.

    Attributes:
        spin_sym : bool
            Whether to use one common potential for both spins. With
            use_homo_condition, the HOMO condition is imposed on the alpha
            HOMO. Default is False.

        See RDFTOEP and UEXXOEP for the other attributes.

    Saved results:
        vref_oep, vrest_oep : ndarray of shape (2, naux)
            OEP basis coefficients of the alpha and beta potentials.
        ip : list
            Alpha and beta ionization potentials of the HOMO condition.

        See RDFTOEP for the other results.
    '''

    def __init__(self, mf, oep_basis, use_homo_condition=False, vh_via_oep=False,
                 space_sym=False, spin_sym=False, xc_homo=None, ip_first_iter=False):
        uexxoep.UEXXOEP.__init__(self, mf, oep_basis, use_homo_condition,
                                 vh_via_oep, space_sym, spin_sym)
        self._init_dft(xc_homo, ip_first_iter)

    def get_homo_target(self, mo_coeff, nocc, vxnl, vj=None, spin=0):
        '''<HOMO|v_ref|HOMO> for which the HOMO energy of spin equals -ip[spin].'''
        c_homo = mo_coeff[:, nocc - 1]
        return -self.ip[spin] - c_homo.dot(self._scf.get_hcore() + vj).dot(c_homo)

    def get_ip(self, mo_coeff):
        '''Alpha and beta ionization potentials E(N-1) - E(N) with the
        functional xc_homo for the frozen orbitals mo_coeff, removing one
        electron from the alpha or beta HOMO. None for a spin without
        electrons.'''
        e_n = self._neutral_energy(mo_coeff)
        ip = []
        for s, dspin in ((0, -1), (1, 1)):
            if self.nocc[s] == 0:
                ip.append(None)
                continue
            mo_occ = numpy.array(self.mo_occ, copy=True)
            mo_occ[s, self.nocc[s] - 1] = 0
            e_ion = self._ion_energy(mo_coeff, mo_occ, self.mol.charge + 1,
                                     self.mol.spin + dspin)
            ip.append(e_ion - e_n)
        return ip

    def eval_vxc(self, coords):
        '''Local exchange-correlation potentials vref + vrest on the grid coords (Bohr).'''
        return self.eval_potential(coords, self.vref_oep + self.vrest_oep)

DFTOEP = UDFTOEP

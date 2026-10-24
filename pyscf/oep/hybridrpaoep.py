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
Hybrid self-consistent random phase approximation optimized effective
potential (hybrid RPA-OEP) method for closed-shell systems.

A fraction alpha of the non-local exact (Hartree-Fock) exchange operator
v_x^NL replaces the same fraction of the local EXX-OEP exchange potential v_x
in the Fock matrix,

    F = h + v_J + alpha * v_x^NL + (1 - alpha) * v_x + v_c,

while v_x and the RPA correlation potential v_c are obtained from the OEP
equations of RPA-OEP (pyscf.oep.rpaoep). The energy is the RPA-OEP energy
functional, the exact-exchange energy plus the RPA correlation energy, of the
resulting orbitals.

References:

    A. Thierbach, D. Schmidtel, A. Görling, J. Chem. Phys. 151, 144117 (2019)
    https://doi.org/10.1063/1.5120587

    E. Trushin, R. Mandalia, A. Görling, J. Chem. Phys. 165, 134115 (2026)
    https://doi.org/10.1063/5.0345799
'''

import numpy

from pyscf.lib import logger
from pyscf.oep import rpaoep


class HybridRPABase:
    '''Fraction of non-local exchange in the Fock matrix, shared by the
    restricted and unrestricted hybrid RPA-OEP classes.'''

    alpha = 0.25

    _keys = {'alpha'}

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.new_logger(self, verbose).info('alpha (fraction of non-local exchange) = %g', self.alpha)
        return self

    def get_local_potential(self, mo_energy, mo_coeff, vj, vxnl):
        '''RPA-OEP potentials, with the fraction alpha of the local exchange
        potential replaced by the non-local exchange operator vxnl.

        Returns:
            Hartree potential matrix, potential matrix entering the Fock
            matrix (AO basis) and a dict of the OEP basis coefficients of the
            local potentials and e_corr, see RRPAOEP.get_local_potential
        '''
        vj, v_ao, pot = super().get_local_potential(mo_energy, mo_coeff, vj, vxnl)
        vx_oep = pot['vref_oep'] + pot['vrest_oep']
        if vx_oep.ndim == 1:
            vx_ao = self.oep2ao(vx_oep)
        else:
            vx_ao = numpy.array([self.oep2ao(v) for v in vx_oep])
        return vj, v_ao + self.alpha * (vxnl - vx_ao), pot


class RHybridRPAOEP(HybridRPABase, rpaoep.RRPAOEP):
    '''Hybrid self-consistent RPA optimized effective potential method for
    closed-shell systems.

    Args:
        mf : RHF or RKS object
        oep_basis : str or dict
            Auxiliary basis in which the exchange and correlation potentials
            are expanded.
        ri_basis : str or dict
            Auxiliary basis for the RPA response function.
        alpha : float
            Fraction of non-local exchange in the Fock matrix. Default is
            0.25. alpha = 0 is RPA-OEP.

    Attributes:
        alpha : float
            Fraction of non-local exchange, see above.

        See RRPAOEP for the other attributes.

    Saved results:
        mo_energy, mo_coeff : ndarray
            Orbitals of the hybrid Fock matrix.
        vref_oep, vrest_oep, vc_oep : ndarray
            OEP basis coefficients of the local exchange and RPA correlation
            potentials, see eval_vx, eval_vc and eval_vxc. The Fock matrix
            contains the exchange potential only with the weight 1 - alpha.

        See RRPAOEP for the other results.

    Examples:

    >>> mol = gto.M(atom='Ne', basis='cc-pvtz')
    >>> mf = scf.RHF(mol).density_fit().run()
    >>> mf_rpa = oep.HybridRPAOEP(mf, 'aug-cc-pvdz-ri', 'cc-pvqz-ri', alpha=0.25).run()
    '''

    def __init__(self, mf, oep_basis, ri_basis, alpha=0.25, use_homo_condition=False,
                 vh_via_oep=False, space_sym=False):
        rpaoep.RRPAOEP.__init__(self, mf, oep_basis, ri_basis, use_homo_condition,
                                vh_via_oep, space_sym)
        self.alpha = alpha

HybridRPAOEP = RHybridRPAOEP

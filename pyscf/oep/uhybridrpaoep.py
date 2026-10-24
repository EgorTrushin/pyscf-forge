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
potential (hybrid RPA-OEP) method for open-shell systems.

References:

    A. Thierbach, D. Schmidtel, A. Görling, J. Chem. Phys. 151, 144117 (2019)
    https://doi.org/10.1063/1.5120587

    E. Trushin, R. Mandalia, A. Görling, J. Chem. Phys. 165, 134115 (2026)
    https://doi.org/10.1063/5.0345799
'''

from pyscf.oep import urpaoep
from pyscf.oep.hybridrpaoep import HybridRPABase


class UHybridRPAOEP(HybridRPABase, urpaoep.URPAOEP):
    '''Hybrid self-consistent RPA optimized effective potential method for
    open-shell systems.

    The Fock matrix of each spin contains the fraction alpha of the non-local
    exchange operator of that spin and the fraction 1 - alpha of its local
    exchange potential, see pyscf.oep.hybridrpaoep.

    Args:
        mf : UHF or UKS object
        oep_basis : str or dict
            Auxiliary basis in which the exchange and correlation potentials
            are expanded.
        ri_basis : str or dict
            Auxiliary basis for the RPA response function.
        alpha : float
            Fraction of non-local exchange in the Fock matrices. Default is
            0.25. alpha = 0 is RPA-OEP.

    Attributes:
        spin_sym : bool
            Whether to use common local exchange and correlation potentials
            for both spins. Default is False.

        See RHybridRPAOEP and URPAOEP for the other attributes.

    Saved results:
        vref_oep, vrest_oep, vc_oep : ndarray of shape (2, naux)
            OEP basis coefficients of the alpha and beta local potentials.

        See RHybridRPAOEP for the other results.
    '''

    def __init__(self, mf, oep_basis, ri_basis, alpha=0.25, use_homo_condition=False,
                 vh_via_oep=False, space_sym=False, spin_sym=False):
        urpaoep.URPAOEP.__init__(self, mf, oep_basis, ri_basis, use_homo_condition,
                                 vh_via_oep, space_sym, spin_sym)
        self.alpha = alpha

HybridRPAOEP = UHybridRPAOEP

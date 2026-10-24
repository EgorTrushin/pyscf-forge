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
Optimized effective potential (OEP) methods in Gaussian basis sets

Simple usage::

    >>> from pyscf import gto, scf, oep
    >>> mol = gto.M(atom='C 0 0 -0.646514; O 0 0 0.484886', basis='cc-pvtz')
    >>> mf = scf.RHF(mol).density_fit().run()
    >>> mf_oep = oep.EXXOEP(mf, 'aug-cc-pvdz-ri').run()

Kohn-Sham inversion of a reference density::

    >>> inv = oep.KSINV(mf, 'aug-cc-pvdz-ri', dm_ref, e_ref).run()
'''

from pyscf import scf
from pyscf.oep import exxoep
from pyscf.oep import uexxoep
from pyscf.oep import ksinv
from pyscf.oep import uksinv
from pyscf.oep.exxoep import REXXOEP
from pyscf.oep.uexxoep import UEXXOEP
from pyscf.oep.ksinv import RKSINV
from pyscf.oep.uksinv import UKSINV


def EXXOEP(mf, oep_basis, **kwargs):
    '''Exact-exchange optimized effective potential method.

    Returns a REXXOEP object for RHF/RKS and a UEXXOEP object for UHF/UKS
    mean-field objects. Open-shell ROHF/ROKS orbitals are converted to UHF/UKS.
    See REXXOEP and UEXXOEP for the keyword arguments.
    '''
    if isinstance(mf, scf.uhf.UHF):
        return uexxoep.UEXXOEP(mf, oep_basis, **kwargs)
    elif isinstance(mf, scf.rohf.ROHF) and mf.mol.spin != 0:
        return uexxoep.UEXXOEP(scf.addons.convert_to_uhf(mf), oep_basis, **kwargs)
    elif isinstance(mf, scf.hf.RHF):
        return exxoep.REXXOEP(mf, oep_basis, **kwargs)
    else:
        raise NotImplementedError('EXXOEP is not available for %s' % mf.__class__)


def KSINV(mf, oep_basis, dm_ref, e_ref=None, ip=None, **kwargs):
    '''Response-function Kohn-Sham inversion of the density dm_ref.

    Returns a RKSINV object for RHF/RKS and a UKSINV object for UHF/UKS
    mean-field objects. Open-shell ROHF/ROKS objects are converted to UHF/UKS.
    See RKSINV and UKSINV for the arguments.
    '''
    if isinstance(mf, scf.uhf.UHF):
        return uksinv.UKSINV(mf, oep_basis, dm_ref, e_ref, ip, **kwargs)
    elif isinstance(mf, scf.rohf.ROHF) and mf.mol.spin != 0:
        return uksinv.UKSINV(scf.addons.convert_to_uhf(mf), oep_basis, dm_ref, e_ref, ip,
                             **kwargs)
    elif isinstance(mf, scf.hf.RHF):
        return ksinv.RKSINV(mf, oep_basis, dm_ref, e_ref, ip, **kwargs)
    else:
        raise NotImplementedError('KSINV is not available for %s' % mf.__class__)

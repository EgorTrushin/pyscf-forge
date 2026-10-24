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

import unittest
import numpy
from pyscf import gto, scf, oep

OEP_BASIS = 'aug-cc-pvdz-ri'
DF_BASIS = 'def2-universal-jkfit'

def setUpModule():
    global mol_h2, mf_h2
    mol_h2 = gto.M(atom='H 0 0 -0.370946; H 0 0 0.370946', basis='cc-pvtz',
                   verbose=5, output='/dev/null')
    mf_h2 = scf.RHF(mol_h2).density_fit(auxbasis=DF_BASIS).run()

def tearDownModule():
    global mol_h2, mf_h2
    mol_h2.stdout.close()
    del mol_h2, mf_h2


class KnownValues(unittest.TestCase):
    def test_hybridrpaoep(self):
        mf_rpa = oep.HybridRPAOEP(mf_h2, OEP_BASIS, 'cc-pvqz-ri')
        self.assertIsInstance(mf_rpa, oep.RHybridRPAOEP)
        self.assertAlmostEqual(mf_rpa.alpha, .25, 14)
        e_tot = mf_rpa.run(conv_tol=1e-11).e_tot
        self.assertTrue(mf_rpa.converged)
        self.assertAlmostEqual(e_tot, -1.201437026163, 8)
        self.assertTrue(mf_rpa.e_corr < 0)
        # The correlation potential is charge-free
        self.assertAlmostEqual(numpy.dot(mf_rpa.y, mf_rpa.vc_oep), 0, 10)

    def test_hybridrpaoep_alpha(self):
        mf_rpa = oep.HybridRPAOEP(mf_h2, OEP_BASIS, 'cc-pvqz-ri', alpha=.5)
        self.assertAlmostEqual(mf_rpa.run(conv_tol=1e-11).e_tot, -1.196035237105, 8)

    def test_alpha_zero_is_rpaoep(self):
        e_hybrid = oep.HybridRPAOEP(mf_h2, OEP_BASIS, 'cc-pvqz-ri', alpha=0).run(conv_tol=1e-11).e_tot
        e_rpa = oep.RPAOEP(mf_h2, OEP_BASIS, 'cc-pvqz-ri').run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_hybrid, e_rpa, 9)

    def test_hybridrpaoep_homo_condition(self):
        mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz', verbose=5, output='/dev/null')
        mf = scf.RHF(mol).density_fit(auxbasis=DF_BASIS).run()
        mf_rpa = oep.HybridRPAOEP(mf, OEP_BASIS, 'cc-pvtz-ri', use_homo_condition=True)
        e_tot = mf_rpa.run(conv_tol=1e-10).e_tot
        self.assertTrue(mf_rpa.converged)
        self.assertAlmostEqual(e_tot, -128.912322187290, 8)

        # <HOMO|v_x|HOMO> = <HOMO|v_x^NL|HOMO> for the orbitals of the hybrid
        # Fock matrix
        c_homo = mf_rpa.mo_coeff[:, mf_rpa.nocc - 1]
        vk = mf.get_k(mol, mf_rpa.make_rdm1()) * -.5
        vx = mf_rpa.oep2ao(mf_rpa.vref_oep + mf_rpa.vrest_oep)
        self.assertAlmostEqual(c_homo.dot(vx).dot(c_homo), c_homo.dot(vk).dot(c_homo), 6)
        mol.stdout.close()


if __name__ == "__main__":
    print("Full Tests for hybrid RPA-OEP")
    unittest.main()

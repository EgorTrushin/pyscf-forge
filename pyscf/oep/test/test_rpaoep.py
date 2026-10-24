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
    def test_rpaoep(self):
        mf_rpa = oep.RPAOEP(mf_h2, OEP_BASIS, 'cc-pvqz-ri')
        self.assertIsInstance(mf_rpa, oep.RRPAOEP)
        e_tot = mf_rpa.run(conv_tol=1e-11).e_tot
        self.assertTrue(mf_rpa.converged)
        self.assertAlmostEqual(e_tot, -1.208018894914, 8)
        self.assertTrue(mf_rpa.e_corr < 0)
        # The correlation potential is charge-free
        self.assertAlmostEqual(numpy.dot(mf_rpa.y, mf_rpa.vc_oep), 0, 10)
        coords = [[0, 0, 1.], [1., 0, 0]]
        self.assertAlmostEqual(abs(mf_rpa.eval_vxc(coords) - mf_rpa.eval_vx(coords)
                                   - mf_rpa.eval_vc(coords)).max(), 0, 12)

    def test_rpaoep_homo_condition(self):
        mf_rpa = oep.RPAOEP(mf_h2, OEP_BASIS, 'cc-pvqz-ri', use_homo_condition=True)
        self.assertAlmostEqual(mf_rpa.run(conv_tol=1e-11).e_tot, -1.208018896512, 8)

    def test_rpaoep_vh_via_oep(self):
        mf_rpa = oep.RPAOEP(mf_h2, OEP_BASIS, 'cc-pvqz-ri', vh_via_oep=True)
        self.assertAlmostEqual(mf_rpa.run(conv_tol=1e-11).e_tot, -1.208018507440, 8)

    def test_rpaoep_ne(self):
        mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz', verbose=5, output='/dev/null')
        mf = scf.RHF(mol).density_fit(auxbasis=DF_BASIS).run()
        mf_rpa = oep.RPAOEP(mf, OEP_BASIS, 'cc-pvtz-ri').run(conv_tol=1e-10)
        self.assertTrue(mf_rpa.converged)
        self.assertAlmostEqual(mf_rpa.e_tot, -128.931575143143, 8)
        mol.stdout.close()


if __name__ == "__main__":
    print("Full Tests for RPA-OEP")
    unittest.main()

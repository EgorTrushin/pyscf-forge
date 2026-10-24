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
RI_BASIS = 'cc-pvtz-ri'
DF_BASIS = 'def2-universal-jkfit'

def setUpModule():
    global mol_n, mf_n
    mol_n = gto.M(atom='N 0 0 0', basis='cc-pvtz', spin=3,
                  verbose=5, output='/dev/null')
    mf_n = scf.UHF(mol_n).density_fit(auxbasis=DF_BASIS).run()

def tearDownModule():
    global mol_n, mf_n
    mol_n.stdout.close()
    del mol_n, mf_n


class KnownValues(unittest.TestCase):
    def test_urpaoep(self):
        mf_rpa = oep.RPAOEP(mf_n, OEP_BASIS, RI_BASIS)
        self.assertIsInstance(mf_rpa, oep.URPAOEP)
        e_tot = mf_rpa.run(conv_tol=1e-10).e_tot
        self.assertTrue(mf_rpa.converged)
        self.assertAlmostEqual(e_tot, -54.609141587988, 8)
        self.assertEqual(mf_rpa.vc_oep.shape, (2, mf_rpa.auxmol.nao))
        for s in (0, 1):
            self.assertAlmostEqual(numpy.dot(mf_rpa.y, mf_rpa.vc_oep[s]), 0, 10)

    def test_urpaoep_spin_sym(self):
        mf_rpa = oep.RPAOEP(mf_n, OEP_BASIS, RI_BASIS, spin_sym=True)
        self.assertAlmostEqual(mf_rpa.run(conv_tol=1e-10).e_tot, -54.607701704575, 8)
        self.assertAlmostEqual(abs(mf_rpa.vc_oep[0] - mf_rpa.vc_oep[1]).max(), 0, 12)

    def test_urpaoep_homo_condition_spin_sym(self):
        mf_rpa = oep.RPAOEP(mf_n, OEP_BASIS, RI_BASIS, use_homo_condition=True,
                            spin_sym=True)
        self.assertAlmostEqual(mf_rpa.run(conv_tol=1e-10).e_tot, -54.605273710842, 8)

    def test_urpaoep_h_atom(self):
        # No beta electron
        mol = gto.M(atom='H 0 0 0', basis='cc-pvtz', spin=1, verbose=5, output='/dev/null')
        mf = scf.UHF(mol).density_fit(auxbasis=DF_BASIS).run()
        e_tot = oep.RPAOEP(mf, OEP_BASIS, RI_BASIS, spin_sym=True).run(conv_tol=1e-10).e_tot
        self.assertAlmostEqual(e_tot, -0.517974697777, 8)
        mol.stdout.close()

    def test_closed_shell_consistency(self):
        # Unrestricted RPA-OEP of a closed-shell molecule equals the
        # restricted one
        mol = gto.M(atom='H 0 0 -0.370946; H 0 0 0.370946', basis='cc-pvtz',
                    verbose=5, output='/dev/null')
        mf = scf.RHF(mol).density_fit(auxbasis=DF_BASIS).run()
        e_r = oep.RPAOEP(mf, OEP_BASIS, RI_BASIS).run(conv_tol=1e-11).e_tot
        mf = scf.UHF(mol).density_fit(auxbasis=DF_BASIS).run()
        e_u = oep.RPAOEP(mf, OEP_BASIS, RI_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_u, e_r, 9)
        mol.stdout.close()


if __name__ == "__main__":
    print("Full Tests for URPA-OEP")
    unittest.main()

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
from pyscf import gto, dft, oep

OEP_BASIS = 'aug-cc-pvdz-ri'

def setUpModule():
    global mol, mf_pbe
    mol = gto.M(atom='N 0 0 0', basis='cc-pvtz', spin=3, verbose=5, output='/dev/null')
    mf_pbe = dft.UKS(mol, xc='pbe').density_fit().run()

def tearDownModule():
    global mol, mf_pbe
    mol.stdout.close()
    del mol, mf_pbe


class KnownValues(unittest.TestCase):
    def test_pbe(self):
        mf_oep = oep.DFTOEP(mf_pbe, OEP_BASIS)
        self.assertIsInstance(mf_oep, oep.UDFTOEP)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertTrue(mf_oep.converged)
        self.assertAlmostEqual(e_tot, -54.529393258708, 8)

    def test_pbe0(self):
        mf = dft.UKS(mol, xc='pbe0').density_fit().run()
        e_tot = oep.DFTOEP(mf, OEP_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -54.541080759965, 8)

    def test_spin_sym(self):
        mf_oep = oep.DFTOEP(mf_pbe, OEP_BASIS, spin_sym=True)
        self.assertAlmostEqual(mf_oep.run(conv_tol=1e-11).e_tot, -54.528254852042, 8)

    def test_homo_condition(self):
        # The alpha HOMOs are degenerate. Which of them is ionized is
        # arbitrary, which affects the ionization potential through the
        # numerical integration at the level of 1e-7.
        mf_oep = oep.DFTOEP(mf_pbe, OEP_BASIS, use_homo_condition=True, xc_homo='GGA_X_PBE')
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -54.529359770798, 7)
        for s in (0, 1):
            self.assertAlmostEqual(mf_oep.mo_energy[s][mf_oep.nocc[s] - 1], -mf_oep.ip[s], 5)


if __name__ == "__main__":
    print("Full Tests for UDFT-OEP")
    unittest.main()

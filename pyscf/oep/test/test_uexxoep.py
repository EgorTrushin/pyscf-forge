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
    global mol_n, mf_n, mol_c, mf_c
    mol_n = gto.M(atom='N 0 0 0', basis='cc-pvtz', spin=3,
                  verbose=5, output='/dev/null')
    mf_n = scf.UHF(mol_n).density_fit(auxbasis=DF_BASIS).run()
    mol_c = gto.M(atom='C 0 0 0', basis='cc-pvtz', spin=2,
                  verbose=5, output='/dev/null')
    mf_c = scf.UHF(mol_c).density_fit(auxbasis=DF_BASIS).run()

def tearDownModule():
    global mol_n, mf_n, mol_c, mf_c
    mol_n.stdout.close()
    mol_c.stdout.close()
    del mol_n, mf_n, mol_c, mf_c


class KnownValues(unittest.TestCase):
    def test_uexxoep(self):
        mf_oep = oep.EXXOEP(mf_n, OEP_BASIS)
        self.assertIsInstance(mf_oep, oep.UEXXOEP)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertTrue(mf_oep.converged)
        self.assertAlmostEqual(e_tot, -54.399423779273, 8)
        self.assertEqual(mf_oep.vref_oep.shape, (2, mf_oep.auxmol.nao))
        for s in (0, 1):
            self.assertAlmostEqual(numpy.dot(mf_oep.y, mf_oep.vref_oep[s]), -1, 10)
            self.assertAlmostEqual(numpy.dot(mf_oep.y, mf_oep.vrest_oep[s]), 0, 10)
        vx = mf_oep.eval_vx([[0, 0, 20.]])
        self.assertAlmostEqual(abs(20 * vx + 1).max(), 0, 4)

    def test_uexxoep_homo_condition(self):
        mf_oep = oep.EXXOEP(mf_n, OEP_BASIS, use_homo_condition=True)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -54.396803935622, 8)

    def test_uexxoep_spin_sym(self):
        mf_oep = oep.EXXOEP(mf_n, OEP_BASIS, spin_sym=True)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -54.396454768182, 8)
        self.assertAlmostEqual(abs(mf_oep.vref_oep[0] - mf_oep.vref_oep[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(mf_oep.vrest_oep[0] - mf_oep.vrest_oep[1]).max(), 0, 12)

    def test_uexxoep_homo_condition_spin_sym(self):
        mf_oep = oep.EXXOEP(mf_n, OEP_BASIS, use_homo_condition=True, spin_sym=True)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -54.394201594996, 8)

    def test_uexxoep_space_sym(self):
        e_tot = oep.EXXOEP(mf_c, OEP_BASIS, space_sym=True).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -37.686603032309, 8)
        e_tot = oep.EXXOEP(mf_c, OEP_BASIS, space_sym=True,
                           spin_sym=True).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -37.676855039909, 8)

    def test_uexxoep_linear_mixing(self):
        mf_oep = oep.EXXOEP(mf_n, OEP_BASIS)
        e_tot = mf_oep.run(max_cycle=100, linear_mixing=.3, conv_tol=1e-11).e_tot
        self.assertTrue(mf_oep.converged)
        self.assertAlmostEqual(e_tot, -54.399423779268, 8)

    def test_uexxoep_from_rohf(self):
        mf = scf.ROHF(mol_n).density_fit(auxbasis=DF_BASIS).run()
        mf_oep = oep.EXXOEP(mf, OEP_BASIS)
        self.assertIsInstance(mf_oep, oep.UEXXOEP)
        self.assertAlmostEqual(mf_oep.run(conv_tol=1e-11).e_tot, -54.399423779273, 8)

    def test_uexxoep_molecule(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvtz', spin=1,
                    verbose=5, output='/dev/null')
        mf = scf.UHF(mol).density_fit(auxbasis=DF_BASIS).run()
        e_tot = oep.EXXOEP(mf, OEP_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -75.417008951088, 8)
        mol.stdout.close()


if __name__ == "__main__":
    print("Full Tests for UEXX-OEP")
    unittest.main()

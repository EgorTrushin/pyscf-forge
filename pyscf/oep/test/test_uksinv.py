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
from pyscf import gto, scf, cc, oep

OEP_BASIS = 'aug-cc-pvdz-ri'
DF_BASIS = 'def2-universal-jkfit'

def setUpModule():
    global mol_n, mf_n, mol_c, mf_c, dm_cc, e_cc
    mol_n = gto.M(atom='N 0 0 0', basis='cc-pvtz', spin=3,
                  verbose=5, output='/dev/null')
    mf_n = scf.UHF(mol_n).density_fit(auxbasis=DF_BASIS).run(conv_tol=1e-12)
    mycc = cc.UCCSD(mf_n).run(conv_tol=1e-11, conv_tol_normt=1e-8)
    dm_cc = numpy.asarray(mycc.make_rdm1(ao_repr=True))
    e_cc = mycc.e_tot
    mol_c = gto.M(atom='C 0 0 0', basis='cc-pvtz', spin=2,
                  verbose=5, output='/dev/null')
    mf_c = scf.UHF(mol_c).density_fit(auxbasis=DF_BASIS).run()

def tearDownModule():
    global mol_n, mf_n, mol_c, mf_c, dm_cc, e_cc
    mol_n.stdout.close()
    mol_c.stdout.close()
    del mol_n, mf_n, mol_c, mf_c, dm_cc, e_cc


class KnownValues(unittest.TestCase):
    def _check_exx_round_trip(self, mf, **kwargs):
        mf_oep = oep.EXXOEP(mf, OEP_BASIS, **kwargs).run(conv_tol=1e-12)
        ip = None
        if mf_oep.use_homo_condition:
            ip = [-mf_oep.mo_energy[s][mf_oep.nocc[s] - 1] for s in (0, 1)]
        inv = oep.KSINV(mf, OEP_BASIS, mf_oep.make_rdm1(), mf_oep.e_tot, ip,
                        vh_via_oep=mf_oep.vh_via_oep, space_sym=mf_oep.space_sym,
                        spin_sym=mf_oep.spin_sym)
        self.assertIsInstance(inv, oep.UKSINV)
        inv.run(conv_tol=1e-12)
        self.assertTrue(inv.converged)
        self.assertAlmostEqual(abs(inv.mo_energy - mf_oep.mo_energy).max(), 0, 8)
        self.assertAlmostEqual(inv.e_c, 0, 8)
        return inv

    def test_exx_round_trip(self):
        inv = self._check_exx_round_trip(mf_n)
        self.assertEqual(inv.vc_oep.shape, (2, inv.auxmol.nao))
        self.assertAlmostEqual(inv.density_error, 0, 6)

    def test_exx_round_trip_homo_condition(self):
        self._check_exx_round_trip(mf_n, use_homo_condition=True)

    def test_exx_round_trip_vh_via_oep(self):
        self._check_exx_round_trip(mf_n, vh_via_oep=True)

    def test_exx_round_trip_spin_sym(self):
        inv = self._check_exx_round_trip(mf_n, spin_sym=True)
        self.assertAlmostEqual(abs(inv.vrest_oep[0] - inv.vrest_oep[1]).max(), 0, 12)

    def test_exx_round_trip_space_sym(self):
        self._check_exx_round_trip(mf_c, space_sym=True)
        self._check_exx_round_trip(mf_c, space_sym=True, spin_sym=True)

    def test_uccsd(self):
        inv = oep.KSINV(mf_n, OEP_BASIS, dm_cc, e_cc).run(conv_tol=1e-12)
        self.assertTrue(inv.converged)
        self.assertAlmostEqual(inv.e_c, -0.125990347020, 8)
        for s in (0, 1):
            self.assertAlmostEqual(numpy.dot(inv.y, inv.vref_oep[s]), -1, 10)
            self.assertAlmostEqual(numpy.dot(inv.y, inv.vrest_oep[s]), 0, 10)

    def test_uccsd_spin_sym(self):
        inv = oep.KSINV(mf_n, OEP_BASIS, dm_cc, e_cc, spin_sym=True).run(conv_tol=1e-12)
        self.assertAlmostEqual(inv.e_c, -0.123514033698, 8)

    def test_dm_ref_shape(self):
        with self.assertRaises(ValueError):
            oep.KSINV(mf_n, OEP_BASIS, dm_cc[0] + dm_cc[1])


if __name__ == "__main__":
    print("Full Tests for UKS inversion")
    unittest.main()

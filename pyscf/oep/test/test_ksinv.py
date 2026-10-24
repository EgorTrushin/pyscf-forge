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
    global mol, mf, dm_cc, e_cc
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz', verbose=5, output='/dev/null')
    mf = scf.RHF(mol).density_fit(auxbasis=DF_BASIS).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf).run(conv_tol=1e-11, conv_tol_normt=1e-8)
    dm_cc = mycc.make_rdm1(ao_repr=True)
    e_cc = mycc.e_tot

def tearDownModule():
    global mol, mf, dm_cc, e_cc
    mol.stdout.close()
    del mol, mf, dm_cc, e_cc


class KnownValues(unittest.TestCase):
    def _check_exx_round_trip(self, **kwargs):
        # Inverting the EXX-OEP density recovers the EXX-OEP potential:
        # same orbital energies and no correlation
        mf_oep = oep.EXXOEP(mf, OEP_BASIS, **kwargs).run(conv_tol=1e-12)
        ip = None
        if mf_oep.use_homo_condition:
            ip = -mf_oep.mo_energy[mf_oep.nocc - 1]
        inv = oep.KSINV(mf, OEP_BASIS, mf_oep.make_rdm1(), mf_oep.e_tot, ip,
                        vh_via_oep=mf_oep.vh_via_oep)
        self.assertIsInstance(inv, oep.RKSINV)
        inv.run(conv_tol=1e-12)
        self.assertTrue(inv.converged)
        self.assertAlmostEqual(abs(inv.mo_energy - mf_oep.mo_energy).max(), 0, 8)
        self.assertAlmostEqual(inv.e_c, 0, 8)
        self.assertAlmostEqual(inv.e_tot, mf_oep.e_tot, 8)
        self.assertAlmostEqual(inv.density_error, 0, 6)
        return inv

    def test_exx_round_trip(self):
        inv = self._check_exx_round_trip()
        self.assertAlmostEqual(abs(inv.eval_vc([[0, 0, 1.], [0, 0, 3.]])).max(), 0, 6)

    def test_exx_round_trip_homo_condition(self):
        self._check_exx_round_trip(use_homo_condition=True)

    def test_exx_round_trip_vh_via_oep(self):
        self._check_exx_round_trip(vh_via_oep=True)

    def test_ccsd(self):
        inv = oep.KSINV(mf, OEP_BASIS, dm_cc, e_cc).run(conv_tol=1e-12)
        self.assertTrue(inv.converged)
        self.assertAlmostEqual(inv.e_c, -0.289452063679, 8)
        s = inv.scf_summary
        self.assertAlmostEqual(s['t_c'] + s['v_c'], inv.e_c, 12)
        # Charge conditions
        self.assertAlmostEqual(numpy.dot(inv.y, inv.vref_oep), -1, 10)
        self.assertAlmostEqual(numpy.dot(inv.y, inv.vrest_oep), 0, 10)
        self.assertAlmostEqual(numpy.dot(inv.y, inv.vx_oep), -1, 10)

    def test_ccsd_homo_condition(self):
        inv = oep.KSINV(mf, OEP_BASIS, dm_cc, e_cc, ip=0.79).run(conv_tol=1e-12)
        self.assertAlmostEqual(inv.e_c, -0.310984207272, 8)
        self.assertAlmostEqual(inv.mo_energy[inv.nocc - 1], -0.79, 10)

    def test_without_e_ref(self):
        inv = oep.KSINV(mf, OEP_BASIS, dm_cc)
        self.assertIsNone(inv.kernel())
        self.assertIsNone(inv.e_c)
        self.assertTrue(inv.converged)


if __name__ == "__main__":
    print("Full Tests for KS inversion")
    unittest.main()

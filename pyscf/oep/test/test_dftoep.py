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
from pyscf import gto, scf, dft, oep

OEP_BASIS = 'aug-cc-pvdz-ri'

def setUpModule():
    global mol
    mol = gto.M(atom='C 0 0 -0.646514; O 0 0 0.484886', basis='cc-pvtz',
                verbose=5, output='/dev/null')

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_ks(xc, mol=None):
    if mol is None:
        mol = globals()['mol']
    return dft.RKS(mol, xc=xc).density_fit().run()


class KnownValues(unittest.TestCase):
    def test_pbe(self):
        mf = run_ks('pbe')
        mo_coeff = mf.mo_coeff.copy()
        mf_oep = oep.DFTOEP(mf, OEP_BASIS)
        self.assertIsInstance(mf_oep, oep.RDFTOEP)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertTrue(mf_oep.converged)
        self.assertAlmostEqual(e_tot, -113.228591213239, 8)
        # The OEP energy is above the Kohn-Sham energy of the functional
        self.assertTrue(e_tot > mf.e_tot)
        # The mean-field object is not modified
        self.assertTrue(numpy.array_equal(mo_coeff, mf.mo_coeff))

    def test_pbe0(self):
        e_tot = oep.DFTOEP(run_ks('pbe0'), OEP_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -113.224831398487, 8)

    def test_homo_condition(self):
        mf_oep = oep.DFTOEP(run_ks('pbe'), OEP_BASIS, use_homo_condition=True,
                            xc_homo='pbe,')
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -113.226067755958, 8)
        # The HOMO energy is minus the ionization potential of the last cycle
        self.assertAlmostEqual(mf_oep.mo_energy[mf_oep.nocc - 1], -mf_oep.ip, 5)

    def test_homo_condition_ip_first_iter(self):
        mf_oep = oep.DFTOEP(run_ks('pbe'), OEP_BASIS, use_homo_condition=True,
                            xc_homo='pbe,', ip_first_iter=True)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -113.225921019219, 8)
        self.assertAlmostEqual(mf_oep.mo_energy[mf_oep.nocc - 1], -mf_oep.ip, 10)

    def test_range_separated(self):
        e_tot = oep.DFTOEP(run_ks('hse06'), OEP_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -113.235800307352, 8)
        mol_ne = gto.M(atom='Ne 0 0 0', basis='cc-pvtz', verbose=5, output='/dev/null')
        mf = run_ks('cam-b3lyp', mol_ne)
        e_tot = oep.DFTOEP(mf, OEP_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -128.938681882995, 8)
        self.assertTrue(e_tot > mf.e_tot)
        mol_ne.stdout.close()

    def test_input_checks(self):
        with self.assertRaises(TypeError):
            oep.DFTOEP(scf.RHF(mol).density_fit().run(), OEP_BASIS)
        mf_oep = oep.DFTOEP(run_ks('pbe'), OEP_BASIS, use_homo_condition=True)
        with self.assertRaises(ValueError):
            mf_oep.kernel()


if __name__ == "__main__":
    print("Full Tests for DFT-OEP")
    unittest.main()

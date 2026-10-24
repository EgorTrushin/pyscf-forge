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
DF_BASIS = 'def2-universal-jkfit'

def setUpModule():
    global mol_co, mf_co, mol_ne, mf_ne
    mol_co = gto.M(atom='C 0 0 -0.646514; O 0 0 0.484886', basis='cc-pvtz',
                   verbose=5, output='/dev/null')
    mf_co = scf.RHF(mol_co).density_fit(auxbasis=DF_BASIS).run()
    mol_ne = gto.M(atom='Ne 0 0 0', basis='cc-pvtz', verbose=5, output='/dev/null')
    mf_ne = scf.RHF(mol_ne).density_fit(auxbasis=DF_BASIS).run()

def tearDownModule():
    global mol_co, mf_co, mol_ne, mf_ne
    mol_co.stdout.close()
    mol_ne.stdout.close()
    del mol_co, mf_co, mol_ne, mf_ne


class KnownValues(unittest.TestCase):
    def test_exxoep(self):
        mf_oep = oep.EXXOEP(mf_co, OEP_BASIS)
        self.assertIsInstance(mf_oep, oep.REXXOEP)
        e_tot = mf_oep.kernel()
        self.assertTrue(mf_oep.converged)
        self.assertAlmostEqual(e_tot, -112.774741623119, 8)
        # OEP energy is above HF
        self.assertTrue(e_tot > mf_co.e_tot)

        # Charge conditions of the potentials
        self.assertAlmostEqual(numpy.dot(mf_oep.y, mf_oep.vref_oep), -1, 10)
        self.assertAlmostEqual(numpy.dot(mf_oep.y, mf_oep.vrest_oep), 0, 10)
        # -1/r asymptotics of the exchange potential, perpendicular to the
        # bond where the dipole contribution vanishes
        r = numpy.array([100., 200.])
        vx = mf_oep.eval_vx(numpy.c_[r, numpy.zeros(2), numpy.zeros(2)])
        self.assertAlmostEqual(abs(r * vx + 1).max(), 0, 3)

    def test_exxoep_homo_condition(self):
        mf_oep = oep.EXXOEP(mf_co, OEP_BASIS, use_homo_condition=True)
        mf_oep.conv_tol = 1e-11
        self.assertAlmostEqual(mf_oep.kernel(), -112.771128876714, 8)

        # <HOMO|v_x|HOMO> = <HOMO|v_x^NL|HOMO>
        nocc = mf_oep.nocc
        c_homo = mf_oep.mo_coeff[:, nocc - 1]
        dm = mf_oep.make_rdm1()
        vk = mf_co.get_k(mol_co, dm) * -.5
        vx = mf_oep.oep2ao(mf_oep.vref_oep + mf_oep.vrest_oep)
        self.assertAlmostEqual(c_homo.dot(vx).dot(c_homo),
                               c_homo.dot(vk).dot(c_homo), 6)

    def test_exxoep_vh_via_oep(self):
        mf_oep = oep.EXXOEP(mf_co, OEP_BASIS, vh_via_oep=True)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -112.774706611367, 8)

    def test_exxoep_from_rks(self):
        mf = dft.RKS(mol_co, xc='pbe').density_fit(auxbasis=DF_BASIS).run()
        e_tot = oep.EXXOEP(mf, OEP_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -112.774741623124, 8)

    def test_exxoep_symmetry(self):
        mol = mol_co.copy()
        mol.symmetry = True
        mol.build()
        mf = scf.RHF(mol).density_fit(auxbasis=DF_BASIS).run()
        e_tot = oep.EXXOEP(mf, OEP_BASIS).run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -112.774741623119, 8)

    def test_exxoep_space_sym(self):
        mf_oep = oep.EXXOEP(mf_ne, OEP_BASIS, space_sym=True)
        e_tot = mf_oep.run(conv_tol=1e-11).e_tot
        self.assertAlmostEqual(e_tot, -128.531501106196, 8)

    def test_exxoep_linear_mixing(self):
        mf_oep = oep.EXXOEP(mf_ne, OEP_BASIS)
        e_tot = mf_oep.run(max_cycle=100, linear_mixing=.3, conv_tol=1e-11).e_tot
        self.assertTrue(mf_oep.converged)
        self.assertAlmostEqual(e_tot, -128.531505078808, 8)


if __name__ == "__main__":
    print("Full Tests for EXX-OEP")
    unittest.main()

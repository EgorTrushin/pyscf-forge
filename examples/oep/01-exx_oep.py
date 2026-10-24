#!/usr/bin/env python

'''
Exact-exchange optimized effective potential (EXX-OEP) for a closed-shell
molecule.

Three basis sets are used: the orbital basis, the OEP basis in which the
local exchange potential is expanded, and the density fitting basis for the
Coulomb and exchange matrices of the mean-field object.

See also E. Trushin, A. Görling, J. Chem. Phys. 155, 054109 (2021)
'''

import numpy
from pyscf import gto, scf, oep

mol = gto.M(atom='C 0 0 -0.646514; O 0 0 0.484886', basis='cc-pvtz')
mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()

mf_oep = oep.EXXOEP(mf, 'aug-cc-pvdz-ri')
mf_oep.kernel()
print('E(EXX-OEP) - E(HF) =', mf_oep.e_tot - mf.e_tot)

nocc = mf_oep.nocc
print('HOMO = %.6f  LUMO-HOMO = %.6f' %
      (mf_oep.mo_energy[nocc - 1], mf_oep.mo_energy[nocc] - mf_oep.mo_energy[nocc - 1]))

#
# The HOMO condition requires <HOMO|v_x|HOMO> = <HOMO|v_x^NL|HOMO> for the
# Kohn-Sham HOMO (Appendix C of the paper above), which brings the HOMO
# energy close to the HF one.
#
mf_oep_homo = oep.EXXOEP(mf, 'aug-cc-pvdz-ri', use_homo_condition=True).run()
print('HOMO with HOMO condition = %.6f, HF HOMO = %.6f' %
      (mf_oep_homo.mo_energy[nocc - 1], mf.mo_energy[nocc - 1]))

#
# The exchange potential v_x = v_ref + v_rest on a line along the bond.
# v_ref is the Fermi-Amaldi type reference potential which carries the -1/r
# asymptotics, v_rest is the charge-free remainder.
#
z = numpy.linspace(-5, 5, 11)
coords = numpy.zeros((z.size, 3))
coords[:, 2] = z
vref = mf_oep.eval_potential(coords, mf_oep.vref_oep)
vrest = mf_oep.eval_potential(coords, mf_oep.vrest_oep)
vx = mf_oep.eval_vx(coords)
print('   z/Bohr       v_ref      v_rest         v_x')
for zi, a, b, c in zip(z, vref, vrest, vx):
    print('%9.3f %11.6f %11.6f %11.6f' % (zi, a, b, c))

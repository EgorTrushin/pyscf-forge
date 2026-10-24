#!/usr/bin/env python

'''
Kohn-Sham inversion of a CCSD density: determines the local Kohn-Sham
potential which reproduces the CCSD density and splits the exchange-
correlation potential into the exact-exchange and the correlation potential.

See also J. Erhard, E. Trushin, A. Görling, J. Chem. Phys. 156, 204124 (2022)
'''

import numpy
from pyscf import gto, scf, cc, oep

mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz')
mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()

mycc = cc.CCSD(mf).run()
# Unrelaxed CCSD density. A relaxed density can be used in the same way.
dm_ccsd = mycc.make_rdm1(ao_repr=True)

#
# The total energy e_ref of the reference method is needed for the
# correlation energy E_c = T_c + V_c of the inverted density. E_c is stored
# in inv.e_c, T_c and V_c in inv.scf_summary.
#
inv = oep.KSINV(mf, 'aug-cc-pvdz-ri', dm_ccsd, mycc.e_tot).run()

#
# HOMO condition: fix the Kohn-Sham HOMO energy to minus the ionization
# potential, here from EOM-IP-CCSD.
#
ip = mycc.ipccsd(nroots=1)[0]
inv_ip = oep.KSINV(mf, 'aug-cc-pvdz-ri', dm_ccsd, mycc.e_tot, ip=ip).run()
print('IP = %.6f  -HOMO = %.6f' % (ip, -inv_ip.mo_energy[inv_ip.nocc - 1]))

#
# Exchange-correlation, exchange and correlation potentials on a radial line
#
r = numpy.linspace(.5, 5, 10)
coords = numpy.zeros((r.size, 3))
coords[:, 2] = r
vxc = inv_ip.eval_vxc(coords)
vx = inv_ip.eval_vx(coords)
vc = inv_ip.eval_vc(coords)
print('   r/Bohr        v_xc         v_x         v_c')
for ri, a, b, c in zip(r, vxc, vx, vc):
    print('%9.3f %11.6f %11.6f %11.6f' % (ri, a, b, c))

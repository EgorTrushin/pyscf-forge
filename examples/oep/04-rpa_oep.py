#!/usr/bin/env python

'''
Self-consistent RPA optimized effective potential (RPA-OEP).

Four basis sets are used: the orbital basis, the OEP basis for the exchange
and correlation potentials, the RI basis for the RPA response function, and
the density fitting basis of the mean-field object.

See also P. Bleiziffer, A. Heßelmann, A. Görling, J. Chem. Phys. 139, 084113
(2013) and E. Trushin et al., Phys. Rev. Lett. 134, 016402 (2025)
'''

import numpy
from pyscf import gto, scf, oep

mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz')
mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()

mf_rpa = oep.RPAOEP(mf, 'aug-cc-pvdz-ri', 'cc-pvqz-ri')
# Settings of the frequency integration (defaults)
mf_rpa.nw = 20
mf_rpa.x0 = 2.5
mf_rpa.run()
print('E(RPA-OEP) = %.10f  E_corr(RPA) = %.10f' % (mf_rpa.e_tot, mf_rpa.e_corr))

#
# Exchange and RPA correlation potentials on a radial line
#
r = numpy.linspace(.5, 5, 10)
coords = numpy.zeros((r.size, 3))
coords[:, 2] = r
vx = mf_rpa.eval_vx(coords)
vc = mf_rpa.eval_vc(coords)
print('   r/Bohr         v_x         v_c')
for ri, a, b in zip(r, vx, vc):
    print('%9.3f %11.6f %11.6f' % (ri, a, b))

#
# Open-shell systems. spin_sym=True gives common potentials for both spins.
#
mol = gto.M(atom='N 0 0 0', basis='cc-pvtz', spin=3)
mf = scf.UHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()
mf_rpa = oep.RPAOEP(mf, 'aug-cc-pvdz-ri', 'cc-pvtz-ri', spin_sym=True).run()

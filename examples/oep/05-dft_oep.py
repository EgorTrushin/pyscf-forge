#!/usr/bin/env python

'''
Density functional theory optimized effective potential (DFT-OEP): the local
Kohn-Sham potential for the orbital-dependent exchange-correlation operator
of a (hybrid) functional.

See also E. Trushin, A. Görling, J. Chem. Theory Comput. 21, 1667 (2025)
'''

from pyscf import gto, dft, oep

mol = gto.M(atom='C 0 0 -0.646514; O 0 0 0.484886', basis='cc-pvtz')

#
# Hybrid functional: the non-local exact exchange of PBE0 is replaced by a
# local potential. The OEP energy is slightly above the generalized
# Kohn-Sham energy of PBE0.
#
mf = dft.RKS(mol, xc='pbe0').density_fit().run()
mf_oep = oep.DFTOEP(mf, 'aug-cc-pvdz-ri').run()
print('E(PBE0-OEP) - E(PBE0) = %.6f' % (mf_oep.e_tot - mf.e_tot))

#
# HOMO condition: the HOMO energy is fixed to minus the ionization potential,
# computed from the frozen-orbital energies of the N and N-1 electron systems
# with the functional xc_homo.
#
mf_oep = oep.DFTOEP(mf, 'aug-cc-pvdz-ri', use_homo_condition=True,
                    xc_homo='0.25*HF + 0.75*GGA_X_PBE').run()
print('IP = %.6f  -HOMO = %.6f' % (mf_oep.ip, -mf_oep.mo_energy[mf_oep.nocc - 1]))

#
# Range-separated hybrids are supported as well
#
mf = dft.RKS(mol, xc='cam-b3lyp').density_fit().run()
mf_oep = oep.DFTOEP(mf, 'aug-cc-pvdz-ri').run()

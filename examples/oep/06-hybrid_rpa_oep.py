#!/usr/bin/env python

'''
Hybrid self-consistent RPA optimized effective potential (hybrid RPA-OEP).

A fraction alpha of the non-local exact exchange replaces the same fraction of
the local exchange potential in the Fock matrix. The local exchange and RPA
correlation potentials and the energy functional are those of RPA-OEP.

See also A. Thierbach, D. Schmidtel, A. Görling, J. Chem. Phys. 151, 144117
(2019) and E. Trushin, R. Mandalia, A. Görling, J. Chem. Phys. 165, 134115
(2026)
'''

from pyscf import gto, scf, oep

mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz')
mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()

#
# alpha = 0 is RPA-OEP
#
for alpha in (0, .25, .5):
    mf_rpa = oep.HybridRPAOEP(mf, 'aug-cc-pvdz-ri', 'cc-pvtz-ri', alpha=alpha).run()
    print('alpha = %.2f  E = %.10f  E_corr(RPA) = %.10f  HOMO = %.6f' %
          (alpha, mf_rpa.e_tot, mf_rpa.e_corr, mf_rpa.mo_energy[mf_rpa.nocc - 1]))

#
# HOMO condition for the local exchange potential
#
mf_rpa = oep.HybridRPAOEP(mf, 'aug-cc-pvdz-ri', 'cc-pvtz-ri', use_homo_condition=True).run()
print('HOMO condition: E = %.10f  HOMO = %.6f' % (mf_rpa.e_tot, mf_rpa.mo_energy[mf_rpa.nocc - 1]))

#
# Open-shell systems. spin_sym=True gives common local potentials for both spins.
#
mol = gto.M(atom='N 0 0 0', basis='cc-pvtz', spin=3)
mf = scf.UHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()
mf_rpa = oep.HybridRPAOEP(mf, 'aug-cc-pvdz-ri', 'cc-pvtz-ri', alpha=.25, spin_sym=True).run()
print('N atom: E = %.10f  E_corr(RPA) = %.10f' % (mf_rpa.e_tot, mf_rpa.e_corr))

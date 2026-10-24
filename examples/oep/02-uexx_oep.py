#!/usr/bin/env python

'''
EXX-OEP for open-shell systems.

See also E. Trushin, A. Görling, J. Chem. Phys. 159, 244109 (2023)
'''

from pyscf import gto, scf, oep

mol = gto.M(atom='N 0 0 0', basis='cc-pvtz', spin=3)
mf = scf.UHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()

#
# Separate exchange potentials for alpha and beta electrons.
# vref_oep and vrest_oep have the shape (2, naux).
#
mf_oep = oep.EXXOEP(mf, 'aug-cc-pvdz-ri').run()

#
# Spin-symmetrized EXX-OEP: one common exchange potential for both spins.
#
mf_oep_ss = oep.EXXOEP(mf, 'aug-cc-pvdz-ri', spin_sym=True).run()

#
# space_sym keeps only the spherically symmetric part of the OEP basis, which
# gives spherical potentials for atoms with a non-spherical density.
#
mol = gto.M(atom='C 0 0 0', basis='cc-pvtz', spin=2)
mf = scf.UHF(mol).density_fit(auxbasis='def2-universal-jkfit').run()
mf_oep = oep.EXXOEP(mf, 'aug-cc-pvdz-ri', space_sym=True, spin_sym=True).run()

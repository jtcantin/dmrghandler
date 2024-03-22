import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import scipy as sp
import sparse

import dmrghandler.dmrg_calc_prepare as dmrg_calc_prepare
import dmrghandler.fcidump_io as fcidump_io

test_rtol = 1e-5
test_atol = 1e-8
test_rsquared_rtol = 1e-1
test_rsquared_atol = 1e-1


class TestFcidumpIO(unittest.TestCase):
    def test_fcidump_write(self):
        fcidump_dir = Path("tests/data_for_testing")
        list_of_fcidump_files = [
            "fcidump.2_co2_6-311++G__",
            "fcidump.7_melact_6-311++G__",
            "fcidump.32_2ru_III_3pl_{'default' _ '6-31+G(d,p)', 'Ru' _ 'lanl2tz' }",
            "fcidump.34_3ruo_IV_2pl_{'Ru' _ 'lanl2tz', 'default' _ '6-31+G(d,p)'}",
            "fcidump.36_1ru_II_2pl_{'default' _ '6-31+G(d,p)', 'Ru' _ 'lanl2tz' }",
        ]

        for fcidump_file in list_of_fcidump_files:
            fcidump_file = Path(fcidump_file)
            fcidump_path = fcidump_dir / fcidump_file

            (
                one_body_tensor,
                two_body_tensor,
                nuc_rep_energy,
                num_orbitals,
                num_spin_orbitals,
                num_electrons,
                two_S,
                two_Sz,
                orb_sym,
                extra_attributes,
            ) = dmrg_calc_prepare.load_tensors_from_fcidump(fcidump_path)

            temp_fcidump_dir = Path("tests/fcidump_temp")
            temp_fcidump_dir.mkdir(exist_ok=True, parents=True)
            temp_fcidump_path = temp_fcidump_dir / fcidump_file.with_suffix(".temp")

            two_body_tensor_sparse = sparse.COO.from_numpy(two_body_tensor)
            one_body_tensor_sparse = sp.sparse.lil_matrix(one_body_tensor)

            output_dict = {
                # Variable: (Description, Default Value)
                "NORB": num_orbitals,
                "NELEC": num_electrons,
                "MS2": two_S,
                "ISYM": 1,
                "ORBSYM": orb_sym,
                # "IPRTIM": ("If 0, print additional CPU timing analysis", -1),
                # "INT": ("Fortran stream from which integrals will be read", 5),
                # "MEMORY": ("Size of workspace array in floating point words", 100000),
                # "CORE": nuc_rep_energy,
                # "MAXIT": ("Maximum number of iterations in Davidson diagonalisation", 25),
                # "THR": (
                #     "Convergence threshold for Davidson diagonalisation (floating point)",
                #     1e-5,
                # ),
                # "THRRES": ("Threshold for printing final CI coefficients (floating point)", 1e-1),
                # "NROOT": ("Number of eigenvalues of Hamiltonian to be found", 1),
            }

            fcidump_io.write_fcidump(
                output_dict=output_dict,
                one_electron_integrals=one_body_tensor_sparse,
                two_electron_integrals=two_body_tensor_sparse,
                core_energy=nuc_rep_energy,
                file_path=temp_fcidump_path,
                real_bool=False,
                verbose=True,
            )

            (
                one_body_tensor_2,
                two_body_tensor_2,
                nuc_rep_energy_2,
                num_orbitals_2,
                num_spin_orbitals_2,
                num_electrons_2,
                two_S_2,
                two_Sz_2,
                orb_sym_2,
                extra_attributes_2,
            ) = dmrg_calc_prepare.load_tensors_from_fcidump(temp_fcidump_path)

            npt.assert_allclose(
                one_body_tensor, one_body_tensor_2, rtol=test_rtol, atol=test_atol
            )
            npt.assert_allclose(
                two_body_tensor, two_body_tensor_2, rtol=test_rtol, atol=test_atol
            )
            npt.assert_allclose(
                nuc_rep_energy, nuc_rep_energy_2, rtol=test_rtol, atol=test_atol
            )
            self.assertEqual(num_orbitals, num_orbitals_2)
            self.assertEqual(num_spin_orbitals, num_spin_orbitals_2)
            self.assertEqual(num_electrons, num_electrons_2)
            self.assertEqual(two_S, two_S_2)
            self.assertEqual(two_Sz, two_Sz_2)
            self.assertEqual(orb_sym, orb_sym_2)

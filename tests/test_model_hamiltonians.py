import logging
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import scipy as sp
import sparse

import dmrghandler.dmrg_calc_prepare as dmrg_calc_prepare
import dmrghandler.fcidump_io as fcidump_io
import dmrghandler.model_hamiltonians as model_hamiltonians

test_rtol = 1e-5
test_atol = 1e-8
test_rsquared_rtol = 1e-1
test_rsquared_atol = 1e-1

# log = logging.getLogger("{Path(config_file_name).stem}")
log = logging.getLogger("dmrghandler")
log.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("model_hamis.log")
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s"
)
fh.setFormatter(formatter)
# add the handlers to the log
log.addHandler(fh)


class TestTightBinding(unittest.TestCase):
    def test_banded_tight_binding_model(self):
        dict_list = [
            {
                "num_orbitals": 3,
                "orbital_energies": [-1.0, 0.0, 1.0],
                "hopping_amplitude": 1.0,
                "bandwidth": 1,
                "num_electrons": 2,
                "two_S": 0,
                "orb_sym": 1,
                "answer": np.array(
                    [
                        [-1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0],
                    ]
                ),
            },
            {
                "num_orbitals": 3,
                "orbital_energies": [-1.0, 0.0, 1.0],
                "hopping_amplitude": 1.0,
                "bandwidth": 2,
                "num_electrons": 2,
                "two_S": 0,
                "orb_sym": 1,
                "answer": np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ),
            },
            {
                "num_orbitals": 5,
                "orbital_energies": [-1.0, 0.0, 1.0, 534, 9.123],
                "hopping_amplitude": 1.0,
                "bandwidth": 3,
                "num_electrons": 3,
                "two_S": 0,
                "orb_sym": 1,
                "answer": np.array(
                    [
                        [-1.0, 1.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 534, 1.0],
                        [0.0, 1.0, 1.0, 1.0, 9.123],
                    ]
                ),
            },
        ]

        for dict in dict_list:
            num_orbitals = dict["num_orbitals"]
            orbital_energies = dict["orbital_energies"]
            hopping_amplitude = dict["hopping_amplitude"]
            bandwidth = dict["bandwidth"]
            num_electrons = dict["num_electrons"]
            two_S = dict["two_S"]
            orb_sym = dict["orb_sym"]

            one_body_tensor = model_hamiltonians.banded_tight_binding_model(
                num_orbitals, orbital_energies, hopping_amplitude, bandwidth
            )

            npt.assert_allclose(
                one_body_tensor, dict["answer"], rtol=test_rtol, atol=test_atol
            )

            # Save the one_body_tensor to a file
            temp_fcidump_path = Path("tests/temp/one_body_tensor.fcidump")
            temp_fcidump_path.parent.mkdir(parents=True, exist_ok=True)

            two_body_tensor_sparse = None
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
                core_energy=0.0,
                file_path=temp_fcidump_path,
                real_bool=False,
                verbose=True,
            )

            # Load the one_body_tensor from the file
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
                np.zeros_like(two_body_tensor_2),
                two_body_tensor_2,
                rtol=test_rtol,
                atol=test_atol,
            )

            # Get the ground state energy directly from the one_body_tensor
            (
                ground_state_energy,
                eigenvalues,
                eigenvectors,
            ) = model_hamiltonians.get_one_body_term_ground_state(
                one_body_tensor, num_electrons
            )

            # Get the ground state energy from DMRG

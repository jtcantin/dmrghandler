"""
This module contains functions for reading and writing FCIDUMP files.
Use these functions cautiously. They are not guaranteed to work with all FCIDUMP file formats.
"""

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

import f90nml
import numpy as np
import scipy as sp
import sparse

AnyPath = None
if TYPE_CHECKING:
    from _typeshed import AnyPath


NAMELIST_DICT = {
    # Variable: (Description, Default Value)
    "NORB": ("Number of orbitals", np.nan),
    "NELEC": ("Number of electrons", np.nan),
    "MS2": ("2S, where S is the spin quantum number", 0),
    "ISYM": ("Spatial symmetry of wavefunction", 1),
    "ORBSYM": ("Spatial symmetries of orbitals, length is NORB", "[1,...,1]"),
    "IPRTIM": ("If 0, print additional CPU timing analysis", -1),
    "INT": ("Fortran stream from which integrals will be read", 5),
    "MEMORY": ("Size of workspace array in floating point words", 100000),
    "CORE": ("Core energy (may also be given in integral file) (floatingpoint)", 0.0),
    "MAXIT": ("Maximum number of iterations in Davidson diagonalisation", 25),
    "THR": (
        "Convergence threshold for Davidson diagonalisation (floating point)",
        1e-5,
    ),
    "THRRES": ("Threshold for printing final CI coefficients (floating point)", 1e-1),
    "NROOT": ("Number of eigenvalues of Hamiltonian to be found", 1),
}


# def read_fcidump(
#     filepath: AnyPath,
#     indicate_defaults: bool = True,
#     real_bool: bool = False,
#     verbose: bool = False,
# ) -> (dict, sp.sparse.lil_matrix, sparse.COO, float):
#     filepath = Path(filepath)
#     raise NotImplementedError("Please use the tools from pyscf to read FCIDUMP files.")
#     # Read the NAMELIST data
#     namelist_dict = f90nml.read(filepath)
#     try:
#         temp_output_dict = namelist_dict["FCI"]
#     except KeyError:
#         raise KeyError("FCI namelist not found in FCIDUMP file.")

#     output_dict = {}
#     for key, value in temp_output_dict.items():
#         output_dict[key.upper()] = value

#     # Read the integral data
#     try:
#         num_orbitals = output_dict["NORB"]
#     except KeyError:
#         raise KeyError("NORB not found in FCIDUMP file.")

#     # one_electron_integrals = np.zeros((num_orbitals, num_orbitals), dtype=np.float64)
#     # two_electron_integrals = np.zeros(
#     #     (num_orbitals, num_orbitals, num_orbitals, num_orbitals), dtype=np.float64
#     # )
#     # Use sparse matrices to save memory
#     one_electron_integrals = sp.sparse.lil_matrix(
#         (num_orbitals, num_orbitals), dtype=np.float64
#     )

#     two_electron_integrals_coord_list = []
#     two_electron_integrals_data_list = []
#     # print(two_electron_integrals.shape)
#     # print(two_electron_integrals)
#     # print(two_electron_integrals[0, 0, 0, 0])
#     # print(two_electron_integrals[0, 0, 1, 0])
#     # input()

#     if "CORE" not in output_dict:
#         core_energy = None
#     else:
#         core_energy = output_dict["CORE"]

#     with open(filepath, "r") as f:
#         lines = f.readlines()
#         # print(lines)
#         hit_end = False

#         num_lines = len(lines)
#         count_check = np.max([num_lines // 100, 1])

#         for line_index, line in enumerate(lines):
#             if verbose:
#                 if line_index % count_check == 0:
#                     print(f"Processing line {line_index+1} of {num_lines}")
#             line = line.strip()
#             if line == "&END" or line == "/":
#                 hit_end = True
#                 continue
#             if not hit_end:
#                 continue
#             if line == "":
#                 continue
#             value, I, J, K, L = line.split()

#             I = int(I)
#             J = int(K)  # J and K are switched in the FCIDUMP file
#             K = int(J)  # J and K are switched in the FCIDUMP file
#             L = int(L)
#             i_ = I - 1
#             j_ = J - 1
#             k_ = K - 1
#             l_ = L - 1
#             # print(line)
#             # print(value, I, J, K, L)

#             if real_bool:
#                 try:
#                     value = float(value)
#                 except ValueError:
#                     raise ValueError(
#                         f"Value {value} could not be converted to a real float."
#                     )
#                 if K != 0:
#                     # Two electron integral
#                     # Record:
#                     # Value i, j, k, l
#                     # Include all 8 permutations for real basis
#                     # two_electron_integrals[i_, j_, k_, l_] = value
#                     # two_electron_integrals[j_, i_, l_, k_] = value
#                     # two_electron_integrals[k_, l_, i_, j_] = value
#                     # two_electron_integrals[l_, k_, j_, i_] = value

#                     # two_electron_integrals[k_, j_, i_, l_] = value
#                     # two_electron_integrals[l_, i_, j_, k_] = value
#                     # two_electron_integrals[i_, l_, k_, j_] = value
#                     # two_electron_integrals[j_, k_, l_, i_] = value

#                     perm_list = [
#                         (i_, j_, k_, l_),
#                         (j_, i_, l_, k_),
#                         (k_, l_, i_, j_),
#                         (l_, k_, j_, i_),
#                         (k_, j_, i_, l_),
#                         (l_, i_, j_, k_),
#                         (i_, l_, k_, j_),
#                         (j_, k_, l_, i_),
#                     ]
#                     for perm in np.unique(perm_list, axis=0).tolist():
#                         two_electron_integrals_coord_list.append(perm)
#                         two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((i_, j_, k_, l_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((j_, i_, l_, k_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((k_, l_, i_, j_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((l_, k_, j_, i_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((k_, j_, i_, l_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((l_, i_, j_, k_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((i_, l_, k_, j_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((j_, k_, l_, i_))
#                     # two_electron_integrals_data_list.append(value)

#                 elif I != 0:
#                     # One electron integral
#                     # Record:
#                     # Value i, j, 0, 0
#                     # Include all 2 permutations for real basis
#                     one_electron_integrals[i_, j_] = value
#                     one_electron_integrals[j_, i_] = value

#                 else:
#                     # Core energy
#                     # Record:
#                     # Value 0, 0, 0, 0
#                     if core_energy is not None:
#                         raise ValueError(
#                             "CORE energy defined multiple times. "
#                             + "CORE energy must be defined only once."
#                         )
#                     else:
#                         core_energy = value
#             else:
#                 # raise NotImplementedError("Complex integrals are not yet supported.")
#                 try:
#                     value = float(value)
#                 except ValueError:
#                     raise ValueError(
#                         f"Value {value} could not be converted to a real float."
#                     )

#                 if K != 0:
#                     # Two electron integral
#                     # Record:
#                     # Value i, j, k, l
#                     # Include all 4 permutations for complex basis
#                     # two_electron_integrals[i_, j_, k_, l_] = value
#                     # two_electron_integrals[j_, i_, l_, k_] = value
#                     # two_electron_integrals[k_, l_, i_, j_] = np.conj(value)
#                     # two_electron_integrals[l_, k_, j_, i_] = np.conj(value)

#                     perm_list = [
#                         (i_, j_, k_, l_),
#                         (j_, i_, l_, k_),
#                         (k_, l_, i_, j_),
#                         (l_, k_, j_, i_),
#                     ]
#                     for perm in np.unique(perm_list, axis=0).tolist():
#                         two_electron_integrals_coord_list.append(perm)
#                         if np.imag(value) != 0.0:
#                             raise ValueError(
#                                 "Complex integral values are not yet supported."
#                             )
#                         two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((i_, j_, k_, l_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((j_, i_, l_, k_))
#                     # two_electron_integrals_data_list.append(value)

#                     # two_electron_integrals_coord_list.append((k_, l_, i_, j_))
#                     # two_electron_integrals_data_list.append(np.conj(value))

#                     # two_electron_integrals_coord_list.append((l_, k_, j_, i_))
#                     # two_electron_integrals_data_list.append(np.conj(value))

#                 elif I != 0:
#                     # One electron integral
#                     # Record:
#                     # Value i, j, 0, 0
#                     # Include all 2 permutations for complex basis
#                     one_electron_integrals[i_, j_] = value
#                     one_electron_integrals[j_, i_] = np.conj(value)

#                 else:
#                     # Core energy
#                     # Record:
#                     # Value 0, 0, 0, 0
#                     if core_energy is not None:
#                         raise ValueError(
#                             "CORE energy defined multiple times. "
#                             + "CORE energy must be defined only once."
#                         )
#                     else:
#                         core_energy = value

#     coords_array = np.array(two_electron_integrals_coord_list)
#     two_electron_integrals = sparse.COO(
#         coords=coords_array.T,
#         data=two_electron_integrals_data_list,
#         shape=(num_orbitals, num_orbitals, num_orbitals, num_orbitals),
#         # dtype=np.float64,
#     )
#     two_electron_integrals = -0.5 * two_electron_integrals
#     # print(two_electron_integrals.shape)
#     # print(coords_array)
#     # print(two_electron_integrals_data_list)
#     # print(two_electron_integrals)
#     # input()

#     if core_energy is not None:
#         output_dict["CORE"] = core_energy

#     # Add any needed default values
#     defaulted_values = []
#     for key, value in NAMELIST_DICT.items():
#         if key not in output_dict:
#             not_present_key = key

#         else:
#             continue

#         if not_present_key == "NORB" or not_present_key == "NELEC":
#             # NORB and NELEC must be present in the FCIDUMP file
#             raise KeyError(
#                 f"Key {not_present_key} not found in FCIDUMP file."
#                 + "This key must be present in the FCIDUMP file."
#             )

#         if not_present_key == "ORBSYM":
#             # ORBSYM is a special case, because it is a list of integers
#             # and not a single integer
#             num_orbitals = output_dict["NORB"]
#             orbsym_array = np.ones(num_orbitals)
#             output_dict["ORBSYM"] = orbsym_array.to_list()
#             defaulted_values.append(not_present_key)

#         else:
#             # Set the default value
#             output_dict[not_present_key] = value[1]
#             defaulted_values.append(not_present_key)

#     if indicate_defaults:
#         print("Defaults set for the following variables:")
#         print(f"      VARIABLE\t:\tDEF VALUE\tDESCRIPTION")
#         for key in defaulted_values:
#             print(f"      {key}\t:\t{NAMELIST_DICT[key][1]}\t\t{NAMELIST_DICT[key][0]}")

#     return output_dict, one_electron_integrals, two_electron_integrals, core_energy


def write_fcidump(
    output_dict: dict,
    one_electron_integrals: Union[sp.sparse.lil_matrix, np.ndarray],
    two_electron_integrals: sparse.COO,
    core_energy: float,
    file_path: AnyPath,
    real_bool: bool = False,
    verbose: bool = False,
    # put_all_two_elec_int_non_zero: bool = False,
) -> AnyPath:
    """Based on Comp. Phys. Commun. 54 75 (1989), https://doi.org/10.1016/0010-4655(89)90033-7
    Note that the two electron integrals are stored in the FCIDUMP file in the chemists' notation,
    with the 1/2 factor convention, i.e. H = h_ij a^+_i a_j + 1/2 h_ijkl a^+_i a^+_k a_l a_j.
    This is also the convention used in the PySCF package.
    """
    file_path = Path(file_path)
    if (one_electron_integrals is not None) and (
        one_electron_integrals.dtype != np.float64
    ):
        raise TypeError(
            "One electron integrals must currently be of type np.float64."
            + "Complex values are not yet supported."
        )
    # if two_electron_integrals.dtype != np.float64:
    #     raise TypeError(
    #         "Two electron integrals must currently be of type np.float64."
    #         + "Complex values are not yet supported."
    #     )
    if (core_energy is not None) and type(core_energy) != float:
        raise TypeError("Core energy must currently be a float.")

    # Check that required variables are present in the output dictionary
    required_variables = ["NORB", "NELEC"]
    for variable in required_variables:
        if variable not in output_dict:
            raise KeyError(
                f"Variable {variable} not found in output dictionary."
                + "This variable must be present in the output dictionary."
            )

    # Check that all dictionary keys are valid
    for key in output_dict:
        if key not in NAMELIST_DICT:
            raise KeyError(
                f"Variable {key} not found in pre-defined NAMELIST_DICT."
                + "Only standard NAMELIST keys can be used."
            )

    # Check that the dimensions of the integrals match the number of orbitals
    num_orbitals = output_dict["NORB"]
    if one_electron_integrals.shape != (num_orbitals, num_orbitals):
        raise ValueError(
            "One electron integrals must be a square matrix with dimensions NORB x NORB."
        )
    if (two_electron_integrals is not None) and (
        two_electron_integrals.shape
        != (
            num_orbitals,
            num_orbitals,
            num_orbitals,
            num_orbitals,
        )
    ):
        raise ValueError(
            "Two electron integrals must be a 4D tensor with dimensions NORB x NORB x NORB x NORB."
        )

    # Check that the core energy is not defined in the dictionary
    if "CORE" in output_dict:
        raise ValueError(
            "CORE energy must not be defined in the output dictionary."
            + "Please only use the function argument to define it."
        )

    # Create namelist
    # proto_namelist_dict = {"FCI": output_dict}
    namelist_obj = f90nml.namelist.Namelist(FCI=output_dict)
    namelist_obj.uppercase = True
    namelist_obj.indent = "  "
    namelist_obj.end_comma = True
    # print(namelist_obj)

    # Write namelist to memory, then replace " = " with "=" so block2 does not crash when reading
    namelist_temp = StringIO()
    namelist_obj.write(namelist_temp, force=True, sort=False)

    namelist_string = namelist_temp.getvalue()
    # print(namelist_string)
    namelist_string = namelist_string.replace(" = ", "=")
    # Replace / with &END
    namelist_string = namelist_string.replace("/", "&END")
    # Ensure that the first line is &FCI and a value
    namelist_string = namelist_string.replace("&FCI\n", "&FCI ")

    # print("----")
    # print(namelist_string)
    # input()

    # Write values to file
    with open(file_path, "w") as f:
        f.write(namelist_string)
        # In the below code, use str(value) to convert to string at full float precision

        # Two electron integrals

        # if not put_all_two_elec_int_non_zero:
        # Get the indices to check for non-zero values, removing permutation symmetry
        indices_to_check = []
        discard_index_list = []
        string_to_write = ""

        if two_electron_integrals is not None:
            # Loop through all non-zero elements
            # A possible optimization is to make a list of all non-zero elements (or a dict)
            # and then loop through that list, removing all future permutations of that element
            # from the list (rather than storing elements to ignore and checking against that list).
            # A dict may be better, as it would allow for easy lookup (and thus removal),
            # as all elements are unique. Not sure how dictionary build would slow down the code though.
            coords = two_electron_integrals.coords
            data = two_electron_integrals.data
            num_elements = len(data)
            count_check = num_elements // 100
            count = 0
            for iiter, jiter, kiter, liter in zip(
                coords[0], coords[1], coords[2], coords[3]
            ):
                if verbose:
                    if count % count_check == 0:
                        print(
                            f"Processing index {count+1} of {num_elements} of Two Electron Integrals"
                        )
                        print(f"discard_index_list length: {len(discard_index_list)}")
                indices = (iiter, jiter, kiter, liter)
                # Check if indices is already in discard list
                # If it is, continue
                if indices in discard_index_list:
                    discard_index_list.remove(indices)
                    count += 1
                    continue

                # Otherwise, add indices to indices_to_check
                indices_to_check.append(indices)
                # Write two electron integral to file
                value = data[count]
                # f.write(f"{str(value)} {iiter+1} {kiter+1} {jiter+1} {liter+1}\n") # J and K are switched in the FCIDUMP file
                # string_to_write += f" {str(value)}    {iiter+1}    {kiter+1}    {jiter+1}    {liter+1}\n"  # J and K are switched in the FCIDUMP file; also factor of -2 diff
                string_to_write += f" {str(value)}    {iiter+1}    {jiter+1}    {kiter+1}    {liter+1}\n"

                # Generate all remaining 7 permutations of indices
                # Ensuring that we don't add the current indices
                if real_bool:
                    permutations = [
                        # (iiter, jiter, kiter, liter), This one we keep
                        (jiter, iiter, liter, kiter),
                        (kiter, liter, iiter, jiter),
                        (liter, kiter, jiter, iiter),
                        (kiter, jiter, iiter, liter),
                        (liter, iiter, jiter, kiter),
                        (iiter, liter, kiter, jiter),
                        (jiter, kiter, liter, iiter),
                    ]
                else:
                    permutations = [
                        # (iiter, jiter, kiter, liter), This one we keep
                        (jiter, iiter, liter, kiter),
                        (kiter, liter, iiter, jiter),
                        (liter, kiter, jiter, iiter),
                        # (kiter, jiter, iiter, liter),
                        # (liter, iiter, jiter, kiter),
                        # (iiter, liter, kiter, jiter),
                        # (jiter, kiter, liter, iiter),
                    ]
                permutations = np.unique(permutations, axis=0).tolist()
                # Convert to tuples
                permutations = [tuple(permutation) for permutation in permutations]
                # Remove indices from permutations
                if indices in permutations:
                    permutations.remove(indices)
                ###################################
                # Add permutations to discard list
                for permutation in permutations:
                    discard_index_list.append(permutation)

                count += 1

            # # Loop through all indices
            # num_elements = num_orbitals**4
            # count = 0
            # for iiter in range(num_orbitals):
            #     for jiter in range(num_orbitals):
            #         for kiter in range(num_orbitals):
            #             for liter in range(num_orbitals):
            #                 if verbose:
            #                     count += 1
            #                     if count % 1000 == 0:
            #                         print(
            #                             f"Processing index {count} of {num_elements} of Two Electron Integrals"
            #                         )
            #                 indices = (iiter, jiter, kiter, liter)
            #                 # Check if indices is already in discard list
            #                 # If it is, continue
            #                 if indices in discard_index_list:
            #                     continue

            #                 # Otherwise, add indices to indices_to_check
            #                 indices_to_check.append(indices)
            #                 # Generate all remaining 7 permutations of indices
            #                 # Ensuring that we don't add the current indices
            #                 if real_bool:
            #                     permutations = [
            #                         # (iiter, jiter, kiter, liter), This one we keep
            #                         (jiter, iiter, liter, kiter),
            #                         (kiter, liter, iiter, jiter),
            #                         (liter, kiter, jiter, iiter),
            #                         (kiter, jiter, iiter, liter),
            #                         (liter, iiter, jiter, kiter),
            #                         (iiter, liter, kiter, jiter),
            #                         (jiter, kiter, liter, iiter),
            #                     ]
            #                 else:
            #                     permutations = [
            #                         # (iiter, jiter, kiter, liter), This one we keep
            #                         (jiter, iiter, liter, kiter),
            #                         (kiter, liter, iiter, jiter),
            #                         (liter, kiter, jiter, iiter),
            #                         # (kiter, jiter, iiter, liter),
            #                         # (liter, iiter, jiter, kiter),
            #                         # (iiter, liter, kiter, jiter),
            #                         # (jiter, kiter, liter, iiter),
            #                     ]
            #                 # print("Indices")
            #                 # print(indices)
            #                 # print(permutations)
            #                 permutations = np.unique(permutations, axis=0).tolist()
            #                 # Convert to tuples
            #                 permutations = [
            #                     tuple(permutation) for permutation in permutations
            #                 ]
            #                 # Remove indices from permutations
            #                 if indices in permutations:
            #                     permutations.remove(indices)
            #                 ###################################
            #                 # Add permutations to discard list
            #                 for permutation in permutations:
            #                     discard_index_list.append(permutation)

            #                 # print(permutations)
            #                 # print(discard_index_list)
            # # else:

            # for iiter, jiter, kiter, liter in indices_to_check:
            #     value = two_electron_integrals[iiter, jiter, kiter, liter]
            #     if value == 0.0:
            #         continue
            #     # Write two electron integrals to file
            #     f.write(f"{str(value)} {iiter+1} {jiter+1} {kiter+1} {liter+1}\n")

        if one_electron_integrals is not None:
            # One electron integrals
            # Get the indices, removing permutation symmetry
            indices_to_check = []
            for iiter in range(num_orbitals):
                for jiter in range(iiter, num_orbitals):
                    indices_to_check.append((iiter, jiter))

            for iiter, jiter in indices_to_check:
                value = one_electron_integrals[iiter, jiter]
                if value == 0.0:
                    continue
                # Write one electron integrals to file
                # f.write(f"{str(value)} {iiter+1} {jiter+1} 0 0\n")
                string_to_write += (
                    f" {str(value)}    {iiter+1}    {jiter+1}    0    0\n"
                )

        if core_energy is not None:
            # Write core energy to file
            # f.write(f"{str(core_energy)} 0 0 0 0\n")
            string_to_write += f" {str(core_energy)}    0    0    0    0\n"

        f.write(string_to_write)

    return file_path


if __name__ == "__main__":
    # file_to_test_with = "fcidump.2_co2_6-311++G__"  # 6 KB # Works
    # file_to_test_with = "fcidump.7_melact_6-311++G__"  # 45 KB # Works
    file_to_test_with = "fcidump.8_melact_6-311++G__"  # 235 KB # Works
    # file_to_test_with = "fcidump.0_ru_macho_{'Ru'_ 'cc-pVTZ-PP', 'default'_ '6-311++G__'}"  # 397 KB # Works, ~20seconds
    # file_to_test_with = "fcidump.30_4a_{'Mo'_ 'def2-SVP', 'I'_ 'def2-SVP', 'Cl'_ 'def2-SVP', 'default'_ '6-311+G(d,p)'}"  # 1,919 KB Works, ~ 7 min
    # file_to_test_with = (
    #     "fcidump.6_ts_ru_macho_melact_{'Ru'_ 'cc-pVTZ-PP', 'default'_ '6-311++G__'}" #29,299 KB # Too slow to test, 12hrs+
    # )
    (
        output_dict,
        one_electron_integrals,
        two_electron_integrals,
        core_energy,
    ) = read_fcidump(
        file_to_test_with, indicate_defaults=True, real_bool=False, verbose=True
    )
    print(output_dict)
    print(one_electron_integrals.shape)
    print(two_electron_integrals.shape)
    print(core_energy)

    # print(one_electron_integrals)
    # print(two_electron_integrals)

    print("Writing out")
    temp_output_dict = output_dict.copy()
    del temp_output_dict["CORE"]
    del temp_output_dict["INT"]
    del temp_output_dict["MEMORY"]
    del temp_output_dict["IPRTIM"]
    del temp_output_dict["MAXIT"]
    del temp_output_dict["THR"]
    del temp_output_dict["THRRES"]
    del temp_output_dict["NROOT"]
    output_file_path = f"test_fcidump_{file_to_test_with}"
    write_fcidump(
        temp_output_dict,
        one_electron_integrals,
        two_electron_integrals,
        core_energy,
        output_file_path,
        real_bool=False,
        verbose=True,
    )
    # output_dict["CORE"] = core_energy
    print("Reading back in")
    (
        output_dict_new,
        one_electron_integrals_new,
        two_electron_integrals_new,
        core_energy_new,
    ) = read_fcidump(
        output_file_path, indicate_defaults=True, real_bool=False, verbose=True
    )

    print(output_dict_new == output_dict)
    # print(np.allclose(one_electron_integrals_new, one_electron_integrals, atol=1e-13))
    # print(np.allclose(two_electron_integrals_new, two_electron_integrals, atol=1e-13))
    print(np.max(np.abs(one_electron_integrals_new - one_electron_integrals)))
    print(np.max(np.abs(two_electron_integrals_new - two_electron_integrals)))
    print(np.isclose(core_energy_new, core_energy, atol=1e-13))

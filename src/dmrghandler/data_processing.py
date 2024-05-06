from pathlib import Path

import h5py
import numpy as np
import openpyxl as px
import openpyxl.chart as px_chart


def get_data_from_incomplete_processing(data_file):
    # Get DMRG energy, bond dimensions, and truncation error for each loop
    dmrg_energies = []
    bond_dimensions = []
    discarded_weights = []
    loop_cpu_times_s = []
    loop_wall_times_s = []
    with h5py.File(data_file, "r") as file_obj:
        # Load for first preloop calculation
        preloop = file_obj["first_preloop_calc"]["dmrg_results"]
        dmrg_energy = float(preloop["dmrg_ground_state_energy"][()])
        bond_dimension = int(preloop["sweep_bond_dims"][()][-1])
        discarded_weight = float(preloop["sweep_max_discarded_weight"][()][-1])
        loop_cpu_time_s = float(preloop["cpu_single_qchem_dmrg_calc_time_s"][()])
        loop_wall_time_s = float(preloop["wall_single_qchem_dmrg_calc_time_s"][()])
        dmrg_energies.append(dmrg_energy)
        bond_dimensions.append(bond_dimension)
        discarded_weights.append(discarded_weight)
        loop_cpu_times_s.append(loop_cpu_time_s)
        loop_wall_times_s.append(loop_wall_time_s)

        # Load for second preloop calculation
        preloop = file_obj["second_preloop_calc"]["dmrg_results"]
        dmrg_energy = float(preloop["dmrg_ground_state_energy"][()])
        bond_dimension = int(preloop["sweep_bond_dims"][()][-1])
        discarded_weight = float(preloop["sweep_max_discarded_weight"][()][-1])
        loop_cpu_time_s = float(preloop["cpu_single_qchem_dmrg_calc_time_s"][()])
        loop_wall_time_s = float(preloop["wall_single_qchem_dmrg_calc_time_s"][()])
        dmrg_energies.append(dmrg_energy)
        bond_dimensions.append(bond_dimension)
        discarded_weights.append(discarded_weight)
        loop_cpu_times_s.append(loop_cpu_time_s)
        loop_wall_times_s.append(loop_wall_time_s)

        # Load for main loop calculations
        for i in range(1, 1000):
            group_name = f"dmrg_loop_{i:03d}"
            if group_name not in file_obj:
                last_loop = i - 1
                print(f"Last loop included = {last_loop}")

                break
            loop = file_obj[group_name]["dmrg_results"]
            dmrg_energy = float(loop["dmrg_ground_state_energy"][()])
            bond_dimension = int(loop["sweep_bond_dims"][()][-1])
            discarded_weight = float(loop["sweep_max_discarded_weight"][()][-1])
            loop_cpu_time_s = float(loop["cpu_single_qchem_dmrg_calc_time_s"][()])
            loop_wall_time_s = float(loop["wall_single_qchem_dmrg_calc_time_s"][()])
            dmrg_energies.append(dmrg_energy)
            bond_dimensions.append(bond_dimension)
            discarded_weights.append(discarded_weight)
            loop_cpu_times_s.append(loop_cpu_time_s)
            loop_wall_times_s.append(loop_wall_time_s)

        if "final_dmrg_results" in file_obj:
            print("Processed results available")

            final = file_obj["final_dmrg_results"]
            processed_dmrg_energies = final["past_energies_dmrg"][()]
            processed_bond_dimensions = final["bond_dims_used"][()]
            processed_discarded_weights = final["past_discarded_weights"][()]

            print("Checking that processed results match raw results.")
            assert np.allclose(dmrg_energies, processed_dmrg_energies)
            assert np.allclose(bond_dimensions, processed_bond_dimensions)
            assert np.allclose(discarded_weights, processed_discarded_weights)

    num_loops = last_loop
    num_dmrg_calculations = len(dmrg_energies)
    assert num_loops == num_dmrg_calculations - 2
    return (
        dmrg_energies,
        bond_dimensions,
        discarded_weights,
        num_loops,
        num_dmrg_calculations,
        loop_cpu_times_s,
        loop_wall_times_s,
    )


def process_time(time_s):
    if time_s > 3600 * 24:
        return f"{time_s / 3600 / 24:.2f} d"
    elif time_s > 3600:
        return f"{time_s / 3600:.2f} h"
    elif time_s > 60:
        return f"{time_s / 60:.2f} m"
    else:
        return f"{time_s:.2f} s"


def add_dmrg_data_chart(
    x_vals_ref,
    y_vals_ref,
    coordinates,
    worksheet,
    x_title="none",
    y_title="none",
    series_name=None,
    chart=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    marker_symbol="circle",
    marker_fill="156082",
    order=0,
):
    new_chart = False
    if chart is None:
        chart = px_chart.ScatterChart()
        new_chart = True
    chart.style = None
    chart.x_axis.title = x_title
    chart.y_axis.title = y_title

    series = px_chart.Series(
        values=y_vals_ref,
        xvalues=x_vals_ref,
        title=series_name,
    )
    series.marker.symbol = marker_symbol
    series.marker.graphicalProperties.solidFill = marker_fill  # Marker filling
    series.marker.graphicalProperties.line.solidFill = marker_fill  # Marker outline
    series.marker.graphicalProperties.shadow = "None"  # Marker shadow

    series.graphicalProperties.line.noFill = True

    chart.series.append(series)

    chart.x_axis.scaling.min = x_min
    chart.x_axis.scaling.max = x_max

    chart.y_axis.scaling.min = y_min
    chart.y_axis.scaling.max = y_max

    chart.x_axis.crosses = "min"
    chart.y_axis.crosses = "min"

    if series_name is None:
        chart.legend = None

    if new_chart:
        worksheet.add_chart(chart, coordinates)

    return chart


def add_dmrg_processing_basic(
    workbook,
    dmrg_energies,
    bond_dimensions,
    discarded_weights,
    loop_cpu_times_s,
    loop_wall_times_s,
    data_dict,
):
    # Get worksheet name from fcidump name
    # take after . and before {
    fcidump_name = data_dict["fcidump"]
    worksheet_name = fcidump_name.split(".")[1].split("{")[0]

    # Create a new worksheet
    ws = workbook.create_sheet(title=worksheet_name)

    # Add header and corresponding data
    ws.append(list(data_dict.keys()))
    ws.append(list(data_dict.values()))

    ws["J3"] = "Estimated Final Energy"
    ws["M3"] = "DON'T USE THIS FOR COMPARISONS, USE E_DMRG INSTEAD (CELL S3)"
    ws["J4"] = "Finish Reason:"
    ws["J5"] = "Loop Results"

    ws["S3"] = "E_DMRG"
    ws["U3"] = (
        "USE THIS E_DMRG FOR COMPARISONS, Estimated Final Energy IS NOT YET RELIABLE"
    )
    ws["S4"] = "DW Linear Slope"
    ws["S5"] = "DW Linear Intercept"

    data_header_dict = {
        "K": "Loop",
        "L": "DMRG Energy",
        "M": "Bond Dimension",
        "N": "Discarded Weights",
        "O": "1/BD",
        "P": "log10 1/BD",
        "Q": "ln DW",
        "R": "Abs Rel Energy",
        "S": "ln AR Energy",
        "T": "Pred E_DMRG",
        "U": "Pred ln AR Energy",
        "V": "CPU Time (s)",
        "W": "Wall Time (s)",
        "X": "CPU Time (processed)",
        "Y": "Wall Time (processed)",
    }
    ws.append(data_header_dict)

    # Fill loop column
    ws["K7"] = -1
    ws["K8"] = 0
    for i in range(len(dmrg_energies) - 2):
        ws[f"K{7+2+i}"] = i + 1

    # Fill DMRG energy column
    for i, energy in enumerate(dmrg_energies):
        ws[f"L{7+i}"] = energy
        ws[f"L{7+i}"].number_format = "0.0000"

    # Fill bond dimension column
    for i, bond_dim in enumerate(bond_dimensions):
        ws[f"M{7+i}"] = bond_dim

    # Fill discarded weights column
    for i, discarded_weight in enumerate(discarded_weights):
        ws[f"N{7+i}"] = discarded_weight
        ws[f"N{7+i}"].number_format = "0.00E+00"

    # Fill 1/BD column with formula
    for i in range(len(bond_dimensions)):
        ws[f"O{7+i}"] = f"=1/M{7+i}"
        ws[f"O{7+i}"].number_format = "0.00E+00"

    # Fill log10 1/BD column with formula
    for i in range(len(bond_dimensions)):
        ws[f"P{7+i}"] = f"=LOG10(O{7+i})"
        ws[f"P{7+i}"].number_format = "0.0000"

    # Fill ln DW column with formula
    for i in range(len(discarded_weights)):
        ws[f"Q{7+i}"] = f"=LN(N{7+i})"
        ws[f"Q{7+i}"].number_format = "0.0000"

    # Fill Abs Rel Energy column with formula
    for i in range(len(dmrg_energies)):
        ws[f"R{7+i}"] = f"=ABS((L{7+i}-$L$3)/$L$3)"
        ws[f"R{7+i}"].number_format = "0.00E+00"

    # Fill ln AR Energy column with formula
    for i in range(len(dmrg_energies)):
        ws[f"S{7+i}"] = f"=LN(R{7+i})"
        ws[f"S{7+i}"].number_format = "0.0000"

    # Fill CPU Time column
    for i, cpu_time in enumerate(loop_cpu_times_s):
        ws[f"V{7+i}"] = cpu_time
        ws[f"V{7+i}"].number_format = "0.00"

    # Fill Wall Time column
    for i, wall_time in enumerate(loop_wall_times_s):
        ws[f"W{7+i}"] = wall_time
        ws[f"W{7+i}"].number_format = "0.00"

    # Fill processed CPU Time column
    for i, cpu_time in enumerate(loop_cpu_times_s):
        ws[f"X{7+i}"] = process_time(cpu_time)

    # Fill processed Wall Time column
    for i, wall_time in enumerate(loop_wall_times_s):
        ws[f"Y{7+i}"] = process_time(wall_time)

    last_data_row = 7 + len(dmrg_energies) - 1
    # E_DMRG formula
    ws["T3"] = f"=MIN(L7:L{last_data_row})"
    ws["T3"].number_format = "0.0000"

    # linear slope and intercept
    ws["T4"] = f"=SLOPE(S7:S{last_data_row},Q7:Q{last_data_row})"
    ws["T4"].number_format = "0.0000"

    ws["T5"] = f"=INTERCEPT(S7:S{last_data_row},Q7:Q{last_data_row})"
    ws["T5"].number_format = "0.0000"

    # Fill Pred E_DMRG column with formula
    for i in range(len(dmrg_energies)):
        ws[f"T{7+i}"] = f"=ABS($L$3)*EXP($T$5)*N{7+i}^$T$4+($L$3)"
        ws[f"T{7+i}"].number_format = "0.0000"

    # Fill Pred ln AR Energy column with formula
    for i in range(len(dmrg_energies)):
        ws[f"U{7+i}"] = f"=$T$4*Q{7+i}+$T$5"
        ws[f"U{7+i}"].number_format = "0.0000"

    # Put in the estimated final energy
    ws["L3"] = dmrg_energies[-1] - 1e-3
    ws["L3"].number_format = "0.000000"

    E_DMRG_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!L7:L{last_data_row}",
    )

    bond_dim_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!M7:M{last_data_row}",
    )

    chart_row_start = 5

    add_dmrg_data_chart(
        x_vals_ref=bond_dim_ref,
        y_vals_ref=E_DMRG_ref,
        coordinates=f"A{chart_row_start}",
        worksheet=ws,
        x_title="Bond Dimension",
        y_title="DMRG Energy (Hartree)",
        series_name=None,
        chart=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )

    inv_bond_dim_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!O7:O{last_data_row}",
    )
    chart_row_start += 15
    add_dmrg_data_chart(
        x_vals_ref=inv_bond_dim_ref,
        y_vals_ref=E_DMRG_ref,
        coordinates=f"A{chart_row_start}",
        worksheet=ws,
        x_title="1/(Bond Dimension)",
        y_title="DMRG Energy (Hartree)",
        series_name=None,
        chart=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )

    log_inv_bond_dim_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!P7:P{last_data_row}",
    )
    chart_row_start += 15
    add_dmrg_data_chart(
        x_vals_ref=log_inv_bond_dim_ref,
        y_vals_ref=E_DMRG_ref,
        coordinates=f"A{chart_row_start}",
        worksheet=ws,
        x_title="log10(1/(Bond Dimension))",
        y_title="DMRG Energy (Hartree)",
        series_name=None,
        chart=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )

    discarded_weight_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!N7:N{last_data_row}",
    )
    E_DMRG_pred_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!T7:T{last_data_row}",
    )
    chart_row_start += 15
    dw_e_dmrg_chart = add_dmrg_data_chart(
        x_vals_ref=discarded_weight_ref,
        y_vals_ref=E_DMRG_pred_ref,
        coordinates=f"A{chart_row_start}",
        worksheet=ws,
        x_title="Discarded Weight",
        y_title="DMRG Energy (Hartree)",
        series_name="Prediction",
        chart=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        marker_symbol="square",
        marker_fill="00B050",
    )

    add_dmrg_data_chart(
        x_vals_ref=discarded_weight_ref,
        y_vals_ref=E_DMRG_ref,
        coordinates=f"A{chart_row_start}",
        worksheet=ws,
        x_title="Discarded Weight",
        y_title="DMRG Energy (Hartree)",
        series_name="Calculation",
        chart=dw_e_dmrg_chart,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )

    ln_discarded_weight_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!Q7:Q{last_data_row}",
    )
    ln_AR_Energy_pred_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!U7:U{last_data_row}",
    )
    ln_AR_Energy_ref = px.chart.Reference(
        worksheet=ws,
        range_string=f"{ws.title}!S7:S{last_data_row}",
    )
    chart_row_start += 15
    dw_e_dmrg_chart = add_dmrg_data_chart(
        x_vals_ref=ln_discarded_weight_ref,
        y_vals_ref=ln_AR_Energy_pred_ref,
        coordinates=f"A{chart_row_start}",
        worksheet=ws,
        x_title="ln(Discarded Weight)",
        y_title="ln(Abs Relative Energy)",
        series_name="Prediction",
        chart=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        marker_symbol="square",
        marker_fill="00B050",
    )

    add_dmrg_data_chart(
        x_vals_ref=ln_discarded_weight_ref,
        y_vals_ref=ln_AR_Energy_ref,
        coordinates=f"A{chart_row_start}",
        worksheet=ws,
        x_title="ln(Discarded Weight)",
        y_title="ln(Abs Relative Energy)",
        series_name="Calculation",
        chart=dw_e_dmrg_chart,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )


def setup_workbook(data_file_path, data_dict_list, workbook, csv_storage_path="./"):
    for data_dict in data_dict_list:
        data_file = (
            data_file_path / Path(data_dict["Calc UUID"]) / Path("dmrg_results.hdf5")
        )
        # Get DMRG energy, bond dimensions, and truncation error for each loop
        (
            dmrg_energies,
            bond_dimensions,
            discarded_weights,
            num_loops,
            num_dmrg_calculations,
            loop_cpu_times_s,
            loop_wall_times_s,
        ) = get_data_from_incomplete_processing(data_file)

        if "Calc UUID Small BD" in data_dict:
            data_file_small_bd = (
                data_file_path
                / Path(data_dict["Calc UUID Small BD"])
                / Path("dmrg_results.hdf5")
            )
            (
                dmrg_energies_small_bd,
                bond_dimensions_small_bd,
                discarded_weights_small_bd,
                num_loops_small_bd,
                num_dmrg_calculations_small_bd,
                loop_cpu_times_s_small_bd,
                loop_wall_times_s_small_bd,
            ) = get_data_from_incomplete_processing(data_file_small_bd)
            dmrg_energies = np.hstack((dmrg_energies_small_bd, dmrg_energies))
            bond_dimensions = np.hstack((bond_dimensions_small_bd, bond_dimensions))
            discarded_weights = np.hstack(
                (discarded_weights_small_bd, discarded_weights)
            )
            num_loops = num_loops_small_bd + num_loops
            num_dmrg_calculations = (
                num_dmrg_calculations_small_bd + num_dmrg_calculations
            )
            loop_wall_times_s = np.hstack(
                (loop_wall_times_s_small_bd, loop_wall_times_s)
            )
            loop_cpu_times_s = np.hstack((loop_cpu_times_s_small_bd, loop_cpu_times_s))

        add_dmrg_processing_basic(
            workbook=workbook,
            dmrg_energies=dmrg_energies,
            bond_dimensions=bond_dimensions,
            discarded_weights=discarded_weights,
            loop_cpu_times_s=loop_cpu_times_s,
            loop_wall_times_s=loop_wall_times_s,
            data_dict=data_dict,
        )

        # Save performance metrics to csv
        csv_storage_path = Path(csv_storage_path)
        csv_storage_path.mkdir(parents=True, exist_ok=True)
        fcidump_name = data_dict["fcidump"]
        csv_filename = Path(fcidump_name + ".csv")
        csv_data_array = np.vstack(
            [
                dmrg_energies,
                bond_dimensions,
                discarded_weights,
                loop_cpu_times_s,
                loop_wall_times_s,
            ]
        )
        csv_data_array = csv_data_array.T
        header = "DMRG Energy, Bond Dimension, Discarded Weights, CPU Time (s), Wall Time (s)"
        np.savetxt(
            csv_storage_path / csv_filename,
            csv_data_array,
            fmt="%.18e",
            delimiter=",",
            newline="\n",
            header=header,
            footer="",
            comments="# ",
            encoding=None,
        )

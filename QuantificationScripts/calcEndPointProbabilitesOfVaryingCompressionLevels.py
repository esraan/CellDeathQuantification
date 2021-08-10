# todo: perform for all files and select apoptosis and ferroptosis models to show.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from global_parameters import *
from Visualization import *
from NucleationAndPropagationMeasurements import *


def compare_single_file_compressed_time_resolution_endpoint_readouts(file_name: str,
                                                                     compression_range_to_compare: Tuple[int, int],
                                                                     **kwargs) -> Tuple[np.array, np.array, str]:
    """
    extracts and aggregates the temporal resolution compressed versions of the file and compares
    them using a simple scatter plot (when the kwargs.visualize argument is set to true - defaultive).

    :param file_name: str, the name of the experiment to compare (including .csv).
    :param compression_range_to_compare: tuple [int, int]
    :param kwargs:
    :return:
    """
    visualize_flag = kwargs.get('visualize', True)
    only_recent_flag = kwargs.get('only_recent_death_flag_for_neighbors_calc', RECENT_DEATH_ONLY_FLAG)

    compression_factor_range_values = np.arange(compression_range_to_compare[0], compression_range_to_compare[1], 1)

    # get original time resolution and treatment
    org_file_full_path = os.sep.join([NON_COMPRESSED_FILE_MAIN_DIR, file_name])
    exp_treatment, org_explicit_temporal_resolution = \
        get_exp_treatment_type_and_temporal_resolution(org_file_full_path.split(os.sep)[-1],
                                                       compressed_flag=False)

    endpoint_readouts_values_p_nuc = list()
    endpoint_readouts_values_p_prop = list()

    # get none compressed endpoint readouts
    p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
    all_frames_nucleators_mask, all_frames_propagators_mask, accumulated_death_fraction_by_time = \
        calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_provided_temporal_resolution(
            single_exp_full_path=org_file_full_path,
            temporal_resolution=org_explicit_temporal_resolution,
            **kwargs)

    endpoint_readouts_values_p_nuc.append(p_nuc_global)
    endpoint_readouts_values_p_prop.append(p_prop_global)

    for compression_factor in compression_factor_range_values:
        # create the path for the compressed file
        compressed_file_path = os.sep.join([COMPRESSED_FILE_MAIN_DIR, f'x{compression_factor}', file_name])
        p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
        all_frames_nucleators_mask, all_frames_propagators_mask, accumulated_death_fraction_by_time = \
            calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_provided_temporal_resolution(
                single_exp_full_path=compressed_file_path,
                temporal_resolution=org_explicit_temporal_resolution * compression_factor,
                **kwargs)

        endpoint_readouts_values_p_nuc.append(p_nuc_global)
        endpoint_readouts_values_p_prop.append(p_prop_global)

    if visualize_flag:
        temporal_resolution_axis = compression_factor_range_values * org_explicit_temporal_resolution
        temporal_resolution_axis = np.concatenate([[org_explicit_temporal_resolution], temporal_resolution_axis])
        plot_endpoint_readout_for_compressed_temporal_resolution(temporal_resolution_axis=temporal_resolution_axis,
                                                                 endpoint_readouts_values_p_nuc=endpoint_readouts_values_p_nuc,
                                                                 endpoint_readouts_values_p_prop=endpoint_readouts_values_p_prop,
                                                                 exp_treatment=exp_treatment,
                                                                 exp_name=file_name.replace('.csv', ''),
                                                                 full_path_to_save_fig=None,
                                                                 **kwargs)

    return endpoint_readouts_values_p_nuc, endpoint_readouts_values_p_prop, exp_treatment


def compare_all_files_compressed_time_resolution_endpoint_readouts(dir_path: str,
                                                                   compression_range_to_compare: Tuple[int, int],
                                                                   **kwargs) -> List[Tuple[np.array, np.array, str]]:
    """
    for each file in the directory pointed in dir_path argument:
        extracts and aggregates the temporal resolution compressed versions of the file and compares
        them using a simple scatter plot (when the kwargs.visualize argument is set to true - defaultive).
    :param dir_path: str, path to original time resolution experiments.
    :param compression_range_to_compare: tuple [int, int]
    :param kwargs:
    :return:
    """
    print_progression_flag = kwargs.get('print_progression', False)

    all_files_full_paths, only_exp_names = get_all_paths_csv_files_in_dir(dir_path=dir_path)
    all_exp_names_with_file_type = list(map(lambda x: f'{x}.{kwargs.get("file_type", "csv")}', only_exp_names))

    for file_idx, file_name in enumerate(all_exp_names_with_file_type):
        if print_progression_flag:
            print(f'analyzing file: {file_name} {file_idx+1}/{len(only_exp_names)}')

        compare_single_file_compressed_time_resolution_endpoint_readouts(file_name=file_name,
                                                                         compression_range_to_compare=compression_range_to_compare,
                                                                         **kwargs)



if __name__ == '__main__':
    # single file comparison
    # file_name = '20160828_10AsgCx43_FB_xy01.csv'
    # compare_single_file_compressed_time_resolution_endpoint_readouts(file_name=file_name,
    #                                                                  compression_range_to_compare=(2, 10),
    #                                                                  visualize=True,
    #                                                                  only_recent_death_flag_for_neighbors_calc=True)
    # all files comparisons
    compare_all_files_compressed_time_resolution_endpoint_readouts(dir_path=NON_COMPRESSED_FILE_MAIN_DIR,
                                                                   compression_range_to_compare=(2, 10),
                                                                   visualize=True,
                                                                   only_recent_death_flag_for_neighbors_calc=True,
                                                                   print_progression=True)
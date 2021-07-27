import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import IntEnum, unique
from utils import *
from global_parameters import *
from Visualization import *
from sklearn.metrics import mean_squared_error as mse


@unique
class NucOrProp(IntEnum):
    PROP = 1
    NUCLEATION = 2


def get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death: np.array,
                                                                      cells_neighbors: List[List],
                                                                      timeframe_to_analyze: int,
                                                                      nuc_or_prop_mode: NucOrProp,
                                                                      recently_dead_only_mode: bool = False) -> Tuple[
    np.array, np.array]:
    """
    returns the indices and mask of all cells that either neighbors of dead cells and are alive (mode=NucOrProp.PROP)
    or the indices and mask of all cells that are not neighbors of dead cells and are alive (mode=NucOrProp.NUCLEATION).
    If recently_dead_only_mode argument is set to 'True', invokes the
    'get_all_neighbors_or_not_neighbors_of_recently_dead_cells_indices_and_mask' function:
    This function takes into account only recently dead cells (previous time frame and current timeframe only)
    as cells which promote cell death in their proximity.
    :param recently_dead_only_mode:
    :param cells_times_of_death:
    :param cells_neighbors:
    :param timeframe_to_analyze:
    :param nuc_or_prop_mode: NucOrProp
    :return: dead_cells_neighbors_indices list, dead_cells_neighbors_mask
    """
    if recently_dead_only_mode:
        return get_all_neighbors_or_not_neighbors_of_recently_dead_cells_indices_and_mask(
            cells_times_of_death=cells_times_of_death,
            cells_neighbors=cells_neighbors,
            timeframe_to_analyze=timeframe_to_analyze,
            explicit_temporal_resolution=None,
            nuc_or_prop_mode=nuc_or_prop_mode
        )

    dead_cells_neighbors_mask = np.zeros_like(cells_times_of_death, dtype=bool)

    # get all cells idxs that are alive after timeframe_to_analyze
    alive_cells_indices = np.where(cells_times_of_death > timeframe_to_analyze)[0]
    alive_cells_mask = cells_times_of_death > timeframe_to_analyze
    # get all cells idxs that are dead at timeframe_to_analyze
    dead_cells_indices = np.where(cells_times_of_death <= timeframe_to_analyze)[0]

    for dead_cell_idx in dead_cells_indices:
        for neighbor_idx in cells_neighbors[dead_cell_idx]:
            # check if the neighbor is alive at timeframe_to_analyze, if alive, is a neighbor of dead cell
            if alive_cells_mask[neighbor_idx]:
                dead_cells_neighbors_mask[neighbor_idx] = True
    if nuc_or_prop_mode == 1:  # PROP
        dead_cells_neighbors_indices = np.where(dead_cells_neighbors_mask)[0]
        return dead_cells_neighbors_indices, dead_cells_neighbors_mask
    elif nuc_or_prop_mode == 2:  # NUCLEATION
        not_neighbors_of_dead_cells_mask = (~dead_cells_neighbors_mask) * alive_cells_mask
        not_neighbors_of_dead_cells_indices = np.where(not_neighbors_of_dead_cells_mask)[0]
        return not_neighbors_of_dead_cells_indices, not_neighbors_of_dead_cells_mask


def get_all_neighbors_or_not_neighbors_of_recently_dead_cells_indices_and_mask(cells_times_of_death: np.array,
                                                                               cells_neighbors: List[List],
                                                                               timeframe_to_analyze: int,
                                                                               explicit_temporal_resolution: int = None,
                                                                               nuc_or_prop_mode: NucOrProp = NucOrProp.PROP) -> \
        Tuple[
            np.array, np.array]:
    """
    returns the indices and mask of all cells that either neighbors of recently dead cells that are alive (mode=NucOrProp.PROP)
    or the indices and mask of all cells that are not neighbors of dead cells and are alive (mode=NucOrProp.NUCLEATION)
    Recently dead cells are cells that died in the current or previous frame. cells that die before are considered as
    cells that do not diffuse any more lethal substances.
    :param explicit_temporal_resolution:
    :param cells_times_of_death:
    :param cells_neighbors:
    :param timeframe_to_analyze:
    :param nuc_or_prop_mode: NucOrProp
    :return: dead_cells_neighbors_indices list, dead_cells_neighbors_mask
    """
    if explicit_temporal_resolution is None:
        unique_times_of_death = np.unique(cells_times_of_death)
        explicit_temporal_resolution = abs(unique_times_of_death[1] - unique_times_of_death[0])

    dead_cells_neighbors_mask = np.zeros_like(cells_times_of_death, dtype=bool)

    # get all cells idxs that are alive after timeframe_to_analyze
    alive_cells_indices = np.where(cells_times_of_death > timeframe_to_analyze)[0]
    alive_cells_mask = cells_times_of_death > timeframe_to_analyze
    # get all cells idxs that are dead at timeframe_to_analyze or at timeframe_to_analyze-temporal_res
    dead_cells_indices = np.where((cells_times_of_death == timeframe_to_analyze) |
                                  (cells_times_of_death == (timeframe_to_analyze - explicit_temporal_resolution)))[0]

    for dead_cell_idx in dead_cells_indices:
        for neighbor_idx in cells_neighbors[dead_cell_idx]:
            # check if the neighbor is alive at timeframe_to_analyze, if alive, is a neighbor of dead cell
            if alive_cells_mask[neighbor_idx]:
                dead_cells_neighbors_mask[neighbor_idx] = True
    if nuc_or_prop_mode == 1:  # PROP
        dead_cells_neighbors_indices = np.where(dead_cells_neighbors_mask)[0]
        return dead_cells_neighbors_indices, dead_cells_neighbors_mask
    elif nuc_or_prop_mode == 2:  # NUCLEATION
        not_neighbors_of_dead_cells_mask = (~dead_cells_neighbors_mask) * alive_cells_mask
        not_neighbors_of_dead_cells_indices = np.where(not_neighbors_of_dead_cells_mask)[0]
        return not_neighbors_of_dead_cells_indices, not_neighbors_of_dead_cells_mask


def possible_nucleators_blobs_generation(possible_nucleators_indices: np.array,
                                         cells_neighbors: List[List]) -> List[np.array]:
    blobs = []
    already_in_blobs = set()
    for possible_nucleator_idx in possible_nucleators_indices:
        if possible_nucleator_idx in already_in_blobs:
            continue
        possible_nucleator_neighbors = np.array(cells_neighbors[possible_nucleator_idx])
        in_blob_cells_indices = np.intersect1d(possible_nucleators_indices, possible_nucleator_neighbors)
        in_blob_cells_indices = [possible_nucleator_idx] + in_blob_cells_indices.tolist()
        blobs.append(in_blob_cells_indices)
        already_in_blobs.update(in_blob_cells_indices)

    return blobs


def get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe(cells_times_of_death: np.array,
                                                                         cells_neighbors: List[List],
                                                                         timeframe_to_analyze: int,
                                                                         temporal_resolution: int,
                                                                         **kwargs) -> Tuple[np.array,
                                                                                            np.array,
                                                                                            np.array,
                                                                                            np.array,
                                                                                            float,
                                                                                            np.array,
                                                                                            np.array]:
    """
    calculates and returns for a given timeframe
    nucleation candidates indices, nucleation candidates mask,
    nucleators indices, nucleators mask,
    and p(nuc). also returns the propagators detected in blobs - propagators_to_add_indices and propagators_to_add_mask

    :param cells_times_of_death:
    :param cells_neighbors:
    :param timeframe_to_analyze:
    :param temporal_resolution:
    :return:
    """
    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc', False)
    nucleation_candidates_indices, nucleation_candidates_mask = \
        get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                          cells_neighbors=cells_neighbors,
                                                                          timeframe_to_analyze=timeframe_to_analyze,
                                                                          nuc_or_prop_mode=NucOrProp.NUCLEATION,
                                                                          recently_dead_only_mode=
                                                                          only_recent_death_flag_for_neighbors_calc)
    # get all cells indexes that die next timeframe
    dead_cells_indices_at_next_time_frame = \
        np.where(cells_times_of_death == timeframe_to_analyze + temporal_resolution)[0]
    # get nucleators by intersecting cells that die on next frame, and nucleation candidates
    possible_nucleators_indices = np.intersect1d(dead_cells_indices_at_next_time_frame, nucleation_candidates_indices)

    # if dead cells are neighbors, collect them into blobs
    blobs = possible_nucleators_blobs_generation(possible_nucleators_indices=possible_nucleators_indices,
                                                 cells_neighbors=cells_neighbors)
    # for each blob, only the first one is a nucleator, the rest are propagators
    nucleators_indices = []
    propagators_to_add_indices = []
    for blob_idx, blob in enumerate(blobs):
        for idx, possible_nucleator_in_blob_idx in enumerate(blob):
            if idx == 0:
                nucleators_indices.append(possible_nucleator_in_blob_idx)
            else:
                propagators_to_add_indices.append(possible_nucleator_in_blob_idx)

    propagators_to_add_indices = np.array(propagators_to_add_indices)

    # create the nucleators mask
    nucleators_mask = np.zeros_like(cells_times_of_death, dtype=bool)
    nucleators_mask = calc_mask_from_indices(empty_mask=nucleators_mask,
                                             indices=nucleators_indices)

    # create the propagators to add mask
    propagators_to_add_mask = np.zeros_like(cells_times_of_death, dtype=bool)
    propagators_to_add_mask = calc_mask_from_indices(empty_mask=propagators_to_add_mask,
                                                     indices=propagators_to_add_indices)

    # calc p_nuc
    p_nuc = calc_fraction_from_candidates(dead_cells_at_time_indices=nucleators_indices,
                                          candidates_indices=nucleation_candidates_indices)

    return nucleation_candidates_indices, nucleation_candidates_mask, nucleators_indices, nucleators_mask, p_nuc, \
           propagators_to_add_indices, propagators_to_add_mask


def get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe(cells_times_of_death: np.array,
                                                                            cells_neighbors: List[List],
                                                                            timeframe_to_analyze: int,
                                                                            temporal_resolution: int,
                                                                            **kwargs) -> Tuple[np.array,
                                                                                               np.array,
                                                                                               np.array,
                                                                                               np.array,
                                                                                               float]:
    """
    calculates and returns for a given timeframe
    propagation candidates indices, propagation candidates mask,
    propagators indices, propagators mask,
    and p(prop)
    IMPORTANT NOTE - the propagation indices, masks and probability DO NOT include propagators from blobs
    detected by the 'get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe' function
    :param cells_times_of_death:
    :param cells_neighbors:
    :param timeframe_to_analyze:
    :param temporal_resolution:
    :return:
    """
    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc', False)

    propagation_candidates_indices, propagation_candidates_mask = \
        get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                          cells_neighbors=cells_neighbors,
                                                                          timeframe_to_analyze=timeframe_to_analyze,
                                                                          nuc_or_prop_mode=NucOrProp.PROP,
                                                                          recently_dead_only_mode=
                                                                          only_recent_death_flag_for_neighbors_calc)
    # get all cells indexes that die next timeframe
    dead_cells_indices_at_next_time_frame = \
        np.where(cells_times_of_death == timeframe_to_analyze + temporal_resolution)[0]
    # get propagators by intersecting cells that die on next frame, and propagation candidates
    propagators_indices = np.intersect1d(dead_cells_indices_at_next_time_frame, propagation_candidates_indices)

    # create the propagators mask
    propagators_mask = np.zeros_like(cells_times_of_death, dtype=bool)
    propagators_mask = calc_mask_from_indices(empty_mask=propagators_mask, indices=propagators_indices)

    # calc p_prop
    p_prop = calc_fraction_from_candidates(dead_cells_at_time_indices=propagators_indices,
                                           candidates_indices=propagation_candidates_indices)

    return propagation_candidates_indices, propagation_candidates_mask, propagators_indices, propagators_mask, p_prop


def calc_single_time_frame_p_nuc_p_prop_probabilities_and_nucleators_and_propagators(cells_times_of_death: np.array,
                                                                                     cells_neighbors: List[List],
                                                                                     timeframe_to_analyze: int,
                                                                                     temporal_resolution: int,
                                                                                     **kwargs) -> Tuple[
    float,
    float,
    np.array,
    np.array,
    np.array,
    np.array,
    float]:
    """
    calculate the P(Nuc) & P(Prop) probabilities for a single timeframe.
    This function considers propagators & propagation candidates derived from the blobs.
    the blobs are detected and analyzed in the
    'get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe' function.
    The function returns the probabilities P(Prop) & P(Nuc) and the indices of propagators and nucleators detected
    in the frame. The function also returns the indices of all dead cells in next frame,
    and the indices of all alive in current frame.
    returns the accumulated death fraction up to the point
    :param cells_xy:
    :param cells_times_of_death:
    :param cells_neighbors:
    :param timeframe_to_analyze:
    :param temporal_resolution:
    :return: p_prop, p_nuc, propagators_indices, nucleators_indices, total_dead_indices, total_alive_indices,
     accumulated time of death
    """

    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc', False)

    # get total death & alive cell indices
    total_alive_in_current_frame_indices = np.where(cells_times_of_death > timeframe_to_analyze)[0]
    total_dead_in_next_frame_indices = np.where(cells_times_of_death == timeframe_to_analyze + temporal_resolution)[0]

    # propagation extracted data
    propagation_candidates_indices, \
    propagation_candidates_mask, \
    propagators_indices, \
    propagators_mask, \
    p_prop = get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe(
        cells_times_of_death=cells_times_of_death,
        cells_neighbors=cells_neighbors,
        timeframe_to_analyze=timeframe_to_analyze,
        temporal_resolution=temporal_resolution,
        only_recent_death_flag_for_neighbors_calc=only_recent_death_flag_for_neighbors_calc
    )

    # nucleation extracted data
    nucleation_candidates_indices, \
    nucleation_candidates_mask, \
    nucleators_indices, \
    nucleators_mask, \
    p_nuc, \
    propagators_to_add_indices, \
    propagators_to_add_mask = get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe(
        cells_times_of_death=cells_times_of_death,
        cells_neighbors=cells_neighbors,
        timeframe_to_analyze=timeframe_to_analyze,
        temporal_resolution=temporal_resolution,
        only_recent_death_flag_for_neighbors_calc=only_recent_death_flag_for_neighbors_calc
    )

    propagators_indices = np.array(propagators_indices.tolist() + propagators_to_add_indices.tolist())
    propagators_mask = propagators_mask + propagators_to_add_mask
    p_prop = calc_fraction_from_candidates(dead_cells_at_time_indices=propagators_indices,
                                           candidates_indices=np.array(propagation_candidates_indices.tolist() +
                                                                       propagators_to_add_indices.tolist()))

    accumulated_fraction_of_death = (cells_times_of_death <= (timeframe_to_analyze + temporal_resolution)).sum() / len(
        cells_times_of_death)

    return p_prop, p_nuc, propagators_indices, \
           nucleators_indices, \
           total_dead_in_next_frame_indices, total_alive_in_current_frame_indices, accumulated_fraction_of_death


def calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts(single_exp_full_path: str, **kwargs) -> \
        Tuple[
            np.array, np.array, float, float, np.array, np.array, np.array]:
    """
    calculates the experiment P(Nuc) & P(Prop) about time and endpoint readouts.
    also aggregates and returns masks for nucleators and propagators cells (endpoint readout as well).
    returns the p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, nucleators_mask, propagators_mask
    and accumulated_death_fraction_by_time
    :param single_exp_full_path:
    :return:
    """
    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc', False)

    cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(exp_full_path=single_exp_full_path)
    all_death_times_unique = np.unique(cells_times_of_death)

    compressed_flag = False
    if 'compressed' in single_exp_full_path.lower():
        compressed_flag = True

    exp_treatment, explicit_temporal_resolution = \
        get_exp_treatment_type_and_temporal_resolution(single_exp_full_path.split(os.sep)[-1],
                                                       compressed_flag=compressed_flag)

    # adds a fake frame before the start of the experiment to calc nucleation and propagation events at first frame
    all_death_times_unique = np.arange(-explicit_temporal_resolution, all_death_times_unique.max(),
                                       explicit_temporal_resolution)
    # all_death_times_unique = np.array([-explicit_temporal_resolution] + all_death_times_unique.tolist())

    dist_threshold = kwargs.get('dist_threshold', DIST_THRESHOLD_IN_PIXELS)
    cells_neighbors_lvl1, cells_neighbors_lvl2, cells_neighbors_lvl3 = get_cells_neighbors(XY=cells_loci,
                                                                                           threshold_dist=dist_threshold)

    p_nuc_by_time = np.zeros_like(all_death_times_unique, dtype=float)
    p_prop_by_time = np.zeros_like(all_death_times_unique, dtype=float)
    accumulated_death_fraction_by_time = np.zeros_like(all_death_times_unique, dtype=float)

    all_frames_nucleators_mask = np.zeros(len(cells_loci), dtype=bool)
    all_frames_propagators_mask = np.zeros(len(cells_loci), dtype=bool)

    for time_frame_idx, current_time in enumerate(all_death_times_unique):
        # print(f'analyzing frame #{time_frame_idx}')
        single_frame_p_prop, \
        single_frame_p_nuc, \
        single_frame_propagators_indices, \
        single_frame_nucleators_indices, \
        single_frame_total_dead_in_next_frame_indices, \
        single_frame_total_alive_in_current_frame_indices, \
        accumulated_death_fraction = \
            calc_single_time_frame_p_nuc_p_prop_probabilities_and_nucleators_and_propagators(
                cells_times_of_death=cells_times_of_death,
                cells_neighbors=cells_neighbors_lvl1,
                timeframe_to_analyze=current_time,
                temporal_resolution=explicit_temporal_resolution,
                only_recent_death_flag_for_neighbors_calc=only_recent_death_flag_for_neighbors_calc)

        p_prop_by_time[time_frame_idx] = single_frame_p_prop
        p_nuc_by_time[time_frame_idx] = single_frame_p_nuc
        accumulated_death_fraction_by_time[time_frame_idx] = accumulated_death_fraction

        curr_frame_propagators_mask = np.zeros(len(cells_loci), dtype=bool)
        curr_frame_propagators_mask = calc_mask_from_indices(empty_mask=curr_frame_propagators_mask,
                                                             indices=single_frame_propagators_indices)
        all_frames_propagators_mask += curr_frame_propagators_mask

        curr_frame_nucleators_mask = np.zeros(len(cells_loci), dtype=bool)
        curr_frame_nucleators_mask = calc_mask_from_indices(empty_mask=curr_frame_nucleators_mask,
                                                            indices=single_frame_nucleators_indices)
        all_frames_nucleators_mask += curr_frame_nucleators_mask

    p_nuc_global = all_frames_nucleators_mask.sum() / len(cells_loci)
    p_prop_global = all_frames_propagators_mask.sum() / len(cells_loci)

    return p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
           all_frames_nucleators_mask, all_frames_propagators_mask, accumulated_death_fraction_by_time


def calc_and_visualize_all_experiments_csvs_in_dir(dir_path: str = None, limit_exp_num: int = float('inf'), **kwargs) -> \
        Tuple[
            np.array, np.array, np.array]:
    """

    :param dir_path:
    :param limit_exp_num:
    :param kwargs:
    :return:
    """
    # todo: add documentation
    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc',
                                                           RECENT_DEATH_ONLY_FLAG)

    visualize_flag = kwargs.get('visualize', True)

    use_log_flag = kwargs.get('use_log', USE_LOG)

    # LOG FILE: clean and write headline
    with open('../experiments_with_bad_results.txt', 'w') as f:
        f.write(f'analyzed directory path:{dir_path}\nExperiments with P(Nuc)+P(Prop)<1 : \n')

    if dir_path is None:
        dir_path = os.sep.join(
            os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV', 'OriginalTimeMinutesData'])

    all_files_to_analyze_full_paths, all_files_to_analyze_only_exp_names = \
        get_all_paths_csv_files_in_dir(dir_path=dir_path)

    all_global_p_nuc, all_global_p_prop = list(), list()

    all_treatment_types = list()
    all_temporal_resolutions = list()

    total_exps = len(all_files_to_analyze_only_exp_names)

    for exp_idx, exp_details in enumerate(zip(all_files_to_analyze_full_paths, all_files_to_analyze_only_exp_names)):
        if exp_idx > limit_exp_num:
            break

        file_full_path, exp_name = exp_details
        print(f'analyzing exp {exp_name} | {exp_idx + 1}/{total_exps}')

        compressed_flag = False
        if 'compressed' in file_full_path.lower():
            compressed_flag = True

        exp_treatment, exp_temporal_res = get_exp_treatment_type_and_temporal_resolution(
            exp_file_name=exp_name + '.csv', compressed_flag=compressed_flag)

        p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
        all_frames_nucleators_mask, all_frames_propagators_mask, \
        accumulated_fraction_of_death_by_time = \
            calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts(single_exp_full_path=file_full_path,
                                                                                   only_recent_death_flag_for_neighbors_calc=only_recent_death_flag_for_neighbors_calc)

        all_treatment_types.append(exp_treatment)
        all_temporal_resolutions.append(exp_temporal_res)
        all_global_p_nuc.append(p_nuc_global)
        all_global_p_prop.append(p_prop_global)
        # LOG FILE to detect in runtime experiments with bad results (i.e., global_p(nuc)+global_p(prop)!=1)
        if abs((p_nuc_global + p_prop_global) - 1) > 0.01:
            with open('../experiments_with_bad_results.txt', 'a') as f:
                f.write(f'exp:{exp_name}| pnuc={p_nuc_global}, pprop={p_prop_global}\n')

        if visualize_flag:
            visualize_cell_death_in_time(xyt_full_path=file_full_path,
                                         nucleators_mask=all_frames_nucleators_mask,
                                         propagators_maks=all_frames_propagators_mask,
                                         exp_treatment=exp_treatment, exp_name=exp_name)
            plot_measurements_by_time(p_nuc_by_time, p_prop_by_time, accumulated_fraction_of_death_by_time,
                                      temporal_resolution=exp_temporal_res, exp_name=exp_name,
                                      exp_treatment=exp_treatment)
    if visualize_flag:
        visualize_endpoint_readouts_by_treatment(x_readout=np.array(all_global_p_nuc),
                                                 y_readout=np.array(all_global_p_prop),
                                                 treatment_per_readout=np.array(all_treatment_types),
                                                 use_log=use_log_flag)

    return np.array(all_global_p_nuc), np.array(all_global_p_prop), np.array(all_treatment_types)


def calc_all_experiments_csvs_in_dir_with_altering_flag_values(dir_path: str = None,
                                                               limit_exp_num: int = float('inf'),
                                                               flag_key: str = 'only_recent_death_flag_for_neighbors_calc',
                                                               flag_values: Tuple = (True, False),
                                                               **kwargs) -> Tuple[List[Tuple], List]:
    """

    :param dir_path:
    :param limit_exp_num:
    :param flag_key:
    :param flag_values:
    :param kwargs:
    :return:
    """
    # todo: add documentation
    all_global_p_nuc_p_prop_tuples_list = list()
    all_treatment_types_list = list()

    for flag_value in flag_values:
        flag_kwarg = {flag_key: flag_value}
        if flag_key == 'dir_path':
            all_global_p_nuc, \
            all_global_p_prop, \
            all_treatment_types = calc_and_visualize_all_experiments_csvs_in_dir(**flag_kwarg,
                                                                                 limit_exp_num=limit_exp_num,
                                                                                 visualize=False)
        else:
            all_global_p_nuc, \
            all_global_p_prop, \
            all_treatment_types = calc_and_visualize_all_experiments_csvs_in_dir(**flag_kwarg,
                                                                                 dir_path=dir_path,
                                                                                 limit_exp_num=limit_exp_num,
                                                                                 visualize=False)

        all_global_p_nuc_p_prop_tuples_list.append((all_global_p_nuc, all_global_p_prop))
        all_treatment_types_list.append(all_treatment_types)

    return all_global_p_nuc_p_prop_tuples_list, all_treatment_types_list


def calc_rmse_between_experiments_results_of_altering_flag_values(dir_path: str = None,
                                                                  limit_exp_num: int = float('inf'),
                                                                  flag_key: str = 'only_recent_death_flag_for_neighbors_calc',
                                                                  flag_values: Tuple = (True, False),
                                                                  **kwargs):
    """

    :param dir_path:
    :param limit_exp_num:
    :param flag_key:
    :param flag_values:
    :param kwargs:
    :return:
    """
    # todo: add documentation
    all_global_p_nuc_p_prop_tuples_list, \
    all_treatment_types_list = calc_all_experiments_csvs_in_dir_with_altering_flag_values(
        dir_path=dir_path,
        limit_exp_num=limit_exp_num,
        flag_key=flag_key,
        flag_values=flag_values,
        **kwargs)

    by_treatment_scores_first_flag_value_p_nuc = dict()
    by_treatment_scores_second_flag_value_p_nuc = dict()
    by_treatment_scores_first_flag_value_p_prop = dict()
    by_treatment_scores_second_flag_value_p_prop = dict()

    by_treatment_rmse_score_p_nuc = dict()
    by_treatment_rmse_score_p_prop = dict()

    for single_exp_idx, treatment_name in enumerate(all_treatment_types_list[0]):
        # by_treatment_rmse_score[treatment_name] = by_treatment_rmse_score.get(treatment_name) + \
        #                                           all_global_p_nuc_p_prop_tuples_list[0][single_exp_idx]
        by_treatment_scores_first_flag_value_p_nuc[treatment_name] = \
            by_treatment_scores_first_flag_value_p_nuc.get(treatment_name, []) + \
            [all_global_p_nuc_p_prop_tuples_list[0][0][single_exp_idx]]
        by_treatment_scores_second_flag_value_p_nuc[treatment_name] = \
            by_treatment_scores_second_flag_value_p_nuc.get(treatment_name, []) + \
            [all_global_p_nuc_p_prop_tuples_list[1][0][single_exp_idx]]

        by_treatment_scores_first_flag_value_p_prop[treatment_name] = \
            by_treatment_scores_first_flag_value_p_prop.get(treatment_name, []) + \
            [all_global_p_nuc_p_prop_tuples_list[0][1][single_exp_idx]]
        by_treatment_scores_second_flag_value_p_prop[treatment_name] = \
            by_treatment_scores_second_flag_value_p_prop.get(treatment_name, []) + \
            [all_global_p_nuc_p_prop_tuples_list[1][1][single_exp_idx]]

    for treatment_name in by_treatment_scores_first_flag_value_p_nuc.keys():
        by_treatment_rmse_score_p_nuc[treatment_name] = \
            mse(y_true=by_treatment_scores_first_flag_value_p_nuc[treatment_name],
                y_pred=by_treatment_scores_second_flag_value_p_nuc[treatment_name], squared=False)
        by_treatment_rmse_score_p_prop[treatment_name] = \
            mse(y_true=by_treatment_scores_first_flag_value_p_prop[treatment_name],
                y_pred=by_treatment_scores_second_flag_value_p_prop[treatment_name], squared=False)
    if kwargs.get('visualize_flag', True):
        visualize_rmse_of_altering_flag_values_by_treatment(p_nuc_rmse_by_treatment=by_treatment_rmse_score_p_nuc,
                                                            p_prop_rmse_by_treatment=by_treatment_rmse_score_p_prop,
                                                            flag_name=flag_key[:18])

    return by_treatment_rmse_score_p_nuc, by_treatment_rmse_score_p_prop


def calc_and_visualize_all_experiments_csvs_in_dir_with_altering_flag_values(dir_path: str = None,
                                                                             limit_exp_num: int = float('inf'),
                                                                             flag_key: str = 'only_recent_death_flag_for_neighbors_calc',
                                                                             flag_values: Tuple = (True, False),
                                                                             **kwargs):
    """

    :param dir_path:
    :param limit_exp_num:
    :param flag_key:
    :param flag_values:
    :param kwargs:
    :return:
    """
    # todo: add documentation
    all_global_p_nuc_p_prop_tuples_list, \
    all_treatment_types_list = calc_all_experiments_csvs_in_dir_with_altering_flag_values(dir_path=dir_path,
                                                                                          limit_exp_num=limit_exp_num,
                                                                                          flag_key=flag_key,
                                                                                          flag_values=flag_values,
                                                                                          **kwargs)

    visualize_endpoint_readouts_by_treatment_to_varying_calculation_flags(
        xy1_readout_tuple=all_global_p_nuc_p_prop_tuples_list[0],
        treatment_per_readout1=all_treatment_types_list[0],
        xy2_readout_tuple=all_global_p_nuc_p_prop_tuples_list[1],
        treatment_per_readout2=all_treatment_types_list[1],
        first_flag_type_name_and_value=f'{flag_key}={flag_values[0]}',
        second_flag_type_name_and_value=f'{flag_key}={flag_values[1]}')


if __name__ == '__main__':
    # recent dead only flag
    # calc_rmse_between_experiments_results_of_altering_flag_values(dir_path=os.sep.join(
    #     os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV',
    #                                       'OriginalTimeMinutesData']))  #, limit_exp_num=10)
    # compression flag
    # calc_rmse_between_experiments_results_of_altering_flag_values(flag_key='dir_path',
    #                                                               flag_values=(os.sep.join(
    #                                                                   os.getcwd().split(os.sep)[:-1] + ['Data',
    #                                                                                                     'Experiments_XYT_CSV',
    #                                                                                                     'OriginalTimeMinutesData']),
    #                                                                            os.sep.join(
    #                                                                                os.getcwd().split(os.sep)[:-1] + [
    #                                                                                    'Data', 'Experiments_XYT_CSV',
    #                                                                                    'CompressedTime_XYT_CSV'])))
    # compression flag
    # calc_and_visualize_all_experiments_csvs_in_dir_with_altering_flag_values(flag_key='dir_path',
    #                                                                          flag_values=(os.sep.join(
    #     os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV',
    #                                       'OriginalTimeMinutesData']), os.sep.join(
    #     os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV',
    #                                       'CompressedTime_XYT_CSV'])))
    #
    #
    # calc_and_visualize_all_experiments_csvs_in_dir(dir_path=os.sep.join(
    #     os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV',
    #                                       'OriginalTimeMinutesData']))

    # # single experiment testing
    # path = '..\\Data\\Experiments_XYT_CSV\\CompressedTime_XYT_CSV\\20160909_b16f10_aMSH_xy37.csv'
    # p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
    # all_frames_nucleators_mask, all_frames_propagators_mask,\
    #     accumulated_fraction_of_death_by_time = calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts(single_exp_full_path=path)
    #
    # visualize_cell_death_in_time(xyt_full_path=path, nucleators_mask=all_frames_nucleators_mask, exp_treatment='just_treatment', exp_name='just name')
    #
    # plot_measurements_by_time(p_nuc_by_time, p_prop_by_time, accumulated_fraction_of_death_by_time,
    #                           temporal_resolution=15, exp_name='20180620_HAP1_erastin_xy1', exp_treatment='test_treat')
    pass

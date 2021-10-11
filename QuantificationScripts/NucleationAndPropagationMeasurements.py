import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import IntEnum, unique
from utils import *
from global_parameters import *
from Visualization import *


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


def calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_provided_temporal_resolution(
        single_exp_full_path: str, temporal_resolution: int = None, **kwargs) -> \
        Tuple[
            np.array, np.array, float, float, np.array, np.array, np.array]:
    """
    calculates the experiment P(Nuc) & P(Prop) about time and endpoint readouts.
    also aggregates and returns masks for nucleators and propagators cells (endpoint readout as well).
    returns the p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, nucleators_mask, propagators_mask
    and accumulated_death_fraction_by_time.
    MUST BE PROVIDE TEMPORAL RESOLUTION
    :param single_exp_full_path: str
    :param temporal_resolution: int
    :return:
    """
    assert temporal_resolution, f'temporal resolution must not be None or negative! the value is f{temporal_resolution}'

    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc', False)

    cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(exp_full_path=single_exp_full_path)
    all_death_times_unique = np.unique(cells_times_of_death)

    # adds a fake frame before the start of the experiment to calc nucleation and propagation events at first frame
    all_death_times_unique = np.arange(-temporal_resolution, all_death_times_unique.max(),
                                       temporal_resolution)
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
                temporal_resolution=temporal_resolution,
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


def calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_explicit_temporal_resolution(
        single_exp_full_path: str, **kwargs) -> \
        Tuple[
            np.array, np.array, float, float, np.array, np.array, np.array]:
    """
    calculates the experiment P(Nuc) & P(Prop) about time and endpoint readouts.
    also aggregates and returns masks for nucleators and propagators cells (endpoint readout as well).
    returns the p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, nucleators_mask, propagators_mask
    and accumulated_death_fraction_by_time
    :param single_exp_full_path: str
    :return:
    """

    compressed_flag = False
    if 'compressed' in single_exp_full_path.lower():
        compressed_flag = True
    meta_data_path = os.sep.join(single_exp_full_path.split(os.sep)[:-2] + ['ExperimentsMetaData.csv'])

    exp_treatment, explicit_temporal_resolution = \
        get_exp_treatment_type_and_temporal_resolution(single_exp_full_path.split(os.sep)[-1],
                                                       compressed_flag=compressed_flag,
                                                       meta_data_file_full_path=meta_data_path)
    return calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_provided_temporal_resolution(
        single_exp_full_path=single_exp_full_path,
        temporal_resolution=explicit_temporal_resolution,
        **kwargs
    )


def calc_and_visualize_all_experiments_csvs_in_dir(dir_path: str = None,
                                                   limit_exp_num: int = float('inf'), **kwargs) -> \
        Tuple[
            np.array, np.array, np.array]:
    """
    calculates temporal and endpoint readouts probabilities for all csv files in a directory.
    all csv files in the directory must contain XYT coordinates with column names ['cell_x', 'cell_y', 'death_time'].
    the function supports the following flags (within the kwargs argument) for different calculations:
    1. only_recent_death_flag_for_neighbors_calc - considers neighbors of dead cells as propagation candidates only
        if the death occured in time T and timeframe-temporalResolution<T<=timeframe .
    2. visualize - whether to visualize (plot) the readouts (both temporal and endpoint).
    3. use_log - when visualizing endpoint readouts, whether to use the log of values calculated or the
        values themselves.
    :param dir_path: str, the directory path which contains all csv files.
    :param limit_exp_num: int, maximum number of experiments to analyze, default is infinity (to analyze all
        experiments in the directory.
    :param kwargs:
    :return: endpoint readouts:
        all_global_p_nuc - np.array, all_global_p_prop - np.array, all_treatment_types - np.array
    """
    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc',
                                                           RECENT_DEATH_ONLY_FLAG)

    visualize_flag = kwargs.get('visualize_flag', True)

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

        meta_data_file_path = os.sep.join(dir_path.split(os.sep)[:-1] + ['ExperimentsMetaData.csv'])
        exp_treatment, exp_temporal_res = get_exp_treatment_type_and_temporal_resolution(
            exp_file_name=exp_name + '.csv', meta_data_file_full_path=meta_data_file_path,
            compressed_flag=compressed_flag)

        p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
        all_frames_nucleators_mask, all_frames_propagators_mask, \
        accumulated_fraction_of_death_by_time = \
            calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_explicit_temporal_resolution(
                single_exp_full_path=file_full_path,
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
        visualize_endpoint_readouts_by_treatment_about_readouts(x_readout=np.array(all_global_p_nuc),
                                                                y_readout=np.array(all_global_p_prop),
                                                                treatment_per_readout=np.array(all_treatment_types),
                                                                use_log=use_log_flag,
                                                                plot_about_treatment=True)

    return np.array(all_global_p_nuc), np.array(all_global_p_prop), np.array(all_treatment_types)


def calc_distance_metric_between_experiments_results_of_altering_flag_values(dir_path: str = None,
                                                                             limit_exp_num: int = float('inf'),
                                                                             flag_key: str = 'only_recent_death_flag_for_neighbors_calc',
                                                                             flag_values: Tuple = (True, False),
                                                                             **kwargs):
    """
    aggregates endpoint readouts calculated by calc_all_experiments_csvs_in_dir_with_altering_flag_values function.
    compares the results of each analysis according the the flag value and calculates a distance metric between
    the results (default is rmse), the distance metric is given in the kwargs under 'distance_metric' key.
    the flags arguments are identical to calc_all_experiments_csvs_in_dir_with_altering_flag_values function's arguments.
    9/08/2021 - supports the metrics appearing in utils.py/calc_distance_metric_between_signals function.
    if visualize_flag in kwargs is set to true (which is the default value) also plots the distance metric results.
    :param dir_path: str, the directory path which contains all csv files.
    :param limit_exp_num: int, maximum number of experiments to analyze, default is infinity (to analyze all
        experiments in the directory.
    :param flag_key: str, the flag name to alternate its value and analyze by.
    :param flag_values: Tuple, values of flags to analyze by.
    :param kwargs:
    :return: by_treatment_distance_metric_score_p_nuc - np.array, by_treatment_distance_metric_score_p_prop - np.array
    """
    all_global_p_nuc_p_prop_tuples_list, \
    all_treatment_types_list = calc_all_experiments_csvs_in_dir_with_altering_flag_values(
        dir_path=dir_path,
        limit_exp_num=limit_exp_num,
        flag_key=flag_key,
        flag_values=flag_values)

    visualize_flag = kwargs.get('visualize_flag', True)
    metric_to_use = kwargs.get('distance_metric', 'rmse')

    by_treatment_scores_first_flag_value_p_nuc = dict()
    by_treatment_scores_second_flag_value_p_nuc = dict()
    by_treatment_scores_first_flag_value_p_prop = dict()
    by_treatment_scores_second_flag_value_p_prop = dict()

    by_treatment_distance_metric_score_p_nuc = dict()
    by_treatment_distance_metric_score_p_prop = dict()

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
        by_treatment_distance_metric_score_p_nuc[treatment_name] = \
            calc_distance_metric_between_signals(
                y_true=np.array(by_treatment_scores_first_flag_value_p_nuc[treatment_name]),
                y_pred=np.array(by_treatment_scores_second_flag_value_p_nuc[treatment_name]),
                metric=metric_to_use)
        by_treatment_distance_metric_score_p_prop[treatment_name] = \
            calc_distance_metric_between_signals(
                y_true=np.array(by_treatment_scores_first_flag_value_p_prop[treatment_name]),
                y_pred=np.array(by_treatment_scores_second_flag_value_p_prop[treatment_name]),
                metric=metric_to_use)
    if visualize_flag:
        visualize_specific_treatments = kwargs.get('visualize_specific_treatments', 'all')
        # todo: change visualization to a a more informative plot, focus on specific treatments.
        visualize_distance_metric_of_altering_flag_values_by_treatment(
            p_nuc_distance_by_treatment=by_treatment_distance_metric_score_p_nuc,
            p_prop_distance_by_treatment=by_treatment_distance_metric_score_p_prop,
            flag_name=flag_key[:18],
            distance_metric_name=metric_to_use,
            visualize_specific_treatments=visualize_specific_treatments
        )

    return by_treatment_distance_metric_score_p_nuc, by_treatment_distance_metric_score_p_prop


def calc_all_experiments_csvs_in_dir_with_altering_flag_values(dir_path: str = None,
                                                               limit_exp_num: int = float('inf'),
                                                               flag_key: str = 'only_recent_death_flag_for_neighbors_calc',
                                                               flag_values: Tuple = (True, False)) \
        -> Tuple[List[Tuple], List]:
    """
    calculates temporal and endpoint readouts probabilities for all csv files in a directory under various flag values.
    all csv files in the directory must contain XYT coordinates with column names ['cell_x', 'cell_y', 'death_time'].
    the flags supported are the same as in calc_and_visualize_all_experiments_csvs_in_dir function.
    the flag key argument is the name of the flag and the values are a tuple of the flag values to analyze the files by.
    :param dir_path: str, the directory path which contains all csv files.
    :param limit_exp_num: int, maximum number of experiments to analyze, default is infinity (to analyze all
        experiments in the directory.
    :param flag_key: str, the flag name to alternate its value and analyze by.
    :param flag_values: Tuple, values of flags to analyze by.
    :return: endpoint readouts for all experiments in the directory, 1st element is the readouts for the 1st
        flag value given, 2nd element is for the 2nd flag value given and so on (if exists).
    """
    all_global_p_nuc_p_prop_tuples_list = list()
    all_treatment_types_list = list()

    for flag_value in flag_values:
        flag_kwarg = {flag_key: flag_value}
        if flag_key == 'dir_path':
            all_global_p_nuc, \
            all_global_p_prop, \
            all_treatment_types = calc_and_visualize_all_experiments_csvs_in_dir(**flag_kwarg,
                                                                                 limit_exp_num=limit_exp_num,
                                                                                 visualize_flag=False)
        else:
            all_global_p_nuc, \
            all_global_p_prop, \
            all_treatment_types = calc_and_visualize_all_experiments_csvs_in_dir(**flag_kwarg,
                                                                                 dir_path=dir_path,
                                                                                 limit_exp_num=limit_exp_num,
                                                                                 visualize_flag=False)

        all_global_p_nuc_p_prop_tuples_list.append((all_global_p_nuc, all_global_p_prop))
        all_treatment_types_list.append(all_treatment_types)

    return all_global_p_nuc_p_prop_tuples_list, all_treatment_types_list


def calc_and_visualize_all_experiments_csvs_in_dir_with_altering_flag_values(dir_path: str = None,
                                                                             limit_exp_num: int = float('inf'),
                                                                             flag_key: str = 'only_recent_death_flag_for_neighbors_calc',
                                                                             flag_values: Tuple = (True, False),
                                                                             **kwargs):
    """
    calculates (using calc_all_experiments_csvs_in_dir_with_altering_flag_values function) & visualizes
    temporal and endpoint readouts probabilities for all csv files in a directory under various flag values.
    all csv files in the directory must contain XYT coordinates with column names ['cell_x', 'cell_y', 'death_time'].
    the flags supported are the same as in calc_and_visualize_all_experiments_csvs_in_dir function.
    the flag key argument is the name of the flag and the values are a tuple of the flag values to analyze the files by.
    :param dir_path: str, the directory path which contains all csv files.
    :param limit_exp_num: int, maximum number of experiments to analyze, default is infinity (to analyze all
        experiments in the directory.
    :param flag_key: str, the flag name to alternate its value and analyze by.
    :param flag_values: Tuple, values of flags to analyze by.
    :param kwargs:
    :return:
    """
    all_global_p_nuc_p_prop_tuples_list, \
    all_treatment_types_list = calc_all_experiments_csvs_in_dir_with_altering_flag_values(dir_path=dir_path,
                                                                                          limit_exp_num=limit_exp_num,
                                                                                          flag_key=flag_key,
                                                                                          flag_values=flag_values)

    visualize_endpoint_readouts_by_treatment_to_varying_calculation_flags(
        xy1_readout_tuple=all_global_p_nuc_p_prop_tuples_list[0],
        treatment_per_readout1=all_treatment_types_list[0],
        xy2_readout_tuple=all_global_p_nuc_p_prop_tuples_list[1],
        treatment_per_readout2=all_treatment_types_list[1],
        first_flag_type_name_and_value=f'{flag_key}={flag_values[0]}',
        second_flag_type_name_and_value=f'{flag_key}={flag_values[1]}')


def calc_slopes_and_probabilities_per_unit_of_time_single_experiment(exp_full_path: str,
                                                                     exp_temporal_resolution: int,
                                                                     unit_of_time_min: int = 60,
                                                                     consider_majority_of_death_only: bool = True,
                                                                     **kwargs) -> Tuple[np.array,
                                                                                        np.array,
                                                                                        np.array,
                                                                                        Tuple[float, float],
                                                                                        Tuple[float, float],
                                                                                        Tuple[float, float]]:
    """
    calculates the p(nuc), p(prop) and accumulated death probabilities for a given time
    interval (in minutes) specified by 'unit_of_time_min' argument.
    If only the majority of death is of interest (consider only a defined portion of the death process),
    set the 'consider_majority_of_death_only' argument to True (default). The lower and upper bounds
    of the majority of death is in terms of overall cell death fraction (found in the accumulated death variable).
    The values of the bounds is set from the kwargs argument (lower_bound_percentile, upper_bound_percentile attributes)
    and their default values are set in the global_variables script.
    the function retrieves the mean probability for each unit of time (for each probability)
    and the slope and intercept of the probability signal (of the values before the unit of time partitioning).

    :param exp_full_path: str, the experiments csv file full path.
    :param exp_temporal_resolution: int, the temporal resolution of the experiment.
    :param unit_of_time_min: int, the unit of time to calculate the mean probabilities for. default 60)
    :param consider_majority_of_death_only: boolean, whether to use the entire death
            probabilities signals or just a subset
    :param kwargs: possible kwargs -
        'only_recent_death_flag_for_neighbors_calc': , dafault =
        'lower_bound_percentile': , dafault =
        'upper_bound_percentile': , dafault =
    :return:
    """
    # get all kwargs into local variables
    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc',
                                                           RECENT_DEATH_ONLY_FLAG)
    lower_bound_death_percentile = kwargs.get('lower_bound_percentile', LOWER_DEATH_PERCENTILE_BOUNDARY)
    upper_bound_death_percentile = kwargs.get('upper_bound_percentile', UPPER_DEATH_PERCENTILE_BOUNDARY)

    # calc experiment's readouts
    p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
    all_frames_nucleators_mask, all_frames_propagators_mask, \
    accumulated_fraction_of_death_by_time = \
        calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_explicit_temporal_resolution(
            single_exp_full_path=exp_full_path,
            only_recent_death_flag_for_neighbors_calc=only_recent_death_flag_for_neighbors_calc)

    # if considering only the majority of death, removes all probabilities that are not a part of the majority
    # of the death process (leaves only data starting from the point in time where the lower bound is found, and ending
    # in the point in time where the upper bound is met).
    if consider_majority_of_death_only:
        tenth_percentile_idx = np.where(accumulated_fraction_of_death_by_time >=
                                        lower_bound_death_percentile)[0][0]
        nineteenth_percentile_idx = np.where(accumulated_fraction_of_death_by_time >
                                             upper_bound_death_percentile)[0][0]
        p_nuc_by_time = p_nuc_by_time[tenth_percentile_idx: nineteenth_percentile_idx]
        p_prop_by_time = p_prop_by_time[tenth_percentile_idx: nineteenth_percentile_idx]
        accumulated_fraction_of_death_by_time = \
            accumulated_fraction_of_death_by_time[tenth_percentile_idx: nineteenth_percentile_idx]

    # calculate the slope and intercept of all probabilities
    nuc_slope, nuc_intercept = calc_signal_slope_and_intercept(x=None, y=p_nuc_by_time)
    prop_slope, prop_intercept = calc_signal_slope_and_intercept(x=None, y=p_prop_by_time)
    accumulated_slope, accumulated_intercept = calc_signal_slope_and_intercept(x=None,
                                                                               y=accumulated_fraction_of_death_by_time)

    # collect probabilities per unit of time
    num_frames_within_time_unit = int(unit_of_time_min / exp_temporal_resolution)

    if num_frames_within_time_unit < 1:
        Warning(ValueError('unit_of_time_min can not be smaller the'
                           ' experiments temporal resolution! returning empty values!'))
        return np.array([]), np.array([]), np.array([]), \
               (float('inf'), float('inf')), \
               (float('inf'), float('inf')), \
               (float('inf'), float('inf'))

    num_of_units_of_time_in_exp = int(len(p_nuc_by_time) / num_frames_within_time_unit)
    # in case there are too few timeframes
    num_of_units_of_time_in_exp = 1 if num_of_units_of_time_in_exp < 1 else num_of_units_of_time_in_exp
    # collect pairs of indices for each time unit
    indices_to_collect = list()
    for idx in range(num_of_units_of_time_in_exp):
        indices_to_collect.append((idx * num_frames_within_time_unit, (idx + 1) * num_frames_within_time_unit))

    if (idx + 1) * num_frames_within_time_unit < len(p_nuc_by_time):
        indices_to_collect.append(((idx + 1) * num_frames_within_time_unit, len(p_nuc_by_time)))

    mean_p_nuc_per_unit_of_time = list()
    mean_p_prop_per_unit_of_time = list()
    mean_p_accumulated_death_per_unit_of_time = list()

    for indices in indices_to_collect:
        st_idx, end_idx = indices
        mean_p_nuc_per_unit_of_time.append(p_nuc_by_time[st_idx: end_idx].mean())
        mean_p_prop_per_unit_of_time.append(p_prop_by_time[st_idx: end_idx].mean())
        mean_p_accumulated_death_per_unit_of_time.append(accumulated_fraction_of_death_by_time[st_idx: end_idx].mean())

    mean_p_nuc_per_unit_of_time = np.array(mean_p_nuc_per_unit_of_time)
    mean_p_prop_per_unit_of_time = np.array(mean_p_prop_per_unit_of_time)
    mean_p_accumulated_death_per_unit_of_time = np.array(mean_p_accumulated_death_per_unit_of_time)

    return mean_p_nuc_per_unit_of_time, \
           mean_p_prop_per_unit_of_time, \
           mean_p_accumulated_death_per_unit_of_time, \
           (nuc_slope, nuc_intercept), \
           (prop_slope, prop_intercept), \
           (accumulated_slope, accumulated_intercept)


def calc_slopes_and_probabilities_per_unit_of_time_entire_dir(dir_full_path: str,
                                                              treatments_to_include: Union[List[str], str],
                                                              limit_exp_num: int = float('inf'),
                                                              unit_of_time_min: int = 60,
                                                              consider_majority_of_death_only: bool = True,
                                                              **kwargs
                                                              ) -> Tuple[dict, dict, dict]:
    compressed_flag = False
    if 'compressed' in dir_full_path.lower():
        compressed_flag = True

    meta_data_file_full_path = os.sep.join(dir_full_path.split(os.sep)[:-1] + ['ExperimentsMetaData.csv'])

    # get all kwargs into local variables
    visualize_flag = kwargs.get('visualize_flag', True)
    use_log = kwargs.get('use_log', USE_LOG)
    only_recent_death_flag_for_neighbors_calc = kwargs.get('only_recent_death_flag_for_neighbors_calc',
                                                           RECENT_DEATH_ONLY_FLAG)

    lower_bound_death_percentile = kwargs.get('lower_bound_percentile', LOWER_DEATH_PERCENTILE_BOUNDARY)
    upper_bound_death_percentile = kwargs.get('upper_bound_percentile', UPPER_DEATH_PERCENTILE_BOUNDARY)

    # get all paths to the experiments' files.
    all_files_full_paths, all_files_only_exp_names = get_all_paths_csv_files_in_dir(dir_path=dir_full_path)
    total_exps = len(all_files_only_exp_names)

    #
    all_nuc_slopes, all_nuc_intercepts = list(), list()
    all_prop_slopes, all_prop_intercepts = list(), list()
    all_accumulated_death_slopes, all_accumulated_death_intercepts = list(), list()

    treatment_per_readout = list()

    exps_mean_per_time_unit_by_treatment = {}

    for exp_idx, single_exp_full_path in enumerate(all_files_full_paths):
        if limit_exp_num < exp_idx + 1:
            break

        exp_name = all_files_only_exp_names[exp_idx]

        exp_treatment, explicit_temporal_resolution = \
            get_exp_treatment_type_and_temporal_resolution(exp_file_name=single_exp_full_path.split(os.sep)[-1],
                                                           meta_data_file_full_path=meta_data_file_full_path,
                                                           compressed_flag=compressed_flag)
        # skip un-wanted treatments
        if treatments_to_include != 'all' and exp_treatment.lower() not in treatments_to_include:
            continue

        print(f'analyzing exp {exp_name} | {exp_idx + 1}/{total_exps}')

        exp_mean_p_nuc_per_unit_of_time, \
        exp_mean_p_prop_per_unit_of_time, \
        exp_mean_p_accumulated_death_per_unit_of_time, \
        exp_nuc_slope_and_intercept, \
        exp_prop_slope_and_intercept, \
        exp_accumulated_death_slope_and_intercept, \
            = calc_slopes_and_probabilities_per_unit_of_time_single_experiment(exp_full_path=single_exp_full_path,
                                                                               exp_temporal_resolution=explicit_temporal_resolution,
                                                                               unit_of_time_min=unit_of_time_min,
                                                                               consider_majority_of_death_only=consider_majority_of_death_only,
                                                                               **kwargs)
        # aggregating mean probabilities by treatment name
        exps_mean_per_time_unit_by_treatment[exp_treatment] = \
            exps_mean_per_time_unit_by_treatment.get(exp_treatment, []) + \
            [[exp_mean_p_nuc_per_unit_of_time, exp_mean_p_prop_per_unit_of_time]]

        # un-packing slope and intercept
        exp_nuc_slope, exp_nuc_intercept = exp_nuc_slope_and_intercept
        exp_prop_slope, exp_prop_intercept = exp_prop_slope_and_intercept
        exp_accumulated_death_slope, exp_accumulated_death_intercept = exp_accumulated_death_slope_and_intercept

        all_nuc_slopes.append(exp_nuc_slope)
        all_nuc_intercepts.append(exp_nuc_intercept)

        all_prop_slopes.append(exp_prop_slope)
        all_prop_intercepts.append(exp_prop_intercept)

        all_accumulated_death_slopes.append(exp_accumulated_death_slope)
        all_accumulated_death_intercepts.append(exp_accumulated_death_intercept)

        treatment_per_readout.append(exp_treatment)

    if visualize_flag:
        # plotting the slopes and intercepts of all_experiments
        kwargs['set_y_lim'] = False
        if only_recent_death_flag_for_neighbors_calc:
            path_to_save_figure_dir_only = os.sep.join(
                os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                  'Only recent death considered for neighbors results',
                                                  'Global_P_Nuc_VS_P_Prop'])
        else:
            path_to_save_figure_dir_only = os.sep.join(
                os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                  'Global_P_Nuc_VS_P_Prop'])
        path_to_save_figure = os.sep.join([path_to_save_figure_dir_only, 'nuc_prop_slopes_about_treatment'])
        # slopes:
        visualize_endpoint_readouts_by_treatment_about_readouts(x_readout=all_nuc_slopes,
                                                                y_readout=all_prop_slopes,
                                                                treatment_per_readout=treatment_per_readout,
                                                                x_label='P(Nuc) Slope',
                                                                y_label='P(Prop) Slope',
                                                                use_log=use_log,
                                                                plot_about_treatment=True,
                                                                full_path_to_save_fig=path_to_save_figure,
                                                                **kwargs)
        path_to_save_figure = os.sep.join([path_to_save_figure_dir_only, 'nuc_prop_intercepts_about_treatment'])
        # intercepts:
        visualize_endpoint_readouts_by_treatment_about_readouts(x_readout=all_nuc_intercepts,
                                                                y_readout=all_prop_intercepts,
                                                                treatment_per_readout=treatment_per_readout,
                                                                x_label='P(Nuc) Intercepts',
                                                                y_label='P(Prop) Intercepts',
                                                                use_log=use_log,
                                                                plot_about_treatment=True,
                                                                full_path_to_save_fig=path_to_save_figure,
                                                                **kwargs)

        # plotting the mean probability per unit of time (? todo: need to calculate and aggregate means first)
        for treatment_name in np.unique(np.array(treatment_per_readout)):
            readouts = exps_mean_per_time_unit_by_treatment[treatment_name]
            plot_temporal_readout_for_entire_treatment(readouts=readouts,
                                                       labels=[f'P(Nuc) per {unit_of_time_min} min',
                                                               f'P(Prop) per {unit_of_time_min} min'],
                                                       treatment=treatment_name,
                                                       unit_of_time=unit_of_time_min)

        # visualize_endpoint_readouts_by_treatment_about_readouts(plot_about_treatment=True)

    nucleation_data = {'slopes': all_nuc_slopes,
                       'intercepts': all_nuc_intercepts}

    propagation_data = {'slopes': all_prop_slopes,
                        'intercepts': all_prop_intercepts}

    accumulated_death_data = {'slopes': all_accumulated_death_slopes,
                              'intercepts': all_accumulated_death_intercepts}

    return nucleation_data, propagation_data, accumulated_death_data


def calc_fraction_of_adjacent_dying_cells_in_time_window(window_start_min: int,
                                                         window_end_min: int,
                                                         cells_neighbors: np.array,
                                                         cells_times_of_death: np.array,
                                                         **kwargs) -> Tuple[float, set]:
    """
    calculates the fraction of cells which die in the given window in approximation to already dead cells ('death seeders').
    if the flag 'consider_death_within_window_only_flag' is set to True, the death seeding cells are only considered as
    such if they died within the window timeframe, else, all dead cells before and within the window timeframe
    are considered as death seeders.
    death which occur in the window's end time argument ('window_end_time') value, are not considered as
    part of the death within the window's timeframe.
    :param window_start_min:
    :param window_end_min:
    :param cells_neighbors:
    :param cells_times_of_death:
    :param kwargs:
    :return:
    """
    consider_death_within_window_only_flag = kwargs.get('consider_death_within_window_only_flag', True)
    dead_cells_in_window_mask = None
    dead_cells_in_window_indices = None

    # death_mask = np.zeros_like(cells_times_of_death)
    if consider_death_within_window_only_flag:
        dead_cells_in_window_mask = (cells_times_of_death >= window_start_min) * (cells_times_of_death < window_end_min)
    else:
        dead_cells_in_window_mask = (cells_times_of_death < window_end_min)

    # dead_cells_in_window_mask = cells_times_of_death[death_mask]
    dead_cells_in_window_indices = np.where(dead_cells_in_window_mask)[0]
    dead_cells_in_window_indices_set = set(dead_cells_in_window_indices)
    # check which of the cells that are dead have neighbors that are also dead in the time window (/up to window included)
    adjacent_dead_cells = set()
    for dead_cell_idx in dead_cells_in_window_indices:
        dead_cell_neighbors_indices = cells_neighbors[dead_cell_idx]
        dead_cell_neighbors_indices_set = set(dead_cell_neighbors_indices)
        adjacent_dead_cells.update(dead_cells_in_window_indices_set.intersection(dead_cell_neighbors_indices_set))

    fraction_of_adjacent_dying_cells_in_time_window = len(adjacent_dead_cells) / len(
        dead_cells_in_window_indices) if len(dead_cells_in_window_indices) > 0 else 0
    indices_of_adjacent_dying_cells_in_time_window_set = adjacent_dead_cells
    return fraction_of_adjacent_dying_cells_in_time_window, indices_of_adjacent_dying_cells_in_time_window_set


def calc_time_difference_of_adjacent_death_in_time_window(window_start_min: int,
                                                          window_end_min: int,
                                                          cells_neighbors: np.array,
                                                          cells_times_of_death: np.array,
                                                          **kwargs) -> Tuple[float, set]:
    raise NotImplemented('Difference in time of death is not suitable for sliding windows measurements!')


def calc_single_exp_measurement_in_sliding_time_window(cells_xy: np.array,
                                                       cells_times_of_death: np.array,
                                                       exp_temporal_resolution: int,
                                                       cells_neighbors: List[List[int]],
                                                       sliding_window_size_in_minutes: int,
                                                       exp_treatment: str,
                                                       exp_name: str,
                                                       **kwargs):
    visualize_flag = kwargs.get('visualize_flag', True)
    calculation_type = kwargs.get('calculation_type', 'fraction_of_adjacent_death')

    # default is fraction_of_adjacent_death (performed in else as well)
    if calculation_type == 'fraction_of_adjacent_death':
        dir_of_calc = ['TemporalMeasurementsPlots', 'FractionOfAdjacentDeathsInSlidingTimeWindows']
    elif calculation_type == 'adjacent_death_time_difference':
        dir_of_calc = ['TemporalMeasurementsPlots', 'AdjacentDeathTimeDifference']
    else:
        dir_of_calc = ['TemporalMeasurementsPlots', 'FractionOfAdjacentDeathsInSlidingTimeWindows']

    death_times_by_min = np.arange(cells_times_of_death.min(), cells_times_of_death.max(), exp_temporal_resolution)
    sliding_windows_indices_by_min = [(window_start, window_start + sliding_window_size_in_minutes) for window_start in
                                      death_times_by_min]
    measurement_for_time_windows = []
    indices_of_measurement_in_single_window_set = set()
    for sliding_window_start, sliding_window_end in sliding_windows_indices_by_min:
        if calculation_type == 'fraction_of_adjacent_death':
            measurement_in_single_window, indices_of_measurement_in_single_window = calc_fraction_of_adjacent_dying_cells_in_time_window(
                window_start_min=sliding_window_start,
                window_end_min=sliding_window_end,
                cells_neighbors=cells_neighbors,
                cells_times_of_death=cells_times_of_death,
                **kwargs)
        elif calculation_type == 'adjacent_death_time_difference':
            measurement_in_single_window, indices_of_measurement_in_single_window = calc_time_difference_of_adjacent_death_in_time_window(
                window_start_min=sliding_window_start,
                window_end_min=sliding_window_end,
                cells_neighbors=cells_neighbors,
                cells_times_of_death=cells_times_of_death,
                **kwargs
            )
        else:
            measurement_in_single_window, indices_of_measurement_in_single_window = calc_fraction_of_adjacent_dying_cells_in_time_window(
                window_start_min=sliding_window_start,
                window_end_min=sliding_window_end,
                cells_neighbors=cells_neighbors,
                cells_times_of_death=cells_times_of_death,
                **kwargs)

        measurement_for_time_windows.append(measurement_in_single_window)
        indices_of_measurement_in_single_window_set.update(indices_of_measurement_in_single_window)

    if visualize_flag:
        path_for_fig = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Results'] +
                                   dir_of_calc +
                                   [f'{exp_treatment}'])
        plot_measurements_by_time(measurement1_by_time=measurement_for_time_windows,
                                  temporal_resolution=exp_temporal_resolution,
                                  exp_treatment=exp_treatment,
                                  exp_name=exp_name,
                                  full_path_to_save_fig=path_for_fig)
    if calculation_type == 'fraction_of_adjacent_death':
        measurement_endpoint_readout = len(indices_of_measurement_in_single_window_set) / len(
            cells_times_of_death)
    elif calculation_type == 'adjacent_death_time_difference':
        measurement_endpoint_readout = np.array(measurement_for_time_windows).mean()
    else:
        measurement_endpoint_readout = len(indices_of_measurement_in_single_window_set) / len(
            cells_times_of_death)

    return sliding_windows_indices_by_min, \
           np.array(measurement_for_time_windows), \
           indices_of_measurement_in_single_window_set, \
           measurement_endpoint_readout


def calc_time_difference_of_adjacent_death_in_single_experiment(
        cells_neighbors: np.array,
        cells_times_of_death: np.array,
        exp_name: str,
        exp_treatment: str,
        **kwargs) -> Tuple[np.array, float]:
    """
    calculates the distribution and mean value of time differences between adjacent cells' deaths.
    all temporal values are in minutes
    :param cells_neighbors: list of all cells' neighbors (by their indices), list of list
    :param cells_times_of_death: np.array of times of death in minutes for each cell.
    :param kwargs:
    :return: np.array adjacent death time differences distribution(hist), flaot - mean value of distribution.
    """
    visualize_flag = kwargs.get('visualize_flag', False)
    bins_as_minutes = kwargs.get('bins_of_adjacent_death_diff', None)
    min_max_normalization_on_hist_flag = kwargs.get('min_max_normalization_on_hist_flag', False)

    if bins_as_minutes is None:
        bins_as_minutes = kwargs.get('number_of_adjacent_death_diff_hist_bins', 10)
    # to avoid taking into account the same cells multiple times, a set of examined cells
    # is kept
    examined_cells = set()
    total_adjacent_death_time_diffs = []
    for curr_cell_idx, curr_cell_death in enumerate(cells_times_of_death):
        curr_cell_neighbors = cells_neighbors[curr_cell_idx]
        cell_adjacent_death_times = []
        for neighbor_idx in curr_cell_neighbors:
            if neighbor_idx in examined_cells:
                continue
            cell_adjacent_death_times.append(cells_times_of_death[neighbor_idx])
        cell_adjacent_death_times = np.array(cell_adjacent_death_times)
        cell_adjacent_death_times_diff_from_curr_cell_death = cell_adjacent_death_times - curr_cell_death
        total_adjacent_death_time_diffs += cell_adjacent_death_times_diff_from_curr_cell_death.tolist()
        examined_cells.update([curr_cell_idx])

    total_adjacent_death_time_diffs = np.array(total_adjacent_death_time_diffs)
    mean_of_adjacent_death_diff = total_adjacent_death_time_diffs.mean()
    total_adjacent_death_time_diffs_hist = np.histogram(total_adjacent_death_time_diffs, bins=bins_as_minutes)[0]

    if min_max_normalization_on_hist_flag:
        total_adjacent_death_time_diffs_hist = (total_adjacent_death_time_diffs_hist-total_adjacent_death_time_diffs_hist.min())/(total_adjacent_death_time_diffs_hist.max()-total_adjacent_death_time_diffs_hist.min())

    if visualize_flag:
        exp_treatment = clean_string_from_bad_chars(treatment_name=exp_treatment)
        dir_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Results',
                                                                 'AdjacentDeathTimeDifferences', f'{exp_treatment}'])

        x_ticks, x_tick_labels = None, None
        if isinstance(bins_as_minutes, Iterable):
            x_ticks = bins_as_minutes[:-1]
            x_tick_labels = [f'{bins_as_minutes[bin_idx]}-{bins_as_minutes[bin_idx+1]}' for bin_idx in range(len(bins_as_minutes[:-1]))]

        visualize_histogram_of_values(hist_values=total_adjacent_death_time_diffs_hist,
                                      title=f'{exp_treatment}\n{exp_name}',
                                      x_label='Time differences between adjacent deaths',
                                      y_label='Time (Min)',
                                      x_tick_labels=x_tick_labels,
                                      path_to_dir_to_save=dir_path,
                                      fig_name=f'{exp_name}')
    return total_adjacent_death_time_diffs_hist, mean_of_adjacent_death_diff


def calc_multiple_exps_measurements(main_exp_dir_full_path: str,
                                    limit_exp_num: int = float('inf'),
                                    **kwargs):
    ######
    # getting all kwargs
    visualize_flag = kwargs.get('visualize_flag', False)
    treatments_to_include = kwargs.get('treatments_to_include', 'all')
    use_log_flag = kwargs.get('use_log', USE_LOG)
    meta_data_file_full_path = kwargs.get('metadata_file_full_path', METADATA_FILE_FULL_PATH)
    compressed_flag = kwargs.get('use_compressed_exps_data_flag', False)
    neighbors_threshold_dist = kwargs.get('neighbors_threshold_dist', DIST_THRESHOLD_IN_PIXELS)
    use_sliding_time_window = kwargs.get('use_sliding_time_window', False)
    if use_sliding_time_window:
        sliding_time_window_size_in_min = kwargs.get('sliding_time_window_size_in_min', 100)
        type_of_measurement = 'fraction_of_adjacent_death'
    else:
        type_of_measurement = kwargs.get('type_of_measurement', 'adjacent_death_time_difference')
        bins_of_adjacent_death_diff = kwargs.get('bins_of_adjacent_death_diff', None)
    #####

    # # LOG FILE: clean and write headline
    # with open('../experiments_with_bad_results.txt', 'w') as f:
    #     f.write(f'analyzed directory path:{main_exp_dir_full_path}\nExperiments with P(Nuc)+P(Prop)<1 : \n')

    if main_exp_dir_full_path is None:
        main_exp_dir_full_path = os.sep.join(
            os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV', 'OriginalTimeMinutesData'])

    all_files_to_analyze_full_paths, all_files_to_analyze_only_exp_names = \
        get_all_paths_csv_files_in_dir(dir_path=dir_path)

    total_exps = len(all_files_to_analyze_only_exp_names)

    all_endpoint_readouts_by_experiment, all_exps_names, all_exps_treatments = list(), list(), list()

    for exp_idx, single_exp_full_path in enumerate(all_files_to_analyze_full_paths):
        if limit_exp_num < exp_idx + 1:
            break

        exp_name = all_files_to_analyze_only_exp_names[exp_idx]

        exp_treatment, explicit_temporal_resolution = \
            get_exp_treatment_type_and_temporal_resolution(exp_file_name=single_exp_full_path.split(os.sep)[-1],
                                                           meta_data_file_full_path=meta_data_file_full_path,
                                                           compressed_flag=compressed_flag)
        # skip un-wanted treatments
        if treatments_to_include != 'all' and sum([treatment_shortname in exp_treatment.lower() for treatment_shortname in treatments_to_include]) == 0:
            continue

        all_exps_names.append(exp_name)
        all_exps_treatments.append(exp_treatment)

        print(f'analyzing exp {exp_name} | {exp_idx + 1}/{total_exps}')
        exp_df = pd.read_csv(single_exp_full_path)
        cells_xy, cells_times_of_death = exp_df.loc[:, ['cell_x', 'cell_y']].values, exp_df.loc[:,
                                                                                     ['death_time']].values
        cells_neighbors_lvl1, cells_neighbors_lvl2, cells_neighbors_lvl3 = get_cells_neighbors(cells_xy,
                                                                                  threshold_dist=neighbors_threshold_dist)
        # if treatments_to_include == 'all':
        #     kwargs['visualize_flag'] = False
        if use_sliding_time_window:
            sliding_windows_indices_by_min, \
            measurement_temporal_readout_in_windows, \
            indices_of_measurement_in_windows_set, \
            measurement_endpoint_readout = calc_single_exp_measurement_in_sliding_time_window(
                cells_xy=cells_xy,
                cells_times_of_death=cells_times_of_death,
                cells_neighbors=cells_neighbors_lvl1,
                exp_temporal_resolution=explicit_temporal_resolution,
                sliding_window_size_in_minutes=sliding_time_window_size_in_min,
                exp_treatment=exp_treatment,
                exp_name=exp_name,
                **kwargs)
        elif type_of_measurement == 'adjacent_death_time_difference':
            total_adjacent_death_time_diffs_hist, measurement_endpoint_readout = calc_time_difference_of_adjacent_death_in_single_experiment(
                cells_neighbors=cells_neighbors_lvl1,
                cells_times_of_death=cells_times_of_death,
                exp_name=exp_name,
                exp_treatment=exp_treatment,
                **kwargs
            )

        kwargs['visualize_flag'] = True

        all_endpoint_readouts_by_experiment.append(measurement_endpoint_readout)

    if visualize_flag:
        if type_of_measurement == 'adjacent_death_time_difference':
            path_for_fig = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                                         'MeanTimeOfAdjacentDeath'])
        else:
            path_for_fig = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                                     'FractionOfAdjacentDeathsInSlidingTimeWindows'])
        visualize_endpoint_readouts_by_treatment_about_readouts(x_readout=all_endpoint_readouts_by_experiment,
                                                                y_readout=np.ones_like(
                                                                    all_endpoint_readouts_by_experiment),
                                                                treatment_per_readout=all_exps_treatments,
                                                                full_path_to_save_fig=path_for_fig, x_label='Treatment',
                                                                y_label='Fraction of adjacent deaths',
                                                                plot_about_treatment=True,
                                                                set_y_lim=False)


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
    # MULTIPLE EXPERIMENTS
    # calc_and_visualize_all_experiments_csvs_in_dir(limit_exp_num=float('inf'),
    #                                                dir_path=os.sep.join(
    #                                                    os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV',
    #                                                                                      'OriginalTimeMinutesData']))

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

    # plot kl-divergence between recent death only flags endpoint readouts
    # kwargs = dict()
    # kwargs['distance_metric'] = 'kl_divergence'
    # kwargs['visualize_flag'] = True
    # kwargs['visualize_specific_treatments'] = ['tnf', 'erastin', 'trail']
    # dir_path = os.sep.join(
    #     os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV',
    #                                  'OriginalTimeMinutesData'])
    # by_treatment_distance_metric_score_p_nuc, by_treatment_distance_metric_score_p_prop = \
    #     calc_distance_metric_between_experiments_results_of_altering_flag_values(
    #     dir_path=dir_path, **kwargs)

    # single simulation xyt results testing
    # path = '..\\Data\\Simulations_XYT_CSV\\apoptosis_attempts\\xyt_ferroptosis_attempt.csv'
    # temporal_resolution = 15
    # p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
    # all_frames_nucleators_mask, all_frames_propagators_mask,\
    #     accumulated_fraction_of_death_by_time = \
    #     calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_provided_temporal_resolution(
    #         single_exp_full_path=path, temporal_resolution=temporal_resolution)
    #
    # visualize_cell_death_in_time(xyt_full_path=path, nucleators_mask=all_frames_nucleators_mask, exp_treatment='just_treatment', exp_name='just name')
    #
    # plot_measurements_by_time(p_nuc_by_time, p_prop_by_time, accumulated_fraction_of_death_by_time,
    #                           temporal_resolution=temporal_resolution, exp_name='simulation_test',
    #                           exp_treatment='simulation')

    ## single experiment probabilities per unit of time testing
    # exp_path = 'C:\\Users\\User\\PycharmProjects\\CellDeathQuantification\\Data\\Experiments_XYT_CSV\\OriginalTimeMinutesData\\20160820_10A_FB_xy13.csv'
    #
    # calc_slopes_and_probabilities_per_unit_of_time_single_experiment(exp_full_path=exp_path,
    #                                                                      exp_temporal_resolution=10,
    #                                                                      exp_treatment='C dots',
    #                                                                      unit_of_time_min=60,
    #                                                                      consider_majority_of_death_only=True)

    # multiple experiments slopes and probabilities per unit of time testing
    # dir_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV', 'OriginalTimeMinutesData'])
    # calc_slopes_and_probabilities_per_unit_of_time_entire_dir(dir_full_path=dir_path,
    #                                                           treatments_to_include='all',
    #                                                           unit_of_time_min=60,
    #                                                           consider_majority_of_death_only=True)

    # p_prop measurement as a fraction of adjacent dying cells in window:
    # dir_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV', 'OriginalTimeMinutesData'])
    # calc_multiple_exps_measurements(main_exp_dir_full_path=dir_path,
    #                                 limit_exp_num=float('inf'),
    #                                 use_sliding_time_window=True,
    #                                 sliding_time_window_size_in_min=60,
    #                                 visualize_flag=True,
    #                                 consider_death_within_window_only_flag=True)
    # time of adjacent death diff measurement
    # time of death differences between adjacent cells
    dir_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data', 'Experiments_XYT_CSV', 'OriginalTimeMinutesData'])
    calc_multiple_exps_measurements(main_exp_dir_full_path=dir_path,
                                    limit_exp_num=float('inf'),
                                    use_sliding_time_window=False,
                                    type_of_measurement='adjacent_death_time_difference',
                                    bins_of_adjacent_death_diff=np.arange(0, 201, 10),
                                    treatments_to_include='all',#['superkiller', 'fac', 'erastin'],
                                    visualize_flag=True
                                    )
    pass

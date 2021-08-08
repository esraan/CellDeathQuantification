import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from NucleatorsProbabilities import *
from utils import *
from global_parameters import *
from Visualization import *


def calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths: np.array,
                                                         cells_neighbors_lists: List[List]):
    neighbors_of_dead_cells_probability_to_die_at_all_frames = []
    implicit_temporal_resolution = cells_times_of_deaths[1] - cells_times_of_deaths[0]
    unique_times_of_death = np.unique(cells_times_of_deaths)
    for timeframe in unique_times_of_death:
        # get all dead cells at time = timeframe
        dead_cells_at_timeframe = np.where(cells_times_of_deaths == timeframe)[0]
        alive_cells_at_timeframe = np.where(cells_times_of_deaths > timeframe)[0]
        # get all dead cells at time = timeframe + implicit_temporal_resolution
        dead_cells_at_next_timeframe = np.where(cells_times_of_deaths == (timeframe + implicit_temporal_resolution))[0]
        # get all neighbors of dead_cells_at_timeframe
        neighbors_of_dead_cells = set()
        for dead_cell_idx in dead_cells_at_timeframe:
            neighbors_of_dead_cells.update(cells_neighbors_lists[dead_cell_idx])
        # keep only alive cells as neighbors at timeframe
        neighbors_of_dead_cells = neighbors_of_dead_cells.intersection(
            set(alive_cells_at_timeframe))

        # get all cells in neighbors_of_dead_cell that are dead in next timeframe and were alive before
        dead_neighbors_at_next_timeframe_of_dead_cells = neighbors_of_dead_cells.copy()
        # intersect with dead cells at timeframe_ + implicit_temporal_resolution
        dead_neighbors_at_next_timeframe_of_dead_cells = dead_neighbors_at_next_timeframe_of_dead_cells.intersection(
            set(dead_cells_at_next_timeframe))
        if len(neighbors_of_dead_cells) == 0 and len(dead_neighbors_at_next_timeframe_of_dead_cells) != 0:
            raise RuntimeWarning(
                f'number of alive neighbors of dead cells is zero, but number of dead neighbors of dead cells is not! timeframe:{timeframe}')
        neighbors_of_dead_cells_probability_to_die_at_curr_frame = len(
            dead_neighbors_at_next_timeframe_of_dead_cells) / (len(neighbors_of_dead_cells) + EPSILON)
        neighbors_of_dead_cells_probability_to_die_at_all_frames.append(
            neighbors_of_dead_cells_probability_to_die_at_curr_frame)
    return neighbors_of_dead_cells_probability_to_die_at_all_frames


def calc_pnuc_at_varying_distances_of_neighbors(exp_filename,
                                                exp_main_directory_path,
                                                file_details_full_path,
                                                path_to_save_fig_no_type=''):
    """
    plots p(nuc) at two different distances from dead cells for a single experiment
    :param exp_filename: the experiment XYT file name (including .csv)
    :param exp_main_directory_path: the directory in which the file resides.
    :param file_details_full_path: the full path to the file details csv file.
    :param func_mode: either single or multi. multi returns the signal instead of plotting it.
    :param plot_kwargs:
    :return:
    """
    exp_path = os.sep.join([exp_main_directory_path, exp_filename])
    exp_xyt = pd.read_csv(exp_path)
    exp_details_df = pd.read_csv(file_details_full_path)
    full_x = exp_xyt["cell_x"].values
    full_y = exp_xyt["cell_y"].values
    # n_instances = len(full_x)
    die_times = exp_xyt["death_time"].values
    XY = np.column_stack((full_x, full_y))
    exp_temporal_resolution = exp_details_df[exp_details_df['File Name'] == exp_filename]['Time Interval (min)'].values[
        0]
    exp_treatment_type = exp_details_df[exp_details_df['File Name'] == exp_filename]['Treatment'].values[0]
    jump_interval = exp_temporal_resolution
    # time_window_size = WINDOW_SIZE * jump_interval

    # get neighbors list of all cells (topological by Voronoi)
    neighbors_list, neighbors_list2, neighbors_list3 = get_cells_neighbors(XY=XY,
                                                                           threshold_dist=DIST_THRESHOLD_IN_PIXELS)

    nuc_probas_calculator_lvl_1 = np.asarray(
        calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths=die_times,
                                                             cells_neighbors_lists=neighbors_list))

    nuc_probas_calculator_lvl_2 = np.asarray(
        calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths=die_times,
                                                             cells_neighbors_lists=neighbors_list2))
    nuc_probas_calculator_lvl_3 = np.asarray(
        calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths=die_times,
                                                             cells_neighbors_lists=neighbors_list3))
    org_path = path_to_save_fig_no_type
    path_to_save_fig_no_type = org_path + '_2on3'
    scatter_with_linear_regression_line(nuc_probas_calculator_lvl_2, nuc_probas_calculator_lvl_3,
                                        x_label='Nucleation probability at level 2 neighborhood',
                                        y_label='Nucleation probability at level 3 neighborhood',
                                        title=f'experiment treatment: {exp_treatment_type}',
                                        path_to_save_fig=path_to_save_fig_no_type)

    path_to_save_fig_no_type = org_path + '_1on2'
    scatter_with_linear_regression_line(nuc_probas_calculator_lvl_1, nuc_probas_calculator_lvl_2,
                                        x_label='Nucleation probability at level 1 neighborhood',
                                        y_label='Nucleation probability at level 2 neighborhood',
                                        title=f'experiment treatment: {exp_treatment_type}',
                                        path_to_save_fig=path_to_save_fig_no_type)

    path_to_save_fig_no_type = org_path + '_1on3'
    scatter_with_linear_regression_line(nuc_probas_calculator_lvl_1, nuc_probas_calculator_lvl_3,
                                        x_label='Nucleation probability at level 1 neighborhood',
                                        y_label='Nucleation probability at level 3 neighborhood',
                                        title=f'experiment treatment: {exp_treatment_type}',
                                        path_to_save_fig=path_to_save_fig_no_type)


if __name__ == '__main__':
    exp_main_dir_path = '../Data/Experiments_XYT_CSV/OriginalTimeMinutesData'
    exp_meta_data_full_path = '../Data/Experiments_XYT_CSV/ExperimentsMetaData.csv'
    for filename in filter(lambda x: x.endswith('.csv'), os.listdir(exp_main_dir_path)):
        path_to_save_fig = os.sep.join(
            ['../Results', 'NucleationProbabilitiesForVaryingLevelsOfNeighborhoods', filename.replace('.csv', '')])
        calc_pnuc_at_varying_distances_of_neighbors(exp_filename=filename,
                                                    exp_main_directory_path=exp_main_dir_path,
                                                    file_details_full_path=exp_meta_data_full_path,
                                                    path_to_save_fig_no_type=path_to_save_fig)

    # calc_pnuc_at_varying_distances_of_neighbors(exp_filename='20180620_HAP1_erastin_xy7.csv',
    #                                             exp_main_directory_path=exp_main_dir_path,
    #                                             file_details_full_path=exp_meta_data_full_path)
    # calc_pnuc_at_varying_distances_of_neighbors(exp_filename='20181227_MCF10A_SKT_xy4.csv',
    #                                             exp_main_directory_path=exp_main_dir_path,
    #                                             file_details_full_path=exp_meta_data_full_path)

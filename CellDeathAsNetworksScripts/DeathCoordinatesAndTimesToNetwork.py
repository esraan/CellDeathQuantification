import os
import sys
import shutil
import math
import warnings
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import graph_tool.all as gt
from utils import get_cells_neighbors
from OldCodeBase_15072021.NucliatorsCount import NucleatorsCounter

def convert_cell_deaths_to_graph_tool_networks(
        coordinates_and_times: pd.DataFrame,
        **kwargs) -> List[gt.Graph]:
    times_of_death = coordinates_and_times.loc[:, kwargs.get("times_col_name", "time_of_death")].values
    cells_coordinates = coordinates_and_times.loc[:, kwargs.get("coordinates_col_names", ["x", "y"])].values
    time_frames = np.unique(times_of_death)

    vertices_indices = np.arange(0, times_of_death.size, 1)
    initial_graph = gt.Graph()  # the graph representing only cell locations prior to any death
    initial_graph.add_vertex(len(vertices_indices))   # add all vertices
    cells_coordinates_prop_map = initial_graph.new_vertex_property("vector<double>", vals=cells_coordinates)
    initial_graph.vp[kwargs.get("cell_position_property_name", "coordinates")] = cells_coordinates_prop_map
    by_frame_networks = [initial_graph]
    neighbors_list_lvl1, neighbors_list_lvl2, neighbors_list_lvl3 = get_cells_neighbors(XY=cells_coordinates,
                                                                                        threshold_dist=kwargs.get("neighbors_max_dist_threshold_px", 100))

    exp_nc = NucleatorsCounter(XY=cells_coordinates, TIMES=times_of_death, neighbors_list=neighbors_list_lvl1,
                               dist_threshold_nucleators_detection=100)
    exp_nucleators = exp_nc.calc_nucleators()
    vertices_colors_by_nucleators = ["red" if cell_is_nuc else "blue" for cell_is_nuc in exp_nucleators]
    cells_colors_prop_map = initial_graph.new_vertex_property("string", vals=vertices_colors_by_nucleators)
    initial_graph.vp[kwargs.get('vertices_is_nuc_property', 'vertex_is_nuc_color')] = cells_colors_prop_map

    prev_graph = initial_graph.copy()
    # todo: add property map for edges - to include weight of ∆TOD for each edge. consider doing this only with a flag (because it will create an edge between all pairs of neighbors).
    curr_time_frame = time_frames[0]
    prev_frame_cells_dead_in_frame_mask = times_of_death == curr_time_frame
    for curr_time_frame in time_frames[1:]:
        cells_dead_in_frame_mask = times_of_death == curr_time_frame
        cells_dead_in_frame_indices = np.where(cells_dead_in_frame_mask)[0]
        # cells_dead_prev_to_frame_mask = times_of_death[times_of_death < curr_time_frame]
        for curr_frame_dead_cell_idx in cells_dead_in_frame_indices:
            curr_frame_dead_cell_neighbors = neighbors_list_lvl1[curr_frame_dead_cell_idx]
            for curr_frame_dead_cell_neighbor_idx in curr_frame_dead_cell_neighbors:
                # if neighbor died in current frame as well, create an edge between both vertices
                if cells_dead_in_frame_mask[curr_frame_dead_cell_neighbor_idx]:
                    prev_graph.add_edge(curr_frame_dead_cell_idx, curr_frame_dead_cell_neighbor_idx, add_missing=False)
                # if neighbor died in previous frame, create an edge between both vertices
                if prev_frame_cells_dead_in_frame_mask[curr_frame_dead_cell_neighbor_idx]:
                    prev_graph.add_edge(curr_frame_dead_cell_idx, curr_frame_dead_cell_neighbor_idx, add_missing=False)
                # if neighbor did not die in previous frames and in current frame, do nothing
        prev_graph.set_directed(is_directed=False)
        gt.remove_parallel_edges(prev_graph)
        by_frame_networks.append(prev_graph)
        prev_frame_cells_dead_in_frame_mask = cells_dead_in_frame_mask.copy()
        prev_graph = prev_graph.copy()

    return by_frame_networks


def visualize_list_of_networks(networks_lst: List[gt.Graph],
                               dir_path_to_save_networks_illustrations: Union[str, os.PathLike],
                               **kwargs) -> None:
    if not os.path.isdir(dir_path_to_save_networks_illustrations):
        os.makedirs(dir_path_to_save_networks_illustrations, exist_ok=False)

    for frame_idx, frame_network in enumerate(networks_lst):
        frame_time = frame_idx * kwargs.get("temporal_resolution", 1)
        network_fname = f"{kwargs.get('network_name', 'test')}_time-{frame_time}.{kwargs.get('network_visualization_format', 'png')}"
        network_fpath = os.path.join(dir_path_to_save_networks_illustrations, network_fname)
        gt.graph_draw(frame_network,
                      pos=frame_network.vp[kwargs.get("cell_position_property_name", "coordinates")],
                      vertex_fill_color=frame_network.vp[kwargs.get('vertices_is_nuc_property', 'vertex_is_nuc_color')],
                      output=network_fpath,
                      vertex_size=5,
                      edge_pen_width=2,
                      bg_color="white")


if __name__ == '__main__':
    # test_exp_dir_path = "/Users/yishaiazabary/PycharmProjects/University/CellDeathQuantification/Data/Experiments_XYT_CSV/OriginalTimeFramesData"
    # test_exp_fname = "20180620_HAP1_erastin_xy1.csv"
    # test_exp_path = os.path.join(test_exp_dir_path, test_exp_fname)
    test_exp_fname = "20230314_ML_Sytox1_RawTrackmateOutputSpotsCoordinates.csv"
    test_exp_path = "/Users/yishaiazabary/Downloads/ML_Sytox1_TIFF_Stack/20230314_ML_Sytox1_RawTrackmateOutputSpotsCoordinates.csv"
    test_exp_df = pd.read_csv(test_exp_path)
    list_of_networks_by_frames = convert_cell_deaths_to_graph_tool_networks(coordinates_and_times=test_exp_df,
                                                                            times_col_name="death_time",
                                                                            coordinates_col_names=["cell_x","cell_y"])
    dir_path_to_save_results = os.path.join("..", "Results", "CellDeathAsNetworksByFrames", test_exp_fname)
    visualize_list_of_networks(networks_lst=list_of_networks_by_frames,
                               dir_path_to_save_networks_illustrations=dir_path_to_save_results,
                               network_name=test_exp_fname,
                               network_visualization_format='png')

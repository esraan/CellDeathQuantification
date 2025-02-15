import os
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


class NucleatorsCounter:
    def __init__(self, XY, TIMES, neighbors_list, dist_threshold_nucleators_detection):
        self.XY = XY
        self.TIMES = TIMES
        self.neighbors_list = neighbors_list
        self.dist_threshold_nucleators_detection = dist_threshold_nucleators_detection

    @staticmethod
    def get_real_distance(cell1_xy, cell2_xy):
        cell1_x, cell1_y = cell1_xy
        cell2_x, cell2_y = cell2_xy
        return ((cell1_x - cell2_x)**2 + (cell1_y - cell2_y)**2)**.5

    def calc_nucleators(self, curr_TOD=0, max_TOD=None):

        # def nucleator_in_single_time_frame(self_ob: NucleatorsCounter, start, end, curr_propagated):
        #     current_dead_mask = (self_ob.TIMES == start)
        #     next_time_frame_dead_mask = (self_ob.TIMES == end)
        #
        #     # collect all dead cells neighbors
        #     all_neighbors_of_dead = set()
        #     for curr_dead_idx, curr_dead_stat in enumerate(current_dead_mask):
        #         if not curr_dead_stat:
        #             continue
        #         curr_cell_xy = self_ob.XY[curr_dead_idx]
        #         dead_cells_neighbors = self_ob.neighbors_list[curr_dead_idx]
        #         dead_cells_neighbors = list(filter(lambda neighbor_idx: NucleatorsCounter.get_real_distance(curr_cell_xy, self_ob.XY[neighbor_idx]) <= self_ob.dist_threshold_nucleators_detection, dead_cells_neighbors))
        #         all_neighbors_of_dead.update(dead_cells_neighbors)
        #
        #     # dead and in neighbors are propagated
        #     for nei_of_dead in all_neighbors_of_dead:
        #         if current_dead_mask[nei_of_dead]:
        #             curr_propagated

        def nucleator_in_single_time_frame(self_ob: NucleatorsCounter, start, end, curr_propagated):
            current_dead_mask = (self_ob.TIMES == start)
            next_time_frame_dead_mask = (self_ob.TIMES == end)
            # collect all dead cells neighbors
            all_neighbors_of_dead = set()
            for curr_dead_idx, curr_dead_stat in enumerate(current_dead_mask):
                if not curr_dead_stat:
                    continue
                curr_cell_xy = self_ob.XY[curr_dead_idx]
                dead_cells_neighbors = self_ob.neighbors_list[curr_dead_idx]
                dead_cells_neighbors = list(filter(lambda neighbor_idx: NucleatorsCounter.get_real_distance(curr_cell_xy, self_ob.XY[neighbor_idx]) <= self_ob.dist_threshold_nucleators_detection, dead_cells_neighbors))
                all_neighbors_of_dead.update(dead_cells_neighbors)

            dead_and_marked = set()
            # dead and in neighbors are propagated
            for nei_of_dead in all_neighbors_of_dead:
                if current_dead_mask[nei_of_dead]:
                    curr_propagated[nei_of_dead] = True
                    dead_and_marked.add(nei_of_dead)
            # for neighbor_of_dead_in_curr_frame in all_neighbors_of_dead:
            #     death_of_neighbor = self_ob.TIMES[neighbor_of_dead_in_curr_frame]
            #     if end > death_of_neighbor >= start:
            #         curr_propagated[neighbor_of_dead_in_curr_frame] = True
            #         dead_and_marked.add(neighbor_of_dead_in_curr_frame)
            # all dead cells neighbors that are dead in next time frame are propagated
            for neighbor_of_dead_in_curr_frame in all_neighbors_of_dead:
                is_dead_in_next_frame = next_time_frame_dead_mask[neighbor_of_dead_in_curr_frame]
                if not is_dead_in_next_frame:
                    continue
                curr_propagated[neighbor_of_dead_in_curr_frame] = True
                dead_and_marked.add(neighbor_of_dead_in_curr_frame)

            # go through neighbors of marked as propagated, if neighbor is dead in same frame, it is propagated
            dead_and_marked_cpy = dead_and_marked.copy()
            for marked in dead_and_marked_cpy:
                marked_neighbor_xy = self_ob.XY[marked]
                marked_neighbors = self_ob.neighbors_list[marked]
                dead_in_next_frame_neighbors = list(filter(lambda neighbor_idx: NucleatorsCounter.get_real_distance(marked_neighbor_xy, self_ob.XY[neighbor_idx]) <= self_ob.dist_threshold_nucleators_detection, marked_neighbors))
                for nei in dead_in_next_frame_neighbors:
                    if self_ob.TIMES[nei] == self_ob.TIMES[marked]:
                        curr_propagated[nei] = True
                        dead_and_marked.add(nei)

            # go through dead in next frame, if not marked as propagated, divide into blobs
            blobs = []
            for dead_in_next_idx, death_stat in enumerate(next_time_frame_dead_mask):
                if not death_stat or dead_in_next_idx in dead_and_marked:
                    continue
                # check if the cell or its neighbors are in one of the blobs
                found_blob = False
                for blob in blobs:
                    dead_in_next_XY = self_ob.XY[dead_in_next_idx]
                    dead_in_next_frame_neighbors = self_ob.neighbors_list[dead_in_next_idx]
                    dead_in_next_frame_neighbors = list(filter(lambda neighbor_idx: NucleatorsCounter.get_real_distance(dead_in_next_XY, self_ob.XY[neighbor_idx]) <= self_ob.dist_threshold_nucleators_detection, dead_in_next_frame_neighbors))
                    for dead_in_next_frame_neighbor in dead_in_next_frame_neighbors:
                        if dead_in_next_frame_neighbor in blob:
                            blob.append(dead_in_next_idx)
                            found_blob = True
                # if no blob was found to put the cell in, create a new blob
                if not found_blob:
                    blobs.append([dead_in_next_idx])

            # for each blob, choose a single nucleator and the rest are labeled as propagators.
            for blob in blobs:
                choose_random = False
                for cell_idx in blob:
                    if not choose_random:
                        # skip first cell as to be nucleator
                        choose_random = True
                        continue
                    curr_propagated[cell_idx] = True
            return curr_propagated

            # todo: refactor into this function to use in frame by frame calculations
        implicit_temporal_resolution = np.unique(self.TIMES)[1] - np.unique(self.TIMES)[0]
        # curr_TOD = 0
        max_time = self.TIMES.max() - implicit_temporal_resolution if max_TOD is None else max_TOD
        propagated = {key:False for key in range(len(self.TIMES))}
        while curr_TOD <= max_time:
            propagated = nucleator_in_single_time_frame(self_ob=self, start=curr_TOD, end=curr_TOD+implicit_temporal_resolution, curr_propagated=propagated.copy())
            curr_TOD += implicit_temporal_resolution
        if max_TOD is not None:
            times_mask = self.TIMES>curr_TOD+implicit_temporal_resolution
            single_frame_prop = np.array(np.array(list(propagated.values()))-1, dtype=bool)
            single_frame_prop[times_mask] = False
            return single_frame_prop
        return np.array(np.array(list(propagated.values()))-1, dtype=bool)#, propagated, adjacent_nucleator_candidates/len(XY)

    @staticmethod
    def get_neighbors(XY):
        vor = Voronoi(XY)
        neighbors = vor.ridge_points
        neighbors_list = []
        neighbors_list2 = []
        neighbors_list3 = []
        for i in range(len(XY)):
            neighbors_list.append([])
            neighbors_list2.append([])
            neighbors_list3.append([])
        for x in neighbors:
            neighbors_list[x[0]].append(x[1])
            neighbors_list[x[1]].append(x[0])
        for i in range(len(XY)):
            for j in neighbors_list[i]:
                neighbors_list2[i] = list(set(neighbors_list2[i]+neighbors_list[j]))
        for i in range(len(XY)):
            for j in neighbors_list2[i]:
                neighbors_list3[i] = list(set(neighbors_list3[i]+neighbors_list2[j]))
        return neighbors_list, neighbors_list2, neighbors_list3



# file_name = '20181229_HAP1-920H_FB+PEG1450_GCAMP_xy51.csv'

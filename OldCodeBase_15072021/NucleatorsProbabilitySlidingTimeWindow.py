import json
import os
import math
import numpy as np
import numpy.polynomial.polynomial as nppoly
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, linregress
from NucliatorsCount import NucleatorsCounter
from sklearn import metrics


ALWAYS = True
MAX_EXP_TO_PROCESS = 300
FONT_SIZE = 10
SHOWFIG = False
SAVEFIG = True
WINDOW_SIZE_IN_FRAMES = 3


class NucleatorsProbabilitySlidingTimeWindow:
    def __init__(self, time_window_size=3, exp_xyt_full_path=None, exp_details_full_path=None):
        if exp_xyt_full_path is None:
            raise ValueError('must have path to experiment xyt data')
        if exp_details_full_path is None:
            raise ValueError('must have path to experiments details data')
        self.time_window_size = time_window_size
        self.exp_xyt_full_path = exp_xyt_full_path
        self.exp_xyt_full_data = pd.read_csv(self.exp_xyt_full_path)
        self.exp_details_path = exp_details_full_path
        self.exp_details_data = pd.read_csv(self.exp_details_path)
        self.full_x = self.exp_xyt_full_data["cell_x"].values
        self.full_y = self.exp_xyt_full_data["cell_y"].values
        self.n_instances = len(self.full_x)
        self.die_times = self.exp_xyt_full_data["death_time"].values
        self.XY = np.column_stack((self.full_x, self.full_y))
        self.data_partitioned_by_window = NucleatorsProbabilitySlidingTimeWindow.divide_by_time_window(self.XY, self.die_times, self.time_window_size, jump_interval=1)

    def calc_nucliation_proba_in_windows(self, jump_interval=1, calc_version='split', use_nucleators=1):
        """
        LEGACY, DON'T USE
        USED FOR A SINGLE EXPERIMENT
        calculates nucleation and propagation probabilities (Laplace correction applied):
        p(nuc) - in each frame/window, all nucleators which died
            divided by all living cells at and after the frame point in time.
        P(prop) - in each frame/window, all non-nucleators which died
            divided by all living cells at and after the frame point in time.
        :param jump_interval: within a window, consecutive frames are 'jump interval' apart (time wise).
        :param calc_version: see appropriate documentation in the code itself.
        :param use_nucleators: 1 - yes, 0 no.
        :return: nucleation_proba_in_window, propagation_proba_in_window - the probabilities
            within all windows sorted by time.
        """
        if use_nucleators:
            nucleation_proba_in_window = []
            propagation_proba_in_window = []
            accumulated_death_proba_in_window = []
            neighbors_list, neighbors_list2, neighbors_list3 = NucleatorsCounter.get_neighbors(XY=self.XY)
            nc = NucleatorsCounter(XY=self.XY, TIMES=self.die_times, neighbors_list=neighbors_list, dist_threshold_nucleators_detection=200)
            all_nucleators = nc.calc_nucleators()
            all_non_nucleators = np.array(all_nucleators-1, dtype=bool)
            maximal_time = np.max(self.die_times)
            sliding_window_start = 0
            sliding_window_end = self.time_window_size

            while sliding_window_end <= maximal_time:
                nuc_proba = 0
                propag_proba = 0
                accu_death_proba = 0
                if calc_version == 'full':
                    # calc probabilities considering all frames in window as one.
                    alive_cells_mask = (self.die_times >= sliding_window_start)
                    dead_cells_mask = (self.die_times < sliding_window_start)
                    window_dietimes_mask = (self.die_times < sliding_window_end) * (self.die_times >= sliding_window_start)
                    window_nucliators_mask = window_dietimes_mask * all_nucleators
                    window_propagated_death_mask = window_dietimes_mask * all_non_nucleators
                    nuc_proba = (np.sum(window_nucliators_mask)+1)/(np.sum(alive_cells_mask)+1)
                    propag_proba = (np.sum(window_propagated_death_mask)+1)/(np.sum(alive_cells_mask)+1)
                    accu_death_proba = (np.sum(dead_cells_mask)+1)/(np.sum(dead_cells_mask) + np.sum(alive_cells_mask) + 1)
                else:
                    # calc probabilities for each frame in window and averaging
                    # the probabilities throughout the whole window.
                    for i in range(self.time_window_size):
                        j = self.time_window_size - i - 1
                        alive_cells_mask = (self.die_times >= sliding_window_start + i)
                        dead_cells_mask = (self.die_times < sliding_window_start + i)
                        window_dietimes_mask = (self.die_times < sliding_window_end - j) * (self.die_times >= sliding_window_start + i)
                        window_nucliators_mask = window_dietimes_mask * all_nucleators
                        window_propagated_death_mask = window_dietimes_mask * all_non_nucleators
                        nuc_proba += (np.sum(window_nucliators_mask)+1)/(np.sum(alive_cells_mask)+1)
                        propag_proba += (np.sum(window_propagated_death_mask)+1)/(np.sum(alive_cells_mask)+1)
                        accu_death_proba += (np.sum(dead_cells_mask)+1)/(np.sum(dead_cells_mask) + np.sum(alive_cells_mask) + 1)
                    nuc_proba = nuc_proba/self.time_window_size
                    propag_proba = propag_proba/self.time_window_size
                    accu_death_proba = accu_death_proba/self.time_window_size
                nucleation_proba_in_window.append(nuc_proba)
                propagation_proba_in_window.append(propag_proba)
                accumulated_death_proba_in_window.append(accu_death_proba)
                sliding_window_start += jump_interval
                sliding_window_end += jump_interval

            return np.array(nucleation_proba_in_window), np.array(propagation_proba_in_window), np.array(accumulated_death_proba_in_window)
        else:
            nucleation_proba_in_window = []
            propagation_proba_in_window = []
            accumulated_death_proba_in_window = []
            neighbors_list, neighbors_list2, neighbors_list3 = NucleatorsCounter.get_neighbors(XY=self.XY)
            nc = NucleatorsCounter(XY=self.XY, TIMES=self.die_times, neighbors_list=neighbors_list, dist_threshold_nucleators_detection=200)
            all_nucleators = nc.calc_nucleators()
            all_non_nucleators = np.array(all_nucleators-1, dtype=bool)
            maximal_time = np.max(self.die_times)
            sliding_window_start = 0
            sliding_window_end = self.time_window_size

            while sliding_window_end <= maximal_time:
                nuc_proba = 0
                propag_proba = 0
                accu_death_proba = 0
                if calc_version == 'full':
                    # calc probabilities considering all frames in window as one.
                    alive_cells_mask = (self.die_times >= sliding_window_start)
                    dead_cells_mask = (self.die_times < sliding_window_start)
                    window_dietimes_mask = (self.die_times < sliding_window_end) * (self.die_times >= sliding_window_start)
                    window_nucliators_mask = window_dietimes_mask * all_nucleators
                    window_propagated_death_mask = window_dietimes_mask * all_non_nucleators
                    nuc_proba = (sum(window_nucliators_mask)+1)/(sum(alive_cells_mask)+1)
                    propag_proba = (sum(window_propagated_death_mask)+1)/(sum(alive_cells_mask)+1)
                    accu_death_proba = (np.sum(dead_cells_mask)+1)/(np.sum(dead_cells_mask) + np.sum(alive_cells_mask) + 1)
                else:
                    # calc probabilities for each frame in window and averaging
                    # the probabilities throughout the whole window.
                    for i in range(self.time_window_size):
                        j = self.time_window_size - i - 1
                        alive_cells_mask = (self.die_times >= sliding_window_start + i)
                        dead_cells_mask = (self.die_times < sliding_window_start + i)
                        window_dietimes_mask = (self.die_times < sliding_window_end - j) * (self.die_times >= sliding_window_start + i)
                        window_nucliators_mask = window_dietimes_mask * all_nucleators
                        window_propagated_death_mask = window_dietimes_mask * all_non_nucleators
                        nuc_proba += (sum(window_nucliators_mask)+1)/(sum(alive_cells_mask)+1)
                        propag_proba += (sum(window_propagated_death_mask)+1)/(sum(alive_cells_mask)+1)
                        accu_death_proba += (np.sum(dead_cells_mask)+1)/(np.sum(dead_cells_mask) + np.sum(alive_cells_mask) + 1)
                    nuc_proba = nuc_proba/self.time_window_size
                    propag_proba = propag_proba/self.time_window_size
                    accu_death_proba = accu_death_proba/self.time_window_size
                nucleation_proba_in_window.append(nuc_proba)
                propagation_proba_in_window.append(propag_proba)
                accumulated_death_proba_in_window.append(accu_death_proba)
                sliding_window_start += jump_interval
                sliding_window_end += jump_interval
            return np.array(nucleation_proba_in_window), np.array(propagation_proba_in_window), np.array(accumulated_death_proba_in_window)

    @staticmethod
    def get_cells_neighboring_dead_cells(dead_cells_mask, neighbors, lvl_of_neighbors=1, xy=None, threshold=200):
        """
        returns a boolean array stating for each cell (by index) whether it neighbors a dead cell (dead
        cells are according to dead_cells_mask). Does not consider if a given neighbor is already dead,
        to calc all alive neighbors, multiply the result of this function with a mask of alive cells.
        :param dead_cells_mask:
        :param neighbors:
        :param lvl_of_neighbors:
        :param xy:
        :param threshold:
        :return:
        """
        around_dead_cells = np.zeros(dead_cells_mask.shape)
        for cell_idx, is_dead in enumerate(dead_cells_mask):
            if is_dead:
                curr_neighbors = neighbors[cell_idx]
                for neighbor_idx in curr_neighbors:
                    if xy is not None:
                        dist = NucleatorsCounter.get_real_distance(cell1_xy=xy[cell_idx], cell2_xy=xy[neighbor_idx])
                        around_dead_cells[neighbor_idx] = (True) * (dist < threshold)
                        if lvl_of_neighbors == 2 or lvl_of_neighbors == 3:
                            curr_neighbor_lvl2 = neighbors[neighbor_idx]
                            for neigbors_2lvl_idx in curr_neighbor_lvl2:
                                dist = NucleatorsCounter.get_real_distance(cell1_xy=xy[cell_idx], cell2_xy=xy[neigbors_2lvl_idx])
                                around_dead_cells[neigbors_2lvl_idx] = (True) * (dist < threshold)
                                if lvl_of_neighbors == 3:
                                    curr_neighbor_lvl3 = neighbors[neigbors_2lvl_idx]
                                    for neigbors_3lvl_idx in curr_neighbor_lvl3:
                                        dist = NucleatorsCounter.get_real_distance(cell1_xy=xy[cell_idx], cell2_xy=xy[neigbors_3lvl_idx])
                                        around_dead_cells[neigbors_3lvl_idx] = (True) * (dist < threshold)

                    else:
                        # if we  want to consider only first lvl neighbors as candidates
                        around_dead_cells[neighbor_idx] = True
                        if lvl_of_neighbors == 2:
                            # if we want to consider first & second lvl neighbors as candidates
                            curr_neighbor_lvl2 = neighbors[neighbor_idx]
                            for neigbors_2lvl_idx in curr_neighbor_lvl2:
                                around_dead_cells[neigbors_2lvl_idx] = True

        return around_dead_cells

    def calc_tighter_nucleation_propagation_proba_in_windows(self, jump_interval=1, calc_version='split'):
        """
        USED FOR A SINGLE EXPERIMENT
        calculates nucleation and propagation probabilities (Laplace correction applied):
        p(nuc) - in each frame/window, all nucleators which died
            divided by all living cells at and after the frame point in time.
        P(prop) - in each frame/window, all non-nucleators which died
            divided by all living cells at and after the frame point in time.
        :param jump_interval: within a window, consecutive frames are 'jump interval' apart (time wise).
        :param calc_version: see appropriate documentation in the code itself.
        :return: nucleation_proba_in_window, propagation_proba_in_window, accumulated_death_prob_in_window - the probabilities
            within all windows sorted by time.
        """
        nucleation_proba_in_window = []
        propagation_proba_in_window = []
        accumulated_death_prob_in_window = []
        # get neighbors list of all cells (topological by Voronoi)
        neighbors_list, neighbors_list2, neighbors_list3 = NucleatorsCounter.get_neighbors(XY=self.XY)

        maximal_time = np.max(self.die_times)
        sliding_window_start = 0
        sliding_window_end = self.time_window_size

        while sliding_window_end <= maximal_time:
            if calc_version == 'full':
                # calc probabilities considering all frames in window as one.
                alive_cells_mask = (self.die_times >= sliding_window_start)
                dead_cells_mask = (self.die_times < sliding_window_start)
                neighbors_of_dead = self.get_cells_neighboring_dead_cells(dead_cells_mask, neighbors_list)
                around_dead_cells_and_alive = neighbors_of_dead * alive_cells_mask
                not_around_dead_cells_and_alive = np.array(neighbors_of_dead-1, dtype=bool) * alive_cells_mask
                window_dietimes_mask = (self.die_times < sliding_window_end) * (self.die_times >= sliding_window_start)
                window_nucleators_mask = window_dietimes_mask * not_around_dead_cells_and_alive
                window_propagated_death_mask = window_dietimes_mask * around_dead_cells_and_alive
                nuc_proba = (window_nucleators_mask.sum()+1)/(not_around_dead_cells_and_alive.sum()+1)
                propag_proba = (window_propagated_death_mask.sum()+1)/(around_dead_cells_and_alive.sum()+1)
                accu_death_proba = (dead_cells_mask.sum()+1)/(alive_cells_mask.sum()+dead_cells_mask.sum()+1)
            else:
                temp_nuc = []
                temp_prog = []
                temp_accu_death = []
                for i in range(0, self.time_window_size, jump_interval):
                    curr_time = i + sliding_window_start
                    if curr_time > maximal_time:
                        continue
                    nuc_prob, prop_prob, accu_prob = self.calc_nucleator_in_frame(curr_time, neighbors_list)
                    temp_nuc.append(nuc_prob)
                    temp_prog.append(prop_prob)
                    temp_accu_death.append(accu_prob)

                # nuc_proba = nuc_proba/(self.time_window_size/jump_interval)
                nuc_proba = np.array(temp_nuc).mean()
                # propag_proba = propag_proba/(self.time_window_size/jump_interval)
                propag_proba = np.array(temp_prog).mean()
                # accu_death_proba = accu_death_proba/(self.time_window_size/jump_interval)
                accu_death_proba = np.array(temp_accu_death).mean()

            nucleation_proba_in_window.append(nuc_proba)
            propagation_proba_in_window.append(propag_proba)
            accumulated_death_prob_in_window.append(accu_death_proba)
            sliding_window_start += jump_interval
            sliding_window_end += jump_interval

        return np.array(nucleation_proba_in_window), np.array(propagation_proba_in_window), np.array(accumulated_death_prob_in_window)

    def calc_nucleator_in_frame(self, curr_time, neighbors_list, dist_threshold=200):
        """
        17/03/2021 - calculate nucleation&propagation probabilities in a single frame.
        nucleators are calculated as cells which die in frame but are not adjacent to an
        already dead cell. if nucleator candidates are themselves adjacent to one another, the inner
        function 'clean_adjacent_nucleators' removes all adjacent nucleator candidates but one
        in each subgroup of adjacent nucleators and adds them into propagated candidates mask.
        :param curr_time: the frame index to calc on.
        :param neighbors_list: neighbors list for all cells, calculated by the 'NucleatorsCounter.get_neighbors' method
        :return: tuple of floats, probabilities for nucleation, propagation, accumulated death
        """
        def clean_adjacent_nucleators(nuc_cand, all_neighbors, all_xy, propagation_cand, dist_thresh):
            for cand_idx, single_nuc_candidate in enumerate(nuc_cand):
                if single_nuc_candidate:
                    cand_xy = all_xy[cand_idx]
                    cand_neighbors = all_neighbors[cand_idx]
                    for cand_neighbor_idx in cand_neighbors:
                        cand_neighbor_xy = all_xy[cand_neighbor_idx]
                        cand_neighbor_true_dist_condition = \
                            NucleatorsCounter.get_real_distance(cand_xy, cand_neighbor_xy) < dist_thresh
                        if nuc_cand[cand_neighbor_idx] and cand_neighbor_true_dist_condition:
                            nuc_cand[cand_neighbor_idx] = False
                            propagation_cand[cand_neighbor_idx] = True
                else:
                    continue
            return nuc_cand, propagation_cand

        alive_cells_mask = (self.die_times >= curr_time)
        dead_cells_mask = (self.die_times < curr_time)
        dead_in_curr_frame_mask = (self.die_times == curr_time)
        # propagation candidates that are alive
        propagation_candidates = self.get_cells_neighboring_dead_cells(dead_cells_mask, neighbors_list, lvl_of_neighbors=1, xy=self.XY)
        # get nucleators candidates, all cells not adjacent to a dead cell. cells must be alive
        nucleator_candidates = np.array(propagation_candidates-1, dtype=bool) * (alive_cells_mask)
        propagation_candidates = propagation_candidates * (alive_cells_mask)
        # remove all adjacent nucleator candidates and insert to the propagation candidates
        nucleators_in_frame = nucleator_candidates * dead_in_curr_frame_mask
        nucleators_in_frame, propagation_candidates = clean_adjacent_nucleators(nucleators_in_frame,
                                                        neighbors_list,
                                                        self.XY,
                                                        propagation_candidates,
                                                        dist_threshold)

        propagators_in_frame = propagation_candidates * dead_in_curr_frame_mask
        # todo current: nuc_proba_in_frame = (nucleators_in_frame.sum() + 1) / (nucleator_candidates.sum() + 1)

        # TEST
        unique_times = np.unique(self.die_times)
        implicit_time_res = abs(unique_times[1] - unique_times[0])
        nuc_ctr = NucleatorsCounter(XY=self.XY, TIMES=self.die_times, neighbors_list=neighbors_list, dist_threshold_nucleators_detection=dist_threshold)
        nuc_proba_in_frame = nuc_ctr.calc_nucleators(curr_TOD=curr_time, max_TOD=curr_time + implicit_time_res)
        nuc_proba_in_frame = (nucleators_in_frame.sum() + 1) / (nucleator_candidates.sum() + 1)
        # END OF TEST

        prop_proba_in_frame = (propagators_in_frame.sum() + 1) / (propagation_candidates.sum() + 1)
        accumulated_death_proba = (dead_cells_mask.sum() + 1)/(alive_cells_mask.sum() + dead_cells_mask.sum() + 1)
        return nuc_proba_in_frame, prop_proba_in_frame, accumulated_death_proba


    def plot_single_nucleation_proba_in_window(self,
                                               showfig=True,
                                               savefig=False,
                                               fig_fname='NucliationProbabilitySlidingTimeWindow.png'):
        exp_treatment_type = self.exp_details_data[self.exp_details_data['File Name'] == self.exp_xyt_full_path.split(os.sep)[-1]]['Treatment'].values[0]
        nucleation_proba_in_window, propagation_proba_in_window, accumulated_death_in_window = self.calc_nucliation_proba_in_windows()
        windows = list(range(len(nucleation_proba_in_window)))
        fig, ax1 = plt.subplots()
        ax1_clr = 'tab:blue'
        ax1.set_title('Nucliation Vs. Propagation Probabilities exp: {}\nTreatment: {}'.format(self.exp_xyt_full_path.split(os.sep)[-1][:-4], exp_treatment_type))
        ax1.plot(windows, nucleation_proba_in_window, label='Nucleation')
        ax1.set_ylabel('P(Nuc)', color=ax1_clr)
        ax1.set_xlabel('Time Window (in frames)')
        ax1.set_xticks(list(range(len(windows))))
        ax1.set_xticklabels(['{}-{}'.format(x, x+self.time_window_size) for x in range(len(windows))])
        ax1.tick_params(axis='y', labelcolor=ax1_clr)
        ax2_color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.plot(windows, propagation_proba_in_window, label='Propagation', color=ax2_color)
        ax2.set_ylabel('P(Prop)', color=ax2_color)
        ax2.tick_params(axis='y', labelcolor=ax2_color)
        # ax1.legend()
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        if showfig:
            plt.show()
        if savefig:
            fig.savefig(fig_fname, dpi=200)

    @staticmethod
    def divide_by_time_window(XY, die_times, time_window_size, jump_interval=1):
        maximal_time = np.max(die_times)
        sliding_window_start = 0
        sliding_window_end = time_window_size
        partitioned_data = []
        while sliding_window_end <= maximal_time:
            window_mask = (die_times < sliding_window_end) * (die_times >= sliding_window_start)
            die_times_in_window = die_times[window_mask]
            xy_in_window = XY[window_mask, :]
            cell_num_in_window = len(xy_in_window)
            partitioned_data.append((xy_in_window, die_times_in_window, cell_num_in_window))
            sliding_window_start += jump_interval
            sliding_window_end += jump_interval
        return partitioned_data

    @staticmethod
    def get_window_probabilities_dict(filtered_list_of_files_to_analyze,
                                      tighter_bounds,
                                      time_window_size,
                                      file_details_fname,
                                      forced_temporal_res):
        exp_details_df = pd.read_csv(file_details_fname)
        subplots_per_treatment = {}
        for f_idx, file_name in enumerate(filtered_list_of_files_to_analyze):
            if f_idx > MAX_EXP_TO_PROCESS:
                break
            if file_name.find('.csv') != -1:
                print("file idx: %d file name: %s" % (f_idx, file_name))
                single_file_fname = os.sep.join([main_fname, file_name])
                exp_temporal_resolution = int(exp_details_df[exp_details_df['File Name'] == file_name]['Time Interval (min)'].values[0] if forced_temporal_res == -1 else forced_temporal_res)
                npstw = NucleatorsProbabilitySlidingTimeWindow(time_window_size=time_window_size * exp_temporal_resolution,
                                                               exp_xyt_full_path=single_file_fname,
                                                               exp_details_full_path=file_details_fname)
                exp_treatment_type = npstw.exp_details_data[npstw.exp_details_data['File Name'] == npstw.exp_xyt_full_path.split(os.sep)[-1]]['Treatment'].values[0]

                if not tighter_bounds:
                    nucleation_proba_in_window, propagation_proba_in_window, accumulated_death_prob_in_window = npstw.calc_nucliation_proba_in_windows(calc_version='split',
                                                                                                                     jump_interval=exp_temporal_resolution)
                else:
                    nucleation_proba_in_window, propagation_proba_in_window, accumulated_death_prob_in_window = npstw.calc_tighter_nucleation_propagation_proba_in_windows(
                        calc_version='split',
                        jump_interval=exp_temporal_resolution)

                windows = list(range(0,len(nucleation_proba_in_window)*exp_temporal_resolution, exp_temporal_resolution))
                subplots_per_treatment[exp_treatment_type] = subplots_per_treatment.get(exp_treatment_type, []) + [{
                    'exp_name':file_name,
                    'nucleation': nucleation_proba_in_window.tolist(),
                    'propagation': propagation_proba_in_window.tolist(),
                    'time_windows': windows,
                    'accumulated_death': accumulated_death_prob_in_window.tolist(),
                    'exp_temporal_resolution': exp_temporal_resolution
                }]
        return subplots_per_treatment

    @classmethod
    def plot_multiple_exps_nucleation_probability_time_window(cls,
                                                              main_fname='ExperimentsXYT_CSVFiles',
                                                              file_details_fname='File details.csv',
                                                              time_window_size=5,
                                                              tighter_bounds=True,
                                                              forced_temporal_res=-1,
                                                              plot_single: bool = False,
                                                              **plot_kwargs):
        filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1), os.listdir(main_fname))
        filtered_list_of_files_to_analyze = list(filtered_list_of_files_to_analyze)
        subplots_per_treatment = cls.calc_subplots_per_treatment(filtered_list_of_files_to_analyze,
                                                                 tighter_bounds,
                                                                 time_window_size,
                                                                 file_details_fname,
                                                                 forced_temporal_res)
        # subplots_per_treatment = cls.get_window_probabilities_dict(filtered_list_of_files_to_analyze,
        #                                                            tighter_bounds,
        #                                                            time_window_size,
        #                                                            file_details_fname,
        #                                                            forced_temporal_res)

        # plotting
        print("#############\nPlotting\n#############")
        mpl.rcParams['font.size'] = FONT_SIZE
        mpl.rcParams['figure.subplot.hspace'] = 0.2
        # mpl.rcParams['figure.subplot.wspace'] = 0.1
        max_time = 0
        for treatment_name, treatment_subplots_data in subplots_per_treatment.items():
            for exp_idx, exp_data_dict in enumerate(treatment_subplots_data):
                windows = exp_data_dict['time_windows']
                if max(windows) > max_time:
                    max_time = max(windows)
        time_ticks = list(range(0, max_time, 50))

        for treatment_name, treatment_subplots_data in subplots_per_treatment.items():
            # ax_to_share = None
            subplots_cnt = len(treatment_subplots_data)
            if subplots_cnt < 2:
                fig, axis = plt.subplots(sharex=True)
            else:
                fig, axis = plt.subplots(subplots_cnt//2+1, 2, figsize=(15, 15), sharex=True)
            for exp_idx, exp_data_dict in enumerate(treatment_subplots_data):
                windows = exp_data_dict['time_windows']
                if max(windows) > max_time:
                    max_time = max(windows)

                exp_name = exp_data_dict['exp_name']
                exp_temporal_resolution = exp_data_dict['exp_temporal_resolution']
                nucleation_proba_in_window = exp_data_dict['nucleation']
                propagation_proba_in_window = exp_data_dict['propagation']
                accumulated_death_prob_in_window = exp_data_dict['accumulated_death']

                # SINGLE EXP IN EACH PLOT
                if plot_single:
                    plt.clf()
                    fig, ax = plt.subplots()
                    ax.set_title('Treatment: {}'.format(treatment_name))
                    ax1_clr = 'tab:blue'
                    ax.plot(windows, nucleation_proba_in_window, label='Nucleation')
                    ax.set_ylabel('P(Nuc)', color=ax1_clr)
                    ax.set_xlabel('Time Window (in min)')
                    ax.tick_params(axis='y', labelcolor=ax1_clr)
                    ax.set_ylim(0, 1.01)
                    ax.set_xticks(time_ticks)
                    ax.set_xticklabels(['{}-{}'.format(x, x+100) for x in time_ticks])

                    ax2_color = 'tab:red'
                    ax2 = ax.twinx()
                    ax2.plot(windows, propagation_proba_in_window, label='Propagation', color=ax2_color)
                    ax2.set_ylabel('P(Prop)', color=ax2_color)
                    ax2.tick_params(axis='y', labelcolor=ax2_color)
                    ax2.set_ylim(0, 1)
                    # plotting accumulated death
                    ax.plot(windows, accumulated_death_prob_in_window, label='Accumulated Death', color='black', marker='p')

                    if plot_kwargs.get('showfig', False):
                        plt.show()
                    if plot_kwargs.get('savefig', False):
                        dir_path = 'PropagationAndNucleationProbabilityInTimeWindow/PlotsForSingleExp'
                        im_path = os.sep.join([dir_path, exp_name+'.png'])
                        plt.savefig(im_path, dpi=200)
                    plt.clf()


                else:
                    if len(treatment_subplots_data) < 2:
                        curr_ax = axis
                    else:
                        curr_ax = axis[exp_idx//2, exp_idx % 2]

                    if plot_kwargs.get('enable_correlation', False):
                        correlation, p_val = pearsonr(nucleation_proba_in_window, propagation_proba_in_window)
                        curr_ax.set_title('EXP:{}\nP-corr: {:.2f}, P-val:{:.2f}'.format(exp_name, correlation, p_val))
                    else:
                        curr_ax.set_title(exp_name)
                    ax1_clr = 'tab:blue'
                    curr_ax.plot(windows, nucleation_proba_in_window, label='Nucleation')
                    curr_ax.set_ylabel('P(Nuc)', color=ax1_clr)
                    curr_ax.set_xlabel('Time Window (in min)')
                    curr_ax.tick_params(axis='y', labelcolor=ax1_clr)
                    curr_ax.set_ylim(0, 1.01)
                    curr_ax.set_xticks(time_ticks)
                    curr_ax.set_xticklabels(['{}-{}'.format(x, x+100) for x in time_ticks])
                    ax2_color = 'tab:red'
                    ax2 = curr_ax.twinx()
                    ax2.plot(windows, propagation_proba_in_window, label='Propagation', color=ax2_color)
                    ax2.set_ylabel('P(Prop)', color=ax2_color)
                    ax2.tick_params(axis='y', labelcolor=ax2_color)
                    ax2.set_ylim(0, 1)
                    # plotting accumulated death
                    curr_ax.plot(windows, accumulated_death_prob_in_window, label='Accumulated Death', color='black', marker='p')

                fig.suptitle(treatment_name)
                fig.autofmt_xdate(rotation=45)
                if plot_kwargs.get('showfig', False):
                    plt.show()
                if plot_kwargs.get('savefig', False):
                    def_fig_fname = os.sep.join(['PropagationAndNucleationProbabilityInTimeWindow',
                                                 'semi_window_analysis',
                                                 'SingleAdjacentNucleators',
                                                 'propagationAndNucleationProbabilities{}|Treat={}.png'.format('_tight_bounds' if tighter_bounds else '', treatment_name.replace(os.sep, '|'))])

                    fig.savefig(plot_kwargs.get('figname', def_fig_fname))

    @staticmethod
    def get_measurement_fit(subplots_per_treatment, measurement='nucleation', model=linregress):
        """
        for each experiment in subplots_per_treatment, calc the model fit (according to the model given)
        for the measurement detailed, then return a dictionary with treatments as primary keys.
        Each value is a list of dictionaries. Each of the nested dictionaries is the results
        of the model fit (slope, intercept, correlation, pvalue and std error).
        :param subplots_per_treatment: dictionary which holds for each treatment (the key) a list of experiments data,
            held in dictionaries, with keys: 'time_windows' + 'exp_name' + 'nucleation'/'propagation' (by measurement)
        :param measurement: string (either 'nucleation' or 'propagation')
        :param model: a model which fits the api
        :return:
        """
        measurement_fit_per_treatment = {}
        for treatment_name, treatment_subplots_data in subplots_per_treatment.items():
            subplots_cnt = len(treatment_subplots_data)
            for exp_idx, exp_data_dict in enumerate(treatment_subplots_data):
                windows = exp_data_dict['time_windows']
                exp_name = exp_data_dict['exp_name']
                measurement_proba_in_window = exp_data_dict[measurement]
                if getattr(model, '__name__', None) is not None and model.__name__ == 'linregress':
                    slope, intercept, r, p, se = model(windows, measurement_proba_in_window)
                    exp_fit_dict = {
                        'exp_name': exp_name,
                        'slope': slope,
                        'intercept': intercept,
                        'correlation': r,
                        'pvalue': p,
                        'stderr': se
                    }
                else:
                    objective_equation = None
                    if model == 'polyfit3':
                        objective_equation = NucleatorsProbabilitySlidingTimeWindow.poly_equation_3nd_order
                    elif model == 'polyfit2':
                        objective_equation = NucleatorsProbabilitySlidingTimeWindow.poly_equation_2nd_order
                    popt, _ = curve_fit(objective_equation, windows, measurement_proba_in_window)
                    measurement_proba_poly_fit = []
                    if model == 'polyfit3':
                        a, b, c, d = popt
                        measurement_proba_poly_fit = objective_equation(np.array(windows), a, b, c, d)
                    elif model == 'polyfit2':
                        a, b, c = popt
                        measurement_proba_poly_fit = objective_equation(np.array(windows), a, b, c)
                    resid = np.array(measurement_proba_in_window) - np.array(measurement_proba_poly_fit)

                    exp_fit_dict = {
                        'exp_name': exp_name,
                        'resid': resid.tolist(),
                        'poly_fit_rmse': metrics.mean_squared_error(measurement_proba_in_window, measurement_proba_poly_fit, squared=False),
                        'poly_fit_r^2_score': metrics.r2_score(measurement_proba_in_window, measurement_proba_poly_fit)
                    }

                measurement_fit_per_treatment[treatment_name] = measurement_fit_per_treatment.get(treatment_name, []) + [exp_fit_dict]

        return measurement_fit_per_treatment

    @staticmethod
    def plot_single_helper(to_plot, measurement_fit_per_treatment, all_exps_spi_df, colors, marker_options, tighter_bounds, model, **plot_kwargs):
        treatment_idx = 0
        fig, ax = plt.subplots()
        for treatment_name, exps_fit in measurement_fit_per_treatment.items():
            to_plot_data = []
            spi_data = []
            for exp_fit in exps_fit:
                exp_name = exp_fit['exp_name'][:-4]

                # read pre-calc SPI
                spi_data.append(all_exps_spi_df[all_exps_spi_df['name'] == exp_name]['spatial_propagation_index'].values[0])

                # gather target variable data
                if to_plot == 'slope':
                    to_plot_data.append(exp_fit['slope'])
                elif to_plot == 'intercept':
                    to_plot_data.append(exp_fit['intercept'])
                elif to_plot == 'stderr':
                    to_plot_data.append(exp_fit['stderr'])
                elif to_plot == 'pvalue':
                    to_plot_data.append(exp_fit['pvalue'])
                elif to_plot == 'correlation':
                    to_plot_data.append(exp_fit['correlation'])
                elif to_plot == 'poly_fit_rmse':
                    to_plot_data.append(exp_fit['poly_fit_rmse'])
                elif to_plot == 'poly_fit_r^2_score':
                    to_plot_data.append(exp_fit['poly_fit_r^2_score'])

            ax.scatter(to_plot_data, spi_data, color=colors(treatment_idx), marker="{}".format(marker_options(treatment_name)), label=treatment_name)
            ax.set_xlabel(to_plot)
            ax.set_ylabel('SPI')
            ax.set_title('SPI Vs. {}'.format(to_plot))

            treatment_idx += 1
        ax.grid('on')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.05, 1))
        if plot_kwargs.get('showfig', False):
            plt.show()
        if plot_kwargs.get('savefig', False):
            def_fig_fname = os.sep.join(['PropagationAndNucleationProbabilityInTimeWindow',
                                         'semi_window_analysis',
                                         'propagationAndNucleationProbabilities{}|Treat={}.png'.format('_tight_bounds' if tighter_bounds else '', treatment_name.replace(os.sep, '|'))])
            fig.savefig(plot_kwargs.get('figname', def_fig_fname), bbox_extra_artists=[lgd], bbox_inches='tight', dpi=200)

    @classmethod
    def calc_subplots_per_treatment(cls, filtered_list_of_files_to_analyze, tighter_bounds, time_window_size, file_details_fname, forced_temporal_res=-1):
        try:
            if ALWAYS:
                raise FileNotFoundError()
            subplots_per_treatment = json.load(open('subplots_per_treatment.json', 'r'))
        except FileNotFoundError as e:
            print('no pre-calced suplots_per_treatment file, re calculating')
            subplots_per_treatment = cls.get_window_probabilities_dict(filtered_list_of_files_to_analyze,
                                                                       tighter_bounds,
                                                                       time_window_size,
                                                                       file_details_fname,
                                                                       forced_temporal_res)
            with open('subplots_per_treatment.json', 'w') as f:
                # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
                json.dump(subplots_per_treatment, f)
        return subplots_per_treatment

    @classmethod
    def calc_measurement_fit_per_treatment(cls, measurement_fit_per_treatment, tighter_bounds, time_window_size, measurement, model, file_details_fname, forced_temporal_res):
        """
        if no measurement fit per treatment was given, attempts to read one from the local file system.
        the file must be in the root directory and named 'measurement_fit_per_treatment.json'.
        if no file was found, calculates the p(nuc) and p(prob) and the measurement fit to the 'model'.
        saves the calculated fit.
        :param measurement_fit_per_treatment:
        :param tighter_bounds: boolean, whether to calc by averaging frame's probabilities in window, or consider window as
            single frame.
        :param time_window_size:
        :param measurement:
        :param model:
        :return:
        """
        if measurement_fit_per_treatment is None:
            try:
                if ALWAYS:
                    raise FileNotFoundError()
                measurement_fit_per_treatment = json.load(open('{}_fit_per_treatment.json'.format(measurement), 'r'))
            except FileNotFoundError as e:
                print('no pre-calced measurement file, re calculating')
                filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1), os.listdir(main_fname))
                filtered_list_of_files_to_analyze = list(filtered_list_of_files_to_analyze)
                subplots_per_treatment = cls.calc_subplots_per_treatment(filtered_list_of_files_to_analyze, tighter_bounds, time_window_size, file_details_fname, forced_temporal_res)
                measurement_fit_per_treatment = cls.get_measurement_fit(subplots_per_treatment, measurement=measurement, model=model)
                with open('{}_fit_per_treatment.json'.format(measurement), 'w') as f:
                    json.dump(measurement_fit_per_treatment, f)
        return measurement_fit_per_treatment

    @staticmethod
    def poly_equation_2nd_order(x, a, b, c):
        return (a*x) + (b*x**2) + c

    @staticmethod
    def poly_equation_3nd_order(x, a, b, c, d):
        return (a * x) + (b * x**2) + (c * x**3) + d

    @classmethod
    def plot_model_fit(cls,
                       measurement_fit_per_treatment = None,
                       model=linregress,
                       main_fname='ExperimentsXYT_CSVFiles',
                       all_exp_summation_fn='SPI_DeathWaves_Results/Results/All/data_combined.csv',
                       file_details_fname='File details.csv',
                       time_window_size=5,
                       tighter_bounds=True,
                       to_plot='slope',
                       measurement='nucleation',
                       forced_temporal_res=-1,
                       **plot_kwargs) -> None:
        """
        calculates (or reads) p(nuc) and p(prop) for all experiments, in 'main_fname', then fits 'measurement'
        to a linear 'model' (or any other given 'model' which suits the API).
        The method plots the properties of the fit given in 'to_plot' attribute.
        :param model:
        :param main_fname: main directory which holds the XYT data.
        :param all_exp_summation_fn: the full path for the csv detailing all experiments SPI
        :param file_details_fname: the full path for the csv detailing all experiments' setups
        :param time_window_size: int
        :param tighter_bounds: boolean, whether to calc by averaging frame's probabilities in window, or consider window as
            single frame.
        :param to_plot: string or an iterable of strings
        :param measurement: string (either 'nucleation' or 'propagation')
        :param plot_kwargs: a dictionary for plot names and other attributes.
        :return:None
        """
        measurement_fit_per_treatment = cls.calc_measurement_fit_per_treatment(measurement_fit_per_treatment,
                                                                               tighter_bounds,
                                                                               time_window_size,
                                                                               measurement,
                                                                               model,
                                                                               file_details_fname, forced_temporal_res)
        all_exp_summation_df = pd.read_csv(all_exp_summation_fn)
        all_exps_spi_df = all_exp_summation_df.loc[:, ['name', 'spatial_propagation_index']]
        unique_types = list(measurement_fit_per_treatment.keys())
        colors = plt.get_cmap('plasma', len(unique_types))
        markers = {
            'FAC&BSO':'>',
            "C'":'P',
            'ML162':'x',
            'else': '*'
        }
        marker_options = lambda name: markers['FAC&BSO'] if "FAC&BSO" in name else markers["C'"] if "C'" in name else markers['ML162'] if 'ML162' in name else markers['else']

        if getattr(model,'__name__', None) != None and model.__name__ == 'linregress':
            if isinstance(to_plot, str):
                plot_kwargs['figname'] = 'SPI Vs {} {} linregress.png'.format(measurement, to_plot)
                cls.plot_single_helper(to_plot, measurement_fit_per_treatment, all_exps_spi_df, colors, marker_options, tighter_bounds, model, **plot_kwargs)
            else:
                for sto_plot in to_plot:
                    plot_kwargs['figname'] = os.sep.join(['PropagationAndNucleationProbabilityInTimeWindow',
                                                          'semi_window_analysis',
                                                          'SPI Vs {} {} linregress.png'.format(measurement, sto_plot)])
                    cls.plot_single_helper(sto_plot, measurement_fit_per_treatment, all_exps_spi_df, colors, marker_options, tighter_bounds, model, **plot_kwargs)
        else:
            if isinstance(to_plot, str):
                plot_kwargs['figname'] = os.sep.join(['PropagationAndNucleationProbabilityInTimeWindow',
                                                      'semi_window_analysis',
                                                      'PolyFit',
                                                      'SPI Vs {} {} {}.png'.format(measurement, to_plot, model)])
                cls.plot_single_helper(to_plot=to_plot,
                                       measurement_fit_per_treatment=measurement_fit_per_treatment,
                                       all_exps_spi_df=all_exps_spi_df,
                                       colors=colors,
                                       marker_options=marker_options,
                                       tighter_bounds=tighter_bounds,
                                       model=model,
                                       **plot_kwargs)
            else:
                for sto_plot in to_plot:
                    plot_kwargs['figname'] = os.sep.join(['PropagationAndNucleationProbabilityInTimeWindow',
                                                          'semi_window_analysis',
                                                          'PolyFit',
                                                          'SPI Vs {} {} {}.png'.format(measurement, sto_plot, model)])
                    cls.plot_single_helper(to_plot=sto_plot,
                                           measurement_fit_per_treatment=measurement_fit_per_treatment,
                                           all_exps_spi_df=all_exps_spi_df,
                                           colors=colors,
                                           marker_options=marker_options,
                                           tighter_bounds=tighter_bounds,
                                           model=model,
                                           **plot_kwargs)

    @staticmethod
    def get_largest_derivative_idx(signal, with_idx=False):
        largest_derivative = 0
        largest_derivative_idx = -1
        prev_point = None
        for idx, point in enumerate(signal):
            if prev_point is None:
                prev_point = point
                continue
            temp = abs(point - prev_point)
            if largest_derivative <= temp:
                largest_derivative = temp
                largest_derivative_idx = idx
            prev_point = point
        if with_idx:
            return largest_derivative, largest_derivative_idx/len(signal)
        return largest_derivative


    @classmethod
    def plot_measurement_highest_derivative_index(cls,
                                                  main_fname='ExperimentsXYT_CSVFiles',
                                                  all_exp_summation_fn='SPI_DeathWaves_Results/Results/All/data_combined.csv',
                                                  file_details_fname='File details.csv',
                                                  time_window_size=5,
                                                  tighter_bounds=True,
                                                  measurement='nucleation',
                                                  **plot_kwargs) -> None:
        """
        calculates (or reads) p(nuc) and p(prop) for all experiments, in 'main_fname', then fits 'measurement'
        to a linear 'model' (or any other given 'model' which suits the API).
        The method plots the properties of the fit given in 'to_plot' attribute.
        :param main_fname: main directory which holds the XYT data.
        :param all_exp_summation_fn: the full path for the csv detailing all experiments SPI
        :param file_details_fname: the full path for the csv detailing all experiments' setups
        :param time_window_size: int
        :param tighter_bounds: boolean, whether to calc by averaging frame's probabilities in window, or consider window as
            single frame.
        :param measurement: string (either 'nucleation' or 'propagation')
        :param plot_kwargs: a dictionary for plot names and other attributes.
        :return:None
        """
        filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1), os.listdir(main_fname))
        filtered_list_of_files_to_analyze = list(filtered_list_of_files_to_analyze)
        subplots_per_treatment = cls.calc_subplots_per_treatment(
            filtered_list_of_files_to_analyze=filtered_list_of_files_to_analyze,
            tighter_bounds=tighter_bounds,
            time_window_size=time_window_size,
            file_details_fname=file_details_fname
        )

        all_exp_summation_df = pd.read_csv(all_exp_summation_fn)
        all_exps_spi_df = all_exp_summation_df.loc[:, ['name', 'spatial_propagation_index']]
        unique_types = list(subplots_per_treatment.keys())
        colors = plt.get_cmap('plasma', len(unique_types))
        markers = {
            'FAC&BSO':'>',
            "C'":'P',
            'ML162':'x',
            'else': '*'
        }

        marker_options = lambda name: markers['FAC&BSO'] if "FAC&BSO" in name else markers["C'"] if "C'" in name else markers['ML162'] if 'ML162' in name else markers['else']

        treatment_idx = 0
        largest_derivatives_by_treatment = {}
        spi_by_treatments = {}
        fig, ax = plt.subplots()
        for treatment_name, treatment_subplots_data in subplots_per_treatment.items():
            for exp_idx, exp_data_dict in enumerate(treatment_subplots_data):
                # windows = exp_data_dict['time_windows']
                exp_name = exp_data_dict['exp_name'][:-4]
                measurement_proba_in_windows = exp_data_dict[measurement]
                exp_spi = all_exps_spi_df[all_exps_spi_df['name'] == exp_name]['spatial_propagation_index'].values[0]
                spi_by_treatments[treatment_name] = spi_by_treatments.get(treatment_name, []) + [exp_spi]
                largest_derivatives_by_treatment[treatment_name] = largest_derivatives_by_treatment.get(treatment_name, []) + [cls.get_largest_derivative_idx(measurement_proba_in_windows)]
            # plot the treatment details
            ax.scatter(largest_derivatives_by_treatment[treatment_name], spi_by_treatments[treatment_name], color=colors(treatment_idx), marker="{}".format(marker_options(treatment_name)), label=treatment_name)
            ax.set_xlabel('Largest derivative temporal location')
            ax.set_ylabel('SPI')
            ax.set_title('SPI Vs. {} Derivative'.format(measurement))

            treatment_idx += 1

        ax.grid('on')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.05, 1))
        if plot_kwargs.get('showfig', False):
            plt.show()
        if plot_kwargs.get('savefig', False):
            fig.savefig(plot_kwargs.get('figname', os.sep.join(['PropagationAndNucleationProbabilityInTimeWindow',
                                                                'semi_window_analysis',
                                                                'SPI Vs. {} largest derivative.png'.format(measurement)])), bbox_extra_artists=[lgd], bbox_inches='tight', dpi=200)

    @classmethod
    def plot_measurement_avg_derivative_index(cls,
                                                  main_fname='ExperimentsXYT_CSVFiles',
                                                  all_exp_summation_fn='SPI_DeathWaves_Results/Results/All/data_combined.csv',
                                                  file_details_fname='File details.csv',
                                                  time_window_size=5,
                                                  tighter_bounds=True,
                                                  measurement='nucleation',
                                                  **plot_kwargs) -> None:
        """
        calculates (or reads) p(nuc) and p(prop) for all experiments, in 'main_fname', then fits 'measurement'
        to a linear 'model' (or any other given 'model' which suits the API).
        The method plots the properties of the fit given in 'to_plot' attribute.
        :param main_fname: main directory which holds the XYT data.
        :param all_exp_summation_fn: the full path for the csv detailing all experiments SPI
        :param file_details_fname: the full path for the csv detailing all experiments' setups
        :param time_window_size: int
        :param tighter_bounds: boolean, whether to calc by averaging frame's probabilities in window, or consider window as
            single frame.
        :param measurement: string (either 'nucleation' or 'propagation')
        :param plot_kwargs: a dictionary for plot names and other attributes.
        :return:None
        """
        filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1),
                                                   os.listdir(main_fname))
        filtered_list_of_files_to_analyze = list(filtered_list_of_files_to_analyze)
        subplots_per_treatment = cls.calc_subplots_per_treatment(
            filtered_list_of_files_to_analyze=filtered_list_of_files_to_analyze,
            tighter_bounds=tighter_bounds,
            time_window_size=time_window_size)

        all_exp_summation_df = pd.read_csv(all_exp_summation_fn)
        all_exps_spi_df = all_exp_summation_df.loc[:, ['name', 'spatial_propagation_index']]
        unique_types = list(subplots_per_treatment.keys())
        colors = plt.get_cmap('plasma', len(unique_types))
        markers = {
            'FAC&BSO':'>',
            "C'":'P',
            'ML162':'x',
            'else': '*'
        }

        marker_options = lambda name: markers['FAC&BSO'] if "FAC&BSO" in name else markers["C'"] if "C'" in name else markers['ML162'] if 'ML162' in name else markers['else']

        treatment_idx = 0
        averages_by_treatment = {}
        spi_by_treatments = {}
        fig, ax = plt.subplots()
        for treatment_name, treatment_subplots_data in subplots_per_treatment.items():
            for exp_idx, exp_data_dict in enumerate(treatment_subplots_data):
                # windows = exp_data_dict['time_windows']
                exp_name = exp_data_dict['exp_name'][:-4]
                measurement_proba_in_windows = exp_data_dict[measurement]
                exp_spi = all_exps_spi_df[all_exps_spi_df['name'] == exp_name]['spatial_propagation_index'].values[0]
                spi_by_treatments[treatment_name] = spi_by_treatments.get(treatment_name, []) + [exp_spi]
                averages_by_treatment[treatment_name] = averages_by_treatment.get(treatment_name, []) + [np.average(np.array(measurement_proba_in_windows))]
            # plot the treatment details
            ax.scatter(averages_by_treatment[treatment_name], spi_by_treatments[treatment_name], color=colors(treatment_idx), marker="{}".format(marker_options(treatment_name)), label=treatment_name)
            ax.set_xlabel('{} average'.format(measurement))
            ax.set_ylabel('SPI')
            ax.set_title('SPI Vs. {}'.format(measurement))

            treatment_idx += 1

        ax.grid('on')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.05, 1))
        if plot_kwargs.get('showfig', False):
            plt.show()
        if plot_kwargs.get('savefig', False):
            fig.savefig(plot_kwargs.get('figname', 'SPI Vs. {} largest derivative.png'.format(measurement)), bbox_extra_artists=[lgd], bbox_inches='tight', dpi=200)

    @staticmethod
    def get_cells_not_neighboring_dead_cells(dead_cells_mask, neighbors, neighbors_list2, neighbors_list3, xy=None, threshold=200):
        """
        returns two groups of cells. 1st is all alive cells that are topological neighbors of dead cells' neighbors.
        2nd is the rest of the alive cells which are not direct topological neighbors of any dead cells.
        :param dead_cells_mask:
        :param neighbors:
        :param neighbors_list2:
        :param xy:
        :param threshold:
        :return:
        """
        all_alive_cells = np.array(dead_cells_mask-1, dtype=bool)
        # get all cells neighboring dead cells (propagation candidates)
        around_dead_cells = np.zeros(dead_cells_mask.shape, dtype=bool)
        for cell_idx, is_dead in enumerate(dead_cells_mask):
            if is_dead:
                curr_neighbors = neighbors[cell_idx]
                for neighbor_idx in curr_neighbors:
                    if xy is not None:
                        dist = NucleatorsCounter.get_real_distance(cell1_xy=xy[cell_idx], cell2_xy=xy[neighbor_idx])
                        around_dead_cells[neighbor_idx] = (True) * (dist < threshold)

        # get complementary & alive cells that are not near dead cells
        all_not_around_dead_cells_and_alive = np.array(around_dead_cells-1, dtype=bool) * all_alive_cells
        # divide to two groups at different "neighboring" distances
        not_around_dead_cells_1 = np.zeros(dead_cells_mask.shape, dtype=bool)
        not_around_dead_cells_2 = np.zeros(dead_cells_mask.shape, dtype=bool)
        for cell_idx, is_cell_not_adjacent_to_death in enumerate(all_not_around_dead_cells_and_alive):
            if is_cell_not_adjacent_to_death:
                alive_cell_2nd_lvl_neighbors = neighbors_list2[cell_idx]
                for adjacent_neighbor_idx in alive_cell_2nd_lvl_neighbors:
                    # if the cell(cell_idx) is a 2nd lvl neighbor to a dead cell
                    if dead_cells_mask[adjacent_neighbor_idx]:
                        not_around_dead_cells_1[cell_idx] = True
                        break

                alive_cell_3rd_lvl_neighbors = neighbors_list3[cell_idx]
                for adjacent_neighbor_idx in alive_cell_3rd_lvl_neighbors:
                    # if the cell(cell_idx) is a 3rd lvl neighbor to a dead cell
                    # and not a 2nd lvl neighbor to a dead cell
                    if dead_cells_mask[adjacent_neighbor_idx]:
                        not_around_dead_cells_2[cell_idx] = True * (not not_around_dead_cells_1[cell_idx])
                        break

        # not_around_dead_cells_2 = np.array(not_around_dead_cells_1-1, dtype=bool) * all_alive_cells
        return not_around_dead_cells_1, not_around_dead_cells_2


    @staticmethod
    def calc_pnuc_at_varying_distances_of_neighbors(exp_filename,
                                                    exp_main_directory_path,
                                                    file_details_full_path,
                                                    func_mode='single',
                                                    **plot_kwargs):
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
        n_instances = len(full_x)
        die_times = exp_xyt["death_time"].values
        XY = np.column_stack((full_x, full_y))
        exp_temporal_resolution = exp_details_df[exp_details_df['File Name'] == exp_filename]['Time Interval (min)'].values[0]
        exp_treatment_type = exp_details_df[exp_details_df['File Name'] == exp_filename]['Treatment'].values[0]
        jump_interval = exp_temporal_resolution
        time_window_size = 5 * jump_interval


        nucleation_lvl1_proba_in_window = []
        nucleation_lvl2_proba_in_window = []
        accumulated_death_prob_in_window = []
        # get neighbors list of all cells (topological by Voronoi)
        neighbors_list, neighbors_list2, neighbors_list3 = NucleatorsCounter.get_neighbors(XY=XY)


        maximal_time = np.max(die_times)
        sliding_window_start = jump_interval*2
        sliding_window_end = time_window_size + jump_interval*2
        while sliding_window_end <= maximal_time:
            nuc_lvl1_proba = 0
            nuc_lvl2_proba = 0
            # propag_proba = 0
            accu_death_proba = 0
            for i in range(0, time_window_size, jump_interval):
                curr_time = i + sliding_window_start
                if curr_time == maximal_time:
                    continue
                alive_cells_mask = (die_times >= curr_time)
                dead_cells_mask = (die_times < curr_time)
                all_dead_in_curr_time_frame = (die_times == curr_time)
                # get all dead cells
                all_dead_cells_prior_frame_mask = (die_times < curr_time)
                # get all possible propagation by neighbors of dead cells
                not_around_dead_cells_1, not_around_dead_cells_2 = \
                    NucleatorsProbabilitySlidingTimeWindow.get_cells_not_neighboring_dead_cells(all_dead_cells_prior_frame_mask,
                                                                                                neighbors_list,
                                                                                                neighbors_list2,
                                                                                                neighbors_list3,
                                                                                                xy=XY)
                print("##################")
                print(not_around_dead_cells_1.sum())
                print(not_around_dead_cells_2.sum())
                print("##################")
                nuc_lvl1_proba += ((all_dead_in_curr_time_frame * not_around_dead_cells_1).sum()+1)/(not_around_dead_cells_1.sum()+1)
                nuc_lvl2_proba += ((all_dead_in_curr_time_frame * not_around_dead_cells_2).sum()+1)/(not_around_dead_cells_2.sum()+1)
                accu_death_proba += (dead_cells_mask.sum()+1)/(alive_cells_mask.sum() + dead_cells_mask.sum()+1)

            nuc_lvl1_proba = nuc_lvl1_proba/(time_window_size/jump_interval)
            nuc_lvl2_proba = nuc_lvl2_proba/(time_window_size/jump_interval)
            accu_death_proba = accu_death_proba/(time_window_size/jump_interval)

            nucleation_lvl1_proba_in_window.append(nuc_lvl1_proba)
            nucleation_lvl2_proba_in_window.append(nuc_lvl2_proba)
            accumulated_death_prob_in_window.append(accu_death_proba)
            sliding_window_start += jump_interval
            sliding_window_end += jump_interval

        windows = list(range(0, len(nucleation_lvl1_proba_in_window)*exp_temporal_resolution, exp_temporal_resolution))
        if func_mode == 'multi':
            return {'exp_name': exp_filename,
                    'exp_treatment': exp_treatment_type,
                    'windows': windows,
                    'p_nuc_lvl1': nucleation_lvl1_proba_in_window,
                    'p_nuc_lvl2': nucleation_lvl2_proba_in_window,
                    'accumulated_death': accumulated_death_prob_in_window}
        fig, ax = plt.subplots()
        ax.plot(windows, nucleation_lvl1_proba_in_window, marker='<', label='lvl1 nuc')
        ax.plot(windows, nucleation_lvl2_proba_in_window, marker='p', label='lvl2 nuc')
        # plotting accumulated death
        ax.plot(windows, accumulated_death_prob_in_window, label='Accumulated Death', color='black', marker='p')

        ax.set_title('EXP:{}\nTREAT:{}'.format(exp_filename, exp_treatment_type))

        time_windows_ticks = list(range(0, die_times.max(), time_window_size))
        time_windows_labels = ["{}-{}".format(x, x+time_window_size) for x in time_windows_ticks]
        ax.set_xticks(time_windows_ticks)
        ax.set_xticklabels(time_windows_labels)
        ax.legend()
        if plot_kwargs.get('showfig', False):
            plt.show()
        if plot_kwargs.get('savefig', False):
            fig.savefig(plot_kwargs.get('figname', 'Validations/Nucleation@VaryingNeighboringDistancesApoptosis.png'.format()), bbox_inches='tight', dpi=200)

    @classmethod
    def plot_pnuc_for_varying_distances_multiple_exps(cls,
                                                      exps_main_dir,
                                                      file_details_full_path,
                                                      **plot_kwargs):
        filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1),
                                                   os.listdir(main_fname))
        filtered_list_of_files_to_analyze = list(filtered_list_of_files_to_analyze)

        all_exps_p_nuc_by_treatment = {}
        max_time = float('-inf')
        for exp_idx, exp_filename in enumerate(filtered_list_of_files_to_analyze):
            single_exp_data = cls.calc_pnuc_at_varying_distances_of_neighbors(exp_filename=exp_filename,
                                                                              exp_main_directory_path=exps_main_dir,
                                                                              file_details_full_path=file_details_full_path,
                                                                              func_mode='multi',
                                                                              **plot_kwargs)
            all_exps_p_nuc_by_treatment[single_exp_data['exp_treatment']] = \
                all_exps_p_nuc_by_treatment.get(single_exp_data['exp_treatment'], []) + [single_exp_data]
            if max(single_exp_data['windows']) > max_time:
                max_time = max(single_exp_data['windows'])

#         plotting
        ax1_clr = 'tab:blue'
        ax2_clr = 'tab:green'
        for treatment_name, treatment_subplots_data in all_exps_p_nuc_by_treatment.items():
            fig, axis = plt.subplots(len(treatment_subplots_data)//2+1, 2, figsize=(15,15), sharex=True)
            for exp_idx, exp_data_dict in enumerate(treatment_subplots_data):
                curr_ax = axis[exp_idx//2, exp_idx % 2]

                curr_ax.plot(exp_data_dict['windows'], exp_data_dict['p_nuc_lvl1'], label='Nucleationlvl1', color=ax1_clr)
                curr_ax.plot(exp_data_dict['windows'], exp_data_dict['p_nuc_lvl2'], label='Nucleationlvl2', color=ax2_clr)
                curr_ax.set_ylabel('P(Nuc)')
                curr_ax.set_xlabel('Time Window (in min)')
                time_windows_ticks = list(range(0, max_time, 120))
                time_windows_labels = ["{}-{}".format(x, x+120) for x in time_windows_ticks]
                curr_ax.set_ylim(0, 1.01)
                curr_ax.set_xticks(time_windows_ticks)
                curr_ax.set_xticklabels(time_windows_labels)
                curr_ax.set_title(exp_data_dict['exp_name'])

                curr_ax.plot(exp_data_dict['windows'], exp_data_dict['accumulated_death'], label='Accumulated Death', color='black', marker='p')
                curr_ax.legend()
            fig.suptitle(treatment_name)
            fig.autofmt_xdate(rotation=45)
            if plot_kwargs.get('showfig', False):
                plt.show()
            if plot_kwargs.get('savefig', False):
                fig.savefig(plot_kwargs.get('figname',
                                            os.sep.join(['Validations','Nucleation@VaryingNeighboringDistances|Treatment:{}.png'.format(treatment_name).replace('/', '|')])),
                                            bbox_inches='tight', dpi=200)

    @staticmethod
    def single_exp_led_calc(accumulated_death_rate: np.array, lf_scores:dict={}) -> np.array:
        largest_derivative, largest_derivative_idx= NucleatorsProbabilitySlidingTimeWindow.get_largest_derivative_idx(accumulated_death_rate, with_idx=True)
        led_scores = np.zeros_like(accumulated_death_rate)
        lf_0, d_0, lf_p = lf_scores.get('lf_0', 0), lf_scores.get('d_0', 0), lf_scores.get('lf_p', accumulated_death_rate[0])
        for frame_idx, frame_accu_death in enumerate(accumulated_death_rate):
            led_scores[frame_idx] = lf_0 + (lf_p-lf_0) * (1 - math.exp(-largest_derivative * (frame_idx-d_0)))

        return led_scores

# todo: save figures of different neighboring levels.
if __name__ == '__main__':
    # apoptosis model

    # filename = '20170523_MCF7_ML162_xy6.csv'
    # ferroptosis model
    # filename = '20180620_HAP1_erastin_xy2.csv'
    # rapture inhibition
    filename = '20181229_HAP1-920H_FB+PEG1450_GCAMP_xy51.csv'

    main_fname = 'ExperimentsXYT_CSVFiles/ConvertedTimeXYT'
    # main_fname = 'ExperimentsXYT_CSVFiles/CompressedTimeXYT'
    file_details_fname=os.path.sep.join(['ExperimentsXYT_CSVFiles', 'File details.csv'])
    # single_file_fname = os.sep.join([main_fname, filename])

    #
    #
    # NucleatorsProbabilitySlidingTimeWindow.plot_multiple_exps_nucleation_probability_time_window(time_window_size=WINDOW_SIZE_IN_FRAMES,
    #                                                                                              main_fname=main_fname,
    #                                                                                              file_details_fname=file_details_fname,
    #                                                                                              showfig=SHOWFIG,
    #                                                                                              savefig=SAVEFIG,
    #                                                                                              tighter_bounds=True,
    #                                                                                              enable_correlation=True)# , forced_temporal_res=30)

    NucleatorsProbabilitySlidingTimeWindow.plot_multiple_exps_nucleation_probability_time_window(time_window_size=WINDOW_SIZE_IN_FRAMES,
                                                                                                 main_fname=main_fname,
                                                                                                 file_details_fname=file_details_fname,
                                                                                                 showfig=SHOWFIG,
                                                                                                 savefig=SAVEFIG,
                                                                                                 tighter_bounds=True,
                                                                                                 plot_single=True,
                                                                                                 enable_correlation=True)# , forced_temporal_res=30)
    # NucleatorsProbabilitySlidingTimeWindow.plot_model_fit(main_fname=main_fname,
    #                                                       measurement='propagation',
    #                                                       # measurement='nucleation',
    #                                                       file_details_fname=file_details_fname,
    #                                                       to_plot=['poly_fit_rmse', 'poly_fit_r^2_score'],
    #                                                       # to_plot=['correlation', 'pvalue', 'intercept', 'stderr', 'slope'],
    #                                                       showfig=SHOWFIG,
    #                                                       savefig=SAVEFIG,
    #                                                       tighter_bounds=True,
    #                                                       time_window_size=3,
    #                                                       # model='polyfit2')
    #                                                       model='polyfit3')
    #                                                       # model=linregress)
    # NucleatorsProbabilitySlidingTimeWindow.plot_measurement_highest_derivative_index(main_fname=main_fname,
    #                                                                                  # measurement='propagation',
    #                                                                                  measurement='nucleation',
    #                                                                                  file_details_fname=file_details_fname,
    #                                                                                  showfig=SHOWFIG,
    #                                                                                  savefig=SAVEFIG,
    #                                                                                  tighter_bounds=True)
    # NucleatorsProbabilitySlidingTimeWindow.calc_pnuc_at_varying_distances_of_neighbors(exp_filename=filename,
    #                                                                                    exp_main_directory_path=main_fname,
    #                                                                                    file_details_full_path=file_details_fname,
    #                                                                                    showfig=SHOWFIG,
    #                                                                                    savefig=SAVEFIG)
    # NucliatorsProbabilitySlidingTimeWindow.plot_pnuc_for_varying_distances_multiple_exps(exps_main_dir=main_fname,
    #                                                                                      file_details_full_path=file_details_fname,
    #                                                                                      showfig=SHOWFIG,
    #                                                                                      savefig=SAVEFIG)



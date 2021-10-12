import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
from global_parameters import *


class BaselineCell2CellInfluenceAnalyzer:
    def __init__(self, file_full_path: str, **kwargs):
        self.file_full_path = file_full_path
        self.exp_name = self.file_full_path.split(os.sep)[-1]
        self.exp_df = pd.read_csv(self.file_full_path)
        self.full_path_to_experiments_metadata_file = kwargs.get('metadata_file_full_path', METADATA_FILE_FULL_PATH)
        self.exp_treatment_name, self.exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(
            exp_file_name=self.exp_name, meta_data_file_full_path=self.full_path_to_experiments_metadata_file)

        self._cell_number = len(self.exp_df)
        self._cells_xy = self.exp_df.loc[:, ['cell_x', 'cell_y']].values
        self._cells_death_times = self.exp_df.loc[:, ['death_time']].values

        self.frames_number = len(self._cells_death_times)

        self.all_frames_idx = np.arange(0, self.frames_number, 1)
        self.all_frames_by_minutes = np.arange(self._cells_death_times.min(), self._cells_death_times.max(),
                                               self.exp_temporal_resolution)

        self.cells_neighbors_distance_threshold = kwargs.get('threshold_dist', DIST_THRESHOLD_IN_PIXELS)
        self._cells_neighbors_list1, self._cells_neighbors_list2, self._cells_neighbors_list3 = \
            get_cells_neighbors(XY=self._cells_xy, threshold_dist=self.cells_neighbors_distance_threshold)

        self._distance_metric = kwargs.get('distance_metric_from_true_times_of_death', 'rmse')

        self.median_by_cell_dist = 0
        self.mean_by_cell_dist = 0

        self._base_line_calculated = False

    def calc_baseline(self):
        median_by_cell = list()
        mean_by_cell = list()
        cells_with_no_neighbors_indices = list()

        for curr_cell_idx, curr_cell_time_of_death in enumerate(self._cells_death_times):
            curr_cell_neighbors_indices = self._cells_neighbors_list1[curr_cell_idx]
            curr_cell_neighbors_times_of_death = self._cells_death_times[[curr_cell_neighbors_indices]].flatten()
            # verify that curr cell has neighbors ("lone cell"), if it does not, ignore that cell as we can not perform prediction based on zero data
            # to ignore that cell in future calculations, we remove it from the cell death list entirely by keeping a list of cells' indices which had no neighbors
            if len(curr_cell_neighbors_times_of_death) == 0:
                cells_with_no_neighbors_indices.append(curr_cell_idx)
                continue

            median_by_cell.append(np.median(curr_cell_neighbors_times_of_death))
            mean_by_cell.append(np.mean(curr_cell_neighbors_times_of_death))

        median_by_cell = np.array(median_by_cell)
        mean_by_cell = np.array(mean_by_cell)

        loner_cells_mask = np.ones(len(self._cells_death_times), dtype=bool)
        loner_cells_mask[[cells_with_no_neighbors_indices]] = False
        cells_times_of_death_with_no_loner_cells = self._cells_death_times[loner_cells_mask, ...]

        self.median_by_cell_dist = calc_distance_metric_between_signals(y_true=cells_times_of_death_with_no_loner_cells,
                                                                        y_pred=median_by_cell,
                                                                        metric=self._distance_metric)
        self.mean_by_cell_dist = calc_distance_metric_between_signals(y_true=cells_times_of_death_with_no_loner_cells,
                                                                      y_pred=mean_by_cell,
                                                                      metric=self._distance_metric)

        self._base_line_calculated = True

    @staticmethod
    def multiple_experiments_of_treatment_error(treatment_type: str,
                                                meta_data_full_file_path: str,
                                                all_experiments_dir_full_path: str):
        # todo: implement
        pass
    # def visualize_metric(self):
    #     if self._base_line_calculated is not True:
    #         raise RuntimeError(f'baseline results were not calculated, please invoke calc_baseline method')
    #
    #     fig, axis = plt.subplots(1, 2)
    #     axis[0]


if __name__ == '__main__':
    single_file_path = 'C:\\Users\\User\\PycharmProjects\\CellDeathQuantification\\Data\\Experiments_XYT_CSV\\OriginalTimeMinutesData\\20160820_10A_FB_xy14.csv'
    baseline_influence_analyzer = BaselineCell2CellInfluenceAnalyzer(file_full_path=single_file_path)
    baseline_influence_analyzer.calc_baseline()

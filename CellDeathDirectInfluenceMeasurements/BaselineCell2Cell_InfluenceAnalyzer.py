import os
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
        self.exp_treatment_name, self.exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=self.exp_name, meta_data_file_full_path=self.full_path_to_experiments_metadata_file)

        self._cell_number = len(self.exp_df)
        self._cells_xy = self.exp_df.loc[['cell_x', 'cell_y'], :].values
        self._cells_death_times = self.exp_df[['death_time'], :].values

        self.frames_number = len(self._cells_death_times)

        self.all_frames_idx = np.arange(0, self.frames_number, 1)
        self.all_frames_by_minutes = np.arange(self._cells_death_times.min(), self._cells_death_times.max(), self.exp_temporal_resolution)

        self.cells_neighbors_distance_threshold = kwargs.get('threshold_dist', DIST_THRESHOLD_IN_PIXELS)
        self._cells_neighbors_list1, self._cells_neighbors_list2, self._cells_neighbors_list3 = \
            get_cells_neighbors(XY=self._cells_xy, threshold_dist=self.cells_neighbors_distance_threshold)

    def calc_baseline(self):
        median_by_frame = np.zeros_like(self.all_frames_by_minutes)
        mean_by_frame = np.zeros_like(self.all_frames_by_minutes)
        for frame_idx, time_of_frame in enumerate(self.all_frames_by_minutes):
            dead_cells_in_current_frame_mask = self._cells_death_times[self._cells_death_times == time_of_frame]
            dead_cells_upto_frame_mask = self._cells_death_times[self._cells_death_times < time_of_frame]


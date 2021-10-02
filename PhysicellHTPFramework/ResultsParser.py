import os
from typing import *
import numpy as np
import pandas as pd
import pyMCDS
import pyMCDS_timeseries as pyMCDS_ts
# import Visualization as vs


class ResultsParser:
    def __init__(self, simulation_output_path: str, parsed_results_dir_path_to_save: str = None):
        self.output_path = simulation_output_path
        self.ts_mcds_parsed_experiment = pyMCDS_ts.pyMCDS_timeseries(output_path=self.output_path)
        self.parsed_results = None
        self.initial_state = self.ts_mcds_parsed_experiment.timeseries[0].data
        self.all_cells_ids = np.arange(0, self.initial_state['discrete_cells']['ID'].max(), 1, dtype=int)
        self.parsed_results_dir_path_to_save = parsed_results_dir_path_to_save
        if self.parsed_results_dir_path_to_save is None:
            self.parsed_results_dir_path_to_save = 'C:\\Users\\User\\PycharmProjects\\CellDeathQuantification\\Data\\Simulations_XYT_CSV\\apoptosis_attempts'

        self.implicit_temporal_resolution = abs(self.ts_mcds_parsed_experiment.timeseries[0].data['metadata']['current_time'] - self.ts_mcds_parsed_experiment.timeseries[1].data['metadata']['current_time'])

    def get_dead_cells_at_t(self, time_frame_data: pyMCDS, prev_dead_cells_idx: set, mode: str="") -> np.array:
        dead_cells_and_loci = list()
        cells_alive_at_time_frame_t = time_frame_data.data['discrete_cells']
        cells_alive_at_time_frame_t_ids = cells_alive_at_time_frame_t['ID']
        all_dead_cells_up_until_time_frame_t = np.setdiff1d(self.all_cells_ids, cells_alive_at_time_frame_t_ids)
        all_dead_cells_in_time_frame_t = np.array(all_dead_cells_up_until_time_frame_t, dtype=int)
        for cell_idx in all_dead_cells_in_time_frame_t:
            if cell_idx not in prev_dead_cells_idx:
                cell_x, cell_y = self.initial_state['discrete_cells']['position_x'][cell_idx], \
                                 self.initial_state['discrete_cells']['position_y'][cell_idx]
                # x,y,t
                dead_cells_and_loci.append(np.array([cell_x, cell_y, cell_idx]))
        return np.array(dead_cells_and_loci)

    def get_cells_death_time(self, ts_mcds: pyMCDS_ts, temporal_resolution: int = None):
        if temporal_resolution is None:
            temporal_resolution = self.implicit_temporal_resolution
        xyt_df = {'cell_x': [], 'cell_y': [], 'death_time': []}
        time_frames = len(ts_mcds.timeseries)
        dead_cells_indices = set()

        time_from_death_initiation = 0

        for frame_idx in range(time_frames):
            time_frame_data = ts_mcds.timeseries[frame_idx]

            if len(time_frame_data.data['discrete_cells']) == 0:
                break

            dead_cells_in_curr_frame = self.get_dead_cells_at_t(time_frame_data=time_frame_data,
                                                                         prev_dead_cells_idx=dead_cells_indices)
            if len(xyt_df['death_time']) > 0:
                time_from_death_initiation += temporal_resolution

            if dead_cells_in_curr_frame.size == 0:
                continue

            dead_cells_indices.update(dead_cells_in_curr_frame[:, 2].tolist())
            dead_cells_in_curr_frame[:, 2] = time_from_death_initiation
            xyt_df['cell_x'] += dead_cells_in_curr_frame[:, 0].tolist()
            xyt_df['cell_y'] += dead_cells_in_curr_frame[:, 1].tolist()
            xyt_df['death_time'] += dead_cells_in_curr_frame[:, 2].tolist()

        return pd.DataFrame(xyt_df)

    def parse_results_of_entire_experiments(self):
        self.parsed_results = self.get_cells_death_time(self.ts_mcds_parsed_experiment)

    def save_simulation_xyt_df(self, filename: str = None, dir_path_to_save: str = None) -> None:
        if self.parsed_results is None:
            raise RuntimeError('you must parse the results first! use parse_results_of_entire_experiments method')
        if dir_path_to_save is None:
            dir_path_to_save = self.parsed_results_dir_path_to_save
        if os.path.isdir(dir_path_to_save):
            os.makedirs(dir_path_to_save, exist_ok=True)
        if filename is None:
            filename = 'xyt_test.csv'
        if not filename.endswith('.csv'):
            filename = f'{filename}.csv'

        path_to_save = os.sep.join([dir_path_to_save, filename])
        self.parsed_results.to_csv(path_to_save)


if __name__ == '__main__':
    rp = ResultsParser(simulation_output_path='C:/Users/User/PhysiCell/output/')
    rp.parse_results_of_entire_experiments()
    print(rp.parsed_results.head())
    # vs.visualize_cell_death_in_time(xyt_df=rp.parsed_results,
    #                                 exp_treatment='simulation',
    #                                 exp_name='simulation1')
    rp.save_simulation_xyt_df(filename='xyt_ferroptosis_attempt')

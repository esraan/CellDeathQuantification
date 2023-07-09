import os.path
import numpy as np
import pandas as pd
from typing import *
import datetime
from tqdm import tqdm
from scipy.spatial import distance


def remove_reappearing_cells_death_events(death_events_df: pd.DataFrame,
                                          distance_threshold_px: Union[float, int] = 15,
                                          **kwargs) -> pd.DataFrame:
    death_events_df = death_events_df.reset_index(drop=True, inplace=False)
    reseted_idx_df_copy = death_events_df.reset_index(drop=True, inplace=False)
    rows_indices_to_rm = []
    for row_idx, row in tqdm(reseted_idx_df_copy.iterrows(), desc=f"checking for reappearing death events: "):
        trimmed_df_copy = reseted_idx_df_copy.drop(np.arange(0, row_idx + 1, 1))
        for row2_idx, row2 in trimmed_df_copy.iterrows():
            if row_idx >= row2_idx:
                continue
            row_loc = row[kwargs.get("coordinates_col_names", ['cell_x', 'cell_y'])].values
            row2_loc = row2[kwargs.get("coordinates_col_names", ['cell_x', 'cell_y'])].values
            dist_between_locs = distance.euclidean(row_loc, row2_loc)
            if dist_between_locs <= distance_threshold_px:
                rows_indices_to_rm.append(row2_idx)
    clean_df = reseted_idx_df_copy.drop(rows_indices_to_rm, inplace=False)
    return clean_df


def convert_trackmate_spots_data_to_xyt_csv(spots_csv_path,
                                            str_colnames: List[str] = ['ID', 'TRACK_ID'],
                                            float_colnames: List[str] = ['POSITION_X', 'POSITION_Y', 'FRAME'],
                                            n_redundant_lines: int = 3,
                                            temporal_resolution: int = None,
                                            cols_renaming_scheme: Dict[str, str] = {
        'POSITION_X': 'cell_x',
        'POSITION_Y': 'cell_y',
        'FRAME': 'death_time'
    },
                                            **kwargs):
    raw_spots_df = pd.read_csv(spots_csv_path)
    raw_spots_min_columns = raw_spots_df.loc[n_redundant_lines:, str_colnames+float_colnames]
    # converting dtypes:
    raw_spots_min_columns_correct_dtypes = raw_spots_min_columns.copy()
    for col in str_colnames:
        raw_spots_min_columns_correct_dtypes[col] = raw_spots_min_columns_correct_dtypes[col].astype(str)

    for col in float_colnames:
        raw_spots_min_columns_correct_dtypes[col] = raw_spots_min_columns_correct_dtypes[col].astype(float)

    raw_spots_min_columns_correct_dtypes = raw_spots_min_columns_correct_dtypes.convert_dtypes()

    spots_min_columns_correct_dtypes_sorted_by_time = raw_spots_min_columns_correct_dtypes.sort_values(by=['FRAME'])
    spots_min_columns_correct_dtypes_no_duplicates = spots_min_columns_correct_dtypes_sorted_by_time.drop_duplicates(
        subset=['TRACK_ID'], keep='first')

    experiment_xyt_df = spots_min_columns_correct_dtypes_no_duplicates[['POSITION_X', 'POSITION_Y', 'FRAME']].copy()
    experiment_xyt_df.rename(columns=cols_renaming_scheme, inplace=True)
    if kwargs.get('remove_close_death_events', False):
        experiment_xyt_df = remove_reappearing_cells_death_events(
            death_events_df=experiment_xyt_df,
            **kwargs
        )
    if temporal_resolution is not None:
        experiment_xyt_df['death_time_frame_num'] = experiment_xyt_df['death_time'].copy()
        experiment_xyt_df['death_time'] = experiment_xyt_df['death_time_frame_num'] * temporal_resolution
    if kwargs.get('save_converted_df', True):
        org_path_split = spots_csv_path.split(os.sep)
        org_name, org_dir_path = org_path_split[-1], os.path.join(*org_path_split[:-1])
        if kwargs.get('new_csv_name') is None:
            new_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_{org_name}"
        else:
            new_name = kwargs.get('new_csv_name')
        new_path = os.path.join(os.sep, org_dir_path, new_name)
        experiment_xyt_df.to_csv(new_path)

    return experiment_xyt_df


if __name__ == '__main__':
    spots_path = "/Users/yishaiazabary/PycharmProjects/University/CellDeathQuantification/Data/2023 data/OriginalTimeFramesData/5-11-23_10agreennuclei_facnbso+ml162_computational001_crop_2023_05_11.csv" #"/Users/yishaiazabary/Library/CloudStorage/GoogleDrive-yshaayaz@gmail.com/My Drive/אוניברסיטה/Ferroptosis/2023 datasets/Raw videos_Jyotirekha/gene_clone_14fieldofview/trackmate_spots_export.csv"
    convert_trackmate_spots_data_to_xyt_csv(spots_path,
                                            temporal_resolution=8,
                                            remove_close_death_events=True,
                                            new_csv_name="5-11-23_10agreennuclei_facnbso+ml162_computational001_crop_2023_05_11_death_times_no_reapearing_death_events.csv"
    )
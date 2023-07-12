import os
import numpy as np
import pandas as pd

MAIN_DIR_UNC0NVERTED = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles'
MAIN_DIR_CONVERTED = '/ExperimentsXYT_CSVFiles/CompressedTimeXYT'
EXPS_DETAILS_FN = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles/File details.csv'

class XytToTrueTimeConvertor:
    def __init__(self, main_dir_unconverted_full_path, main_dir_converted_full_path, exps_details_file_full_path):
        self.main_dir_unconverted_full_path = main_dir_unconverted_full_path
        self.main_dir_converted_full_path = main_dir_converted_full_path
        self.exps_details_file_full_path = exps_details_file_full_path
        self.exp_details_df = pd.read_csv(self.exps_details_file_full_path)
        self.max_length_in_minutes = float('-inf')

    @staticmethod
    def convert_single_file(original_df: pd.DataFrame, file_temporal_res=30) -> pd.DataFrame:
        converted = original_df.copy()
        converted['death_time'] = converted['death_time'] * file_temporal_res
        return converted

    def get_exp_temporal_res(self, exp_name):
        return self.exp_details_df[self.exp_details_df['File Name'] == exp_name]['Time Interval (min)'].values[0]

    def convert_multiple_files(self, save=True):
        filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1), os.listdir(self.main_dir_unconverted_full_path))
        filtered_list_of_files_to_analyze = list(filtered_list_of_files_to_analyze)
        all_exps_coverted_dfs = {}
        for f_idx, file in enumerate(filtered_list_of_files_to_analyze):
            if f_idx > 200000:
                break
            if file.find('.csv') != -1:
                exp_temporal_res = self.get_exp_temporal_res(file)
                original_df = pd.read_csv(os.sep.join([self.main_dir_unconverted_full_path, file]))
                converted_df = self.convert_single_file(original_df, file_temporal_res=exp_temporal_res)
                if save:
                    new_exp_fn = os.sep.join([self.main_dir_converted_full_path, file])
                    converted_df.to_csv(new_exp_fn)
                all_exps_coverted_dfs[file] = converted_df

        return all_exps_coverted_dfs

    def reduce_temporal_resolution(self):
        # todo:
        pass


if __name__ == '__main__':
    converter = XytToTrueTimeConvertor(MAIN_DIR_UNC0NVERTED, MAIN_DIR_CONVERTED, EXPS_DETAILS_FN)
    converter.convert_multiple_files(save=True)
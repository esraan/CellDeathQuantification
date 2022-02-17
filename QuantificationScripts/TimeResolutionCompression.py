import numpy as np
import pandas as pd
from utils import *
from global_parameters import *


def compress_single_file_temporal_resolution_given_compression_factor(file_path: str = None,
                                                                      compression_factor: int = 2,
                                                                      **kwargs) -> pd.DataFrame:
    """
    compresses the temporal resolution of a given file (death_time column compression only) by the provided
    compression_factor argument.
    if kwargs.save_compressed_df argument is set to True, saves the compressed version under the appropriate
    directory.
    if kwargs.save_df_in_original_dir argument is set to True, saves the compressed version under the original directory
    of the file, adding the "compressed_to_X" postfix, where X=new temporal resolution. The function also adds the
    new filename to the meta data file with the treatment postfix "compressed_to_X" with the same X value.
    :param file_path: str
    :param compression_factor: int, the
    :param kwargs:
    :return: pd.DataFrame, the compressed dataframe.
    """
    # todo: test the function
    assert file_path, f'either file_path or file_df should be provided, but file_path is f{file_path}'
    save_df = kwargs.get('save_compressed_df', False)
    save_df_in_original_dir = kwargs.get('save_df_in_original_dir', False) * save_df
    meta_data_file_path = kwargs.get('meta_data_full_file_path', METADATA_FILE_FULL_PATH)

    file_name = file_path.split(os.sep)[-1]
    file_dir_path = os.sep.join(file_path.split(os.sep)[:-1])

    exp_treatment, exp_explicit_temporal_res = get_exp_treatment_type_and_temporal_resolution(exp_file_name=file_name,
                                                                                              compressed_flag=False)
    new_temporal_res = exp_explicit_temporal_res * compression_factor

    # validate the compressed file directory exists, if not, create it
    compression_level_dir = os.sep.join([COMPRESSED_FILE_MAIN_DIR, f'x{compression_factor}'])
    if not os.path.isdir(compression_level_dir):
        os.makedirs(compression_level_dir)

    file_df = pd.read_csv(file_path).loc[:, ['cell_x', 'cell_y', 'death_time']]
    compressed_df = file_df.copy()

    new_all_timeframes_in_minutes = np.arange(0, file_df.death_time.values.max() + new_temporal_res + 1,
                                              new_temporal_res)

    time_idx = 1
    for time_frame in new_all_timeframes_in_minutes[1:]:
        mask = (compressed_df.death_time <= time_frame) * \
               (compressed_df.death_time > new_all_timeframes_in_minutes[time_idx - 1])
        compressed_df.loc[mask, 'death_time'] = time_frame

        time_idx += 1

    if save_df:
        if not save_df_in_original_dir:
            compressed_file_path = os.sep.join([compression_level_dir, file_name])
        else:
            new_file_name = f"{file_name.replace('.csv', '')}_compressed_to_{new_temporal_res}.csv"
            new_treatment_name = exp_treatment #f"{exp_treatment}_compressed_to_{new_temporal_res}"
            compressed_file_path = os.path.join(file_dir_path, new_file_name)
            # updating the meta data file
            meta_data_df = pd.read_csv(meta_data_file_path, index_col=None)
            meta_data_df = meta_data_df.drop(columns=['Unnamed: 0.1','Unnamed: 0'], errors='ignore')
            new_exp_row = meta_data_df[meta_data_df['File Name']==file_name].copy()
            new_exp_row['File Name'] = new_file_name
            new_exp_row['Treatment'] = new_treatment_name
            new_exp_row['Time Interval (min)'] = new_temporal_res
            meta_data_df = meta_data_df.append(new_exp_row)
            meta_data_df.to_csv(meta_data_file_path)

        compressed_df.to_csv(compressed_file_path)

    return compressed_df


def compress_all_files_in_dir_temporal_resolution_given_compression_factor(dir_path: str = None,
                                                                           compression_factor: int = 2, **kwargs) -> \
        Optional[List[pd.DataFrame]]:
    """
    compresses the temporal resolution of a given file (death_time column compression only) by the provided
    compression_factor argument.
    if kwargs.save_compressed_df argument is set to True, saves the compressed version under the appropriate
    directory.
    :param dir_path: str
    :param compression_factor: int, the
    :param kwargs:
    :return: Optional[List[pd.DataFrame]], the compressed dataframes.
    """
    save_df = kwargs.get('save_compressed_df', False)
    print_progression_flag = kwargs.get('print_progression', False)
    compress_resolutions_constraint = kwargs.get('compress_resolutions_constraint', None)

    all_files_full_paths, only_exp_names = get_all_paths_csv_files_in_dir(dir_path=dir_path)

    if save_df:
        all_compressed_dfs = list()
    else:
        all_compressed_dfs = None

    for file_idx, full_single_file_path in enumerate(all_files_full_paths):
        file_name = full_single_file_path.split(os.sep)[-1]
        exp_treatment, exp_explicit_temporal_res = get_exp_treatment_type_and_temporal_resolution(
            exp_file_name=file_name,
            compressed_flag=False)
        if compress_resolutions_constraint is not None and exp_explicit_temporal_res != compress_resolutions_constraint:
            continue

        if print_progression_flag:
            print(f'file name {full_single_file_path.split(os.sep)[-1]}')
        single_file_compressed_df = compress_single_file_temporal_resolution_given_compression_factor(
            file_path=full_single_file_path,
            compression_factor=compression_factor,
            **kwargs)
        if save_df:
            all_compressed_dfs.append(single_file_compressed_df)

    if save_df:
        return all_compressed_dfs


def compress_all_files_in_dir_temporal_resolution_given_compression_factor_range(dir_path: str = None,
                                                                                 compression_factor_range: tuple = (
                                                                                         2, 6),
                                                                                 **kwargs) -> None:
    """
    compresses the temporal resolution of a given file (death_time column compression only) by the provided
    compression_factor_range argument (compresses for each value withing the range - excluding the last value).
    if kwargs.save_compressed_df argument is set to True, saves the compressed version under the appropriate
    directory.
    :param dir_path: str
    :param compression_factor_range: tuple[int,int] - the range of compression factor to compress by
    :param kwargs:
    :return: pd.DataFrame, the compressed dataframe.
    """
    compression_factor_range_values = np.arange(compression_factor_range[0], compression_factor_range[1], 1)

    print_progression_flag = kwargs.get('print_progression', False)

    for compression_factor in compression_factor_range_values:
        if print_progression_flag:
            print(f'######\ncompression factor {compression_factor}\n#######')
        compress_all_files_in_dir_temporal_resolution_given_compression_factor(dir_path=dir_path,
                                                                               compression_factor=compression_factor,
                                                                               **kwargs)


if __name__ == '__main__':
    # single experiment compression
    # file_path = '..\\Data\\Experiments_XYT_CSV\\OriginalTimeMinutesData\\20160828_10AsgCx43_FB_xy01.csv'
    #
    # compressed_df = compress_single_file_temporal_resolution_given_compression_factor(file_path=file_path,
    #                                                                                   save_compressed_df=True,
    #                                                                                   compression_factor=8)

    # # entire directory compression - single compression factor
    # dir_path = '..\\Data\\Experiments_XYT_CSV\\OriginalTimeMinutesData'
    # compress_all_files_in_dir_temporal_resolution_given_compression_factor(dir_path=dir_path,
    #                                                                        compression_factor=2,
    #                                                                        save_compressed_df=True)
    #
    # # entire directory compression - range of compression factor
    # dir_path = '..\\Data\\Experiments_XYT_CSV\\OriginalTimeMinutesData'
    # compress_all_files_in_dir_temporal_resolution_given_compression_factor_range(dir_path=dir_path,
    #                                                                              compression_factor_range=(2, 10),
    #                                                                              save_compressed_df=True,
    #                                                                              print_progression=True)

    # only 15 minutes experiments compression
    dir_path = '..\\Data\\Experiments_XYT_CSV\\OriginalTimeMinutesData'
    meta_data_full_file_path = 'C:\\Users\\User\\PycharmProjects\\CellDeathQuantification\\Data\\Experiments_XYT_CSV\\ExperimentsMetaData.csv'
    compress_all_files_in_dir_temporal_resolution_given_compression_factor(
        dir_path=dir_path,
        compression_factor=2,
        save_compressed_df=True,
        save_df_in_original_dir=True,
        meta_data_full_file_path=meta_data_full_file_path,
        compress_resolutions_constraint=15
    )

    pass

import os
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt


def mat_to_csv(mat_path, times_key:str = 'numBlobs', save_df=False, rank=False):
    data = io.loadmat(mat_path)

    original_XY = data["Centroids2"]
    x_array, y_array = [x[0] for x in original_XY], [x[1] for x in original_XY]
    num_of_dead_cells = data[times_key]
    # extract times
    times = []
    for u in range(len(num_of_dead_cells)):
        for x in range(num_of_dead_cells[u][0]):
            times.append(u)

    df = pd.DataFrame(
        {
            'cell_x': x_array,
            'cell_y': y_array,
            'die_times': times
        }
    )
    df = pd.DataFrame(df)
    if rank:
        df['die_times'] = df['die_times'].rank()
    if save_df:
        df.to_csv(mat_path.replace('.mat', '.csv'))
    return df

def scatter_cells_death(df: pd.DataFrame, log_times=False)->None:
    plt.scatter(df['cell_x'], df['cell_y'], c=np.log(df['die_times'] if log_times else df['die_times']))
    plt.show()

def multiple_mat_to_csv_by_folder(directory_path: str)->None:
    for file in os.listdir(directory_path):
        if file.find('.mat') != -1:
            print('converting file {}'.format(file))
            mat_to_csv(os.sep.join([directory_path, file]), save_df=True)

def compress_time_resolution_from_df(exp_df, compression_factor=2) -> pd.DataFrame:
    try:
        die_times = exp_df["die_times"].values
    except AttributeError as e:
        die_times = exp_df["die_times"]
    curr_time_frames = np.unique(die_times)
    org_temporal_res = abs(curr_time_frames[-2] - curr_time_frames[-1])
    new_temporal_res = compression_factor * org_temporal_res
    curr_max_time = new_temporal_res
    curr_min_time = 1
    new_die_times = np.zeros_like(die_times)
    # incase the compression factor is bigger than the maximum current time
    if curr_max_time > die_times.max():
        curr_max_time = die_times.max()
    while curr_max_time < die_times.max()+new_temporal_res:
        within_mask = (die_times >= curr_min_time) * (die_times <= curr_max_time)
        new_die_times[within_mask] = curr_max_time
        curr_min_time += new_temporal_res
        curr_max_time += new_temporal_res

    exp_df['die_times'] = new_die_times
    return exp_df

def compress_time_resolution_multiple_files(main_fn,
                                            compress_fn,
                                            lst_files_to_compress,
                                            compression_factor=2):
    for filename in lst_files_to_compress:
        exp_details_df = pd.read_csv(os.sep.join([main_fn, filename]))
        compressed_df = compress_time_resolution_from_df(exp_details_df, compression_factor)
        compressed_df.to_csv(os.sep.join([compress_fn, filename]))

# multiple_mat_to_csv_by_folder('XYTMatFilesToConvert')
# df = mat_to_csv('ExperimentsXYT_CSVFiles/20200311_HAP1_ML162_xy4.mat', save_df=True, rank=False)
# scatter_cells_death(df)
# t = pd.read_csv('/Users/yishaiazabary/Downloads/prescreen_data.csv')
# t = pd.read_csv('/Users/yishaiazabary/Downloads/prescreen_data (1).csv')
# t

if __name__ == '__main__':
    main_fn = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles'
    converted_main_fn = os.sep.join([main_fn, 'ConvertedTimeXYT'])
    compressed_main_fn = os.sep.join([main_fn, 'CompressedTimeXYT'])
    # lst_of_files_to_compress = ['20181227_MCF10A_SKT_xy2.csv', '20170213_U937_FB_xy2.csv',
    # '20160828_10AsgCx43_FB_xy04.csv',
    # '20160828_10Awt_FB_xy03.csv',
    # '20161129_MCF7_FB_xy15.csv',
    # '20180620_HAP1_erastin_xy5.csv',
    # '20181227_MCF10A_SKT_xy2.csv',
    # '20181229_HAP1-920H_FB+PEG3350_GCAMP_xy58.csv',
    # '20200311_HAP1_ML162_xy3.csv']
    filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1), os.listdir(converted_main_fn))
    lst_of_files_to_compress = list(filtered_list_of_files_to_analyze)


    compress_time_resolution_multiple_files(converted_main_fn,
                                            compressed_main_fn,
                                            lst_of_files_to_compress)
    # mat_to_csv('/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/XYTMatFilesToConvert/ORG_MAT_XYT_files/20171008_MCF7_H2O2_xy5.mat')
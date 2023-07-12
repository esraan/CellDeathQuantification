import json
import os
import numpy as np
import numpy.polynomial.polynomial as nppoly
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, linregress
from NucliatorsCount import NucleatorsCounter
from sklearn import metrics
from DataPreprocessing import *
from NucleationProbabilityAndSPI import NucleationProbabilityAndSPI
import copy


ALWAYS = True
MAX_EXP_TO_PROCESS = 30000
FONT_SIZE = 10
SHOWFIG = True
SAVEFIG = True
WINDOW_SIZE_IN_FRAMES = 3
MARKERS_BY_TREATMENT = {
    'PEG': 'o',
    'FAC&BSO':'.',
    "C'":'v',
    'ML162':'^',
    'erastin': '<',
    'H2O2': '>',
    'zVAD': 's',
    'TRAIL': '+',
}


def get_all_files_path_in_dir(main_fn):
    filtered_list_of_files_to_analyze = filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1), os.listdir(main_fn))
    full_paths = [os.sep.join([main_fn, x]) for x in filtered_list_of_files_to_analyze]
    return full_paths


def get_exp_data(exp_full_path, exp_details_full_path):
    exp_name = exp_full_path.split(os.sep)[-1].replace('MANIPULATED_', '')
    exp_xyt_full_data = pd.read_csv(exp_full_path)
    exp_details_path = exp_details_full_path
    exp_details_data = pd.read_csv(exp_details_path)
    full_x = exp_xyt_full_data["cell_x"].values
    full_y = exp_xyt_full_data["cell_y"].values
    die_times = exp_xyt_full_data["death_time"].values
    XY = np.column_stack((full_x, full_y))
    treatment = exp_details_data[exp_details_data['File Name'] == exp_name]['Treatment'].values[0]
    temporal_res = exp_details_data[exp_details_data['File Name'] == exp_name]['Time Interval (min)'].values[0]
    neighbors_list, neighbors_list2, neighbors_list3 = NucleatorsCounter.get_neighbors(XY=XY)
    return {
        'exp_name': exp_name,
        'XY': XY,
        'die_times': die_times,
        'treatment': treatment,
        'neighbors_list': neighbors_list,
        'temporal_res': temporal_res
    }


def calc_adjacent_nucleator_scenraio_ctr(XY, death_times, neighbors_list_1lvl, dist_threshold=200):
    adjacent_nucleator_candidates = 0
    leaders = {}
    propagated = {}
    for curr_cell_idx, curr_cell_death in enumerate(death_times):
        curr_cell_neighbors = neighbors_list_1lvl[curr_cell_idx]
        curr_cell_xy = XY[curr_cell_idx]
        curr_cell_near_neighbors = filter(lambda neighbor_idx: NucleatorsCounter.get_real_distance(curr_cell_xy, XY[neighbor_idx]) <= dist_threshold, curr_cell_neighbors)
        # assume curr_cell is nucleator
        leaders[curr_cell_idx] = True
        propagated[curr_cell_idx] = False
        # Check if any of the neighbors are nucleators/died before the curr cell
        for adjacent_cell_idx in curr_cell_near_neighbors:
            adjacent_cell_death = death_times[adjacent_cell_idx]
            if adjacent_cell_death < curr_cell_death:
                propagated[curr_cell_idx] = True
                leaders[curr_cell_idx] = False
                break
            if adjacent_cell_death == curr_cell_death and leaders.get(adjacent_cell_idx, False):
                adjacent_nucleator_candidates += 1
                propagated[curr_cell_idx] = True
                leaders[curr_cell_idx] = False
                break
            if propagated.get(adjacent_cell_idx, False):
                propagated[curr_cell_idx] = True
                leaders[curr_cell_idx] = False
                break

    return leaders, propagated, adjacent_nucleator_candidates/len(XY)


def calc_adjacent_nucleator_scenraio_ctr_multiple_exps(main_fn, exps_details_full_path):
    all_exps_fns = get_all_files_path_in_dir(main_fn=main_fn)
    all_exps_data = {}
    for exp_idx, exp_full_path in enumerate(all_exps_fns):
        exp_data = get_exp_data(exp_full_path, exps_details_full_path)
        exp_leaders, exp_propagated, adjacent_nucleator_candidates_cnt = calc_adjacent_nucleator_scenraio_ctr(exp_data['XY'], exp_data['die_times'], exp_data['neighbors_list'])
        exp_data['leaders'] = exp_leaders
        exp_data['propagated'] = exp_propagated
        exp_data['adjacent_nucleator_candidates_cnt'] = adjacent_nucleator_candidates_cnt
        all_exps_data[exp_data['exp_name']] = exp_data
    return all_exps_data

def get_marker_per_treatment(treatment_name):
    marker_keys = MARKERS_BY_TREATMENT.keys()
    for key in marker_keys:
        if key in treatment_name:
            return MARKERS_BY_TREATMENT[key]
    return '$f$'

def plot_adjacent_nucleator_multiple(exps_data, dimensions_to_plot=['adjacent_nucleator_candidates_cnt',
                                                                    'temporal_res',
                                                                    'treatment']):
    first_field = list(map(lambda exp_d: exp_d[dimensions_to_plot[0]], exps_data.values()))
    sec_field = list(map(lambda exp_d: exp_d[dimensions_to_plot[1]], exps_data.values()))
    third_field = list(map(lambda exp_d: exp_d[dimensions_to_plot[2]], exps_data.values()))
    fig, ax = plt.subplots()
    by_treatment={}
    for exp_idx in range(len(first_field)):
        by_treatment[third_field[exp_idx]] = by_treatment.get(third_field[exp_idx], []) + [{
            'x-axis':first_field[exp_idx],
            'y-axis':sec_field[exp_idx]
        }]
        # ax.scatter(first_field[exp_idx], sec_field[exp_idx], marker=get_marker_per_treatment(third_field[exp_idx]), label=third_field[exp_idx])
    for treatment_name, treatment_data in by_treatment.items():
        x_axis = [exp_data['x-axis'] for exp_data in treatment_data]
        y_axis = [exp_data['y-axis'] for exp_data in treatment_data]
        ax.scatter(x_axis, y_axis, marker=get_marker_per_treatment(treatment_name), label=treatment_name)
    ax.set_xlabel(dimensions_to_plot[0])
    ax.set_ylabel(dimensions_to_plot[1])
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.05, 1))
    plt.show()
    fig.savefig(os.sep.join(['TemporalResolutionInspectionResults',
                             '{}|{}|BY{}.png'.format(dimensions_to_plot[0],
                                                     dimensions_to_plot[1],
                                                     dimensions_to_plot[2])]),
                bbox_extra_artists=[lgd], bbox_inches='tight', dpi=200)
    return by_treatment



def plot_cells_and_nucleators(XY, nucleators, die_times:np.array, plt_title=''):
    fig, ax = plt.subplots()
    max_x, max_y = XY[:, 0].max(), XY[:, 1].max()
    n_colors = die_times.max()-die_times.min()
    clrmap = cm.get_cmap('gnuplot2', n_colors)
    for cell_idx, cell_xy in enumerate(XY):
        ax.set_aspect(1)
        curr_cell_x, curr_cell_y = cell_xy[0] / max_x, cell_xy[1] / max_y
        cell_death = die_times[cell_idx]
        # circle_color = 'red' if nucleators[cell_idx] else 'green'
        circle_clr = clrmap(cell_death)
        circle_radius = 0.008
        if nucleators[cell_idx]:
            cc = plt.Circle((curr_cell_x, curr_cell_y), circle_radius, color=circle_clr)
        else:
            cc = plt.Rectangle((curr_cell_x, curr_cell_y), width=circle_radius, height=circle_radius, color=circle_clr)
        ax.add_artist(cc)

        # drawing the connectivity line
        # for connected_candidate_idx, connected_candidate in enumerate(cells_connectivity_matrix[cell_idx, :]):
        #     if connected_candidate_idx == cell_idx or connected_candidate == 0:
        #         continue
        #     connected_candidate_x, connected_candidate_y = tuple(self.XY[connected_candidate_idx])
        #     connected_candidate_x, connected_candidate_y = connected_candidate_x / max_x, connected_candidate_y / max_y
        #     connectivity_line = plt.Line2D(xdata=(curr_cell_x, connected_candidate_x),
        #                                    ydata=(curr_cell_y, connected_candidate_y),
        #                                    c='blue', linestyle='--', markersize=3, mec='red')
        #     ax.add_line(connectivity_line)
    clr_bar = fig.colorbar(cm.ScalarMappable(cmap=clrmap), ax=ax)
    # clr_bar.ax.set_yticklabels(['{}'.format(x) for x in np.unique(die_times)])
    # clr_bar.ax.set_yticklabels(['{}'.format(x) for x in np.unique(die_times)[::int(n_colors/5)]])

    ax.set_title(plt_title)
    plt.show()

def spi_nucp_for_varied_compression(single_exp_full_path, details_full_path):
    def get_exp_spi_pnuc_from_exp_dict(exp_data):
        exp_temp_res = exp_data['temporal_res']
        exp_xy = exp_data['XY']
        exp_die_times = exp_data['die_times']
        exp_treatment = exp_data['treatment']
        return NucleationProbabilityAndSPI(XY=exp_xy,
                                                die_times=exp_die_times,
                                                time_frame=exp_temp_res,
                                                treatment=exp_treatment,
                                                n_scramble=1000,
                                                draw=False,
                                                dist_threshold_nucliators_detection=200)
    org_exp_data = get_exp_data(single_exp_full_path, details_full_path)
    org_spi_pnuc = get_exp_spi_pnuc_from_exp_dict(org_exp_data)
    compression_fold = np.arange(2, 10, 1)
    compressed_results = {}
    org_temp_res = org_exp_data['temporal_res']

    compressed_results[1] = org_spi_pnuc.get_spi_nucleators()
    curr_xy = org_spi_pnuc.XY
    curr_nucleators = org_spi_pnuc.nucliators
    plot_cells_and_nucleators(curr_xy, curr_nucleators, die_times=org_exp_data['die_times'], plt_title='{}\nTempRes={}'.format(exp_full_path.split(os.sep)[-1].replace('.csv', ''), org_temp_res))

    for compression_idx, compression_lvl in enumerate(compression_fold):
        # todo: debug/check compression > 2
        compressed_exp_data = copy.deepcopy(org_exp_data)
        compressed_exp_df = compress_time_resolution_from_df(exp_df=compressed_exp_data, compression_factor=compression_lvl)
        compressed_exp_data['die_times'] = compressed_exp_df['die_times']
        compressed_exp_data['temporal_res'] = compressed_exp_data['temporal_res'] * compression_lvl
        if compressed_exp_data['temporal_res'] > compressed_exp_df['die_times'].max():
            break
        nuc_cnt_instance = get_exp_spi_pnuc_from_exp_dict(compressed_exp_data)
        curr_nucleators = nuc_cnt_instance.nucliators
        curr_xy = nuc_cnt_instance.XY
        # plot_cells_and_nucleators(curr_xy, curr_nucleators, plt_title='{}\nTempRes={}'.format(exp_full_path.split(os.sep)[-1].replace('.csv', ''), compression_lvl*org_temp_res))
        compressed_results[compression_lvl] = nuc_cnt_instance.get_spi_nucleators()

    fig, ax = plt.subplots()

    ax.scatter([org_temp_res] + [x * org_temp_res for x in range(2, len(compressed_results)+1)], [x['spi'] for x in compressed_results.values()], marker='*', label='SPI')
    ax.scatter([org_temp_res] + [x * org_temp_res for x in range(2, len(compressed_results)+1)], [x['p_nuc'] for x in compressed_results.values()], marker='p', label='P(Nucleator)')
    ax.set_title('Exp:{}\nSPI&P(Nucleators) about time compression'.format(exp_full_path.split(os.sep)[-1].replace('.csv', '')))
    ax.set_xticks([org_temp_res] + [x * org_temp_res for x in range(2, len(compressed_results)+1)])
    ax.set_xticklabels(['{}'.format(org_temp_res)] + ['{}'.format(x * org_temp_res) for x in range(2, len(compressed_results)+1)])
    ax.set_xlabel('Temporal Resolution (Min)')
    plt.legend()
    plt.show()
    plt.savefig('TemporalResolutionInspectionResults/SPI_PNUC_VaryingCompressionLvls4.png', dpi=200)

    return compressed_results


# todo: for model experiments, compress time at varying levels and compare.
# def compressed_time_comparison()
if __name__ == '__main__':
    main_regular_time_fn = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles/ConvertedTimeXYT'
    main_compressed_time_fn = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles/CompressedTimeXYT'
    main_to_inspect = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles/ExperimentsInspection'
    exps_details_full_path = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles/File details.csv'

    # exp_fn = 'CSVForTimeInspection/MANIPULATED_20180620_HAP1_erastin_xy3.csv'
    exp_fn = '20160828_10AsgCx43_FB_xy10.csv'
    # exp_fn = '20171008_MCF7_H2O2_xy2.csv'
    # exp_fn = '20180620_HAP1_erastin_xy4.csv'
    # exp_fn = '20180514_HAP1_FB_xy3.csv'
    # exp_fn = '20170213_U937_FB_xy3.csv'
    # exp_fn = '20181227_MCF10A_SKT_xy4.csv'
    exp_full_path = os.sep.join([main_regular_time_fn, exp_fn])
    # exp_data = get_exp_data(exp_full_path, exps_details_full_path)
    # exp_leaders, exp_propagated, adjacent_nucleator_candidates = calc_adjacent_nucleator_scenraio_ctr(exp_data['XY'], exp_data['die_times'], exp_data['neighbors_list'])
    # plot_cells_and_nucleators(XY=exp_data['XY'], nucleators=exp_leaders)

    # exps_data = calc_adjacent_nucleator_scenraio_ctr_multiple_exps(main_regular_time_fn, exps_details_full_path)
    # plot_adjacent_nucleator_multiple(exps_data)
    spi_nucp_for_varied_compression(single_exp_full_path=exp_full_path, details_full_path=exps_details_full_path)




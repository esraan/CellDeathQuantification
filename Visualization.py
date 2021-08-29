import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from global_parameters import *
from utils import *

# todo: add documentation to all functions


def visualize_distance_metric_of_altering_flag_values_by_treatment(p_nuc_distance_by_treatment: dict,
                                                                   p_prop_distance_by_treatment: dict,
                                                                   flag_name: str, full_path_to_save_fig: str = None,
                                                                   **kwargs):
    plt.clf()
    distance_metric_name = kwargs.get('distance_metric_name', 'RMSE')
    visualize_specific_treatments = kwargs.get('visualize_specific_treatments', 'all')

    if visualize_specific_treatments != 'all' and type(visualize_specific_treatments) == type(list()):
        full_treatment_names = []
        for full_treatment_name in p_prop_distance_by_treatment.keys():
            for given_treatment_name in visualize_specific_treatments:
                if given_treatment_name.lower() in full_treatment_name.lower():
                    full_treatment_names.append(full_treatment_name)

        p_nuc_distance_by_treatment = {key: val for key, val in p_nuc_distance_by_treatment.items() if key in full_treatment_names}
        p_prop_distance_by_treatment = {key: val for key, val in p_prop_distance_by_treatment.items() if key in full_treatment_names}

    fig, ax = plt.subplots()
    treatments_axis = np.arange(0, len(p_nuc_distance_by_treatment), 1)
    ax.bar(x=treatments_axis, height=list(p_nuc_distance_by_treatment.values()), color=(1,0,0,.8),
           label='Fraction of Nucleators')
    ax.set_xticks(treatments_axis)
    ax.set_xticklabels(list(p_nuc_distance_by_treatment.keys()))
    ax.tick_params(axis='x', labelrotation=-45, labelsize=kwargs.get('x_tick_label_size', 8))
    ax.set_ylabel(f'{distance_metric_name} distance between {flag_name} results')
    ax.set_title('P(Nuc)')

    ax.bar(x=treatments_axis, height=list(p_prop_distance_by_treatment.values()), color=(0,1,1,.8),
           label='Fraction of Propagators')
    ax.set_xticks(treatments_axis)
    ax.set_xticklabels(list(p_prop_distance_by_treatment.keys()))
    ax.tick_params(axis='x', labelrotation=-45, labelsize=kwargs.get('x_tick_label_size', 8))
    ax.set_ylabel(f'{distance_metric_name} distance between {flag_name} results')
    ax.set_title('P(Prop)')

    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
    plt.tight_layout()

    if SHOWFIG:
        plt.show()
    elif SAVEFIG:
        path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
            os.sep.join(
                os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                  'Comparison_between_flags',
                                                  'Global_P_Nuc_VS_P_Prop'])
        if not os.path.isdir(path_for_plot_dir):
            os.makedirs(path_for_plot_dir)

        path_for_plot = os.sep.join([path_for_plot_dir,
                                     f'global_p_nuc_vs_p_prop_all_exps_flag={flag_name}_{distance_metric_name}_distance_scores'])

        plt.savefig(path_for_plot+'.png', dpi=200)
        plt.savefig(path_for_plot + '.eps', dpi=200)

    plt.close(fig)


def visualize_endpoint_readouts_by_treatment_to_varying_calculation_flags(xy1_readout_tuple: Tuple[np.array, np.array],
                                                                          treatment_per_readout1: np.array,
                                                                          xy2_readout_tuple: Tuple[np.array, np.array],
                                                                          treatment_per_readout2: np.array,
                                                                          full_path_to_save_fig: str = None,
                                                                          x_label: str = 'Fraction of Nucleators',
                                                                          y_label: str = 'Fraction of Propagators',
                                                                          use_log: bool = False,
                                                                          **kwargs):
    plt.clf()

    first_flag_value_color = kwargs.get('first_flag_value_color', (1, 0, 0, .5))
    second_flag_value_color = kwargs.get('second_flag_value_color', (0, 0, 1, .5))
    first_flag_type_name_and_value = kwargs.get('first_flag_type_name_and_value', 'Recent death=True')
    second_flag_type_name_and_value = kwargs.get('second_flag_type_name_and_value', 'Recent death=False')

    flag_name = first_flag_type_name_and_value.split('=')[0]

    if use_log:
        x1_readout = np.log10(xy1_readout_tuple[0])
        y1_readout = np.log10(xy1_readout_tuple[1])
        x2_readout = np.log10(xy2_readout_tuple[0])
        y2_readout = np.log10(xy2_readout_tuple[1])
    else:
        x1_readout = xy1_readout_tuple[0]
        y1_readout = xy1_readout_tuple[1]
        x2_readout = xy2_readout_tuple[0]
        y2_readout = xy2_readout_tuple[1]

    fig, ax = plt.subplots(figsize=(10, 10))
    marker1_per_point, color_per_point, treatment1_to_marker, treatment1_to_color = \
        get_marker_per_treatment_list(treatment_per_readout1)
    marker2_per_point, color_per_point, treatment2_to_marker, treatment2_to_color = \
        get_marker_per_treatment_list(treatment_per_readout2)

    # plotting the results of the first flag value
    for point_idx, xy in enumerate(zip(x1_readout, y1_readout)):
        x, y = xy
        marker = marker1_per_point[point_idx]
        # color = color_per_point[point_idx]
        # marker_size = 10 if 'tnf' in treatment_per_readout1[point_idx].lower() else 5
        ax.plot(x, y, marker=marker, color=first_flag_value_color,
                label=treatment_per_readout1[point_idx])  # ,ms=marker_size)

    # plotting the results of the second flag value
    for point_idx, xy in enumerate(zip(x2_readout, y2_readout)):
        x, y = xy
        marker = marker2_per_point[point_idx]
        # color = color_per_point[point_idx]
        # marker_size = 10 if 'tnf' in treatment_per_readout1[point_idx].lower() else 5
        ax.plot(x, y, marker=marker, color=second_flag_value_color,
                label=treatment_per_readout2[point_idx])  # ,ms=marker_size)

    custom_handles, custom_lables = \
        get_custom_legend_artists(labels_to_colors={key: 'black' for key in treatment1_to_color.keys()},
                                  labels_to_markers=treatment1_to_marker)

    lgd = ax.legend(handles=custom_handles, labels=custom_lables, loc='best', bbox_to_anchor=(1.05, 1))
    handles, labels = ax.get_legend_handles_labels()
    ax.grid('on')
    ax.set_xlabel('Log' + x_label if use_log else x_label)
    ax.set_ylabel('Log' + y_label if use_log else y_label)

    add_to_title = f'{first_flag_value_color} is {first_flag_type_name_and_value}\n{second_flag_value_color} is {second_flag_type_name_and_value}'
    ax.set_title(f'Log(Fraction) of Nucleators & Propagators\n{add_to_title}' if use_log else f'Fraction of Nucleators & Propagators\n{add_to_title}')

    if not use_log:
        ax.set_xticks([0.1, .30])
        ax.set_yticks([0.6, 0.95])
    else:
        ax.set_xticks([-1, 0])
        ax.set_yticks([-.4, 0])
    plt.tight_layout()

    if SHOWFIG:
        plt.show()
    elif SAVEFIG:
        path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
            os.sep.join(
                os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                  'Comparison_between_flags',
                                                  'Global_P_Nuc_VS_P_Prop'])
        if not os.path.isdir(path_for_plot_dir):
            os.makedirs(path_for_plot_dir)

        if use_log:
            path_for_plot = os.sep.join([path_for_plot_dir, f'LOG_global_p_nuc_vs_p_prop_all_exps_flag={flag_name}.png'])
        else:
            path_for_plot = os.sep.join([path_for_plot_dir, f'global_p_nuc_vs_p_prop_all_exps_flag={flag_name}.png'])
        plt.savefig(path_for_plot, dpi=200, bbox_extra_artists=[lgd], bbox_inches='tight')

    plt.close(fig)


def visualize_endpoint_readouts_by_treatment_about_readouts(x_readout: np.array,
                                                            y_readout: np.array,
                                                            treatment_per_readout: np.array,
                                                            full_path_to_save_fig: str = None,
                                                            x_label: str = 'Fraction of Nucleators',
                                                            y_label: str = 'Fraction of Propagators',
                                                            use_log: bool = False,
                                                            plot_about_treatment: bool = False,
                                                            **kwargs):
    if use_log:
        x_readout = np.log10(x_readout)
        y_readout = np.log10(y_readout)
    if plot_about_treatment:
        fig, axis = plt.subplots(1, 2)
        unique_treatments = set(treatment_per_readout)
        x_axis_dict = {treatment_name:idx for idx, treatment_name in enumerate(unique_treatments)}
        # x_axis = np.arange(0, len(x_readout), 1)
    else:
        fig, ax = plt.subplots()  # figsize=(10, 10))

    marker_per_point, color_per_point, treatment_to_marker, treatment_to_color = \
        get_marker_per_treatment_list(treatment_per_readout)

    for point_idx, xy in enumerate(zip(x_readout, y_readout)):
        x, y = xy
        marker = marker_per_point[point_idx]
        color = color_per_point[point_idx]
        treatment_name = treatment_per_readout[point_idx]
        marker_size = 10 if 'tnf' in treatment_name.lower() else 5

        if plot_about_treatment:
            treatment_idx = x_axis_dict[treatment_name]
            axis[0].plot(treatment_idx, x, ms=marker_size, marker=marker,
                         color=color, label=treatment_per_readout[point_idx])
            axis[1].plot(treatment_idx, y, ms=marker_size, marker=marker,
                         color=color, label=treatment_per_readout[point_idx])
        else:
            ax.plot(x, y, ms=marker_size, marker=marker,
                    color=color, label=treatment_per_readout[point_idx])

    custom_handles, custom_lables = get_custom_legend_artists(labels_to_colors=treatment_to_color,
                                                              labels_to_markers=treatment_to_marker)

    if plot_about_treatment:
        # temporal_unit = 'Minutes' if temporal_resolution != 1 else 'Frame#'
        axis[0].grid('on')
        axis[0].set_xlabel(f'Treatment type')
        axis[0].set_title('Log ' + x_label if use_log else x_label)
        axis[1].grid('on')
        axis[1].set_xlabel(f'Treatment type')
        axis[1].set_title('Log ' + y_label if use_log else y_label)

        if not use_log:
            axis[0].set_xticks(np.arange(0, len(x_axis_dict), 1))
            axis[0].set_xticklabels(list(x_axis_dict.keys()))
            axis[0].tick_params(axis='x', labelrotation=-45, labelsize=kwargs.get('x_tick_label_size', 4))
            axis[1].set_xticks(np.arange(0, len(x_axis_dict), 1))
            axis[1].set_xticklabels(list(x_axis_dict.keys()))
            axis[1].tick_params(axis='x', labelrotation=-45, labelsize=kwargs.get('x_tick_label_size', 4))

        else:
            axis[0].set_xticks(np.arange(0, len(x_axis_dict), 1))
            axis[0].set_xticklabels(list(x_axis_dict.keys()))
            axis[0].tick_params(axis='x', labelrotation=-45, labelsize=kwargs.get('x_tick_label_size', 4))
            axis[1].set_xticks(np.arange(0, len(x_axis_dict), 1))
            axis[1].set_xticklabels(list(x_axis_dict.keys()))
            axis[1].tick_params(axis='x', labelrotation=-45, labelsize=kwargs.get('x_tick_label_size', 4))

        plt.setp(axis[0].xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
        plt.setp(axis[1].xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")

        if not use_log:
            if RECENT_DEATH_ONLY_FLAG:
                y_tick_limits_x_label = [0.15, 0.6]
                y_tick_limits_y_label = [0.35, 0.9]
            else:
                y_tick_limits_x_label = [0.1, 0.3]
                y_tick_limits_y_label = [0.7, 0.95]
        else:
            if RECENT_DEATH_ONLY_FLAG:
                y_tick_limits_x_label = [-0.8, -0.2]
                y_tick_limits_y_label = [-0.4, 0]
            else:
                y_tick_limits_x_label = [-1, -0.5]
                y_tick_limits_y_label = [-0.2, 0]

        axis[0].set_ylim(y_tick_limits_x_label)
        axis[1].set_ylim(y_tick_limits_y_label)

    else:
        ax.grid('on')
        ax.set_xlabel('Log ' + x_label if use_log else x_label)
        ax.set_ylabel('Log ' + y_label if use_log else y_label)
        ax.set_title('Log(Fraction) of Nucleators & Propagators' if use_log else 'Fraction of Nucleators & Propagators')
        if not use_log:
            x_tick_limits = [0.1, .30]
            y_tick_limits = [0.6, 0.95]
        else:
            x_tick_limits = [-1, 0]
            y_tick_limits = [-.4, 0]

        ax.set_xlim(x_tick_limits)
        ax.set_ylim(y_tick_limits)

    plt.tight_layout()

    if SHOWFIG:
        plt.show()
    elif SAVEFIG:
        if RECENT_DEATH_ONLY_FLAG:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
                os.sep.join(
                    os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                      'Only recent death considered for neighbors results',
                                                      'Global_P_Nuc_VS_P_Prop'])
        else:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else os.sep.join(
                os.getcwd().split(os.sep)[:-1] + ['Results', 'MeasurementsEndpointReadoutsPlots',
                                                  'Global_P_Nuc_VS_P_Prop'])
        if not os.path.isdir(path_for_plot_dir):
            os.makedirs(path_for_plot_dir)

        plot_about_time_path_addition = '_about_treatment' if plot_about_treatment else ''
        if use_log:
            path_for_plot = os.sep.join([path_for_plot_dir, f'LOG_global_p_nuc_vs_p_prop_all_exps{plot_about_time_path_addition}'])
        else:
            path_for_plot = os.sep.join([path_for_plot_dir, f'global_p_nuc_vs_p_prop_all_exps{plot_about_time_path_addition}'])
        plt.savefig(path_for_plot + '.png', dpi=200)  # , bbox_extra_artists=[lgd], bbox_inches='tight')
        plt.savefig(path_for_plot + '.eps', dpi=200)  # , bbox_extra_artists=[lgd], bbox_inches='tight')

    plt.close(fig)
def visualize_cell_death_in_time(xyt_df: pd.DataFrame = None,
                                 xyt_full_path: str = None,
                                 nucleators_mask: np.array = None,
                                 propagators_maks: np.array = None,
                                 full_path_to_save_fig: str = None,
                                 exp_treatment: str = None, exp_name: str = None) -> None:
    """

    :param xyt_df:
    :param xyt_full_path:
    :param nucleators_mask:
    :param full_path_to_save_fig:
    :param exp_treatment:
    :param exp_name:
    :return:
    """
    plt.clf()

    if xyt_df is None and xyt_full_path is None:
        raise ValueError('either df or path must be not None!')

    if exp_treatment is None or exp_name is None:
        raise ValueError('must provide exp treatment and name!')

    # clean exp_name and treatment from bad characters
    exp_name, exp_treatment = exp_name.replace('/', ''), exp_treatment.replace('/', '')

    data = xyt_df if xyt_df is not None else pd.read_csv(xyt_full_path)
    death_times = data['death_time'].values
    x, y = data['cell_x'].values, data['cell_y'].values

    min_time_of_death, max_time_of_death = death_times.min(), death_times.max()
    fig, ax = plt.subplots()

    cmap = mpl.cm.__builtin_cmaps[12]
    ax.scatter(x, y, c=death_times, cmap=cmap)

    # mark nucleators
    if nucleators_mask is not None:
        nucleators_indices = np.where(nucleators_mask)[0]
        ax.scatter(x[(nucleators_indices)], y[(nucleators_indices)], marker='x', color=(0, 1, 0, .5))
    # mark propagators
    if propagators_maks is not None:
        propagators_indices = np.where(propagators_maks)[0]
        ax.scatter(x[(propagators_indices)], y[(propagators_indices)], marker='+', color=(1, 0, 1, .5))

    ax.set_xticks([])
    ax.set_yticks([])
    # colorbar
    norm = mpl.colors.Normalize(vmin=min_time_of_death, vmax=max_time_of_death)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='time of death')

    plt.tight_layout()

    if SHOWFIG:
        plt.show()
    elif SAVEFIG:
        if RECENT_DEATH_ONLY_FLAG:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
                os.sep.join(
                    os.getcwd().split(os.sep)[:-1] + ['Results', 'CellDeathVisualizations',
                                                      'Only recent death considered for neighbors results',
                                                      f'{exp_treatment}'])
        else:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else os.sep.join(
                os.getcwd().split(os.sep)[:-1] + ['Results', 'CellDeathVisualizations', f'{exp_treatment}'])
        if not os.path.isdir(path_for_plot_dir):
            os.makedirs(path_for_plot_dir)

        path_for_plot = os.sep.join([path_for_plot_dir, f'{exp_name}.png'])
        plt.savefig(path_for_plot, dpi=200)

    plt.close(fig)


def plot_measurements_by_time(p_nuc_by_time: np.array,
                              p_prop_by_time: np.array,
                              accumulated_fraction_of_death_by_time: np.array,
                              temporal_resolution: int,
                              exp_name: str, exp_treatment: str,
                              full_path_to_save_fig: str = None) -> None:
    """

    :param p_nuc_by_time:
    :param p_prop_by_time:
    :param accumulated_fraction_of_death_by_time:
    :param temporal_resolution:
    :param exp_name:
    :param exp_treatment:
    :param full_path_to_save_fig:
    :return:
    """
    plt.clf()

    # clean exp_name and treatment from bad characters
    exp_name, exp_treatment = exp_name.replace('/', ''), exp_treatment.replace('/', '')

    max_time = len(p_nuc_by_time) * temporal_resolution
    time_axis = np.arange(0, max_time, temporal_resolution)

    fig, ax = plt.subplots()

    ax.set_title(f'Treatment: {exp_treatment}\nexp:{exp_name}')

    ax1_clr = 'tab:blue'
    ax.plot(time_axis, p_nuc_by_time, label='Nucleation')
    ax.set_ylabel('P(Nuc)', color=ax1_clr)

    ax.tick_params(axis='y', labelcolor=ax1_clr)
    ax.set_ylim(0, 1)
    ax.set_xticks([time_axis[0], time_axis[len(time_axis) // 2], time_axis[-1]])
    # ax.set_xticklabels(['{}-{}'.format(x, x + 100) for x in time_ticks])
    ax2_color = 'tab:red'
    ax2 = ax.twinx()
    ax2.plot(time_axis, p_prop_by_time, label='Propagation', color=ax2_color)
    ax2.set_ylabel('P(Prop)', color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    ax2.set_ylim(0, 1)
    # plotting accumulated death
    ax.set_xlabel('Time Window (Minutes)')
    ax.plot(time_axis, accumulated_fraction_of_death_by_time, label='Accumulated Death', color='black', marker='p')

    # plot lines to visualize 10%-90% cells deaths
    tenth_percentile = np.where(accumulated_fraction_of_death_by_time >= .1)[0][0]
    plt.axvline(x=time_axis[tenth_percentile], c='black', linestyle='--', linewidth=0.7)

    nineteenth_percentile = np.where(accumulated_fraction_of_death_by_time >= .9)[0][0]
    plt.axvline(x=time_axis[nineteenth_percentile], c='black', linestyle='--', linewidth=0.7)

    plt.tight_layout()

    if SHOWFIG:
        plt.show()
    elif SAVEFIG:
        if RECENT_DEATH_ONLY_FLAG:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
                os.sep.join(
                    os.getcwd().split(os.sep)[:-1] + ['Results', 'TemporalMeasurementsPlots',
                                                      'Only recent death considered for neighbors results',
                                                      f'{exp_treatment}'])
        else:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
                os.sep.join(
                    os.getcwd().split(os.sep)[:-1] + ['Results', 'TemporalMeasurementsPlots', f'{exp_treatment}'])
        if not os.path.isdir(path_for_plot_dir):
            os.makedirs(path_for_plot_dir)

        path_for_plot = os.sep.join([path_for_plot_dir, f'{exp_name}'])
        plt.savefig(path_for_plot+'.png', dpi=200)
        plt.savefig(path_for_plot + '.eps', dpi=200)

    plt.close(fig)


def scatter_with_linear_regression_line(x: np.array, y: np.array, x_label: str, y_label: str, title: str,
                                        path_to_save_fig: str, plot_linear_regression: bool = False,
                                        color_map: dict = None, colors: np.array = None):
    """

    :param colors:
    :param color_map:
    :param plot_linear_regression:
    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :param title:
    :param path_to_save_fig:
    :return:
    """
    plt.clf()

    fig, ax = plt.subplots()
    # plot probabilities for each level of neighborhoods
    if color_map is None or colors is None:
        ax.scatter(x, y, s=50, color=(0, 1, 0, 0.8), marker='*')
    else:
        ax.scatter(x, y, s=50, marker='*', c=colors, cmap=color_map)

    if plot_linear_regression:
        # plot linear regression line
        regression_line_x, regression_line_y = get_linear_regression_line_between_two_signals(x, y)
        ax.plot(regression_line_x, regression_line_y)
    else:
        regression_line = np.linspace(0, 1, num=len(x), endpoint=True)
        ax.plot(regression_line, regression_line)
    # plot decoration
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))

    if colors is not None and color_map is not None:
        norm = mpl.colors.Normalize(vmin=colors.min(), vmax=colors.max())
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, label='time of death')

    plt.tight_layout()

    if SAVEFIG:
        plt.savefig(f'{path_to_save_fig}.png', dpi=200)
        plt.savefig(f'{path_to_save_fig}.eps', dpi=200)
    elif SHOWFIG:
        plt.show()
    plt.close(fig=fig)


if __name__ == '__main__':
    # visualize_cell_death_in_time(xyt_path='Data/Experiments_XYT_CSV/20180620_HAP1_erastin_xy6.csv',
    #                              to_save_path=os.sep.join(['Results/CellDeathVisualizations', '20180620_HAP1_erastin_xy6_cell_death.png']))
    #
    # visualize_cell_death_in_time(xyt_path='Data/Experiments_XYT_CSV/20181227_MCF10A_SKT_xy1.csv',
    #                              to_save_path=os.sep.join(
    #                                  ['Results/CellDeathVisualizations', '20181227_MCF10A_SKT_xy1_cell_death.png']))
    for path in get_all_paths_csv_files_in_dir('Data/Simulations_XYT_CSV/erstin/0.0005'):
        visualize_cell_death_in_time(xyt_full_path=path,
                                     full_path_to_save_fig=os.sep.join(
                                         ['Results/CellDeathVisualizations',
                                          path.split(os.sep)[-1].replace('.csv', '_erastin_cell_death.png')]))
    for path in get_all_paths_csv_files_in_dir('Data/Simulations_XYT_CSV/SKT/0.005'):
        visualize_cell_death_in_time(xyt_full_path=path,
                                     full_path_to_save_fig=os.sep.join(
                                         ['Results/CellDeathVisualizations',
                                          path.split(os.sep)[-1].replace('.csv', '_skt_cell_death.png')]))


def plot_endpoint_readout_for_compressed_temporal_resolution(temporal_resolution_axis: np.array,
                                                             endpoint_readouts_values_p_nuc: np.array,
                                                             endpoint_readouts_values_p_prop: np.array,
                                                             exp_treatment: str,
                                                             exp_name: str,
                                                             full_path_to_save_fig: str = None,
                                                             **kwargs):
    fig, axis = plt.subplots(1, 2)
    fig.suptitle(f'Experiment:{exp_name}\nTreatment{exp_treatment}')
    x_axis = np.arange(0, len(temporal_resolution_axis), 1)
    axis[0].scatter(x_axis, endpoint_readouts_values_p_nuc)
    axis[0].set_xticks(x_axis)
    axis[0].set_xticklabels([f'{x}' for x in temporal_resolution_axis])
    axis[0].set_xlabel('Temporal resolution (Minutes)')
    axis[0].set_ylabel('Fraction of Nucleators')
    axis[0].set_ylim((0, 1))

    axis[1].scatter(x_axis, endpoint_readouts_values_p_prop)
    axis[1].set_xticks(x_axis)
    axis[1].set_xticklabels([f'{x}' for x in temporal_resolution_axis])
    axis[1].set_xlabel('Temporal resolution (Minutes)')
    axis[1].set_ylabel('Fraction of Propagators')
    axis[1].set_ylim((0, 1))

    plt.tight_layout()

    if SHOWFIG:
        plt.show()
    elif SAVEFIG:
        exp_treatment = exp_treatment.replace('\\', '_').replace('/', '_')
        recent_death_only = kwargs.get('only_recent_death_flag_for_neighbors_calc', RECENT_DEATH_ONLY_FLAG)
        if recent_death_only:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
                os.sep.join(
                    os.getcwd().split(os.sep)[:-1] + ['Results',
                                                      'Compressed_MeasurementsEndpointReadoutsPlots',
                                                      'Only recent death considered for neighbors results',
                                                      'SingleExperimentsCompressionComparisons',
                                                      f'{exp_treatment}'])
        else:
            path_for_plot_dir = full_path_to_save_fig if full_path_to_save_fig is not None else \
                os.sep.join(
                    os.getcwd().split(os.sep)[:-1] + ['Results',
                                                      'Compressed_MeasurementsEndpointReadoutsPlots',
                                                      'SingleExperimentsCompressionComparisons',
                                                      f'{exp_treatment}'])
        if not os.path.isdir(path_for_plot_dir):
            os.makedirs(path_for_plot_dir)

        path_for_plot = os.sep.join([path_for_plot_dir, f'{exp_name}'])
        plt.savefig(f'{path_for_plot}.png', dpi=200)
        plt.savefig(f'{path_for_plot}.eps', dpi=200)

    plt.close(fig)



import os
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


class DeathProbabilityNearDeadCellsByTime:
    def __init__(self, XY, TIMES, neighbors_list):
        self.XY = XY
        self.TIMES = TIMES
        self.neighbors_list = neighbors_list
    @staticmethod
    def get_neighbors(XY):
        n_instances = len(XY)
        vor = Voronoi(XY)
        neighbors = vor.ridge_points
        neighbors_list = []
        [neighbors_list.append([]) for i in range(n_instances)]
        for x in neighbors:
            neighbors_list[x[0]].append(x[1])
            neighbors_list[x[1]].append(x[0])
        return neighbors_list

    def find_proba(self, time_differences_to_calc, num_of_dead_neighbors_to_calc):
        def calc_proba_for_x_dead_neighbors(neighbors_list, TIMES, x_dead_neighbors, time_to_die_by):
            unique_times = np.unique(self.TIMES)
            alive_with_x_dead_at_time_x1_ctr = set()
            dead_with_x_dead_at_time_x1_ctr = set()
            examined_cells = set()
            for time_x in unique_times:
                alive_cells_at_time_x1 = np.where(TIMES > time_x+time_to_die_by)[0]
                dead_cells_at_time_x = np.where(TIMES <= time_x)[0]
                for cell_idx, cell_neighbors in enumerate(neighbors_list):
                    # check whether cell already dead
                    if cell_idx in dead_cells_at_time_x or cell_idx in examined_cells:
                        continue
                    xy = np.intersect1d(dead_cells_at_time_x,
                                        np.array(cell_neighbors))
                    if len(xy) == x_dead_neighbors and cell_idx in alive_cells_at_time_x1:
                        alive_with_x_dead_at_time_x1_ctr.add(cell_idx)
                        examined_cells.add(cell_idx)
                    if len(xy) == x_dead_neighbors and cell_idx not in alive_cells_at_time_x1:
                        dead_with_x_dead_at_time_x1_ctr.add(cell_idx)
                        examined_cells.add(cell_idx)
            return len(dead_with_x_dead_at_time_x1_ctr)/(len(alive_with_x_dead_at_time_x1_ctr) + len(dead_with_x_dead_at_time_x1_ctr))

        proba_map = np.zeros((len(time_differences_to_calc), len(num_of_dead_neighbors_to_calc)))
        for t_idx, time_difference in enumerate(time_differences_to_calc):
            for n_idx, num_of_dead_neighbors in enumerate(num_of_dead_neighbors_to_calc):
                proba_map[t_idx, n_idx] = \
                    calc_proba_for_x_dead_neighbors(self.neighbors_list,
                                                    self.TIMES,
                                                    x_dead_neighbors=num_of_dead_neighbors,
                                                    time_to_die_by=time_difference)

        return proba_map
    def plot_proba(self,
                   fig_title='Probability to die',
                   cbarlabel='probability to die',
                   showfig=True,
                   savefig=True,
                   fig_fname='ProbToDie.png',
                   time_differences_to_calc=np.arange(1,6,1),
                   num_of_dead_neighbors_to_calc=np.arange(1,6,1),
                   **cbar_kw):
        probability_map = self.find_proba(time_differences_to_calc, num_of_dead_neighbors_to_calc)
        fig, ax = plt.subplots()

        im = ax.imshow(probability_map, cmap="YlGn", vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        ax.set_xticks(np.arange(len(num_of_dead_neighbors_to_calc)))
        ax.set_yticks(np.arange(len(time_differences_to_calc)))
        ax.set_xticklabels(num_of_dead_neighbors_to_calc)
        ax.set_xlabel('number of dead neighbors at death time-1')
        ax.set_yticklabels(time_differences_to_calc)
        ax.set_ylabel('death time difference')
        ax.set_title(fig_title)
        for i in range(len(time_differences_to_calc)):
            for j in range(len(num_of_dead_neighbors_to_calc)):
                text = ax.text(j, i, np.float16(probability_map[i, j]),
                               ha="center", va="center", color="black")
        if showfig:
            plt.show()
        if savefig:
            fig.savefig(fig_fname, dpi=200)

        plt.close()


    @classmethod
    def multiple_plot_prob(cls, main_fname='', file_details_fname='', **plot_proba_kwargs):
        filtered_list_of_files_to_analyze = filter(lambda x: not(x=='.DS_Store' or x.find('File details') != -1), os.listdir(main_fname))
        org_fig_fname = plot_proba_kwargs.get('fig_fname','{}fig_fname.png')
        org_fig_title = plot_proba_kwargs.get('fig_title','{}fig_title')
        all_experiments_details = pd.read_csv(file_details_fname)
        all_experiments_details_names_treatments = all_experiments_details.loc[:, ['File Name', 'Treatment']]
        for f_idx, file in enumerate(filtered_list_of_files_to_analyze):
            if f_idx > 20000:
                break

            if file.find('.csv') != -1:
                print("file name: %s" % file)
                file_treatment = all_experiments_details.loc[all_experiments_details['File Name'] == file]['Treatment'].values[0]
                data_df = pd.read_csv(os.path.sep.join(['ExperimentsXYT_CSVFiles', file]))
                XY = data_df.loc[:, ['cell_x', 'cell_y']].values
                TIMES = data_df.loc[:, ['death_time']].values
                vrn_model = Voronoi(XY)
                neighbors_list = DeathProbabilityNearDeadCellsByTime.get_neighbors(XY)

                dpbt = cls(XY=XY, TIMES=TIMES, neighbors_list=neighbors_list)
                plot_proba_kwargs['fig_fname'] = org_fig_fname.format(os.path.sep.join(['ProbToDiePlots', file.replace('.csv', '')]))
                plot_proba_kwargs['fig_title'] = org_fig_title.format(file.replace('.csv', '')) + '\nTreatment: {}'.format(file_treatment)
                dpbt.plot_proba(**plot_proba_kwargs)
    #todo: convert this to return the map of the average probabilities per treatment
    @classmethod
    def multiple_prob_map_avg_per_treatment(cls, main_fname='', file_details_fname='', **plot_proba_kwargs):
        filtered_list_of_files_to_analyze = filter(lambda x: not(x=='.DS_Store' or x.find('File details') != -1), os.listdir(main_fname))
        org_fig_fname = plot_proba_kwargs.get('fig_fname','{}fig_fname.png')
        org_fig_title = plot_proba_kwargs.get('fig_title','{}fig_title')
        all_experiments_details = pd.read_csv(file_details_fname)
        all_experiments_details_names_treatments = all_experiments_details.loc[:, ['File Name', 'Treatment']]
        file_to_avg_proba = {}
        file_to_proba = {}
        for f_idx, file in enumerate(filtered_list_of_files_to_analyze):
            if f_idx > 200000:
                break

            if file.find('.csv') != -1:
                print("file name: %s" % file)
                file_treatment = all_experiments_details.loc[all_experiments_details['File Name'] == file]['Treatment'].values[0]
                data_df = pd.read_csv(os.path.sep.join(['ExperimentsXYT_CSVFiles', file]))
                XY = data_df.loc[:, ['cell_x', 'cell_y']].values
                TIMES = data_df.loc[:, ['death_time']].values
                vrn_model = Voronoi(XY)
                neighbors_list = DeathProbabilityNearDeadCellsByTime.get_neighbors(XY)

                dpbt = cls(XY=XY, TIMES=TIMES, neighbors_list=neighbors_list)
                plot_proba_kwargs['fig_fname'] = org_fig_fname.format(os.path.sep.join(['ProbToDiePlots', file.replace('.csv', '')]))
                plot_proba_kwargs['fig_title'] = org_fig_title.format(file.replace('.csv', '')) + '\nTreatment: {}'.format(file_treatment)
                time_differences_to_calc = plot_proba_kwargs.get('time_differences_to_calc', np.arange(1,6,1))
                num_of_dead_neighbors_to_calc = plot_proba_kwargs.get('num_of_dead_neighbors_to_calc', np.arange(1,6,1))
                file_to_proba[file] = dpbt.find_proba(time_differences_to_calc, num_of_dead_neighbors_to_calc)
                if plot_proba_kwargs.get('mean_by_axis', False):
                    file_to_avg_proba[file] = file_to_proba[file][:, 2].mean()
                else:
                    file_to_avg_proba[file] = file_to_proba[file].mean()
                # proba_map_for_file = dpbt.find_proba(time_differences_to_calc=, )
        fig, ax = plt.subplots()
        avg_probas = np.array(file_to_avg_proba.values())
        unique_types = np.unique(all_experiments_details_names_treatments.loc[:, ['Treatment']].values)
        colors = plt.get_cmap('plasma', len(unique_types))
        markers = {
            'FAC&BSO':'>',
            "C'":'P',
            'ML162':'x',
            'else': '*'
        }
        marker_options = lambda name: markers['FAC&BSO'] if "FAC&BSO" in name else markers["C'"] if "C'" in name else markers['ML162'] if 'ML162' in name else markers['else']
        idx = 0
        for experiment_name, experiment_avg_proba in file_to_avg_proba.items():
            experiment_treatment = all_experiments_details_names_treatments.loc[all_experiments_details_names_treatments['File Name']==experiment_name]['Treatment'].values[0]
            treatment_idx = np.where(unique_types==experiment_treatment)[0][0]
            ax.scatter(treatment_idx, experiment_avg_proba, color=colors(treatment_idx), marker="{}".format(marker_options(experiment_treatment)))#, label=experiment_treatment[:-4])
            idx += 1
        # handles, labels = ax.get_legend_handles_labels()
        ax.set_title('Mean probability of adjacent cell deaths')
        ax.set_xlabel('Treatment name')
        ax.set_ylabel('Avg probability to die next to dead cells')
        ax.set_xticks(range(len(unique_types)))
        ax.set_xticklabels(unique_types)
        # plt.xticks(rotation=30)
        fig.autofmt_xdate(rotation=45)
        if plot_proba_kwargs.get('savefig', False):
            fig.savefig(plot_proba_kwargs.get('fig_fname','AvgDeathProbaNearDeadCellsByTime'), bbox_inches='tight', dpi=200)
        if plot_proba_kwargs.get('showfig', False):
            plt.show()

    @staticmethod
    def plot_probs_to_die_by_treatment(death_probs_by_treatment_dict: dict, x_length:int = 5, **plot_kwargs):
        num_of_unique_treatments = len(death_probs_by_treatment_dict.keys())
        fig, axis = plt.subplots(num_of_unique_treatments//2 + 1, 2, figsize=(15,50))
        treatment_idx = 0
        dead_neighbors_num = np.arange(1, x_length+1, 1)
        for treatment_name, treatment_probas in death_probs_by_treatment_dict.items():
            for single_exp_data in treatment_probas:
                axis[treatment_idx//2, treatment_idx%2].plot(dead_neighbors_num, single_exp_data)
            axis[treatment_idx//2, treatment_idx % 2].set_title('Treatment %s' % (treatment_name))
            axis[treatment_idx//2, treatment_idx % 2].set_xlabel('# dead neighbors')
            axis[treatment_idx//2, treatment_idx % 2].set_ylabel('Probability to die')
            axis[treatment_idx//2, treatment_idx % 2].set_xticks(np.arange(1, x_length+1, 1))
            axis[treatment_idx//2, treatment_idx % 2].set_xticklabels(["{} DN".format(x) for x in np.arange(1, x_length+1, 1)])
            axis[treatment_idx//2, treatment_idx % 2].set_ylim(0, 1)
            treatment_idx += 1
        plt.tight_layout()
        if plot_kwargs.get('showfig', False):
            plt.show()
        if plot_kwargs.get('savefig', False):
            fig.savefig(plot_kwargs.get('figname', 'deathNextToDeadNeighborsProbaByTreatment2.png'))

    @staticmethod
    def boxplot_probs_to_die_by_treatment(death_probs_by_treatment_dict: dict, x_length:int = 5, **plot_kwargs):
        num_of_unique_treatments = len(death_probs_by_treatment_dict.keys())
        fig, axis = plt.subplots(num_of_unique_treatments//2 + 1, 2, figsize=(15,50))
        treatment_idx = 0
        dead_neighbors_num = np.arange(1, x_length+1, 1)
        _by_treatment = dict()
        # axis.boxplot(x=list(death_probs_by_treatment_dict.values()), labels=list(death_probs_by_treatment_dict.keys()))

        for treatment_name, treatment_probas in death_probs_by_treatment_dict.items():
            _by_treatment[treatment_name] = np.ndarray((len(treatment_probas), len(dead_neighbors_num)))
            for single_exp_idx, single_exp_data in enumerate(treatment_probas):
                for dead_neighbors_cnt, proba in enumerate(single_exp_data):
                    _by_treatment[treatment_name][single_exp_idx, dead_neighbors_cnt] = proba
            axis[treatment_idx//2, treatment_idx % 2].set_title('Treatment %s' % (treatment_name))
            axis[treatment_idx//2, treatment_idx % 2].set_xlabel('# dead neighbors')
            axis[treatment_idx//2, treatment_idx % 2].set_ylabel('Probability to die')
            axis[treatment_idx//2, treatment_idx % 2].set_xticks(np.arange(1, x_length+1, 1))
            # for dead_neighbors_cnt in dead_neighbors_num:
            # axis[treatment_idx//2, treatment_idx % 2].boxplot(_by_treatment[treatment_name][:, dead_neighbors_cnt-1])
            axis[treatment_idx//2, treatment_idx % 2].boxplot(_by_treatment[treatment_name])
            treatment_idx += 1
        plt.tight_layout()
        if plot_kwargs.get('showfig', False):
            plt.show()
        if plot_kwargs.get('savefig', False):
            fig.savefig(plot_kwargs.get('figname', 'deathNextToDeadNeighborsProbaByTreatmentBoxPlot.png'))

    @staticmethod
    def variability_in_probs_to_die_by_treatment(death_probs_by_treatment_dict: dict, x_length:int = 5, **plot_kwargs):
        num_of_unique_treatments = len(death_probs_by_treatment_dict.keys())
        fig, axis = plt.subplots(num_of_unique_treatments//2 + 1, 2, figsize=(15,50))
        treatment_idx = 0
        dead_neighbors_num = np.arange(1, x_length+1, 1)
        _by_treatment = dict()
        # axis.boxplot(x=list(death_probs_by_treatment_dict.values()), labels=list(death_probs_by_treatment_dict.keys()))

        for treatment_name, treatment_probas in death_probs_by_treatment_dict.items():
            _by_treatment[treatment_name] = np.ndarray((len(treatment_probas), len(dead_neighbors_num)))
            for single_exp_idx, single_exp_data in enumerate(treatment_probas):
                for dead_neighbors_cnt, proba in enumerate(single_exp_data):
                    _by_treatment[treatment_name][single_exp_idx, dead_neighbors_cnt] = proba
            axis[treatment_idx//2, treatment_idx % 2].set_title('Treatment %s' % (treatment_name))
            axis[treatment_idx//2, treatment_idx % 2].set_xlabel('# dead neighbors')
            axis[treatment_idx//2, treatment_idx % 2].set_ylabel('Variability in Death Probability')
            axis[treatment_idx//2, treatment_idx % 2].set_xticks(np.arange(1, x_length+1, 1))
            # for dead_neighbors_cnt in dead_neighbors_num:
            # axis[treatment_idx//2, treatment_idx % 2].boxplot(_by_treatment[treatment_name][:, dead_neighbors_cnt-1])
            axis[treatment_idx//2, treatment_idx % 2].scatter(list(range(1, x_length+1, 1)), [np.var(_by_treatment[treatment_name][:, dead_num]) for dead_num in range(x_length)])
            axis[treatment_idx//2, treatment_idx % 2].set_ylim(0, 0.02)
            treatment_idx += 1
        plt.tight_layout()
        if plot_kwargs.get('showfig', False):
            plt.show()
        if plot_kwargs.get('savefig', False):
            fig.savefig(plot_kwargs.get('figname', 'deathNextToDeadNeighborsProbaByTreatmentVariance.png'))


    @classmethod
    def all_exp_probs_to_die(cls, main_fname='',
                             file_details_fname='',
                             time_differences_to_calc=np.arange(1, 6, 1),
                             num_of_dead_neighbors_to_calc=np.arange(1, 6, 1),
                             time_diff_in_death=1,
                             num_of_dead_neighbors_in_death=None,
                             **plot_kwargs):
        filtered_list_of_files_to_analyze = filter(lambda x: not(x=='.DS_Store' or x.find('File details') != -1), os.listdir(main_fname))
        # org_fig_fname = plot_kwargs.get('fig_fname','{}fig_fname.png')
        # org_fig_title = plot_kwargs.get('fig_title','{}fig_title')
        all_experiments_details = pd.read_csv(file_details_fname)
        all_experiments_details_names_treatments = all_experiments_details.loc[:, ['File Name', 'Treatment']]
        treatment_to_probabilities = {}
        for f_idx, file in enumerate(filtered_list_of_files_to_analyze):
            if f_idx > 20000:
                break
            if file.find('.csv') != -1:
                print("file idx: %d file name: %s" % (f_idx, file))
                file_treatment = all_experiments_details.loc[all_experiments_details['File Name'] == file]['Treatment'].values[0]
                data_df = pd.read_csv(os.path.sep.join(['ExperimentsXYT_CSVFiles', file]))
                XY = data_df.loc[:, ['cell_x', 'cell_y']].values
                TIMES = data_df.loc[:, ['death_time']].values
                vrn_model = Voronoi(XY)
                neighbors_list = DeathProbabilityNearDeadCellsByTime.get_neighbors(XY)

                dpbt = cls(XY=XY, TIMES=TIMES, neighbors_list=neighbors_list)
                if treatment_to_probabilities.get(file_treatment, None) is None:
                    treatment_to_probabilities[file_treatment] = []
                if num_of_dead_neighbors_in_death is None:
                    to_append = dpbt.find_proba(time_differences_to_calc, num_of_dead_neighbors_to_calc)[time_diff_in_death-1, :]
                else:
                    to_append = dpbt.find_proba(time_differences_to_calc, num_of_dead_neighbors_to_calc)[:, num_of_dead_neighbors_in_death]
                treatment_to_probabilities[file_treatment].append(to_append)
        # DeathProbabilityNearDeadCellsByTime.plot_probs_to_die_by_treatment(treatment_to_probabilities,
        #                                                                    x_length=len(to_append),
        #                                                                    **plot_kwargs)
        # DeathProbabilityNearDeadCellsByTime.boxplot_probs_to_die_by_treatment(treatment_to_probabilities,
        #                                                                       x_length=len(to_append),
        #                                                                       **plot_kwargs)
        DeathProbabilityNearDeadCellsByTime.variability_in_probs_to_die_by_treatment(treatment_to_probabilities,
                                                                                     x_length=len(to_append),
                                                                                     **plot_kwargs)


if __name__ == '__main__':
    main_fname = 'ExperimentsXYT_CSVFiles'
    # DeathProbabilityNearDeadCellsByTime.multiple_prob_map_avg_per_treatment(main_fname,
    #                                                                         file_details_fname=os.path.sep.join([main_fname, 'File details.csv']),
    #                                                                         showfig=True,
    #                                                                         savefig=True,
    #                                                                         fig_fname='{}_ProbToDieWith3DeadNeighbors.png',
    #                                                                         mean_by_axis=True)
    DeathProbabilityNearDeadCellsByTime.all_exp_probs_to_die(main_fname,
                                                             file_details_fname=os.path.sep.join([main_fname, 'File details.csv']),
                                                             showfig=False,
                                                             savefig=True,
                                                             time_diff_in_death=1,
                                                             time_differences_to_calc=np.arange(1,3,1),
                                                             num_of_dead_neighbors_to_calc=np.arange(1,6,1))
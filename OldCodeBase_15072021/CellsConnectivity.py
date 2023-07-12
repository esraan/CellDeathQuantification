import os
import numpy as np
import pandas as pd
from NucliatorsCount import NucleatorsCounter
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx


ALWAYS = True
MAX_EXP_TO_PROCESS = 3000
FONT_SIZE = 10
SHOWFIG = False
SAVEFIG = True
NUCLEATOR_SIZE = 50
NUCLEATOR_CLR = 'red'
PROPAGATED_SIZE = 20
PROPAGATED_CLR = 'blue'


class CellRep:
    def __init__(self, x_loc, y_loc, is_nucleator):
        self._x_loc = x_loc
        self._y_loc = y_loc
        self._is_nucleator = is_nucleator

    @property
    def x_loc(self):
        return self._x_loc

    @property
    def y_loc(self):
        return self._y_loc

    @property
    def is_nucleator(self):
        return self._is_nucleator

    @property
    def cell_dict_rep(self):
        return (
            (self.x_loc, self._y_loc),
            self._is_nucleator
        )

    def __repr__(self):
        return 'cell_loc: {0} nuc: {1}'.format((self._x_loc, self._y_loc), self._is_nucleator)


class CellsConnectivity:

    def __init__(self, exp_name, XY, die_times, temporal_res, treatment_type, dist_threshold=200, time_threshold=30, weighted_by_time_diff=False):
        self.exp_name = exp_name
        self.XY = XY
        self.die_times = die_times
        self.temporal_res = temporal_res
        self.dist_threshold = dist_threshold
        self.time_threshold = time_threshold
        self.treatment_type = treatment_type
        self.cells_connectivity_matrix = np.zeros((self.die_times.shape[0], self.die_times.shape[0]))
        self.neighborslvl1, self.neighborslvl2, self.neighborslvl3 = NucleatorsCounter.get_neighbors(XY=self.XY)
        self.nc_instance = NucleatorsCounter(self.XY, self.die_times, self.neighborslvl1, self.dist_threshold)
        self.nucleators = self.nc_instance.calc_nucleators()
        self.network_graph = nx.Graph()
        self.weighted_by_time_diff = weighted_by_time_diff

        # calculating graph
        self.calc_connectivity_matrix()
        self.calc_graph()

    def calc_connectivity_matrix(self):
        # todo: try doing min-max normalization instead of max.
        # calc max difference in time of death between neighboring cells for normalization
        if self.weighted_by_time_diff:
            max_time_diff = 0
            min_time_diff = float('inf')
            for cell_idx, cell_death in enumerate(self.die_times):
                cell_neighbors = self.neighborslvl1[cell_idx]
                for neighbor_idx in cell_neighbors:
                    neighbor_death = self.die_times[neighbor_idx]
                    dist_threshold_cond = \
                        NucleatorsCounter.get_real_distance(self.XY[cell_idx], self.XY[neighbor_idx]) < self.dist_threshold
                    if dist_threshold_cond and cell_death <= neighbor_death:
                        max_time_diff = abs(cell_death - neighbor_death) if abs(cell_death - neighbor_death) > max_time_diff else max_time_diff
                        min_time_diff = abs(cell_death - neighbor_death) if abs(cell_death - neighbor_death) < min_time_diff else min_time_diff

        for cell_idx, cell_death in enumerate(self.die_times):
            cell_neighbors = self.neighborslvl1[cell_idx]
            for neighbor_idx in cell_neighbors:
                neighbor_death = self.die_times[neighbor_idx]
                if not self.weighted_by_time_diff:
                    # if the neighbor died subsequently it is connected to the current cell
                    dist_threshold_cond = \
                        NucleatorsCounter.get_real_distance(self.XY[cell_idx], self.XY[neighbor_idx]) < self.dist_threshold
                    if dist_threshold_cond and cell_death > neighbor_death < cell_death + self.time_threshold:
                        self.cells_connectivity_matrix[cell_idx, neighbor_idx] = 1
                else:
                    # connecting all neighbors below self.dist_threshold, edge is weighted by the normalized
                    # time diffrences between cells deaths. edge is directed from the first cell to die to the posterior
                    dist_threshold_cond = \
                        NucleatorsCounter.get_real_distance(self.XY[cell_idx], self.XY[neighbor_idx]) < self.dist_threshold
                    if dist_threshold_cond and cell_death <= neighbor_death:
                        self.cells_connectivity_matrix[cell_idx, neighbor_idx] = \
                            (abs(cell_death - neighbor_death) - min_time_diff+1) / (max_time_diff - min_time_diff + 1)


    @staticmethod
    def convert_xy_nuc_to_cells(XY, nucleators):
        cells_lst = []
        for cell_idx, cell_xy in enumerate(XY):
            cells_lst.append(CellRep(cell_xy[0], cell_xy[1], nucleators[cell_idx]).cell_dict_rep)
        return cells_lst

    def calc_graph(self):
        # adding nodes
        self.network_graph.add_nodes_from(self.convert_xy_nuc_to_cells(self.XY, self.nucleators))
        for cell_idx, cell_rep in enumerate(self.network_graph.nodes):
            for connected_candidate_idx, connected_candidate in enumerate(self.cells_connectivity_matrix[cell_idx, :]):
                if connected_candidate_idx == cell_idx or connected_candidate == 0:
                    continue
                cell1, cell2 = list(self.network_graph.nodes)[cell_idx], list(self.network_graph.nodes)[connected_candidate_idx]
                self.network_graph.add_edge(cell1, cell2, weight=connected_candidate)

    def plot_cells_connectivity_from_graph(self):
        self.plot_single_graph(self.network_graph, plot_title='treatment:{}\nexp:{}'.format(self.treatment_type, self.exp_name))
        plt.show()

    @staticmethod
    def plot_single_graph(network_graph, plot_title='Cells Connectivity Graph', savefig=False):
        node_colors = list(map(lambda cell_rep: NUCLEATOR_CLR if cell_rep[1] else PROPAGATED_CLR, network_graph.nodes))
        node_sizes = list(map(lambda cell_rep: NUCLEATOR_SIZE if cell_rep[1] else PROPAGATED_SIZE, network_graph.nodes))
        edge_colors = list(map(lambda edge_rep: network_graph.edges[edge_rep[0], edge_rep[1]]['weight'], network_graph.edges))
        cmap = plt.cm.summer
        nodes_plot_kwargs = {
            'node_color': node_colors,
            'node_size': node_sizes,
            'edge_color': edge_colors,
            'edge_cmap': cmap
        }
        edge_alphas = [1 for i in range(len(edge_colors))]
        nodes_positions = {cell_rep: cell_rep[0] for cell_rep in network_graph.nodes}

        nx.draw(G=network_graph, pos=nodes_positions, **nodes_plot_kwargs)


        # nx.draw_networkx_nodes(G=network_graph, pos=nodes_positions, **nodes_plot_kwargs)
        # edges = nx.draw_networkx_edges(
        #     network_graph,
        #     nodes_positions,
        #     node_size=node_sizes,
        #     arrowstyle="->",
        #     arrowsize=10,
        #     edge_color=edge_colors,
        #     edge_cmap=cmap,
        #     width=2,
        # )

        # pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        # pc.set_array(edge_colors)
        plt.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=min(edge_colors),
                                      vmax=max(edge_colors),
                                      clip=True),
            cmap=cmap), shrink=0.5)
        plt.suptitle(plot_title)

    def plot_cells_connectivity(self):
        fig, ax = plt.subplots()
        max_x, max_y = self.XY[:, 0].max(), self.XY[:, 1].max()
        for cell_idx, cell_xy in enumerate(self.XY):
            ax.set_aspect(1)
            curr_cell_x, curr_cell_y = cell_xy[0] / max_x, cell_xy[1] / max_y
            circle_color = 'red' if self.nucleators[cell_idx] else 'green'
            cc = plt.Circle((curr_cell_x, curr_cell_y), 0.01, color=circle_color)
            ax.add_artist(cc)
            # drawing the connectivity line
            for connected_candidate_idx, connected_candidate in enumerate(self.cells_connectivity_matrix[cell_idx, :]):
                if connected_candidate_idx == cell_idx or connected_candidate == 0:
                    continue
                connected_candidate_x, connected_candidate_y = tuple(self.XY[connected_candidate_idx])
                connected_candidate_x, connected_candidate_y = connected_candidate_x / max_x, connected_candidate_y / max_y
                connectivity_line = plt.Line2D(xdata=(curr_cell_x, connected_candidate_x),
                                               ydata=(curr_cell_y, connected_candidate_y),
                                               c='blue', linestyle='--', markersize=3, mec='red')
                ax.add_line(connectivity_line)
        plt.show()

    @classmethod
    def calc_graphs_for_multiple_experiments(cls, exps_fn, file_details_fn, weighted_by_time_diff=False, **plot_kwargs):
        all_graphs_by_treatment_name = {}
        exp_details_df = pd.read_csv(file_details_fn)
        filtered_list_of_files_to_analyze = list(filter(lambda x: not(x == '.DS_Store' or x.find('File details') != -1), os.listdir(exps_fn)))
        for f_idx, file_name in enumerate(filtered_list_of_files_to_analyze):
            exp_name = file_name[:-4]
            if f_idx > MAX_EXP_TO_PROCESS:
                break
            if file_name.find('.csv') != -1:
                print("file idx: %d file name: %s" % (f_idx, file_name))
                single_exp_file_fname = os.sep.join([exps_fn, file_name])
                exp_xyt = pd.read_csv(single_exp_file_fname)
                full_x, full_y = exp_xyt["cell_x"].values, exp_xyt["cell_y"].values
                XY = np.column_stack((full_x, full_y))
                die_times = exp_xyt["death_time"].values
                exp_temporal_resolution = int(exp_details_df[exp_details_df['File Name'] == file_name]['Time Interval (min)'].values[0])
                exp_treatment_type = exp_details_df[exp_details_df['File Name'] == file_name]['Treatment'].values[0]
                exp_connectivity = cls(exp_name=exp_name,
                                       XY=XY,
                                       die_times=die_times,
                                       temporal_res=exp_temporal_resolution,
                                       treatment_type=exp_treatment_type,
                                       weighted_by_time_diff=weighted_by_time_diff)
                all_graphs_by_treatment_name[exp_treatment_type] = all_graphs_by_treatment_name.get(exp_treatment_type, []) + [exp_connectivity.network_graph]

                if plot_kwargs.get('showfig', False):
                    cls.plot_single_graph(exp_connectivity.network_graph, plot_title='treatment:{}\nexp:{}'.format(exp_treatment_type, exp_name))

                    plt.show()
                if plot_kwargs.get('savefig', False):
                    def_fig_fname = os.sep.join(['ExpsGraphsRepresentations',
                                                 'exp_graph_rep:{}|Treat={}.png'.format(exp_name, exp_treatment_type.replace(os.sep, '|'))])
                    cls.plot_single_graph(exp_connectivity.network_graph, plot_title='treatment:{}\nexp:{}'.format(exp_treatment_type, exp_name))
                    plt.savefig(plot_kwargs.get('figname', def_fig_fname), dpi=200)
            plt.clf()

        return all_graphs_by_treatment_name

    @classmethod
    def calc_connectivity_about_treatment(cls,
                                          exps_fn,
                                          file_details_fn,
                                          calc_for_treatment_types=['FAC', 'PEG', 'erastin'],
                                          weighted_by_time_diff=False,
                                          **plotkwargs):
        plotkwargscpy = plotkwargs.copy()
        # plotkwargscpy['showfig'] = False
        # plotkwargscpy['savefig'] = False
        graphs_by_treatment_name = cls.calc_graphs_for_multiple_experiments(exps_fn=exps_fn,
                                                                            file_details_fn=file_details_fn,
                                                                            weighted_by_time_diff=weighted_by_time_diff,
                                                                            **plotkwargscpy)
        connected_components_by_treatment = {}
        no_of_connected_components_by_treatment = {}
        avg_of_connected_components_by_treatment = {}
        avg_degree_by_treatment = {}
        for treatment_idx, treatment_name in enumerate(graphs_by_treatment_name.keys()):
            to_calc = True
            if calc_for_treatment_types is not None:
                to_calc = sum([x in treatment_name for x in calc_for_treatment_types])
            if to_calc:
                for exp_in_treat_idx, exp_graph in enumerate(graphs_by_treatment_name[treatment_name]):
                    exp_connected_components = nx.connected_components(exp_graph)
                    connected_components_by_treatment[treatment_name] = \
                        connected_components_by_treatment.get(treatment_name, []) + [exp_connected_components]
                    normalized_no_of_components = len(list(exp_connected_components))/len(exp_graph.nodes)
                    no_of_connected_components_by_treatment[treatment_name] = \
                        no_of_connected_components_by_treatment.get(treatment_name, []) + [normalized_no_of_components]
                    #todo: avg_degree_by_treatment[treatment_name] = \
                    #     avg_degree_by_treatment.get(treatment_name, []) + [np.array(exp_graph.degree).mean()]
                    avg_degree_by_treatment[treatment_name] = \
                            avg_degree_by_treatment.get(treatment_name, []) + [np.array(exp_graph.degree, dtype=object)[:, 1].mean()/np.array(exp_graph.degree, dtype=object)[:, 1].max()]

                avg_of_connected_components_by_treatment[treatment_name] = np.array(no_of_connected_components_by_treatment[treatment_name]).mean()

        return connected_components_by_treatment, no_of_connected_components_by_treatment, avg_of_connected_components_by_treatment, avg_degree_by_treatment


if __name__ == '__main__':
    main_exps_fn = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles/ConvertedTimeXYT'
    file_details_fname = '/Users/yishaiazabary/Desktop/University/Thesis/CellDeathThesis/Code/Ferroptosis/ExperimentsXYT_CSVFiles/File details.csv'


    # single experiment testing
    # filename = '20200311_HAP1_ML162_xy3.csv'
    # filename = '20180620_HAP1_erastin_xy6.csv'
    # filename = '20181229_HAP1-920H_FB+PEG3350_GCAMP_xy57.csv'
    # exp_fn = os.sep.join([main_exps_fn, filename])
    # exp_xyt = pd.read_csv(exp_fn)
    # exp_details_df = pd.read_csv(file_details_fname)
    # full_x = exp_xyt["cell_x"].values
    # full_y = exp_xyt["cell_y"].values
    # n_instances = len(full_x)
    # die_times = exp_xyt["death_time"].values
    # XY = np.column_stack((full_x, full_y))
    # exp_temporal_resolution = exp_details_df[exp_details_df['File Name'] == filename]['Time Interval (min)'].values[0]
    # exp_treatment_type = exp_details_df[exp_details_df['File Name'] == filename]['Treatment'].values[0]
    # cc = CellsConnectivity(XY=XY,
    #                        die_times=die_times,
    #                        temporal_res=exp_temporal_resolution,
    #                        treatment_type=exp_treatment_type)
    # cc.plot_cells_connectivity()
    # cc.plot_cells_connectivity_from_graph()

    # multiple experiments plotting
    # CellsConnectivity.calc_graphs_for_multiple_experiments(exps_fn=main_exps_fn,
    #                                                        file_details_fn=file_details_fname,
    #                                                        savefig=SAVEFIG,
    #                                                        showfig=SHOWFIG)
    CellsConnectivity.calc_connectivity_about_treatment(exps_fn=main_exps_fn,
                                                        file_details_fn=file_details_fname,
                                                        weighted_by_time_diff=True,
                                                        calc_for_treatment_types=None,
                                                        showfig=SHOWFIG,
                                                        savefig=SAVEFIG)
    

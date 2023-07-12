import numpy as np
import pandas as pd
from graphtheory.structures.edges import Edge
from graphtheory.structures.graphs import Graph
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod


# help classes
class SingleCell:
    def __init__(self, x:float, y:float, time_of_death: float):
        self._x, self._y, self._time_of_death = x, y, time_of_death

    def get_x(self):
        return self._x

    def set_x(self, new_x:float):
        self._x = new_x

    def get_y(self):
        return self._y

    def set_y(self, new_y:float):
        self._y = new_y

    def get_time_of_death(self):
        return self._time_of_death

    def set_time_of_death(self, new_time_of_death:float):
        self._time_of_death = new_time_of_death

    def dist(self, other):
        if not isinstance(other, SingleCell):
            raise TypeError("must both be SingleCell type")
        return ((self._x - other.get_x()) ** 2 + (self._y - other.get_y()) ** 2) ** (1 / 2)

    def __hash__(self):
        return hash((self._x, self._y, self._time_of_death))

    def __eq__(self, other):
        if not isinstance(other, SingleCell):
            raise TypeError("must both be SingleCell type")
        return self.dist(other)==0

    def __repr__(self):
        return 'X:{:.2f}|Y:{:.2f}|TOF:{:.1f}'.format(self._x, self._y, self._time_of_death)

def plot_graph(graph: Graph, clr_map):
    fig, ax = plt.subplots()

    for cell_, edges in graph.items():
        for cl_, edge in edges.items():
            source_ = edge.source
            target = edge.target
            # scatter cells
            ax.plot([source_.get_x(), target.get_x()], [source_.get_y(), target.get_y()], marker='o', color=clr_map(target.get_time_of_death()))
            # plt.axline((source_.get_x(),source_.get_y()), (target.get_x(),target.get_y()))
    plt.show()

class DrawableGraph(ABC):
    def __init__(self):
        self._graph = None

    @abstractmethod
    def built_graph(self):
        pass

    def plot_graph(self, clr_map):
        fig, ax = plt.subplots()

        for cell_, edges in self._graph.items():
            for cl_, edge in edges.items():
                source_ = edge.source
                target = edge.target
                # scatter cells
                ax.plot([source_.get_x(), target.get_x()], [source_.get_y(), target.get_y()], marker='o', color=clr_map(target.get_time_of_death()))
                # plt.axline((source_.get_x(),source_.get_y()), (target.get_x(),target.get_y()))
        plt.show()


class GraphByVoronoiNeighbors(DrawableGraph):

    def __init__(self, path_to_file, is_directed=False):
        super().__init__()
        self.path_to_file = path_to_file
        self.is_directed_graph = is_directed
        self.unique_times = None

    def built_graph(self):
        df = pd.read_csv(path_to_file)
        XY = df.loc[:, ['cell_x', 'cell_y']].values
        TIMES = df.loc[:, ['death_time']].values
        self.unique_times = np.unique(TIMES)
        cells_list = []
        for cell_idx, cell_xy in enumerate(XY):
            cells_list.append(SingleCell(cell_xy[0], cell_xy[1], TIMES[cell_idx][0]))

        self._graph = Graph(directed=self.is_directed_graph)

        vrn_model = Voronoi(XY)
        neighbors = vrn_model.ridge_points
        for neighboring_cells in neighbors:
            # first_x, first_y = XY[neighboring_cells[0]]
            # first_cell = SingleCell(first_x, first_y)
            first_cell = cells_list[neighboring_cells[0]]
            # second_x, second_y = XY[neighboring_cells[1]]
            # second_cell = SingleCell(second_x, second_y)
            second_cell = cells_list[neighboring_cells[1]]
            self._graph.add_edge(Edge(source=first_cell, target=second_cell, weight=first_cell.dist(second_cell)))

        clr_map = plt.get_cmap('plasma', len(self.unique_times))
        self.plot_graph(clr_map)

class GraphByDeathLeaders(DrawableGraph):

    def __init__(self, path_to_file, is_directed=False):
        super().__init__()
        self.path_to_file = path_to_file
        self.is_directed_graph = is_directed
        self.unique_times = None

    def built_graph(self):
        df = pd.read_csv(path_to_file)
        XY = df.loc[:, ['x_loc', 'y_loc']].values
        TIMES = df.loc[:, ['die_time']].values
        cell_leaders = df.loc[:, ['cell_leader']].values
        self.unique_times = np.unique(TIMES)
        cells_list = []
        for cell_idx, cell_xy in enumerate(XY):
            cells_list.append(SingleCell(cell_xy[0], cell_xy[1], TIMES[cell_idx][0]))

        self._graph = Graph(directed=self.is_directed_graph)

        for cell_idx, cell in enumerate(cells_list):
            first_cell = cell
            cell_leader = cells_list[cell_leaders[cell_idx][0]]
            if cell_leader is first_cell:
                continue
            self._graph.add_edge(Edge(source=first_cell, target=cell_leader, weight=first_cell.dist(cell_leader)))

        clr_map = plt.get_cmap('jet', len(self.unique_times))
        self.plot_graph(clr_map)


# path_to_file = 'ExperimentsXYT_CSVFiles/20161129_MCF7_FB_xy13.csv'
# g = GraphByVoronoiNeighbors(path_to_file=path_to_file)
# g.built_graph()

path_to_file = 'testsOutput/Results/20161129_MCF7_FB_xy13/Data/File_20161129_MCF7_FB_xy13.csv'
g = GraphByDeathLeaders(path_to_file=path_to_file)
g.built_graph()


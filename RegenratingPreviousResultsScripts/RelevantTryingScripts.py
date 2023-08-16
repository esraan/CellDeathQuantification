import os
import sys
import shutil
import math
import warnings
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sys.path.append("/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/UpdatedCellDeathQuantification/CellDeathQuantification/QuantificationScripts")
from QuantificationScripts.NucleationAndPropagationMeasurements import *
from OldCodeBase_15072021.NucliatorsCount import NucleatorsCounter 
from OldCodeBase_15072021.NucleationProbabilityAndSPI import NucleationProbabilityAndSPI 
from RegenratingPreviousResultsScripts.PreviousResultsUtils import *

#loading np files
np.load("/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldDataResultsGeneration/20170929_MCF7_H2O2_xy30.csv_factor_of_change_map_5_tods_5.npy")


#NRF one expreiment trying things out
exps_dir_name = "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/2023DataSet/CSVs_trackmate_fiji"
meta_data_file_path= "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/ExperimentsMetaData.csv"

exps_results_dicts = calc_factor_of_propagation_by_number_of_dead_neighbors_and_time_from_recent_neighbors_death(
        exp_name="20160820_10A_FB_xy14.csv",
        exps_dir_path="/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/OriginalTimeFramesData",
        max_number_of_dead_neighbors_to_calc=5,
        max_delta_tod_from_recently_dead_neighbor=5,
        meta_data_full_file_path=meta_data_file_path,
        show_fig=True,
        dist_threshold=200,
        number_of_random_permutations=10,
        include_only_treatments=['FAC'],
        fig_v_min=1.,
        fig_v_max=3.
    )



#SPI 

from QuantificationScripts.SPICalculator import *
exps_dir_name = "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/OriginalTimeFramesData"
"/Users/yishaiazabary/PycharmProjects/University/CellDeathQuantification/Data/2023 data/OriginalTimeFramesData"
meta_data_file_full_path= "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/ExperimentsMetaData.csv"
meta_data_extract_exp_names= pd.read_csv(meta_data_file_full_path)
exp_full_path = "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/OriginalTimeMinutesData/20160820_10A_FB_xy13.csv"
cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)
exp_name = '20160820_10A_FB_xy11.csv'
exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                            meta_data_file_full_path=meta_data_file_full_path)

nuc_p_spi = SPICalculator(XY=cells_locis,
                                        die_times=cells_tods,
                                        temporal_resolution=exp_temporal_resolution,
                                        treatment=exp_treatment,
                                        n_scramble=1000,
                                        draw=False,
                                        dist_threshold_nucliators_detection=200)
print(nuc_p_spi.get_spis())


from QuantificationScripts.SPICalculator import *
nuc_p_spi = SPICalculator(XY=cells_locis,
                                        die_times=cells_tods,
                                        temporal_resolution=exp_temporal_resolution,
                                        treatment=exp_treatment,
                                        n_scramble=1000,
                                        draw=False,
                                        dist_threshold_nucliators_detection=200)
print(nuc_p_spi.get_spis())
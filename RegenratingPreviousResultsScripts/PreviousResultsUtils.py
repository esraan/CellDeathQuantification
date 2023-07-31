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
from QuantificationScripts.NucleationAndPropagationMeasurements import *
from OldCodeBase_15072021.NucliatorsCount import NucleatorsCounter 
from OldCodeBase_15072021.NucleationProbabilityAndSPI import NucleationProbabilityAndSPI 
def calc_all_pairs_tods_distances_dependencies_mean(cells_location: list,
                                                    cells_tods:list,
                                                    dist_threshold_nucleators_detection :int,
                                                    **kwargs) -> tuple :
    nu_instance = NucleatorsCounter(cells_location, cells_tods, NucleatorsCounter.get_neighbors( XY=cells_location)[0],dist_threshold_nucleators_detection=dist_threshold_nucleators_detection)
    treatment_pairs_distances = nu_instance.get_all_nighbors_distances()
    treatment_pairs_tods = nu_instance.get_all_nighbors_tods()
    return treatment_pairs_distances, treatment_pairs_tods

def calc_exp_mean_of_pairs_tods_distances_dependencies(
        exp_name: Union[str, List[str]],
        exps_dir_path: str,
        meta_data_full_file_path: str,
        **kwargs) -> np.array:
    if isinstance(exp_name, list):
        results = {}
        for exp in exp_name:
            res = calc_exp_mean_of_pairs_tods_distances_dependencies(
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                meta_data_full_file_path=meta_data_full_file_path, **kwargs
            )
            results[exp] = res
        return results
    fig_title = kwargs.get('fig_title', f'TODs and distances dependencies\n{exp_name}\n')
    cbar_kwargs = kwargs.get('cbar_kwargs', {})

    exp_full_path = os.path.join(exps_dir_path, exp_name)
    exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                            meta_data_file_full_path=meta_data_full_file_path)

    cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)


    treatment_pairs_distances_and_tod_dependencies =\
        calc_all_pairs_tods_distances_dependencies_mean(cells_tods=cells_tods,
                                                                                          cells_location=cells_locis,
                                                                                          dist_threshold_nucleators_detection=200)
    # generate 'number_of_random_permutations' random permutations of cells times of death
    #   and calculate each permutation probability map, then calculate the difference factor
    #   between each on and the original.
    return treatment_pairs_distances_and_tod_dependencies[0][1],treatment_pairs_distances_and_tod_dependencies[1][1]


def values_distribution_of_distances_or_delta_TODs_for_each_nighbors_pairs(
    returned_value: str,
    exp_name: Union[str, List[str]],
        exps_dir_path: str,
        meta_data_full_file_path: str,
        **kwargs) -> np.array:
    if isinstance(exp_name, list):
        results = {}
        for exp in exp_name:
            res = values_distribution_of_distances_or_delta_TODs_for_each_nighbors_pairs(
                returned_value=returned_value,
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                meta_data_full_file_path=meta_data_full_file_path, **kwargs
            )
            results[exp] = res
        return results
    fig_title = kwargs.get('fig_title', f'TODs and distances dependencies\n{exp_name}\n')
    cbar_kwargs = kwargs.get('cbar_kwargs', {})

    exp_full_path = os.path.join(exps_dir_path, exp_name)
    exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                            meta_data_file_full_path=meta_data_full_file_path)

    cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)


    treatment_pairs_distances_and_tod_dependencies =\
        calc_all_pairs_tods_distances_dependencies_mean(cells_tods=cells_tods,
                                                                                          cells_location=cells_locis,
                                                                                          dist_threshold_nucleators_detection=200)
    # generate 'number_of_random_permutations' random permutations of cells times of death
    #   and calculate each permutation probability map, then calculate the difference factor
    #   between each on and the original.
    if returned_value=="distances":
        return treatment_pairs_distances_and_tod_dependencies[0][0]
    elif returned_value=="TODs":
        return np.concatenate(treatment_pairs_distances_and_tod_dependencies[1][0]).ravel().tolist()
    else:
        return treatment_pairs_distances_and_tod_dependencies[0][0]\
    ,np.concatenate(treatment_pairs_distances_and_tod_dependencies[1][0]).ravel().tolist()
    

def replace_ugly_long_name(name, cell_line = ""):
    if "fb" in name.lower() and "peg" not in name.lower():
        if cell_line=="":
            return "FAC&BSO"
        elif "sgCx43" in cell_line:
            return "MCF10A+FB"
        return cell_line+"+FB"
    elif "tsz" in name.lower():
        return "U937+TFNa+SMAC+zVAD"
    elif "ml162" in name.lower():
        if cell_line=="":
            return "ML162"
        elif "HAP1" in cell_line:
            return "ML162+HAP1" 
        return cell_line +"+ML162"
    elif "erastin" in name.lower():
        return "HAP1+erastin"
    elif "peg" in name.lower():
        return "HAP1+FAC&BSO+PEG"
    elif "skt" in name.lower():
        return "MCF10A+superkiller TRAIL"
    elif "amsh" in name.lower():
        return "B16F10+C' dots"
    elif "h2o2" in name.lower():
        return "MCF7+H2O2"
    elif "sparse" or "dense" in name.lower():
        return cell_line+"+FAC&BSO **"
    else:
        return name
    
def calc_all_cells_local_densities(cells_location: list,
                                                    radious_for_density:int,
                                                    **kwargs):
    return NucleatorsCounter.get_cells_density(cells_location,radious_for_density)


def calc_exp_all_cells_local_densities_for_distribution(
        exp_name: Union[str, List[str]],
        exps_dir_path: str,
        meta_data_full_file_path: str,
        radious_for_density:int,
        **kwargs) -> np.array:
    if isinstance(exp_name, list):
        results = {}
        for exp in exp_name:
            res = calc_exp_all_cells_local_densities_for_distribution(
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                radious_for_density=radious_for_density,
                meta_data_full_file_path=meta_data_full_file_path, **kwargs
            )
            results[exp] = res
        return results


    exp_full_path = os.path.join(exps_dir_path, exp_name)
    try:
        pd.read_csv(exp_full_path)
    except FileNotFoundError:
        return

    cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)
    exp_all_cells_local_density_values =\
        calc_all_cells_local_densities(radious_for_density=radious_for_density,
                                                                                          cells_location=cells_locis)
    # generate 'number_of_random_permutations' random permutations of cells times of death
    #   and calculate each permutation probability map, then calculate the difference factor
    #   between each on and the original.
    return exp_all_cells_local_density_values



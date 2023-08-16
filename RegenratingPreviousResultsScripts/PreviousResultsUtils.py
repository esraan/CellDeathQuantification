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
import numpy as np
from matplotlib.collections import EllipseCollection,CircleCollection,PatchCollection
import math
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

    try:
        exp_full_path = os.path.join(exps_dir_path, exp_name)
        cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)
    except FileNotFoundError:
        return


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
        # elif "sgCx43" in cell_line:
        #     return "MCF10A+FB"
        if "dense" or "sparse" in name:
            return "MCF10A+FAC&BSO **"
        return cell_line+"+FB"
    elif "tsz" in name.lower():
        return "U937+TSZ"#"U937+TFNa+SMAC+zVAD"
    elif "ml162" in name.lower():
        if cell_line=="":
            return "ML162"
        elif "HAP1" in cell_line:
            return "HAP1+ML162" 
        return cell_line +"+ML162"
    elif "erastin" in name.lower():
        return "HAP1+erastin"
    elif "peg" in name.lower():
        return "HAP1+FAC&BSO+PEG"
    elif "skt" in name.lower():
        return "MCF10A+TRAIL" #"MCF10A+superkiller TRAIL"
    elif "amsh" in name.lower():
        return "B16F10+C' dots"
    elif "h2o2" in name.lower():
        return "MCF7+H2O2"
    elif "sparse" or "dense" in name.lower():
        return "MCF10A+FAC&BSO **"
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


def calc_experiment_SPI_and_NI(cells_location: list,
                                                    cells_tods:list,
                                                    exp_temporal_resolution:int,
                                                    dist_threshold_nucleators_detection :int,
                                                    exp_treatment,
                                                    n_scramble:int,
                                                    **kwargs) -> int :
    nuc_p_spi_instance = NucleationProbabilityAndSPI(XY=cells_location, die_times=cells_tods, time_frame=exp_temporal_resolution, treatment= exp_treatment, n_scramble=n_scramble, dist_threshold_nucliators_detection=dist_threshold_nucleators_detection,draw=False)

    return nuc_p_spi_instance.get_spi_nucleators()

def calc_all_experiments_SPI_and_NI_for_landscape(
        exp_name: Union[str, List[str]],
        exps_dir_path: str,
        meta_data_full_file_path: str,
        **kwargs) -> np.array:
    if isinstance(exp_name, list):
        results = {}
        for exp in exp_name:
            res = calc_all_experiments_SPI_and_NI_for_landscape(
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                meta_data_full_file_path=meta_data_full_file_path, **kwargs
            )
            results[exp] = res
        return results
    try:
        # print(exp_name)
        exp_full_path = os.path.join(exps_dir_path, exp_name)
        
        exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                                meta_data_file_full_path=meta_data_full_file_path)

        cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)


        SPI_and_NI_dict =\
            calc_experiment_SPI_and_NI(cells_tods=cells_tods,
                                                                                            cells_location=cells_locis,
                                                                                            n_scramble=1000,
                                                                                            exp_temporal_resolution=exp_temporal_resolution,
                                                                                            exp_treatment=exp_treatment,
                                                                                            dist_threshold_nucleators_detection=200)
        # generate 'number_of_random_permutations' random permutations of cells times of death
        #   and calculate each permutation probability map, then calculate the difference factor
        #   between each on and the original.
        return SPI_and_NI_dict["spi"][0], SPI_and_NI_dict["p_nuc"]
    except FileNotFoundError:
        return (None,None)
    
def format_melt_df_distribution_values (exp_names:list,
                                        dict_of_exp_values_of_the_designated_variable_density_deltaTODs_distances: dict,
                                        meta_data_full_file_path: str,
                                        **kwargs)->dict:
    name_of_the_melted_variable = kwargs.get("name_of_the_melted_variable", "Local_density_values")
    Density_needed = kwargs.get("Density_needed", False)
    dict_formated ={"Experiment_name":[],
                    name_of_the_melted_variable:[],
                    "Treatment":[]}
    if Density_needed:
        dict_formated["Density"]=[]
        for exp_name in exp_names:
            try:
                disignated_len = len(dict_of_exp_values_of_the_designated_variable_density_deltaTODs_distances[exp_name]) if isinstance(dict_of_exp_values_of_the_designated_variable_density_deltaTODs_distances[exp_name], list) else 0
                dict_formated["Experiment_name"] += [exp_name]*disignated_len
                print(exp_name)
                dict_formated[name_of_the_melted_variable] += dict_of_exp_values_of_the_designated_variable_density_deltaTODs_distances[exp_name]
                exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                                    meta_data_file_full_path=meta_data_full_file_path)
                dict_formated["Treatment"]+= [exp_treatment]*disignated_len
                dict_formated["Density"] += ["sparse" if "sparse" in exp_name else "dense"]* disignated_len
                
            except Exception:
                continue
    else:
        for exp_name in exp_names:
            try:
                disignated_len = len(dict_of_exp_values_of_the_designated_variable_density_deltaTODs_distances[exp_name]) if isinstance(dict_of_exp_values_of_the_designated_variable_density_deltaTODs_distances[exp_name], list) else 0
                dict_formated["Experiment_name"] += [exp_name]*disignated_len
                # print(exp_name)
                dict_formated[name_of_the_melted_variable] += dict_of_exp_values_of_the_designated_variable_density_deltaTODs_distances[exp_name]
                exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                                    meta_data_file_full_path=meta_data_full_file_path)
                dict_formated["Treatment"]+= [exp_treatment]*disignated_len
            except Exception:
                continue
    
    return dict_formated

def draw_cells_via_fixed_radious_according_to_xy_cordination_of_prevoius_data_single_exp(cells_location: list,
                                                                                         cells_tod:List,
                                                                                         cells_fixed_radious: int,
                                                                                         **kwargs):
    
    # X = [double[0] for double in cells_location]
    # Y = [double[1] for double in cells_location]
    # radious = 2
    # size = [math.pi * (radious**2)]*len(X)
    # patches = [plt.Circle(center, size) for center, size in zip(cells_locis, size)]
    # fig, ax = plt.subplots()
    # coll = PatchCollection(patches)
    # ax.add_collection(coll)
    size = math.pi * (cells_fixed_radious**2)
    color = np.random.rand(len(cells_location))
    cmap = plt.cm.twilight_shifted
    fig, ax = plt.subplots()
    facecolors = [cm.jet(x) for x in cells_tod]
    # offsets = list(zip(X, Y))
    ec = EllipseCollection(widths=size, heights=size, angles=0, units='xy',
                                        facecolors=plt.cm.twilight_shifted(cells_tod), 
                                        offsets=cells_location, transOffset=ax.transData)
    ax.add_collection(ec)
    ax.axis('equal') # set aspect ratio to equal
    # ax.axis([-400, 1800, -200, 1600])
    # fig=plt.scatter(X, Y, c=color, cmap=cmap)
    pos_neg_clipped = ax.imshow(cells_tod, cmap=cmap,
                             interpolation='none')
    cbar = plt.colorbar(pos_neg_clipped,ax=ax)
    cbar.set_label('TOD')
    plt.xlim(0,2900)
    plt.ylim(0,2500)
    plt.show()
    

def draw_cells_via_fixed_radious_according_to_xy_cordination_of_prevoius_data(
            exp_name: Union[str, List[str]],
            exps_dir_path: str,
            meta_data_full_file_path: str,
            **kwargs):
    
    
    
    if isinstance(exp_name, list):
        for exp in exp_name:
            draw_cells_via_fixed_radious_according_to_xy_cordination_of_prevoius_data(
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                meta_data_full_file_path=meta_data_full_file_path,
                **kwargs
            )
        return
        
    try:
        cells_fixed_radious = kwargs.get("cells_fixed_radious", 10)
        exp_full_path = os.path.join(exps_dir_path, exp_name)
        exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                            meta_data_file_full_path=meta_data_full_file_path)
        
        if True not in [to_include.lower() in exp_treatment.lower() for to_include in kwargs.get("treatment_to_include",[])]:
            return
        cells_locis, cells_tod = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)


        draw_cells_via_fixed_radious_according_to_xy_cordination_of_prevoius_data_single_exp(cells_location=cells_locis,
                                                                                             cells_tod= cells_tod,
                                                                                            cells_fixed_radious=cells_fixed_radious,
                                                                                            )
        # generate 'number_of_random_permutations' random permutations of cells times of death
        #   and calculate each permutation probability map, then calculate the difference factor
        #   between each on and the original.
        
    except FileNotFoundError:
        return 
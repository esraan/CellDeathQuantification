from PIL import Image, ImageDraw
import imageio
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import sys
from random import randint
import math
sys.path.append("/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/UpdatedCellDeathQuantification/CellDeathQuantification")
from utils import *
from RegenratingPreviousResultsScripts.PreviousResultsUtils import *
def create_simulated_frames_gif_of_cells_loci_and_tods_old_data(cells_location: list,
                                                         cells_tod:list,
                                                         cells_fixed_radious:int,)-> List[List]:
    """_summary_

    Args:
        cells_location (list): _description_
        cells_tod (list): _description_
        gif_saving_path (str): _description_
    """    
    # Create a white background image
    background = Image.new('RGB', (3759, 2808), (255, 255, 255))
    
    # Create a list to store individual frames
    frames = []

    # Set the circle radius (adjust as needed)
    circle_radius = cells_fixed_radious

    # Create an empty set to keep track of visited coordinates
    visited_coordinates = set()

    # Normalize the time values to map to colors
    norm = Normalize(vmin=min(cells_tod), vmax=max(cells_tod))
    colormap = plt.get_cmap('twilight_shifted')

    # Create a scalar mappable for color mapping
    scalar_mappable = ScalarMappable(cmap=colormap, norm=norm)

    # Iterate through coordinates and times
    for i, (x, y) in enumerate(cells_location):
        # Copy the background for each frame
        frame = background.copy()
        
        # Create a drawing context
        draw = ImageDraw.Draw(frame)
        
        # Mark the current coordinates as visited
        visited_coordinates.add((x, y, int(cells_tod[i])))
        
        # Draw filled circles for all visited coordinates with colors based on time
        for (vx, vy, vt) in visited_coordinates:
            time_color = scalar_mappable.to_rgba(vt)
            color = tuple(int(c * 255) for c in time_color[:3])
            draw.ellipse((vx - circle_radius, vy - circle_radius, vx + circle_radius, vy + circle_radius), fill=color)
            
        # Add text showing the time of appearance
        text = f"Time: {cells_tod[i]}"
        draw.text((x - circle_radius, y + circle_radius), text, fill=(0, 0, 0))  # Black text
        
        # Append the frame to the list
        frames.append(frame)
    return frames



def draw_cells_gif_via_fixed_radious_according_to_xy_cordination_of_prevoius_data(
            exp_name: Union[str, List[str]],
            exps_dir_path: str,
            meta_data_full_file_path: str,
            **kwargs):
    
    
    
    if isinstance(exp_name, list):
        for exp in exp_name:
            draw_cells_gif_via_fixed_radious_according_to_xy_cordination_of_prevoius_data(
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                meta_data_full_file_path=meta_data_full_file_path,
                **kwargs
            )
        return
        
    try:
        cells_fixed_radious = kwargs.get("cells_fixed_radious",randint(1,6))
        exp_full_path = os.path.join(exps_dir_path, exp_name)
        exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                            meta_data_file_full_path=meta_data_full_file_path)
        
        # if True not in [to_include.lower() in exp_treatment.lower() for to_include in kwargs.get("treatment_to_include",[])]:
        #     return
        cells_locis, cells_tod = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)


        frames = create_simulated_frames_gif_of_cells_loci_and_tods_old_data(cells_location=cells_locis,
                                                                                             cells_tod= cells_tod,
                                                                                            cells_fixed_radious=cells_fixed_radious)
        
        if kwargs.get("dir_path_to_save_gif_simulation", None) is not None:
            dir_path_to_save = os.path.join(kwargs.get("dir_path_to_save_gif_simulation"), f'GIF', f'{exp_treatment}')
        else:
            dir_path_to_save = os.path.join('..','Results','NonRandomalityFactorResults', f'GIF', f'{exp_treatment}')
        
        if kwargs.get('save_fig', True):
            exp_name = exp_name.split(".csv")[0]+".csv"
            os.makedirs(dir_path_to_save, exist_ok=True)
            fig_path_GFI = os.path.join(dir_path_to_save, f"{exp_name}.gif")                                                                
            with BytesIO() as output_buffer:
                imageio.mimsave(output_buffer, frames, format="GIF", duration=0.5)  # Adjust duration as needed
                with open(fig_path_GFI, "wb") as gif_file:
                    gif_file.write(output_buffer.getvalue())
        # generate 'number_of_random_permutations' random permutations of cells times of death
        #   and calculate each permutation probability map, then calculate the difference factor
        #   between each on and the original.
        
    except FileNotFoundError:
        return 

# #OLD DATA
# exps_dir_name = "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/OriginalTimeMinutesData"
# meta_data_file_full_path= "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/ExperimentsMetaData.csv"
# meta_data_extract_exp_names= pd.read_csv(meta_data_file_full_path)
# exp_names = list(meta_data_extract_exp_names.iloc[:96,1])
# dir_path_to_save_gif_simulation = "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/simulatedGIF_OldData/"
# draw_cells_gif_via_fixed_radious_according_to_xy_cordination_of_prevoius_data(exp_name=exp_names,
#                                                                               exps_dir_path=exps_dir_name,
#                                                                               meta_data_full_file_path=meta_data_file_full_path,
#                                                                               cells_fixed_radious = 5,
#                                                                               dir_path_to_save_gif_simulation=dir_path_to_save_gif_simulation)   



def death_rate_calculation_in_different_timeframes(Time_list:list,
                                                         cells_tods:list,
                                                         exp_density:int,
                                                         **kwargs):
    cells_counts_per_timeframe_dict = Counter(cells_tods.flatten())
    times =  list(cells_counts_per_timeframe_dict.keys())
    count_in_time = list(cells_counts_per_timeframe_dict.values())
    ratio_of_cells_died_in_timeframes_list = []
    time_scale = Time_list[1]-Time_list[0]
    time_in_timeframe_desired = []
    print(sum(count_in_time))
    last_time = 0
    for idx, value in enumerate(times):
        # last_time = value
    # print(f"{value} occurs {count} times")
        if value%time_scale==0:
            last_time = value
            print(f"index: {idx}, sum till now: {sum(count_in_time[:idx+1])}, ")
            ratio_of_cells_died_in_timeframes_list.append(sum(count_in_time[:idx+1])/sum(count_in_time))
            time_in_timeframe_desired.append(value)
    index_to_refill = Time_list.index(last_time)
    ratio_of_cells_died_in_timeframes_list+=[exp_density/exp_density]*(len(Time_list)-index_to_refill)
    return {"Time":Time_list, "Death rate":ratio_of_cells_died_in_timeframes_list}

def calc_all_experiments_global_death_rate(
        exp_name: Union[str, List[str]],
        exps_dir_path: str,
        meta_data_full_file_path: str,
        **kwargs) -> np.array:
    if isinstance(exp_name, list):
        results = {}
        for exp in exp_name:
            res = calc_all_experiments_global_death_rate(
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                meta_data_full_file_path=meta_data_full_file_path, **kwargs
            )
            results[exp] = res
        return results
    try:
        # print(exp_name)
        exp_full_path = os.path.join(exps_dir_path, exp_name)
        meta_data_file_full_path= "/Users/esraan/Library/CloudStorage/GoogleDrive-esraan@post.bgu.ac.il/My Drive/PhD materials/OldData/Experiments_XYT_CSV/ExperimentsMetaData.csv"
        meta_data_extract_exp_names= pd.read_csv(meta_data_file_full_path)
        exp_cell_line= meta_data_extract_exp_names[meta_data_extract_exp_names["File Name"]== exp_name]["Cell Line"].values[0]
        exp_treatment, exp_temporal_resolution, exp_density = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                                meta_data_file_full_path=meta_data_full_file_path,
                                                                                                get_exp_density=True)
        time_scale = kwargs.get("time_scale", exp_temporal_resolution)
        cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)
        del cells_locis
        max_time = kwargs.get("max_time", 1750)
        Time_list = list(range(0, max_time, time_scale))
        cells_global_death_rate =\
            death_rate_calculation_in_different_timeframes(cells_tods=cells_tods,
                                                                                            Time_list=Time_list,
                                                                                            exp_density=exp_density)
        cells_global_death_rate["Treatment"]=len(cells_global_death_rate.get("Time",[]))*[ replace_ugly_long_name(exp_name,exp_cell_line)]#exp_treatment]
        return cells_global_death_rate#[cells_global_death_rate, len(cells_global_death_rate)*exp_treatment]
    except FileNotFoundError:
        return {"Treatment":[], "Time":[], "Death rate":[]}
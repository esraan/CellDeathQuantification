import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def nucleationVsSPIPlot(df_path:str = 'SPI_DeathWaves_Results/Results/All/data_combined.csv',
                        savefig=False,
                        showfig=True):
    def encode_strings(arr_to_encode:np.array) -> np.array:
        _encodings = {}
        encoded_arr = np.zeros(arr_to_encode.shape[0])
        for idx, elem in enumerate(arr_to_encode):
            _encodings[elem[0]] = _encodings.get(elem[0], len(_encodings))
            encoded_arr[idx] = _encodings[elem[0]]
        return encoded_arr

    field_to_color_by = 'treatment'
    all_exp_summation_df = pd.read_csv(df_path)
    fig, ax = plt.subplots()

    unique_types = np.unique(all_exp_summation_df.loc[:, [field_to_color_by]].values)
    colors = plt.get_cmap('plasma', len(unique_types))
    markers = {
        'FAC&BSO':'>',
        "C'":'P',
        'ML162':'x',
        'else': '*'
    }
    marker_options = lambda name: markers['FAC&BSO'] if "FAC&BSO" in name else markers["C'"] if "C'" in name else markers['ML162'] if 'ML162' in name else markers['else']

    for idx, single_type in enumerate(unique_types):
        all_exp_res_under_type = all_exp_summation_df[all_exp_summation_df[field_to_color_by]==single_type]
        # Nucliation_results = all_exp_res_under_type.loc[:, ['nucliation_index']].values
        # SPI_results = all_exp_res_under_type.loc[:, ['spatial_propagation_index']].values
        Nucliation_results = all_exp_res_under_type.loc[:, ['nucliation_index']].values
        values_mask = Nucliation_results < 0.6
        Nucliation_results = Nucliation_results[values_mask]
        SPI_results = all_exp_res_under_type.loc[:, ['spatial_propagation_index']].values[values_mask]
        exps_names = all_exp_res_under_type.loc[:, ['name']].values
        ax.scatter(Nucliation_results, SPI_results, color=colors(idx), marker="{}".format(marker_options(single_type)), label=single_type)
    ax.set_title('SPI Vs Nucleation\nColor seperation by {}'.format(field_to_color_by))
    ax.set_xlabel('Nucleation Probability')
    ax.set_ylabel('SPI')
    ax.grid('on')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.05, 1))
    if savefig:
        fig.savefig('SPI Vs Nucleation Correct Nuc.png',  bbox_extra_artists=[lgd], bbox_inches='tight', dpi=200)
    if showfig:
        plt.show()


if __name__ == '__main__':
    nucleationVsSPIPlot(savefig=True, showfig=False)

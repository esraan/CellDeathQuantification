import pandas as pd
import os
import scipy.io

file_loc = "C:/Users/cheng/Documents/University/3rd/Lab/SampleData/"

experiments_data = pd.read_csv(file_loc + "File details.csv")
dataDir = file_loc + "data/"
experiments = []
for file in os.listdir(dataDir):
    data = scipy.io.loadmat(dataDir + file)

    original_XY = data["Centroids2"]
    n_instances = len(original_XY)
    x_array, y_array = [x[0] for x in original_XY], [x[1] for x in original_XY]

    num_of_dead_cells = data["numBlobs"]
    length = len(num_of_dead_cells)
    # extract times
    times = []
    for u in range(len(num_of_dead_cells)):
        for x in range(num_of_dead_cells[u][0]):
            times.append(u)

    original_data = pd.DataFrame(
        {
            'X': x_array,
            'Y': y_array,
            'times_of_death': times
        }
    )
    csv_name = file_loc + 'datacsv/' + file[0:-4] + '.csv'
    original_data.to_csv(csv_name)



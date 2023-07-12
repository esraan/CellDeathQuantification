import scipy.io

filepath = 'SimulationsOutputs/Tests/final_cells_physicell.mat'
data = scipy.io.loadmat(filepath)
print(data)
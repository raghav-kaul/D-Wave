import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import re
from distance_matrix_creator import *
from mpl_toolkits import mplot3d
import seaborn as sns

def extract_values_from_log(log_file):
    # Define regular expressions for extracting numerical values
    solver_regex = re.compile(r"solver finished in (\d+\.\d+) seconds")
    energy_regex = re.compile(r"sample energy = (-?\d+\.\d+)")

    # Initialize variables to store extracted values
    solver_time = None
    sample_energy = None

    # Open and read the contents of the log file
    with open(log_file, "r") as file:
        lines = file.readlines()

        # Iterate over each line in the file
        for line in lines:
            # Check if the line contains solver time information
            solver_match = solver_regex.search(line)
            if solver_match:
                solver_time = float(solver_match.group(1))

            # Check if the line contains sample energy information
            energy_match = energy_regex.search(line)
            if energy_match:
                sample_energy = float(energy_match.group(1))

    # Return the extracted numerical values
    return solver_time, abs(sample_energy)

# folder = 'Output_arrays'
# log_files = [folder+'/'+filename for filename in os.listdir(folder) if filename.endswith('.log')]
# npy_files = [folder+'/'+filename for filename in os.listdir(folder) if filename.endswith('.npy') and "layout" in filename]
solver_times = []
weights = []
for i in range(1,81):
    log_file = f'Output_arrays/v2_sweeps_output{i}.log'
    
    solver_time, abs_energy = extract_values_from_log(log_file)
    
    solver_times.append(solver_time)
    weights.append(abs_energy)


print(np.min(weights),np.max(weights))
print("best solution index = ",np.argmin(weights))
print("worst solution index = ",np.argmax(weights))


data = [0]*81
data = [data.copy() for i in range(13)]

for i in range(80):
    npy_file = f'Output_arrays/v2_{i+1}SA_layout_sweeps_400000_IF_10.npy'
    matrix= np.reshape(np.load(npy_file),81)
    for position, facility in enumerate(matrix):
        if facility != 0:
            data[int(facility-1)][int(position)] += 1
# print(data)
color = iter(cm.rainbow(np.linspace(0, 1, len(data))))
positions = [i for i in range(81)]

# fig = plt.figure()
# # ax = plt.axes(projection='3d')
# print('solver time = ',np.mean(solver_time))
# for i in range(len(data)):
#     # fig = plt.figure()
#     # ax = plt.axes()
#     # heights = np.reshape(data[i],(9,9))
#     heights = data[i]
#     plt.bar(x=positions,height=heights,color=next(color),label=f'Machine {i+1}',alpha = 0.7)
#     # x = np.arange(9)
#     # y = np.arange(9)
#     # xx,yy = np.meshgrid(x,y)
#     # top = heights
#     # bottom = np.zeros_like(top)
#     # width = depth = 1
    
#     # ax.bar3d(xx.flatten(),yy.flatten(),bottom,width,depth,top,shade=True, zsort = 'max',color=next(color))
#     # ax.set_alpha(0.8)
#     # ax.set_title(f'Machine {i+1}')
#     # # plt.legend()
#     plt.title(f"Functional Unit {i+1}")
#     plt.xlabel('Positions')
#     plt.ylabel('Occurences')
#     plt.legend()
#     # plt.gca().invert_xaxis()\
        
#     # if i % 3 ==0:
#     #     plt.show()
#     # plt.savefig(f"bar_Machine{i+1}")
# plt.show()

# facility_size = [10,4,7,4,2,1,14,2,14,3,1,3,2]
# # fig,axs = plt.subplots(5,3)
# # for i in range(len(data)):
# #     x = data[i]
# #     vmax = list(np.flip(np.sort(x)))[facility_size[i]]
        
# #     sns.heatmap(np.reshape(data[i],(9,9)),annot=np.reshape(data[i],(9,9)),cbar=True,vmax = vmax).set_title(f'Functional Unit {i+1}')
# #     plt.title(f"Heatmap of Occurences for function unit {i+1}")
# #     plt.savefig(f"heat {i+1}")
# #     plt.show()
# print(np.array(data))
# # transposed_data = np.array(data).T
# # sns.set_theme(style="whitegrid")
# # sns.boxenplot(data=transposed_data)
# # plt.show()



print("mean = ", np.mean(weights))
print('std dev = ', np.std(weights))
print("var = ", np.var(weights))
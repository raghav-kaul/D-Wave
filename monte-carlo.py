import numpy as np
import pandas as pd
import sys
import seaborn as sns
from distance_matrix_creator import rectdistancematrix
factory_rows = 9
factory_collumns = 9

flow_matrix = pd.read_csv("csv_files/flowmatrixbqm.csv", header = None)
flow_matrix = (flow_matrix.to_numpy()/3)**5
flow_matrix = np.round(flow_matrix)
flow_matrix = np.triu(flow_matrix)
flow_matrix = np.array(flow_matrix)
print(flow_matrix)
print(np.shape(flow_matrix))


distance_matrix = rectdistancematrix(factory_rows,factory_collumns)
# distance = np.round(distance,decimals=3)
distance_matrix = np.triu(np.array(distance_matrix))
distance_matrix = (np.array(distance_matrix))#**3)/5
distance_matrix = np.round(distance_matrix,decimals=3)
print(distance_matrix)
print(np.shape(distance_matrix))

facility_size = [10,4,7,4,2,1,14,2,14,3,1,3,2]

def calculate_micro_energy(i,j,ii,jj,distance,flow,binary_matrix):
    return binary_matrix[i][j]*binary_matrix[ii][jj]*(distance*flow)
        
def energy(binary_layout):
    energy = 0
    rows = len(binary_layout)
    cols = len(binary_layout[0])
    
    
    for i in range(rows):
        for j in range(cols):
            for ii in range(rows):
                for jj in range(cols):
                    if i == ii and j == jj:
                        pass
                    else:
                        distance = distance_matrix[j][jj]
                        flow = flow_matrix[i][ii]
                        energy += calculate_micro_energy(i,j,ii,jj,distance,flow,binary_layout)
    
    for i in range(len(binary_layout)):
        if np.sum(binary_layout[i]) != facility_size[i]:
            energy+= 300*abs(np.sum(binary_layout[i])-facility_size[i])
        else:
            pass
    
    for i in range(len(binary_layout.T)):
        if np.sum(binary_layout.T[i]) > 1:
            energy+= 200*(np.sum(binary_layout.T[i])-1)
        else:
            pass
    return energy

binary_layout = pd.read_csv('test.csv', index_col=0)
binary_layout = binary_layout.to_numpy()
print(energy(binary_layout))

iters = 10000000

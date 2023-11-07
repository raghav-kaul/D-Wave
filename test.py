import numpy as np
import pandas as pd
import sys
from distance_matrix_creator import distmatrix
np.set_printoptions(linewidth=1000, suppress=True,threshold= sys.maxsize)
flow = pd.read_csv("csv_files/flowmatrix.csv", header = None)
# flow = (flow.to_numpy()/3)**5
# flow = np.round(flow)
# flow = np.triu(flow)
flow = np.array(flow)
print(flow,"\n\n")


distance = distmatrix(9,9)
# distance = np.round(distance,decimals=3)
# distance = np.triu(np.array(distance))
distance = (np.array(distance))#**3)/5
distance = np.round(distance,decimals=3)
print(distance)
print(np.shape(distance))


Nfacil = 13
Npos = 81
matrL =9 
positions = []
facilities = [i for i in range(Nfacil)]
positions = [i for i in range(Npos)]

facility_size = [10,4,7,4,2,1,14,2,14,3,1,3,2]

x = [13,13,11,10,10,10,9,4,9,9,9,9,9,9,4,6,4,9,9,9,9,9,9,9,8,8,7,7,7,7,7,7,7,7,4,3,7,7,7,7,7,7,3,5,5,0,0,0,0,3,3,3,3,3,2,0,0,2,2,2,1,1,1,12,12,1,1,0,1,1,0,0,12,12,1,0,1,0,0,1,0]
# x = [13,11,10,10,9,6,10,4,4,9,9,9,9,9,9,9,9,9,9,9,13,9,9,8,8,7,7,7,7,7,0,7,7,7,7,4,7,7,7,7,7,0,4,5,5,0,0,0,0,3,3,3,3,3,12,0,12,0,3,3,2,0,2,2,2,0,1,1,1,1,1,1,12,1,1,0,0,0,1,0,1]
print(x)
print(len(x))

def energy(x):
    energy = 0
    hamiltonian = 0
    for j in range(len(x)):
        for jj in range(j+1,len(x)):
            delta = flow[x[j]-1][x[jj]-1] * distance[j][jj]*x[j]*x[jj]/(x[j]*x[jj]+1)
            hamiltonian += delta
    
    penalty = 0
    for i in facilities:
        diff = abs(x.count(i+1)-facility_size[i])
        penalty += diff

    energy = hamiltonian + penalty
    return energy
# energy(x)
print(energy(x))
import concurrent.futures
import time
import sys
from dimod import BinaryQuadraticModel
from dwave.system import LeapHybridSampler,EmbeddingComposite,DWaveSampler 
from dwave.samplers import SimulatedAnnealingSampler
import pandas as pd
import numpy as np
import seaborn as sns
from distance_matrix_creator import *
import matplotlib.pyplot as plt
import dwave.inspector
import sys

# np.set_printoptions(suppress=True,threshold= sys.maxsize, linewidth=1000,formatter={'float_kind':'{}'.format, 'all': lambda x: " {:.0f}. ".format(x)})
np.set_printoptions(linewidth=1000, suppress=True,threshold= sys.maxsize)
flow = pd.read_csv("csv_files/flowmatrixbqm.csv", header = None)
flow = ((flow.to_numpy())/3)**3
flow = np.round(flow)
flow = np.triu(flow)
flow = np.array(flow)


distance = squaredistmatrix(9)
distance = np.triu(np.array(distance))
distance = (np.array(distance))
distance = np.round(distance,decimals=3)
# print(distance)
# print(np.shape(distance))


Nfacil = 13
Npos = 81
matrL =9 
positions = []
facilities = [i for i in range(Nfacil)]
positions = [i for i in range(Npos)]

# print("Facilities = ", facilities)
# print("Positions = ",positions)

facility_size = [10,4,7,4,2,1,14,2,14,3,1,3,2]


# #Building a variable for each Machine

x = []
for f in facilities:
    x.append([f'F{f}P{p}' for p in positions])
# print(np.array(x))


# #Initialise BQM
bqm = BinaryQuadraticModel('BINARY')

# #Objective function
for f in range(len(facilities)):
    for f1 in range(len(facilities)):     
        for p in range(len(positions)):
            for p1 in range(len(positions)):
                if f == f1 and p == p1:
                    pass
                else:
                    bqm.add_quadratic(x[f][p],x[f1][p1],distance[p][p1]*flow[f][f1])


# #constraint 1: only 1 machine is placed per position

for p in positions:
    c1 = [(x[f][p],1) for f in facilities]
    bqm.add_linear_inequality_constraint(
        c1,
        ub = 1,
        lb = 0,
        lagrange_multiplier=200,
        label = 'c1_posi_' + str(p),
    )


# #Constraint 2: Each facility is given correct size
for f in facilities:
    c2 = [(x[f][p],1) for p in positions]
    bqm.add_linear_equality_constraint(
        c2,
        constant=-1*facility_size[f],
        lagrange_multiplier=300,
        # label = "c2_facil_" + str(f)
    )


# Removing 0 bias variables and couplers from BQM
new_bqm = BinaryQuadraticModel(bqm.linear, {interaction: bias for interaction, bias in bqm.quadratic.items() if bias}, bqm.offset, bqm.vartype)
file = open("txt_files/bqm.txt", "w")
file.write(str(bqm))
file.close()

file2 = open("txt_files/new_bqm.txt", "w")
file2.write(str(new_bqm))
file2.close()


# #running the solver

# sampler = LeapHybridSampler()
sampler = SimulatedAnnealingSampler()
numreads = 1 #number of times algorithm is applied
numsweeps = 1000 #number of metropolis updates for simulated annealer
timelimit = 60 #time limit for hybrid sampler

t1 = time.time()

#FOR HYBRID SAMPLER
# sampleset = sampler.sample(new_bqm, time_limit = timelimit)

#FOR SIMULATED ANNEALING
sampleset = sampler.sample(new_bqm, num_reads=numreads,num_sweeps=numsweeps)
t2 = time.time()
print(f'solver finished in {t2-t1} seconds') 


# Printing Output Solutions
t3 = time.time()

# dwave.inspector.show_bqm_sampleset(sampleset=sampleset)

### Post Processing the input matrix to convert to a more readable cartesian layout
printout = []
for f in facilities:
    printouttemp = []
    for p in positions:
        label = f'F{f}P{p}'
        value = sampleset.first.sample[label]
        printouttemp.append(value) 
    printout.append(printouttemp)

layout = np.zeros((matrL,matrL))
ctr = 1
for i in printout:
    for j in range(len(i)):
        if i[j] == 1:
            q = int(j/len(layout))
            r = j%len(layout)
            layout[q][r] = ctr
    ctr+=1

### Plotting the Layout using Heatmap
fig, ax = plt.subplots(figsize=(9, 10))
fig1 = sns.heatmap(layout,annot = layout, vmin = 0, vmax = len(facilities)).set(title = "Final Layout")
# plt.savefig(f'Images/v2_{sys.argv[1]}SA_layout_sweeps_{numsweeps}_IF_10.png')
plt.show()
### Check to ensure all facilities are assigned correct sizes
correct_size = True
printout = np.array(printout)
for i in range(len(printout)):
    true_size = facility_size[i]
    calc_size = np.sum(printout[i])
    if true_size == calc_size:
        print("Facility ", i+1, "\t correct size of", true_size, "\n")
    else:
        print("Facility ", i+1, "\t wrong size.", "True Size = ", true_size, "Calc size = ", calc_size,"\n")
        correct_size = False
if correct_size == True:
    print("all facilities Correct Size")

### Check to ensure same position is not assigned to two facilities
ctr1 = 0
for i in range(len(printout.T)):
    if np.sum(printout.T[i]) >1:
        print("Position ", i+1, " = Overlapped")
    else:
        ctr1+=1

if ctr1 == 81:
    print("No Overlaps")

fig, ax = plt.subplots(figsize=(17, 9))
fig2 = sns.heatmap(printout,annot = printout, vmax= 1, vmin=0 ).set(title = "Final Matrix")
# plt.savefig(f'Images/v2_{sys.argv[1]}SA_matrix_sweeps_{numsweeps}_IF_10.png')
plt.show()
print(f'sample energy = {sampleset.first.energy}')
# np.save(f'Output_arrays/v2_{sys.argv[1]}SA_matrix_sweeps_{numsweeps}_IF_10.npy',printout)
# np.save(f'Output_arrays/v2_{sys.argv[1]}SA_layout_sweeps_{numsweeps}_IF_10.npy',layout)
t4 = time.time()
print(f'postprocess finished in {t4-t3} seconds') 

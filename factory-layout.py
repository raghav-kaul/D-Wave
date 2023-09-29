from dimod import BinaryQuadraticModel
from dwave.system import LeapHybridSampler
import pandas as pd
import numpy as np

flow = pd.read_csv("flowmatrix.csv", header = None)
flow = (flow.to_numpy())*5
# flow = np.triu(flow)

distance = pd.read_csv("distancematrix.csv", header = None)
distance = distance.to_numpy()
# distance = np.triu(distance)

# print("flow Matrix\n", np.shape(flow), '\n')
# print("distance matrix\n", np.shape(distance))

facilities = [0,1,2,3,4,5,6,7,8,9,10,11,12]
positions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

#Building a variable for each Machine
x = []
for f in facilities:
    x.append([f'F{f}P{p}' for p in positions])

#Initialise BQM

bqm = BinaryQuadraticModel('BINARY')

#Objective function
for f in range(len(facilities)):
    for p in range(len(positions)):
        for ff in range(f+1,len(facilities)): 
            for pp in range(p+1,len(positions)):
                bqm.add_quadratic(x[f][p],x[ff][pp],distance[p][pp]*flow[f][ff])

#constraint 1: only 1 machine is placed per position

for p in positions:
    c1 = [(x[f][p],1) for f in facilities]
    bqm.add_linear_equality_constraint(
        c1,
        constant = -1,
        lagrange_multiplier=1
    )

#Constraint 2: only 1 position is chosen per facility
for f in facilities:
    c2 = [(x[f][p],1) for p in positions]
    bqm.add_linear_equality_constraint(
        c2,
        constant=-1,
        lagrange_multiplier=10
    )

#running the solver
sampler = LeapHybridSampler(time_limit = 5)
sampleset = sampler.sample(bqm)

for f in facilities:
    printout = str(f) + "\t"
    for p in positions:
        label = f'F{f}P{p}'
        value = sampleset.first.sample[label]
        printout += str(value)
    print(printout)
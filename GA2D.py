import random as r
import pandas as pd
import numpy as np
import math as m
import cProfile
"""
receiving dep = 1
raw material storage = 2
crushing dept = 3
peeling dept = 4
chop mix dept = 5
chop dept = 6
form cook dept = 7
mix cook dept = 8
blasting dept = 9
packaging dept = 10
filling dept = 11
food court = 12
finished goods = 13
"""

###Importing Relationship Matrix
rel = pd.read_csv("relmatrix.csv", header=None)
rel = rel.to_numpy()
# print(rel)
Total_possible_positions = 16
deptlist = [1,2,3,4,5,6,7,8,9,10,11,12,13]
L = len(deptlist)
free_spaces = Total_possible_positions - L

file = open("Best_Individuals", "w+")
    
def neighborlist(x,y):
    xu = x+1
    xd = x-1
    yu = y+1
    yd = y-1
    nlist = [(xu,y),(xd,y),(x,yu),(x,yd),(xu,yu),(xu,yd),(xd,yu),(xd,yd)]
    nlist = [i for i in nlist if all(np.array(i)>=0) and all(np.array(i)<4)]
    return nlist

def fitness(popl):
    fitness = 0
    for i in range(4):
        for j in range(4):
            if popl[i][j] !=0:
                nlist = neighborlist(i,j)
                for x in nlist:
                    if popl[x[0]][x[1]] != 0:
                        X = int(popl[i][j] - 1)
                        Y = int(popl[x[0]][x[1]] - 1)
                        if rel[X][Y] == 1:
                            fitness += -20
                        else:
                            fitness += rel[X][Y]
    return fitness

def mutate(ind):
    ind2 = ind.copy()
    i = r.randint(0,3)
    j = r.randint(0,3)
    nlist = neighborlist(i,j)
    switch = r.sample(nlist,1)
    ind2[i][j],ind2[switch[0][0]][switch[0][1]] = ind2[switch[0][0]][switch[0][1]],ind2[i][j]
    return ind2

def initpopl(popl_size, popl = []):

    s = deptlist+list(np.zeros((free_spaces)))
    for k in range(popl_size):
        l=np.array(r.sample(s,len(s)))
        popl.append(l.reshape((4,4)))
    return (np.array(popl))

###Initializing first generation (Randomly Generated)

population_size = 500
population = initpopl(population_size)

print("Initial Population\n", np.array(population))

###GENETIC ALGORITHM

alpha = 0

for gen in range(2000):
    print("GENERATION ", gen+1)
    
    ###Finding Fitness of Each Individual in Population
    scorelist = []
    for ind in range(population_size):
        score = fitness(population[ind])
        scorelist.append([score,ind])


    scorelist = np.array(scorelist)

    sorted_score = scorelist[scorelist[:,0].argsort()[::-1]]
    best_inds = []
    for i in range(population_size):
        if sorted_score[i][0] > 262:
            # best_inds.append([population[sorted_score[i][1]]])
            file.write(str(population[sorted_score[i][1]]) + "\n")
            file.write("fitness = " + str(sorted_score[i][0])+"\n\n")
    # print("sorted score\n", sorted_score)
    # print("scorelist \n", scorelist)

### taking top 25% of total population and mutating each individual 4 times
    n = int(population_size/2)
    print("Best layout and fitness in current gen")
    print(population[sorted_score[0][1]])
    print("fitness = ",sorted_score[0][0])
    new_population = []
    for i in range(n):
        index = sorted_score[i][1]
        for j in range(4):
            new_ind = mutate(population[index])
            new_population.append(new_ind)
    population = new_population.copy()

file.close()
print(best_inds)
"""

popl = [[10,  0,  0, 13,],
 [ 8, 11,  0, 12],
 [ 9,  6,  4,  7],
 [ 5,  3,  2,  1]]

print(fitness(popl))

"""
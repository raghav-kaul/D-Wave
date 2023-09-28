import random as r
import pandas as pd
import numpy as np
"""
receiving dep = 0
raw material storage = 1
crushing dept = 2
peeling dept = 3
chop mix dept = 4
chop dept = 5
form cook dept = 6
mix cook dept = 7
blasting dept = 8
packaging dept = 9
filling dept = 10
food court = 11
finished goods = 12
"""
###Importing Relationship Matrix
rel = pd.read_csv("relmatrix.csv", index_col = 0)
rel = rel.to_numpy()
# print(rel)

deptlist = [0,1,2,3,4,5,6,7,8,9,10,11,12]
length = len(deptlist)
def scorer(i,j):
    if rel[i][j] == 1:
         return 10
    else:
         return -1/rel[i][j]
    

def fitness(soln, alpha):
    fitness = 0
    length = len(soln) - 1

    if soln[0] != 0:
        fitness +=15
    if soln[12] != 12:
        fitness +=15

    for i in range(0,length-1):
        fitness += alpha*scorer(soln[i],soln[i+1])
    return fitness

def mutate(soln):
    newsoln = soln
    i = r.randint(1,11)
    j = r.randint(1,11)
    return newsoln

###Initializing first generation (Randomly Generated)


population_size = 500
population = []

for i in range(population_size):
    layout = r.sample(deptlist,length)
    population.append(layout)


###GENETIC ALGORITHM

alpha = 5

for gen in range(10000):
    print("GENERATION", gen+1)
    
    # print ("\n\nGen #",gen,"Population \n", population)

    #Finding Fitness of Each Individual in Population
    scorelist = []
    for soln in population:
        score = fitness(soln,alpha)
        scorelist.append(score)

    # print("scorelist \n", scorelist)
    scorel_sorted = np.sort(scorelist)

    #Taking top n performing individuals (n=2 as of now) and mutating them for next generation
    n = 2
    new_pop = []
    print("Best layout and score of Current Gen\n", population[scorelist.index(scorel_sorted[0])], scorel_sorted[0])
    if scorel_sorted[0] < -43.74:
        break
    for j in range(int(population_size/n)):
        for i in range(n):
            best_sc = scorel_sorted[i]
            next_gen_ind = mutate(population[scorelist.index(best_sc)])
            new_pop.append(next_gen_ind)

    population = new_pop



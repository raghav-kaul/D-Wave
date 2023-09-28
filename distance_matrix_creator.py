import numpy as np


def distance(i,j,ii,jj):
    d = np.sqrt((i-ii)**2+(j-jj)**2)

    return d

def distmatrix(row,column):
    distmat = []
    for i in range(row):
        for j in range(column):
            Dist = []
            for ii in range(row):
                for jj in range(column):
                    dist = distance(i,j,ii,jj)
                    Dist.append(dist)
            distmat.append(Dist)
    return distmat


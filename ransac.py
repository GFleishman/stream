#!/usr/env/python3

# Interest-point based RANSAC
# Initial correspondence found by correlation
# Author: TW
# Date: June 2019

import numpy as np
import cv2, scipy.ndimage, pickle, subprocess, os, sys


def stats(arr):
    means = np.mean(arr, axis=1)
    sqr_means = np.mean(np.square(arr), axis=1)
    stddevs = np.sqrt( sqr_means - np.square(means) )
    return means, stddevs

def correlate(Acent, Astd, Bcent, Bstd):
    correlations = np.empty((len(Astd), len(Bstd)))
    for i in range(len(Astd)):
        correlations[i] = np.mean(Acent[i] * Bcent, axis=1) / Bstd / Astd[i]
    return correlations

def bestPairs(Apos, Bpos, correlations, threshold):
    bestIndcs = np.argmax(correlations, axis=1)
    AIndcs = range(len(Apos))
    keeps = correlations[(AIndcs, bestIndcs)] > threshold
    return Apos[keeps], Bpos[bestIndcs[keeps]]

def get_PKL(filename):
    f=open(filename,'rb')
    PKL=pickle.load(f)
    f.close()
    return(PKL)


cutoff = 0.9
threshold = 2.5 # px error
f1,f2=sys.argv[1:3]

f1 = get_PKL(f1)
f2 = get_PKL(f2)

if len(f1) <= 100 or len(f2) <= 100:
    print("Insufficient spots detected for ransac; writing identity matrix")
    Aff = np.eye(4)[:3]

else:
    Apos = np.array( [x[0] for x in f1] )
    Acon = np.array( [x[1].flatten() for x in f1] )
    Bpos = np.array( [x[0] for x in f2] )
    Bcon = np.array( [x[1].flatten() for x in f2] )
    
    Amean, Astd = stats(Acon)
    Acent = Acon - Amean[..., None]
    Bmean, Bstd = stats(Bcon)
    Bcent = Bcon - Bmean[..., None]
    correlations = correlate(Acent, Astd, Bcent, Bstd)

    pA, pB = bestPairs(Apos, Bpos, correlations, cutoff)
    if len(pA) <= 50 or len(pB) <= 50:
        print("Insufficient spots correlated for ransac; writing identity matrix")
        Aff = np.eye(4)[:3]
    else:
        r, Aff, inline = cv2.estimateAffine3D(pA, pB, ransacThreshold=threshold, confidence=0.999)
        if (np.diag(Aff) < 0.1).any():
            print("Degenerate affine produced; writing identity matrix")
            Aff = np.eye(4)[:3]

np.savetxt(sys.argv[3],Aff,fmt='%.6f',delimiter=' ')


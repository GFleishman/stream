#!/usr/env/python3

# Interest-point based RANSAC
# Initial correspondence found by correlation
# Author: TW
# Date: June 2019

import numpy as np
import cv2, scipy.ndimage, pickle, subprocess, os, sys
import time


def stats(arr):
    means = np.mean(arr, axis=1)
    sqr_means = np.mean(np.square(arr), axis=1)
    stddevs = np.sqrt( sqr_means - np.square(means) )
    return means, stddevs

def correlate(Acent, Astd, Bcent, Bstd):
    correlations = np.empty((len(Astd), len(Bstd)))
    for i in range(len(Astd)):
        t0 = time.clock()  # debug
        correlations[i] = np.mean(Acent[i] * Bcent, axis=1) / Bstd / Astd[i]
    return correlations

def bestPairs(Apos, Bpos, correlations, threshold):
    bestIndcs = np.argmax(correlations, axis=1)
    AIndcs = range(len(Apos))
    keeps = correlations[[AIndcs, bestIndcs]] > threshold
    return Apos[keeps], Bpos[bestIndcs[keeps]]

def get_PKL(filename):
    f=open(filename,'rb')
    PKL=pickle.load(f)
    f.close()
    return(PKL)

if len(sys.argv)==1:
    for i,f1 in enumerate(os.listdir('.')):
        for j,f2 in enumerate(os.listdir('.')):
            if i<j and f1!=f2 and f1.endswith('.pkl') and f2.endswith('.pkl'):
                cmd=['python',__file__,f1,f2,f1.split('.')[0],f2.split('.')[0]]
                print(' ' .join(cmd))
                output = subprocess.Popen(cmd, shell=True)                
else:
    threshold=2.5 # px error
    cutoff=0.9
    f1,f2=sys.argv[1:3]

    f1 = get_PKL(f1)
    Apos = np.array( [x[0] for x in f1] )
    Acon = np.array( [x[1].flatten() for x in f1] )
    f2 = get_PKL(f2)
    Bpos = np.array( [x[0] for x in f2] )
    Bcon = np.array( [x[1].flatten() for x in f2] )

    Amean, Astd = stats(Acon)
    Acent = Acon - Amean[..., None]
    Bmean, Bstd = stats(Bcon)
    Bcent = Bcon - Bmean[..., None]

    correlations = correlate(Acent, Astd, Bcent, Bstd)
    pA, pB = bestPairs(Apos, Bpos, correlations, cutoff)

    r,Aff,inline=cv2.estimateAffine3D(pA,pB, ransacThreshold=threshold, confidence=0.999)
    numMatch=sum(inline.T[0])
    np.savetxt(sys.argv[3],Aff,fmt='%.6f',delimiter=' ')
        

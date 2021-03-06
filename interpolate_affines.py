import sys
import numpy as np
import glob
from itertools import product
from os.path import dirname


def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
        index  = np.array(f.readline().split(' ')).astype(np.int16)
    return offset, extent, index


def get_neighbors(tiledir, index, suffix='/*/coords.txt'):

    bin_strs = ['-100', '100', '0-10', '010', '00-1', '001'] 
    neighbors = {}
    tiles = glob.glob(tiledir + suffix)
    for tile in tiles:
        oo, ee, ii = read_coords(tile)
        key = ''.join( [str(i) for i in ii - index] )
        if key in bin_strs: neighbors[key] = dirname(tile)
    return neighbors


tiledir = sys.argv[1]
tiles = glob.glob(tiledir + '/*[0-9]')
tiles.sort()
no_updates = False

while not no_updates:

    no_updates = True
    for tile in tiles:
        affine = np.loadtxt(tile + '/ransac_affine.mat')
        if (affine == np.eye(4)[:3]).all():
            no_updates = False
            oo, ee, ii = read_coords(tile + '/coords.txt')
            neighbors = get_neighbors(tiledir, ii)
    
            for k in neighbors.keys():
                neighbor_affine = np.loadtxt(neighbors[k] + '/ransac_affine.mat')
                affine[:, -1] += neighbor_affine[:, -1]
            affine[:, -1] /= len(neighbors.keys())
            np.savetxt(tile + '/ransac_affine.mat', affine, fmt='%.6f', delimiter=' ')


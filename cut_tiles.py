#!/groups/scicompsoft/home/fleishmang/bin/miniconda3/bin/python3

import z5py
import json
import sys
import numpy as np
from os.path import abspath
from os import makedirs


def read_reference_grid(n5_path, subpath):
    """Read grid dimensions from n5 file"""
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    return tuple(atts['dimensions'])


def read_n5_spacing(n5_path, subpath):
    """Read voxel spacing from n5 file"""
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    return np.absolute(np.array(atts['pixelResolution']) * np.array(atts['downsamplingFactors']))


def write_coords_file(path, offset, extent, row, column):
    with open(path, 'w') as f:
        print(*offset, file=f)
        print(*extent, file=f)
        print(row, column, file=f)


if __name__ == '__main__':

    ref_img_path        = abspath(sys.argv[1])
    ref_img_subpath     = sys.argv[2]
    tiles_dir           = abspath(sys.argv[3])
    stride              = int(sys.argv[4])
    overlap             = int(sys.argv[5])


    vox       = read_n5_spacing(ref_img_path, ref_img_subpath)
    grid      = read_reference_grid(ref_img_path, ref_img_subpath) * vox
    stride        = stride * vox[:-1]  # for now, using entire z-depth
    overlap       = overlap * vox[:-1]
    extent        = np.array([stride[0], stride[1], grid[-1]])
    offset        = np.array([0., 0., 0.])
    tile_counter  = 0
    column        = 0
    row           = 0

    
    while offset[0] < grid[0]:
        while offset[1] < grid[1]:
            
            ttt = tiles_dir + '/' + str(tile_counter)
            makedirs(ttt, exist_ok=True)

            local_extent = np.minimum(extent, grid - offset)
            write_coords_file(ttt + '/coords.txt', offset, local_extent, row, column)

            row += 1
            tile_counter += 1
            offset[1] += stride[1] - overlap[1]

        row = 0
        column += 1
        offset[1] = 0.
        offset[0] += stride[0] - overlap[0]


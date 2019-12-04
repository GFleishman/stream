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


def write_coords_file(path, offset, extent, index):
    with open(path, 'w') as f:
        print(*offset, file=f)
        print(*extent, file=f)
        print(*index, file=f)


if __name__ == '__main__':

    ref_img_path           = abspath(sys.argv[1])
    ref_img_subpath        = sys.argv[2]
    tiles_dir              = abspath(sys.argv[3])
    xy_stride              = int(sys.argv[4])
    xy_overlap             = int(sys.argv[5])
    z_stride               = int(sys.argv[6])
    z_overlap              = int(sys.argv[7])


    vox           = read_n5_spacing(ref_img_path, ref_img_subpath)
    grid          = read_reference_grid(ref_img_path, ref_img_subpath) * vox
    stride        = np.array([xy_stride, xy_stride, z_stride]) * vox
    overlap       = np.array([xy_overlap, xy_overlap, z_overlap]) * vox
    offset        = np.array([0., 0., 0.])
    index         = np.array([0, 0, 0])
    tile_counter  = 0

    
    while offset[2] + overlap[2] < grid[2]:
        while offset[0] + overlap[0] < grid[0]:
            while offset[1] + overlap[1] < grid[1]:
            
                ttt = tiles_dir + '/' + str(tile_counter)
                makedirs(ttt, exist_ok=True)
    
                local_extent = np.minimum(stride, grid - offset)
                write_coords_file(ttt + '/coords.txt', offset, local_extent, index)
    
                index[0] += 1
                tile_counter += 1
                offset[1] += stride[1] - overlap[1]

            index[0] = 0
            index[1] += 1
            offset[1] = 0.
            offset[0] += stride[0] - overlap[0]

        index[0] = 0
        index[1] = 0
        index[2] += 1
        offset[1] = 0
        offset[0] = 0
        offset[2] += stride[2] - overlap[2]
    

import sys
import numpy as np
import glob
import nrrd
from os.path import dirname, isfile
import z5py
import json
import gc


# READERS
def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
        index  = np.array(f.readline().split(' ')).astype(np.uint16)
    return offset, extent, index


def read_fields(neighbors, suffix):
    fields = {}
    for key in neighbors.keys():
        if neighbors[key] and key != 'pos':
            if isfile(neighbors[key] + suffix):
                fields[key], m = nrrd.read(neighbors[key] + suffix)
    return fields


def read_n5_spacing(n5_path, subpath):
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    vox = np.absolute(np.array(atts['pixelResolution']) * np.array(atts['downsamplingFactors']))
    return vox.astype(np.float64)


def read_n5_reference_grid(n5_path, subpath):
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    return tuple(atts['dimensions'])


# WRITERS
def create_n5_dataset(n5_path, subpath, sh, xy_overlap):
    n5im = z5py.File(n5_path, use_zarr_format=False)
    try:
        n5im.create_dataset('/c0'+subpath, shape=sh[::-1], chunks=(70, xy_overlap, xy_overlap), dtype=np.float32)
        n5im.create_dataset('/c1'+subpath, shape=sh[::-1], chunks=(70, xy_overlap, xy_overlap), dtype=np.float32)
        n5im.create_dataset('/c2'+subpath, shape=sh[::-1], chunks=(70, xy_overlap, xy_overlap), dtype=np.float32)
    except:
        pass
    return n5im


def write_updated_transform(n5im, subpath, updated_warp, oo):
    extent = np.array(updated_warp.shape[:-1])
    ee = oo + extent
    utx = np.moveaxis(updated_warp[..., 0], (0, 2), (2, 0))
    uty = np.moveaxis(updated_warp[..., 1], (0, 2), (2, 0))
    utz = np.moveaxis(updated_warp[..., 2], (0, 2), (2, 0))
    n5im['/c0'+subpath][oo[2]:ee[2], oo[1]:ee[1], oo[0]:ee[0]] = utx
    n5im['/c1'+subpath][oo[2]:ee[2], oo[1]:ee[1], oo[0]:ee[0]] = uty
    n5im['/c2'+subpath][oo[2]:ee[2], oo[1]:ee[1], oo[0]:ee[0]] = utz


def copy_metadata(ref_path, ref_subpath, out_path, out_subpath):
    with open(ref_path + ref_subpath + '/attributes.json') as atts:
        ref_atts = json.load(atts)
    with open(out_path + out_subpath + '/attributes.json') as atts:
        out_atts = json.load(atts)
    for k in ref_atts.keys():
        if k not in out_atts.keys():
            out_atts[k] = ref_atts[k]
    with open(out_path + out_subpath + '/attributes.json', 'w') as atts:
        json.dump(out_atts, atts)


# FOR AFFINE TRANSFORM
def position_grid(sh, dtype=np.uint16):
    """Return a position array in physical coordinates with shape sh"""
    coords = np.array(np.meshgrid(*[range(x) for x in sh], indexing='ij'), dtype=dtype)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def transform_grid(matrix, grid):
    """Apply affine matrix to position grid"""
    mm = matrix[:, :-1]
    tt = matrix[:, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt


# HANDLE OVERLAPS
def get_neighbors(tiledir, index, suffix='/*/coords.txt'):
    
    neighbors = { 'center':False,
                  'right':False,
                  'bottom':False,
                  'right-bottom':False }

    tiles = glob.glob(tiledir + suffix)
    for tile in tiles:
        oo, ee, ii = read_coords(tile)
        tile = dirname(tile)
        index_diff = ii - index
        if   (index_diff == [0, 0]).all(): neighbors['center'] = tile
        elif (index_diff == [1, 0]).all(): neighbors['bottom'] = tile
        elif (index_diff == [0, 1]).all(): neighbors['right'] = tile
        elif (index_diff == [1, 1]).all(): neighbors['right-bottom'] = tile
    return neighbors


def reconcile_warps(lcc, warps, xy_overlap):

    cs = slice(-xy_overlap, None)
    os = slice(None, xy_overlap)
    if 'right' in lcc.keys() and 'right' in warps.keys():
        updates = lcc['center'][cs] < lcc['right'][os]
        warps['center'][cs][updates] = warps['right'][os][updates]
    if 'bottom' in lcc.keys() and 'bottom' in warps.keys():
        updates = lcc['center'][:, cs] < lcc['bottom'][:, os]
        warps['center'][:, cs][updates] = warps['bottom'][:, os][updates]
    if 'right-bottom' in lcc.keys() and 'right-bottom' in warps.keys():
        corner = np.maximum.reduce([lcc['center'][cs, cs], lcc['right'][os, cs],
                                    lcc['bottom'][cs, os], lcc['right-bottom'][os, os]])
        updates = corner == lcc['right'][os, cs]
        warps['center'][cs, cs][updates] = warps['right'][os, cs][updates]
        updates = corner == lcc['bottom'][cs, os]
        warps['center'][cs, cs][updates] = warps['bottom'][cs, os][updates]
        updates = corner == lcc['right-bottom'][os, os]
        warps['center'][cs, cs][updates] = warps['right-bottom'][os, os][updates]
    return warps['center']


# TODO: 
#       modify to accommodate overlaps in Z
#       add simpler overlap reconciliation methods: averaging, weighted averaging
if __name__ == '__main__':

    tile            = sys.argv[1]
    xy_overlap      = int(sys.argv[2])
    z_overlap       = sys.argv[3]
    reference       = sys.argv[4]
    ref_subpath     = sys.argv[5]
    global_affine   = sys.argv[6]
    output          = sys.argv[7]
    invoutput       = sys.argv[8]
    output_subpath  = sys.argv[9]


    # read basic elements
    tiledir = dirname(tile)
    vox = read_n5_spacing(reference, ref_subpath)
    offset, extent, index = read_coords(tile + '/coords.txt')

    # initialize updated warp fields with global affine
    matrix = np.float32(np.loadtxt(global_affine))
    grid = np.round(extent/vox).astype(np.uint16)
    grid = position_grid(grid) * vox + offset
    updated_warp = transform_grid(matrix, grid)

    inv_matrix = np.array([ matrix[0],
                            matrix[1],
                            matrix[2],
                            [0, 0, 0, 1] ])
    inv_matrix = np.linalg.inv(inv_matrix)[:-1]
    updated_invwarp = transform_grid(inv_matrix, grid)
    del grid; gc.collect()

    # handle overlap regions
    neighbors = get_neighbors(tiledir, index)
    if isfile(neighbors['center']+'/final_lcc.nrrd'):
        lcc = read_fields(neighbors, suffix='/final_lcc.nrrd')
        for key in lcc.keys():
            lcc[key][lcc[key] > 1.0] = 0  # typically in noisy regions
        warps = read_fields(neighbors, suffix='/warp.nrrd')
        updated_warp += reconcile_warps(lcc, warps, xy_overlap)
        del warps; gc.collect()  # need space for inv_warps
        inv_warps = read_fields(neighbors, suffix='/invwarp.nrrd')
        updated_invwarp += reconcile_warps(lcc, inv_warps, xy_overlap)
        
    # OPTIONAL: SMOOTH THE OVERLAP REGIONS
    # OPTIONAL: USE WEIGHTED COMBINATION BASED ON LCC AT ALL VOXELS

    # update offset to avoid writing left and top overlap regions
    oo = np.round(offset/vox).astype(np.uint16)
    if index[0] != 0 and index[1] != 0:
        updated_warp = updated_warp[xy_overlap:, xy_overlap:]
        updated_invwarp = updated_invwarp[xy_overlap:, xy_overlap:]
        oo[0:2] += xy_overlap
    elif index[1] == 0:
        updated_warp = updated_warp[:, xy_overlap:]
        updated_invwarp = updated_invwarp[:, xy_overlap:]
        oo[1] += xy_overlap
    elif index[0] == 0:
        updated_warp = updated_warp[xy_overlap:, :]
        updated_invwarp = updated_invwarp[xy_overlap:, :]
        oo[0] += xy_overlap

    # write results
    ref_grid = read_n5_reference_grid(reference, ref_subpath)
    n5im = create_n5_dataset(output, output_subpath, ref_grid, xy_overlap)
    write_updated_transform(n5im, output_subpath, updated_warp, oo)
    n5im = create_n5_dataset(invoutput, output_subpath, ref_grid, xy_overlap)
    write_updated_transform(n5im, output_subpath, updated_invwarp, oo)


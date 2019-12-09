import sys
import numpy as np
import glob
import nrrd
from os.path import dirname, isfile
import z5py
import json
import gc
from itertools import product


# READERS
def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
        index  = np.array(f.readline().split(' ')).astype(np.uint16)
    return offset, extent, index


def read_fields(neighbors, suffix):
    fields = {}
    keys = neighbors.keys()
    for key in keys.sort():
        if neighbors[key]:
            if isfile(neighbors[key] + suffix):
                fields[key], m = nrrd.read(neighbors[key] + suffix)
        else:
            fields[key] = np.zeros_like( fields['000'] )
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
def create_n5_dataset(n5_path, subpath, sh, xy_overlap, z_overlap):
    n5im = z5py.File(n5_path, use_zarr_format=False)
    try:
        n5im.create_dataset('/c0'+subpath, shape=sh[::-1], 
                            chunks=(z_overlap, xy_overlap, xy_overlap), dtype=np.float32)
        n5im.create_dataset('/c1'+subpath, shape=sh[::-1],
                            chunks=(z_overlap, xy_overlap, xy_overlap), dtype=np.float32)
        n5im.create_dataset('/c2'+subpath, shape=sh[::-1],
                            chunks=(z_overlap, xy_overlap, xy_overlap), dtype=np.float32)
    except:
        # TODO: should only pass if it's a "File already exists" exception
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

    bin_strs = [''.join(p) for p in product('10', repeat=3)]
    neighbors = { a:b for a, b in zip(bin_strs, [False,]*8) }
    tiles = glob.glob(tiledir + suffix)
    for tile in tiles:
        oo, ee, ii = read_coords(tile)
        key = ''.join( [str(i) for i in ii - index] )
        if key in neighbors.keys(): neighbors[key] = dirname(tile)
    return neighbors


def slice_dict(step, xy_overlap, z_overlap):

    a = slice(None, None)
    b = slice(-xy_overlap, None)
    c = slice(None, xy_overlap)
    d = slice(-z_overlap, None)
    e = slice(None, z_overlap)

    if step == 1:
        return { '100':{'000':[a, b, a], '100':[a, c, a]},
                 '010':{'000':[b, a, a], '010':[c, a, a]},
                 '001':{'000':[a, a, d], '001':[a, a, e]} }
    if step == 2:
        return { '110':{'000':[b, b, a], '100':[b, c, a], '010':[c, b, a], '110':[c, c, a]},
                 '101':{'000':[a, b, d], '001':[a, b, e], '100':[a, c, d], '101':[a, c, e]},
                 '011':{'000':[b, a, d], '010':[c, a, d], '001':[b, a, e], '011':[c, a, e]} }
    if step == 3:
        return { '000':[b, b, d], '100':[b, c, d], '010':[c, b, d], '001':[b, b, e],
                 '111':[c, c, e], '110':[c, c, d], '101':[b, c, e], '011':[c, b, e] }


def reconcile_one_step_neighbor(lcc, warps, bin_str, xy_overlap, z_overlap):

    SD = slice_dict(1, xy_overlap, z_overlap)[bin_str]
    updates = lcc['000'][ SD['000'] ] < lcc[bin_str][ SD[bin_str] ]
    warps['000'][ SD['000'] ][updates] = warps[bin_str][ SD[bin_str] ][updates]
    return warps['000']


def reconcile_two_step_neighbor(lcc, warps, bin_str, xy_overlap, z_overlap):

    SD = slice_dict(2, xy_overlap, z_overlap)[bin_str]
    corner = np.maximum.reduce( [lcc[k][ SD[k] ] for k in SD.keys()] )
    for key in SD.keys():
        updates = corner == lcc[key][ SD[key] ]
        warps['000'][ SD['000'] ][updates] = warps[key][ SD[key] ][updates]
    return warps['000']


def reconcile_three_step_neighbor(lcc, warps, bin_str, xy_overlap, z_overlap):

    SD = slice_dict(3, xy_overlap, z_overlap)
    corner = np.maximum.reduce( [lcc[k][ SD[k] ] for k in SD.keys()] )
    for key in SD.keys():
        updates = corner == lcc[key][ SD[key] ]
        warps['000'][ SD['000'] ][updates] = warps[key][ SD[key] ][updates]
    return warps['000']


def reconcile_warps(lcc, warps, xy_overlap, z_overlap):

    bin_strs = [''.join(p) for p in product('10', repeat=3)]
    for bin_str in bin_strs:
        bin_str_array = np.array( [int(i) for i in bin_str] )
        if np.sum(bin_str_array) == 1:
            warps['000'] = reconcile_one_step_neighbor(lcc, warps, bin_str, xy_overlap, z_overlap)
    for bin_str in bin_strs:
        bin_str_array = np.array( [int(i) for i in bin_str] )
        if np.sum(bin_str_array) == 2:
            warps['000'] = reconcile_two_step_neighbor(lcc, warps, bin_str, xy_overlap, z_overlap)
    for bin_str in bin_strs:
        bin_str_array = np.array( [int(i) for i in bin_str] )
        if np.sum(bin_str_array) == 3:
            warps['000'] = reconcile_three_step_neighbor(lcc, warps, bin_str, xy_overlap, z_overlap)
    return warps['000']


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
    elif index[1] != 0:
        updated_warp = updated_warp[:, xy_overlap:]
        updated_invwarp = updated_invwarp[:, xy_overlap:]
        oo[1] += xy_overlap
    elif index[0] != 0:
        updated_warp = updated_warp[xy_overlap:, :]
        updated_invwarp = updated_invwarp[xy_overlap:, :]
        oo[0] += xy_overlap

    if index[2] != 0:
        updated_warp = updated_warp[..., z_overlap:]
        updated_invwarp = updated_invwarp[..., z_overlap:]
        oo[2] += z_overlap

    # write results
    ref_grid = read_n5_reference_grid(reference, ref_subpath)
    n5im = create_n5_dataset(output, output_subpath, ref_grid, xy_overlap, z_overlap)
    write_updated_transform(n5im, output_subpath, updated_warp, oo)
    n5im = create_n5_dataset(invoutput, output_subpath, ref_grid, xy_overlap, z_overlap)
    write_updated_transform(n5im, output_subpath, updated_invwarp, oo)


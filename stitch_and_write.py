import sys
import numpy as np
import glob
import nrrd
from os.path import dirname, isfile
import z5py
import json
import gc


def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
        index  = np.array(f.readline().split(' ')).astype(np.uint16)
    return offset, extent, index


def get_neighbors(tiledir, offset, extent, suffix='/*/coords.txt'):
    """inspect coords files, identify position and neighbors"""

    neighbors = {}
    keys = ['center', 'right', 'bottom', 'right-bottom', 'pos']
    for k in keys: neighbors[k] = False
    if (offset[:2] == [0, 0]).all(): neighbors['pos'] = 'top-left'
    elif offset[0] == 0: neighbors['pos'] = 'left'
    elif offset[1] == 0: neighbors['pos'] = 'top'

    tiles = glob.glob(tiledir + suffix)
    between = lambda x, lb, ub: x >= lb and x <= ub
    for tile in tiles:
        oo, ee, ii = read_coords(tile)
        tile = dirname(tile)
        if np.prod(list(map(between, oo, offset, offset+extent))):
            if (oo[:2] == offset[:2]).all(): neighbors['center'] = tile
            elif oo[0] == offset[0]: neighbors['bottom'] = tile
            elif oo[1] == offset[1]: neighbors['right'] = tile
            else: neighbors['right-bottom'] = tile
    return neighbors


def read_fields(neighbors, suffix):

    fields = {}
    for key in neighbors.keys():
        if neighbors[key] and key != 'pos':
            if isfile(neighbors[key] + suffix):
                fields[key], m = nrrd.read(neighbors[key] + suffix)
    return fields


def reconcile_warps(lcc, warps, overlap):

    cs = slice(-overlap, None)
    os = slice(None, overlap)
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





def read_n5_spacing(n5_path, subpath):
    # get the voxel spacing
    # metadata is properly (x, y, z)
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    vox = np.absolute(np.array(atts['pixelResolution']) * np.array(atts['downsamplingFactors']))
    return vox.astype(np.float64)


def position_grid(sh, dtype=np.uint16):
    """Return a position array in physical coordinates with shape sh"""
    coords = np.array(np.meshgrid(*[range(x) for x in sh], indexing='ij'), dtype=dtype)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def transform_grid(matrix, grid):
    """Apply affine matrix to position grid"""
    mm = matrix[:, :-1]
    tt = matrix[:, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt


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


def read_matrix(matrix_path):
    """read affine matrix from file to array"""
    matrix = np.loadtxt(matrix_path)
    return np.float32(matrix)


def create_n5_dataset(n5_path, subpath, sh, overlap):
    n5im = z5py.File(n5_path, use_zarr_format=False)
    try:
        n5im.create_dataset('/c0'+subpath, shape=sh[::-1], chunks=(70, overlap, overlap), dtype=np.float32)
        n5im.create_dataset('/c1'+subpath, shape=sh[::-1], chunks=(70, overlap, overlap), dtype=np.float32)
        n5im.create_dataset('/c2'+subpath, shape=sh[::-1], chunks=(70, overlap, overlap), dtype=np.float32)
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


def read_reference_grid(n5_path, subpath):
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    return tuple(atts['dimensions'])



if __name__ == '__main__':

    tile            = sys.argv[1]
    overlap         = int(sys.argv[2])
    reference       = sys.argv[3]
    ref_subpath     = sys.argv[4]
    global_affine   = sys.argv[5]
    output          = sys.argv[6]
    invoutput       = sys.argv[7]
    output_subpath  = sys.argv[8]
    group_id        = sys.argv[9]


    tiledir = dirname(tile)
    vox = read_n5_spacing(reference, ref_subpath)

    matrix = read_matrix(global_affine)
    grid = np.round(extent/vox).astype(np.uint16)
    grid = position_grid(grid) * vox + offset
    updated_warp = transform_grid(matrix, grid)

    # TODO: should actually be using grid based on moving image coordinates here
    inv_matrix = np.array([ matrix[0],
                            matrix[1],
                            matrix[2],
                            [0, 0, 0, 1] ])
    inv_matrix = np.linalg.inv(inv_matrix)[:-1]
    updated_invwarp = transform_grid(inv_matrix, grid)
    del grid; gc.collect()

    neighbors = get_neighbors(tiledir, offset, extent)
    if isfile(neighbors['center']+'/final_lcc.nrrd'):
        lcc = read_fields(neighbors, suffix='/final_lcc.nrrd')
        for key in lcc.keys():
            lcc[key][lcc[key] > 1.0] = 0  # typically in noisy regions
        warps = read_fields(neighbors, suffix='/warp.nrrd')
        updated_warp += reconcile_warps(lcc, warps, overlap)
        del warps; gc.collect()  # need space for inv_warps
        inv_warps = read_fields(neighbors, suffix='/invwarp.nrrd')
        updated_invwarp += reconcile_warps(lcc, inv_warps, overlap)
        

    # OPTIONAL: SMOOTH THE OVERLAP REGIONS
    # OPTIONAL: USE WEIGHTED COMBINATION BASED ON LCC AT ALL VOXELS

    oo = np.round(offset/vox).astype(np.uint16)
    if not neighbors['pos']: 
        updated_warp = updated_warp[overlap:, overlap:]
        updated_invwarp = updated_invwarp[overlap:, overlap:]
        oo[0:2] += overlap
    elif neighbors['pos'] == 'left':
        updated_warp = updated_warp[:, overlap:]
        updated_invwarp = updated_invwarp[:, overlap:]
        oo[1] += overlap
    elif neighbors['pos'] == 'top':
        updated_warp = updated_warp[overlap:, :]
        updated_invwarp = updated_invwarp[overlap:, :]
        oo[0] += overlap


    # the shape here should be the complete s2 image dimensions
    ref_grid = read_reference_grid(reference, ref_subpath)
    n5im = create_n5_dataset(output, output_subpath, ref_grid, overlap)
    write_updated_transform(n5im, output_subpath, updated_warp, oo)
    n5im = create_n5_dataset(invoutput, output_subpath, ref_grid, overlap)
    write_updated_transform(n5im, output_subpath, updated_invwarp, oo)


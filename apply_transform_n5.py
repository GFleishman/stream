import z5py
import numpy as np
import json
import sys
from scipy.ndimage import map_coordinates
from os.path import splitext, abspath, isdir


def position_grid(sh, dtype=np.uint16):
    """Return a position array in physical coordinates with shape sh"""
    coords = np.array(np.meshgrid(*[range(x) for x in sh], indexing='ij'), dtype=dtype)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def transform_grid(matrix, grid):
    """Apply affine matrix to position grid"""
    mm = matrix[:, :-1]
    tt = matrix[:, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt


def interpolate_image(img, X, order=1):
    """Interpolate image at coordinates X"""
    X = np.moveaxis(X, -1, 0)
    return map_coordinates(img, X, order=order, mode='constant')


def read_n5_data(n5_path, subpath):
    # get the data
    # z5py formats data (z, y, x) by default
    im = z5py.File(n5_path, use_zarr_format=False)[subpath][:, :, :]
    return np.moveaxis(im, (0, 2), (2, 0))

def read_n5_spacing(n5_path, subpath):
    # get the voxel spacing
    # metadata is properly (x, y, z)
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    vox = np.absolute(np.array(atts['pixelResolution']) * np.array(atts['downsamplingFactors']))
    return vox.astype(np.float32)


def read_matrix(matrix_path):
    """read affine matrix from file to array"""
    matrix = np.loadtxt(matrix_path)
    return np.float32(matrix)


def write_aff_n5(n5_path, subpath, aff_im):
    aff_im = np.moveaxis(aff_im, (0, 2), (2, 0))
    im = z5py.File(n5_path, use_zarr_format=False)
    im.create_dataset(subpath, shape=aff_im.shape, chunks=(70, 128, 128), dtype=aff_im.dtype)
    im[subpath][:, :, :] = aff_im


def read_reference_grid(n5_path, subpath):
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    return tuple(atts['dimensions'])


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


def read_n5_transform(n5_path, subpath):
    with open(n5_path + '/c0' + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    grid = tuple(atts['dimensions'])
    txm = np.empty(grid + (3,))
    txm_n5 = z5py.File(n5_path, use_zarr_format=False)
    txm[..., 0] = np.moveaxis(txm_n5['/c0'+subpath][:, :, :], (0, 2), (2, 0))
    txm[..., 1] = np.moveaxis(txm_n5['/c1'+subpath][:, :, :], (0, 2), (2, 0))
    txm[..., 2] = np.moveaxis(txm_n5['/c2'+subpath][:, :, :], (0, 2), (2, 0))
    return txm



if __name__ == '__main__':

    ref_img_path     = sys.argv[1]
    ref_img_subpath  = sys.argv[2]
    mov_img_path     = sys.argv[3]
    mov_img_subpath  = sys.argv[4]
    txm_path         = sys.argv[5]
    out_path         = sys.argv[6]


    ext   = splitext(txm_path)[1]
    vox   = read_n5_spacing(mov_img_path, mov_img_subpath)
    if ext == '.mat':
        matrix     = read_matrix(txm_path)
        grid       = read_reference_grid(ref_img_path, ref_img_subpath)
        grid       = position_grid(grid) * vox
        grid       = transform_grid(matrix, grid)
    elif ext in ['', '.n5']:
        grid       = read_n5_transform(txm_path, '/s2')
 
    im  = read_n5_data(mov_img_path, mov_img_subpath)
    im  = interpolate_image(im, grid/vox)

    write_aff_n5(out_path, ref_img_subpath, im)
    copy_metadata(ref_img_path, ref_img_subpath, out_path, ref_img_subpath)


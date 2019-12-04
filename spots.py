#!/usr/env/python3

# Difference-of-Gaussian spot detection
# Author: TW
# Date: June 2019

# Modified by GF for local testing

import z5py, json
from scipy import spatial
from scipy.ndimage import convolve, map_coordinates
from scipy.ndimage.filters import maximum_filter
from scipy.stats import multivariate_normal
import numpy as np
import sys
import pickle
from skimage.feature.blob import _blob_overlap


def save_PKL(filename,var):
    f=open(filename,'wb')
    pickle.dump(var,f)
    f.close()

def gauss_conv_volume(image, sigmas):
    # implement DOG locally
    x = int(round(2*sigmas[1]))  # sigmas[1] > sigmas[0]
    x = np.arange(-x, x+.5, 1)
    x = np.moveaxis(np.array(np.meshgrid(x, x, x)), 0, -1)
    g1 = multivariate_normal.pdf(x, mean=[0,]*3, cov=np.diag([sigmas[0],]*3))
    g2 = multivariate_normal.pdf(x, mean=[0,]*3, cov=np.diag([sigmas[1],]*3))
    return convolve(image, g1 - g2)

def max_filt_volume(image, min_distance):
    return maximum_filter(image, min_distance)


def get_context(img,pos,window,interpmap=False):
    w=img[pos[0]-window:pos[0]+window+1, pos[1]-window:pos[1]+window+1, pos[2]-window:pos[2]+window+1]
    width=2*window+1
    if w.size!=width**3: #just ignore near edge
        return(False)
    if interpmap:
        return(map_coordinates(w,interpmap,order=3).reshape((width[0],width[1],width[2])))
    else:
        return(w)

def scan(img,spots,window,vox,interpmap=None):
    output=[]   
    for spot in spots:
        w=get_context(img,spot,window,interpmap)
        if type(w)!=bool:
            output.append([spot*vox,w])
    return(output)

def prune_blobs(blobs_array, overlap, distance):
    tree = spatial.cKDTree(blobs_array[:, :-2])
    pairs = np.array(list(tree.query_pairs(distance)))
    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        if _blob_overlap(blob1, blob2) > overlap:
            if blob1[-2] > blob2[-2]:
                blob2[-2] = 0
            else:
                blob1[-2] = 0
    return np.array([b for b in blobs_array if b[-2] > 0])

def get_local_max(image, min_distance=3, threshold=None):
    image_max = max_filt_volume(image, min_distance)
    if threshold:
        mask = np.logical_and(image == image_max, image >= threshold) # threshold of DoG NOT raw!
    else:
        mask = image == image_max
    return(np.column_stack(np.nonzero(mask)))
        
def tw_blob_dog(image, min_sigma=1, sigma_ratio=1.6, threshold=2.0, min_distance=5, overlap=.5):
    DoG = gauss_conv_volume(image, [min_sigma, min_sigma*sigma_ratio])
    coord=get_local_max(DoG, min_distance=min_distance)
    intensities=image[coord[:,0],coord[:,1],coord[:,2]]
    filtered=intensities>threshold
    coord=np.hstack((coord[filtered],np.array([intensities[filtered]]).T,np.full((sum(filtered),1),min_sigma)))
    return(coord)


def read_n5_spacing(n5_path, subpath):
    # get the voxel spacing
    # metadata is properly (x, y, z)
    with open(n5_path + subpath + '/attributes.json') as atts:
        atts = json.load(atts)
    vox = np.absolute(np.array(atts['pixelResolution']) * np.array(atts['downsamplingFactors']))
    return vox.astype(np.float64)


def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
    return offset, extent




# MAIN
# get the data
# z5py formats data (z, y, x) by default
z5im = z5py.File(sys.argv[1], use_zarr_format=False)[sys.argv[2]]
vox = read_n5_spacing(sys.argv[1], sys.argv[2])

if sys.argv[4] != 'coarse':
    offset, extent = read_coords(sys.argv[4])
    oo = np.round(offset/vox).astype(np.uint16)
    ee = oo + np.round(extent/vox).astype(np.uint16)
    im = z5im[oo[2]:ee[2], oo[1]:ee[1], oo[0]:ee[0]]
else:
    im = z5im[:, :, :]

im = np.moveaxis(im, (0, 2), (2, 0))
im = im.astype(np.float64)

# get the spots
coord=tw_blob_dog(im,1,2)
sortIdx=np.argsort(coord[:,-2])[::-1]
spotNum=2000

# prune the spots
if sys.argv[4] == 'coarse':
    sortedSpots=coord[sortIdx,:][:spotNum * 4]
    sortIdx=np.argsort(sortedSpots[:,0])
    sortedSpots=sortedSpots[sortIdx,:][::2]
    sortIdx=np.argsort(sortedSpots[:,1])
    sortedSpots=sortedSpots[sortIdx,:][::2]
else:
    sortedSpots=coord[sortIdx,:][:spotNum]


# final prune and save
min_distance=6
overlap=0.01
window=8
pruned_spots=prune_blobs(sortedSpots,overlap,min_distance)[:,:-2].astype(np.int)
context=scan(im,pruned_spots,window,vox)
save_PKL(sys.argv[3],context)
    

#!/bin/bash


# the fixed n5 image
fixed="/groups/multifish/multifish/Yuhan/Stitch/CEA_3_R6/Stitch/n5"
# the moving n5 image
moving="/groups/multifish/multifish/Yuhan/Stitch/CEA_3_R7/Stitch/n5"
# the folder where you'd like all outputs to be written
outdir="/groups/multifish/multifish/fleishmang/test_overlap_blocksize"


# the channel used to drive registration
channel="c2"
# the scale level for affine alignments
aff_scale="s3"
# the scale level for deformable alignments
def_scale="s2"
# the number of voxels along x/y for registration tiling, must be power of 2
xy_stride=512
# the number of voxels along z for registration tiling, must be power of 2
z_stride=512

# computational parameters
little_gaussian_stddev="later"
big_gaussian_stddev="later"
cc_radius="later"
def_smoothing="later"


# DO NOT EDIT BELOW THIS LINE
big_stream='/groups/multifish/multifish/fleishmang/stream/stream.sh'
bash "$big_stream" "$fixed" "$moving" "$outdir" "$channel" \
     "$aff_scale" "$def_scale" "$xy_stride" "$z_stride"


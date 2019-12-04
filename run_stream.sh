#!/bin/bash


# the fixed n5 image
fixed=""
# the moving n5 image
moving=""
# the folder where you'd like all outputs to be written
outdir=""


# the channel used to drive registration
channel="c2"
# the scale level for affine alignments
aff_scale="s3"
# the scale level for deformable alignments
def_scale="s2"
# the number of voxels along x/y for registration tiling
xy_stride=512
# the number of voxels along x/y to overlap between registration tiles
xy_overlap=51
# the number of voxels along z for registration tiling
z_stride=160
# the number of voxels along z to overlap between registration tiles
z_overlap=16


# DO NOT EDIT BELOW THIS LINE
big_stream='/groups/multifish/multifish/fleishmang/stream/stream.sh'
bash "$big_stream" "$fixed" "$moving" "$outdir" "$channel" \
     "$aff_scale" "$def_scale" "$xy_stride" "$xy_overlap" \
     "$z_stride" "$z_overlap"


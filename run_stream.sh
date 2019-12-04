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
tile_stride=512
# the number of voxels to overlap between registration tiles
tile_overlap=51


# DO NOT EDIT BELOW THIS LINE
big_stream='/groups/multifish/multifish/fleishmang/stream/stream.sh'
bash "$big_stream" "$fixed" "$moving" "$outdir" "$channel" \
     "$aff_scale" "$def_scale" "$tile_stride" "$tile_overlap"


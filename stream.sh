#!/bin/bash


function get_job_dependency {
  dependencies=$( echo ${1?} | tr "," "\n" )
  for dep in ${dependencies[@]}; do
      bjobs_lines=`bjobs -J "${dep}"`
      jobids=`echo "$bjobs_lines" | cut -f 1 -d' ' | tail -n +2 | uniq`
      for jobid in ${jobids[@]}; do
        dependency_string="${dependency_string}ended($jobid)&&"
      done
  done
  dependency_string=${dependency_string::-2}
  echo $dependency_string
}


function submit {
    name=${1?};       shift
    dependency=${1?}; shift
    cores=${1?};      shift
    execute="$@"

    [[ -z "$dependency" ]] || dependency=$(get_job_dependency $dependency)
    [[ -z "$dependency" ]] || dependency="-w $dependency"

    bsub -J $name \
         -n $cores \
         -o ${logdir}/${name}.o \
         -e ${logdir}/${name}.e \
         -P $BILLING \
         $dependency \
         "$execute"
}


function initialize_environment {
  logdir="${outdir}/logs";    mkdir -p $logdir
  affdir="${outdir}/aff";     mkdir -p $affdir
  tiledir="${outdir}/tiles";  mkdir -p $tiledir
  BILLING='multifish'
  PYTHON='/groups/scicompsoft/home/fleishmang/bin/miniconda3/bin/python3'
  SCRIPTS='/groups/multifish/multifish/fleishmang/stream'
  CUT_TILES="$PYTHON ${SCRIPTS}/cut_tiles.py"
  SPOTS="$PYTHON ${SCRIPTS}/spots.py"
  RANSAC="$PYTHON ${SCRIPTS}/ransac.py"
  INTERPOLATE_AFFINES="$PYTHON ${SCRIPTS}/interpolate_affines.py"
  DEFORM="$PYTHON ${SCRIPTS}/deform.py"
  STITCH="$PYTHON ${SCRIPTS}/stitch_and_write.py"
  APPLY_TRANSFORM="$PYTHON ${SCRIPTS}/apply_transform_n5.py"
}


fixed=${1?}; shift
moving=${1?}; shift
outdir=${1?}; shift
channel=${1?}; shift
aff_scale=${1?}; shift
def_scale=${1?}; shift
xy_stride=${1?}; shift
z_stride=${1?}; shift

xy_overlap=$(( $xy_stride / 8 ))
z_overlap=$(( $z_stride / 8 ))


# TODO: add prefix based on fixed/moving paths to job names to avoid
#       dependency conflict between simultaneous runs

initialize_environment

submit "cut_tiles" '' 1 \
$CUT_TILES $fixed /${channel}/${def_scale} $tiledir $xy_stride $xy_overlap $z_stride $z_overlap

submit "coarse_spots" '' 1 \
$SPOTS $fixed /${channel}/${aff_scale} ${affdir}/fixed_spots.pkl coarse

submit "coarse_spots" '' 1 \
$SPOTS $moving /${channel}/${aff_scale} ${affdir}/moving_spots.pkl coarse

submit "coarse_ransac" "coarse_spots" 1 \
$RANSAC ${affdir}/fixed_spots.pkl ${affdir}/moving_spots.pkl \
        ${affdir}/ransac_affine.mat

submit "apply_affine_small" "coarse_ransac" 1 \
$APPLY_TRANSFORM $fixed /${channel}/${aff_scale} $moving /${channel}/${aff_scale} \
                 ${affdir}/ransac_affine.mat ${affdir}/ransac_affine

submit "apply_affine_big" "coarse_ransac" 4 \
$APPLY_TRANSFORM $fixed /${channel}/${def_scale} $moving /${channel}/${def_scale} \
                 ${affdir}/ransac_affine.mat ${affdir}/ransac_affine


while [[ ! -f ${tiledir}/0/coords.txt ]]; do
    sleep 1
done
sleep 5


for tile in $( ls -d ${tiledir}/*[0-9] ); do

  tile_num=`basename $tile`

  submit "spots${tile_num}" '' 1 \
  $SPOTS $fixed /${channel}/${aff_scale} ${tile}/fixed_spots.pkl ${tile}/coords.txt

  submit "spots${tile_num}" "apply_affine_small" 1 \
  $SPOTS ${affdir}/ransac_affine /${channel}/${aff_scale} ${tile}/moving_spots.pkl ${tile}/coords.txt

  submit "ransac${tile_num}" "spots${tile_num}" 1 \
  $RANSAC ${tile}/fixed_spots.pkl ${tile}/moving_spots.pkl ${tile}/ransac_affine.mat
done

submit "interpolate_affines" 'ransac*' 1 \
$INTERPOLATE_AFFINES $tiledir

for tile in $( ls -d ${tiledir}/*[0-9] ); do
  tile_num=`basename $tile`
  submit "deform${tile_num}" "interpolate_affines,apply_affine_big" 1 \
  $DEFORM $fixed /${channel}/${def_scale} ${affdir}/ransac_affine /${channel}/${def_scale} \
          ${tile}/coords.txt ${tile}/warp.nrrd \
          ${tile}/ransac_affine.mat ${tile}/final_lcc.nrrd \
          ${tile}/invwarp.nrrd
done

for tile in $( ls -d ${tiledir}/*[0-9] ); do
  submit "stitch" 'deform*' 2 \
  $STITCH $tile $xy_overlap $z_overlap $fixed /${channel}/${def_scale} ${affdir}/ransac_affine.mat \
          ${outdir}/transform ${outdir}/invtransform /${def_scale}
done

submit "apply_transform" "stitch" 6 \
$APPLY_TRANSFORM $fixed /${channel}/${def_scale} $moving /${channel}/${def_scale} \
                 ${outdir}/transform ${outdir}/warped


#!/usr/bin/env bash


export OMP_NUM_THREADS=2
nframes=(16 32 48 64 80 96)

export MODEL=$1
export ENV=$2
export SEED=$3
export NGPU=$4


function worker () {
    worker_idx=$1
    nframe=$2

    export CUDA_VISIBLE_DEVICES=$((worker_idx % NGPU))

    out_dir="results/$MODEL"
    mkdir "$out_dir"

    python -u -m cbe.eval_tracker \
    --model="$MODEL" \
    --env="$ENV" \
    --n_traj=500 \
    --n_frame="$nframe" \
    --seed="$SEED" \
    --jitter \
    2>&1 \
    | tee "$out_dir/$ENV-$nframe-jitter-$SEED-log.txt"
}


export -f worker
SHELL=$(type -p bash) parallel --tag --linebuffer --link worker ::: $(seq 0 $((${#nframes[@]}-1))) ::: "${nframes[@]}"

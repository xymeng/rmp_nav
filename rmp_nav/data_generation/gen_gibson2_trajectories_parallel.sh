#!/bin/bash


export WORKER_START_IDX=1
export N_TRAJ=10000
export TRAJ_IMG_DIR_PREFIX="traj_imgs"

declare -a EXTRA_ARGS

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --out_dir) OUT_DIR="$2"; shift ;;
    --worker_start_idx) WORKER_START_IDX="$2"; shift ;;
    --n_worker) N_WORKERS="$2"; shift ;;
    *) EXTRA_ARGS+=("$1")
    esac
    shift
done


function worker () {
    worker_id=$1

    traj_img_dir="${OUT_DIR}/${TRAJ_IMG_DIR_PREFIX}_${worker_id}"
    mkdir -p "${traj_img_dir}"

    out_file="${OUT_DIR}/train_${worker_id}.hd5"
    log_file="${OUT_DIR}/log_${worker_id}.txt"

    python -u -m rmp_nav.data_generation.gen_trajectories \
    --out_file="${out_file}" \
    --seed=$((worker_id*987)) \
    --traj_image_dir="${traj_img_dir}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee --append "${log_file}"
}

source "$(which env_parallel.bash)"
env_parallel --tag --linebuffer worker ::: $(seq "${WORKER_START_IDX}" $((WORKER_START_IDX+N_WORKERS-1)))

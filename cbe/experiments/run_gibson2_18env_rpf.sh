#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$PWD" != "$DIR" ]; then
    echo "Please run the script in the script's residing directory"
    exit 0
fi

name=$1
tag=$2

ROOT="$PWD/${name}"

log_file="${ROOT}/$tag-log.txt"

if [ -f "${log_file}" ]; then
    read -p "Log file exists. Overwrite (y/n)?" overwrite
    case "${overwrite}" in
      y|Y )
      echo "Will overwrite log file."
      ;;
      n|N )
      echo "Will rotate log file."
      savelog -c 100 -l "${log_file}"
      ;;
      * )
      echo "Invalid option."
      exit 1
      ;;
    esac
fi

python -u -m cbe.train_rpf \
--dataset_dir "${RMP_NAV_ROOT}/data/gibson2/minirccar_agent_local_240fov_18env_slow" \
--dataset_variant "gibson2" \
--model_variant "default" \
--model_spec "${ROOT}/model_spec.yaml" \
--model_file "${ROOT}/$tag-model" \
--visdom_env "rpf-${name//\//-}-$tag" \
"${@:3}" | tee "${log_file}"

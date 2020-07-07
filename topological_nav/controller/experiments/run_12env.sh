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

python -u -m topological_nav.controller.train_multiframe_dst \
--model_variant "future_pair" \
--dataset_dir "${RMP_NAV_ROOT}/data/minirccar_agent_local_240fov_12env_v2_farwp" \
--model_spec "${ROOT}/model_spec.yaml" \
--model_file "${ROOT}/$tag-model" \
--visdom_env "${name//\//-}-imitation-$tag" \
${@:3} | tee "${log_file}"

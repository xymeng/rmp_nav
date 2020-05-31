#!/usr/bin/env bash
set -u

ENV=$1
GRAPH_CONFIG=$2
SAVE_FILE="$(dirname $GRAPH_CONFIG)"


python -m topological_nav.tools.build_nav_graph \
--env="${ENV}" \
--graph_config="${GRAPH_CONFIG}" \
--save_file="${SAVE_FILE}/graph.pickle" \
| tee "$(dirname $GRAPH_CONFIG)/build_log.txt"

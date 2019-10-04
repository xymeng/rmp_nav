#!/usr/bin/env bash

set -u

PYTHON_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as gp; print(gp()[\"include\"])")

pushd ${RMP_NAV_ROOT}/rmp_nav/common/ > /dev/null
sh compile_math_utils.sh -I"$PYTHON_INCLUDE_DIR"
popd > /dev/null

pushd ${RMP_NAV_ROOT}/rmp_nav/simulation/ > /dev/null
sh compile_map_utils.sh -I"$PYTHON_INCLUDE_DIR"
popd > /dev/null

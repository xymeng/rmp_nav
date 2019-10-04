#!/bin/bash

DIR="$( cd "$( dirname -- "$0" )" && pwd )"

export PYTHONPATH="${DIR}":$PYTHONPATH
export RMP_NAV_ROOT="${DIR}"

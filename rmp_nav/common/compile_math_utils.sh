#!/usr/bin/env bash

g++ -O3 -Wall -fPIC -shared -std=c++11 \
    math_utils.cpp \
    -o math_utils_cpp.so \
    -static-libstdc++ \
    -Wl,-rpath,"\$ORIGIN" \
    $(python3 -m pybind11 --includes) \
    $@

#!/usr/bin/env bash

g++ -O3 -Wall -fPIC -shared -std=c++11 \
map_utils.cpp \
-o map_utils_cpp.so \
-static-libstdc++ \
-Wl,-rpath,"\$ORIGIN" \
-I /usr/local/include \
-L /usr/local/lib \
$@

pushd ${RMP_NAV_ROOT}/third_party/range_libc/pywrapper

TRACE=OFF WITH_CUDA=OFF python setup.py install

popd

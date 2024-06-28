#!/bin/bash

PYPI_CACHE_DIR=$HOME/cache/pypi

# ---- YOU MAY NOT NEED TO EDIT BELLOWS ----------

__AUTHOR__="Goichi Iisaka"
__VERSION__="1.0"
__DATE__="Jun 17 2021, Sunny day"

python -m pip install -U pip wheel

# for Build numpy and scipy
cat <<_EOF_ > $HOME/.numpy-site.cfg
[mkl]
library_dirs = $HOME/.local/lib
include_dirs = $HOME/.local/include
mkl_libs = mkl_rt
runtime_library_dirs = $HOME/.local/lib
lapack_libs =
extra_link_args = -Wl,--rpath,$HOME/.local/lib -Wl,--no-as-needed -lmkl_rt -ldl -lpthread -lm
_EOF_

export LD_LIBRARY_PATH=$HOME/.local/lib:${LD_LIBRARY_PATH}

[ -d "${PYPI_CACHE_DIR}" ] || mkdir -p "${PYPI_CACHE_DIR}"

pip download -d ${PYPI_CACHE_DIR} \
    mkl-devel mkl-include pybind11 cython
pip install --no-index --find-link ${PYPI_CACHE_DIR} \
    mkl-devel mkl-include pybind11 cython

# Fix MKL libs
for F in $HOME/.local/lib/lib*.so.[0-9]
do
    [ -f ${F%.*} ] || ln -s $F ${F%.*}
done

pip wheel --no-binary :all: numpy && {
    rm -f ${PYPI_CACHE_DIR}/numpy-*.whl
    mv numpy-*whl ${PYPI_CACHE_DIR}
    pip install --no-index --find-link ${PYPI_CACHE_DIR} --force-reinstall numpy
}

pip wheel --no-deps --no-binary :all: scipy && {
    rm -f ${PYPI_CACHE_DIR}/scipy-*.whl
    mv scipy-*whl ${PYPI_CACHE_DIR}
    pip install --no-index --find-link ${PYPI_CACHE_DIR} --force-reinstall scipy
}

pip wheel --no-deps --no-binary :all: bottleneck && {
    rm -f ${PYPI_CACHE_DIR}/Rottleneck-*
    mv Bottleneck-*whl ${PYPI_CACHE_DIR}
    pip install --no-index --find-link ${PYPI_CACHE_DIR} --force-reinstall bottleneck
}

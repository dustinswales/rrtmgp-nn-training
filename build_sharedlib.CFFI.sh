#!/bin/bash

ml purge
ml intel/20.0.2
ml netcdf/4.7.4
ml

export FC=ifort
export NFHOME=/apps/spack2/opt/spack/linux-centos7-x86_64/intel-20.0.2/netcdf-4.7.4/
export NCHOME=${NFHOME}
export FCFLAGS="-O3 -g -traceback -shared -fpic -check bounds"
export RRTMGP_ROOT=/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/libs

mkdir -p libs
cd libs
cp ${RRTMGP_ROOT}/build/*.mod .
cp ${RRTMGP_ROOT}/build/*.o .

${FC} ${FCFLAGS} mo_rte_kind.o mo_rte_util_array.o     ${RRTMGP_ROOT}/rrtmgp/kernels/mo_gas_optics_kernels.F90 -o mo_gas_optics_kernels.so
${FC} ${FCFLAGS} mo_rte_kind.o mo_rrtmgp_util_string.o ../mo_gas_optics_init.F90 -o mo_gas_optics_init.so

cd ..

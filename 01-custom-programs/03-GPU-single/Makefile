ARCH=sm_80
NVCC_OPTIONS=-arch=${ARCH} -lnvidia-ml -Xcompiler -fopenmp
all:
	nvcc -O3 ${NVCC_OPTIONS} main.cu -o prog

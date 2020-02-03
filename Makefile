ARCH=70

NVFLAGS=-std=c++11 -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 --expt-relaxed-constexpr -Xcompiler="-Wall -Wextra"

HEADERS = \
	src/bb_bin.cuh \
	src/bb_comput_common.cuh \
	src/bb_comput_l_keys.cuh \
	src/bb_comput_l.cuh \
	src/bb_comput_s_keys.cuh \
	src/bb_comput_s.cuh \
	src/bb_dispatch_keys.cuh \
	src/bb_dispatch.cuh \
	src/bb_exch_keys.cuh \
	src/bb_exch.cuh \
	src/bb_segsort_common.cuh \
	src/bb_segsort_keys.cuh \
	src/bb_segsort.cuh

.PHONY: all clean

all: main.out

main.out: $(HEADERS) main.cu
	nvcc $(NVFLAGS) main.cu -o main.out

clean:
	rm main.out

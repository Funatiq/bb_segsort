ARCH=70

NVFLAGS=-std=c++11 -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 --expt-relaxed-constexpr -Xcompiler="-Wall -Wextra"

HEADERS = \
	bb_bin.cuh \
	bb_comput_l_common.cuh \
	bb_comput_l_keys.cuh \
	bb_comput_l.cuh \
	bb_comput_s_common.cuh \
	bb_comput_s_keys.cuh \
	bb_comput_s.cuh \
	bb_exch_keys.cuh \
	bb_exch.cuh \
	bb_segsort_common.cuh \
	bb_segsort_keys.cuh \
	bb_segsort.cuh

.PHONY: all clean

all: main.out

main.out: $(HEADERS) main.cu
	nvcc $(NVFLAGS) main.cu -o main.out

clean:
	rm main.out

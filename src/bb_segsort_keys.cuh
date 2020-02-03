/*
* (c) 2015-2019 Virginia Polytechnic Institute & State University (Virginia Tech)
*          2020 Robin Kobus (kobus@uni-mainz.de)
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
*/

#ifndef _H_BB_SEGSORT_KEYS
#define _H_BB_SEGSORT_KEYS

#include <utility>

#include "bb_bin.cuh"
#include "bb_dispatch_keys.cuh"
#include "bb_segsort_common.cuh"


template<class K>
void bb_segsort_run(
    K *keys_d, K *keysB_d,
    const int *d_segs, const int num_segs,
    int *d_bin_segs_id, int *h_bin_counter, int *d_bin_counter,
    cudaStream_t stream, cudaEvent_t event)
{
    bb_bin(d_segs, num_segs,
        d_bin_segs_id, d_bin_counter, h_bin_counter,
        stream, event);

    // sort small segments
    dispatch_kernels(
        keys_d, keysB_d,
        d_segs, d_bin_segs_id, d_bin_counter,
        stream);

    // wait for copy to host
    cudaEventSynchronize(event);

    int max_segsize = h_bin_counter[13];
    // std::cout << "max segsize: " << max_segsize << '\n';

    // sort long segments
    int subwarp_num = num_segs-h_bin_counter[12];
    if(subwarp_num > 0)
    gen_grid_kern_r2049(
        keys_d, keysB_d,
        d_segs, d_bin_segs_id, d_bin_counter+11, max_segsize,
        stream);
}


template<class K>
int bb_segsort(
    K * & keys_d, const int num_elements,
    const int *d_segs, const int num_segs)
{
    cudaError_t cuda_err;

    int *h_bin_counter;
    int *d_bin_counter;
    int *d_bin_segs_id;
    cuda_err = cudaMallocHost((void **)&h_bin_counter, (SEGBIN_NUM+1) * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc h_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_counter, (SEGBIN_NUM+1) * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_segs_id, num_segs * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_segs_id");

    K *keysB_d;
    cuda_err = cudaMalloc((void **)&keysB_d, num_elements * sizeof(K));
    CUDA_CHECK(cuda_err, "alloc keysB_d");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t event;
    cudaEventCreate(&event);

    bb_segsort_run(
        keys_d, keysB_d,
        d_segs, num_segs,
        d_bin_segs_id, h_bin_counter, d_bin_counter,
        stream, event);

    cudaStreamSynchronize(stream);

    std::swap(keys_d, keysB_d);

    cuda_err = cudaFreeHost(h_bin_counter);
    CUDA_CHECK(cuda_err, "free h_bin_counter");
    cuda_err = cudaFree(d_bin_counter);
    CUDA_CHECK(cuda_err, "free d_bin_counter");
    cuda_err = cudaFree(d_bin_segs_id);
    CUDA_CHECK(cuda_err, "free d_bin_segs_id");
    cuda_err = cudaFree(keysB_d);
    CUDA_CHECK(cuda_err, "free keysB");

    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    return 1;
}

#endif

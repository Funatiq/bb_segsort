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

#ifndef _H_BB_SEGSORT
#define _H_BB_SEGSORT

#include <utility>
#include <limits>

#include "bb_bin.cuh"
#include "bb_dispatch.cuh"
#include "bb_segsort_common.cuh"


template<class K, class T>
void bb_segsort_run(
    K *keys_d, T *vals_d, K *keysB_d, T *valsB_d,
    const int *d_segs, const int num_segs,
    int *d_bin_segs_id, int *d_bin_counter,
    const int max_segsize,
    cudaStream_t stream)
{
    bb_bin(d_segs, num_segs,
        d_bin_segs_id, d_bin_counter,
        stream);

    // sort small segments
    dispatch_kernels(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter,
        stream);

    // sort long segments
    gen_grid_kern_r2049(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+11, max_segsize,
        stream);
}


template<class K, class T>
int bb_segsort(
    K * & keys_d, T * & vals_d, const int num_elements,
    const int *d_segs, const int num_segs, int max_segsize = std::numeric_limits<int>::max())
{
    if(max_segsize > num_elements)
        max_segsize = num_elements;

    cudaError_t cuda_err;

    int *d_bin_counter;
    int *d_bin_segs_id;
    cuda_err = cudaMalloc((void **)&d_bin_counter, (SEGBIN_NUM+1) * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_segs_id, num_segs * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_segs_id");

    K *keysB_d;
    T *valsB_d;
    cuda_err = cudaMalloc((void **)&keysB_d, num_elements * sizeof(K));
    CUDA_CHECK(cuda_err, "alloc keysB_d");
    cuda_err = cudaMalloc((void **)&valsB_d, num_elements * sizeof(T));
    CUDA_CHECK(cuda_err, "alloc valsB_d");

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    bb_segsort_run(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, num_segs,
        d_bin_segs_id, d_bin_counter,
        max_segsize,
        stream);

    cudaStreamSynchronize(stream);

    std::swap(keys_d, keysB_d);
    std::swap(vals_d, valsB_d);

    cuda_err = cudaFree(d_bin_counter);
    CUDA_CHECK(cuda_err, "free d_bin_counter");
    cuda_err = cudaFree(d_bin_segs_id);
    CUDA_CHECK(cuda_err, "free d_bin_segs_id");
    cuda_err = cudaFree(keysB_d);
    CUDA_CHECK(cuda_err, "free keysB");
    cuda_err = cudaFree(valsB_d);
    CUDA_CHECK(cuda_err, "free valsB");

    cudaStreamDestroy(stream);
    return 1;
}

#endif

/*
* (c) 2015 Virginia Polytechnic Institute & State University (Virginia Tech)
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

#include <iostream>
#include <vector>
#include <algorithm>

#include "bb_bin.cuh"
#include "bb_comput_s.cuh"
#include "bb_comput_l.cuh"

#include "bb_segsort_common.cuh"

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }

template<class K, class T>
void bb_segsort_run(
    K *keys_d, T *vals_d, K *keysB_d, T *valsB_d, const int num_elements,
    const int *d_segs, int *d_bin_segs_id, const int num_segs,
    int *h_bin_counter, int *d_bin_counter,
    cudaStream_t stream, cudaEvent_t event)
{
    // std::cout << "num_elements: " << num_elements << '\n';
    // std::cout << "num_segs: " << num_segs << '\n';

    bb_bin(d_bin_segs_id, d_bin_counter, d_segs, num_segs, num_elements, h_bin_counter, stream, event);

    int max_segsize = h_bin_counter[13];
    // std::cout << "max segsize: " << max_segsize << '\n';

    int subwarp_size, subwarp_num, factor;
    dim3 blocks(256, 1, 1);
    dim3 grids(1, 1, 1);

    blocks.x = 256;
    subwarp_num = h_bin_counter[1]-h_bin_counter[0];
    grids.x = (subwarp_num+blocks.x-1)/blocks.x;
    if(subwarp_num > 0)
    gen_copy<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[0], subwarp_num, num_segs);

    blocks.x = 256;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[2]-h_bin_counter[1];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp2_tc1_r2_r2_orig<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[1], subwarp_num, num_segs);

    blocks.x = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[3]-h_bin_counter[2];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc2_r3_r4_orig<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[2], subwarp_num, num_segs);

    blocks.x = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[4]-h_bin_counter[3];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc4_r5_r8_orig<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[3], subwarp_num, num_segs);

    blocks.x = 128;
    subwarp_size = 4;
    subwarp_num = h_bin_counter[5]-h_bin_counter[4];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp4_tc4_r9_r16_strd<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[4], subwarp_num, num_segs);

    blocks.x = 128;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[6]-h_bin_counter[5];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp8_tc4_r17_r32_strd<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[5], subwarp_num, num_segs);

    blocks.x = 128;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[7]-h_bin_counter[6];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp16_tc4_r33_r64_strd<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[6], subwarp_num, num_segs);

    blocks.x = 256;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[8]-h_bin_counter[7];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp8_tc16_r65_r128_strd<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[7], subwarp_num, num_segs);

    blocks.x = 256;
    subwarp_size = 32;
    subwarp_num = h_bin_counter[9]-h_bin_counter[8];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp32_tc8_r129_r256_strd<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[8], subwarp_num, num_segs);

    blocks.x = 128;
    subwarp_num = h_bin_counter[10]-h_bin_counter[9];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk128_tc4_r257_r512_orig<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[9], subwarp_num, num_segs);

    blocks.x = 256;
    subwarp_num = h_bin_counter[11]-h_bin_counter[10];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk256_tc4_r513_r1024_orig<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[10], subwarp_num, num_segs);

    blocks.x = 512;
    subwarp_num = h_bin_counter[12]-h_bin_counter[11];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk512_tc4_r1025_r2048_orig<<<grids, blocks, 0, stream>>>(keys_d, vals_d, keysB_d, valsB_d,
        num_elements, d_segs, d_bin_segs_id+h_bin_counter[11], subwarp_num, num_segs);

    // sort long segments
    subwarp_num = num_segs-h_bin_counter[12];
    if(subwarp_num > 0)
    gen_grid_kern_r2049(keys_d, vals_d, keysB_d, valsB_d, num_elements,
        d_segs, d_bin_segs_id+h_bin_counter[12], subwarp_num, num_segs, max_segsize,
        stream);

    cudaStreamSynchronize(stream);
}


template<class K, class T>
int bb_segsort(K * & keys_d, T * & vals_d, const int num_elements, const int *d_segs, const int num_segs)
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
    T *valsB_d;
    cuda_err = cudaMalloc((void **)&keysB_d, num_elements * sizeof(K));
    CUDA_CHECK(cuda_err, "alloc keysB_d");
    cuda_err = cudaMalloc((void **)&valsB_d, num_elements * sizeof(T));
    CUDA_CHECK(cuda_err, "alloc valsB_d");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t event;
    cudaEventCreate(&event);

    bb_segsort_run(
        keys_d, vals_d, keysB_d, valsB_d, num_elements,
        d_segs, d_bin_segs_id, num_segs,
        h_bin_counter, d_bin_counter,
        stream, event);

    std::swap(keys_d, keysB_d);
    std::swap(vals_d, valsB_d);

    cuda_err = cudaFreeHost(h_bin_counter);
    CUDA_CHECK(cuda_err, "free h_bin_counter");
    cuda_err = cudaFree(d_bin_counter);
    CUDA_CHECK(cuda_err, "free d_bin_counter");
    cuda_err = cudaFree(d_bin_segs_id);
    CUDA_CHECK(cuda_err, "free d_bin_segs_id");
    cuda_err = cudaFree(keysB_d);
    CUDA_CHECK(cuda_err, "free keysB");
    cuda_err = cudaFree(valsB_d);
    CUDA_CHECK(cuda_err, "free valsB");

    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    return 1;
}

#endif

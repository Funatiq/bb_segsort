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

#ifndef _H_BB_DISPATCH
#define _H_BB_DISPATCH

#include "bb_comput_s.cuh"
#include "bb_comput_l.cuh"


template<class K, class T, class Offset>
void dispatch_kernels(
    K *keys_d, T *vals_d, K *keysB_d, T *valsB_d,
    const Offset *d_segs, const int *d_bin_segs_id, const int *d_bin_counter,
    cudaStream_t stream)
{
    constexpr int num_blocks_default = 512;

    // int subwarp_size, factor;
    int threads_per_block;
    int num_blocks;

    threads_per_block = 256;
    // num_blocks = (num_blocks_default+threads_per_block-1)/threads_per_block;
    num_blocks = num_blocks_default/32;
    gen_copy<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter);

    threads_per_block = 256;
    // subwarp_size = 2;
    // factor = threads_per_block/subwarp_size; // 128
    num_blocks = num_blocks_default/16;
    gen_bk256_wp2_tc1_r2_r2_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+0);

    threads_per_block = 128;
    // subwarp_size = 2;
    // factor = threads_per_block/subwarp_size; // 64
    num_blocks = num_blocks_default/8;
    gen_bk128_wp2_tc2_r3_r4_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+1);

    threads_per_block = 128;
    // subwarp_size = 2;
    // factor = threads_per_block/subwarp_size; // 64
    num_blocks = num_blocks_default/8;
    gen_bk128_wp2_tc4_r5_r8_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+2);

    threads_per_block = 128;
    // subwarp_size = 4;
    // factor = threads_per_block/subwarp_size; // 32
    num_blocks = num_blocks_default/4;
    gen_bk128_wp4_tc4_r9_r16_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+3);

    threads_per_block = 128;
    // subwarp_size = 8;
    // factor = threads_per_block/subwarp_size; // 16
    num_blocks = num_blocks_default/2;
    gen_bk128_wp8_tc4_r17_r32_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+4);

    threads_per_block = 128;
    // subwarp_size = 16;
    // factor = threads_per_block/subwarp_size; // 8
    num_blocks = num_blocks_default;
    gen_bk128_wp16_tc4_r33_r64_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+5);

    threads_per_block = 256;
    // subwarp_size = 8;
    // factor = threads_per_block/subwarp_size; // 32
    num_blocks = num_blocks_default/4;
    gen_bk256_wp8_tc16_r65_r128_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+6);

    threads_per_block = 256;
    // subwarp_size = 32;
    // factor = threads_per_block/subwarp_size; // 8
    num_blocks = num_blocks_default;
    gen_bk256_wp32_tc8_r129_r256_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+7);

    threads_per_block = 128;
    num_blocks = num_blocks_default;
    gen_bk128_tc4_r257_r512_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+8);

    threads_per_block = 256;
    num_blocks = num_blocks_default;
    gen_bk256_tc4_r513_r1024_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+9);

    threads_per_block = 512;
    num_blocks = num_blocks_default;
    gen_bk512_tc4_r1025_r2048_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id, d_bin_counter+10);
}


template<class K, class T, class Offset>
__global__
void gen_grid_kern_r2049(
    K *keys, T *vals, K *keysB, T *valsB,
    const Offset *segs, const int *bins, const int *bin_counter, const int *max_segsize)
{
    if(*max_segsize < 2049) return;

    constexpr cudaStream_t stream = 0;
    constexpr int workloads_per_block = 2048;

    const int *bin = bins + bin_counter[0];
    const int bin_size = bin_counter[1]-bin_counter[0];

    dim3 block_per_grid(1, 1, 1);
    block_per_grid.x = bin_size;
    block_per_grid.y = (*max_segsize+workloads_per_block-1)/workloads_per_block;

    int threads_per_block = 512;
    kern_block_sort<<<block_per_grid, threads_per_block, 0, stream>>>(
        keys, vals, keysB, valsB,
        segs, bin,
        workloads_per_block);

    swap(keys, keysB);
    swap(vals, valsB);
    int cnt_swaps = 1;

    threads_per_block = 128;
    for(int stride = workloads_per_block;
        stride < *max_segsize;
        stride <<= 1)
    {
        kern_block_merge<<<block_per_grid, threads_per_block, 0, stream>>>(
            keys, vals, keysB, valsB,
            segs, bin,
            stride, workloads_per_block);
        swap(keys, keysB);
        swap(vals, valsB);
        cnt_swaps++;
    }

    if((cnt_swaps&1)) {
        swap(keys, keysB);
        swap(vals, valsB);
    }

    threads_per_block = 128;
    kern_copy<<<block_per_grid, threads_per_block, 0, stream>>>(
        keys, vals, keysB, valsB,
        segs, bin,
        workloads_per_block);
}

#endif

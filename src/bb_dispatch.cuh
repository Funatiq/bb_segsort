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
    const Offset *d_seg_begins, const Offset *d_seg_ends, const int *d_bin_segs_id, const int *d_bin_counter,
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
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter);

    threads_per_block = 256;
    // subwarp_size = 2;
    // factor = threads_per_block/subwarp_size; // 128
    num_blocks = num_blocks_default/16;
    gen_bk256_wp2_tc1_r2_r2_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+0);

    threads_per_block = 128;
    // subwarp_size = 2;
    // factor = threads_per_block/subwarp_size; // 64
    num_blocks = num_blocks_default/8;
    gen_bk128_wp2_tc2_r3_r4_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+1);

    threads_per_block = 128;
    // subwarp_size = 2;
    // factor = threads_per_block/subwarp_size; // 64
    num_blocks = num_blocks_default/8;
    gen_bk128_wp2_tc4_r5_r8_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+2);

    threads_per_block = 128;
    // subwarp_size = 4;
    // factor = threads_per_block/subwarp_size; // 32
    num_blocks = num_blocks_default/4;
    gen_bk128_wp4_tc4_r9_r16_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+3);

    threads_per_block = 128;
    // subwarp_size = 8;
    // factor = threads_per_block/subwarp_size; // 16
    num_blocks = num_blocks_default/2;
    gen_bk128_wp8_tc4_r17_r32_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+4);

    threads_per_block = 128;
    // subwarp_size = 16;
    // factor = threads_per_block/subwarp_size; // 8
    num_blocks = num_blocks_default;
    gen_bk128_wp16_tc4_r33_r64_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+5);

    threads_per_block = 256;
    // subwarp_size = 8;
    // factor = threads_per_block/subwarp_size; // 32
    num_blocks = num_blocks_default/4;
    gen_bk256_wp8_tc16_r65_r128_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+6);

    threads_per_block = 256;
    // subwarp_size = 32;
    // factor = threads_per_block/subwarp_size; // 8
    num_blocks = num_blocks_default;
    gen_bk256_wp32_tc8_r129_r256_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+7);

    threads_per_block = 128;
    num_blocks = num_blocks_default;
    gen_bk128_tc4_r257_r512_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+8);

    threads_per_block = 256;
    num_blocks = num_blocks_default;
    gen_bk256_tc4_r513_r1024_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+9);

    threads_per_block = 512;
    num_blocks = num_blocks_default;
    gen_bk512_tc4_r1025_r2048_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends,
        d_bin_segs_id, d_bin_counter+10);
}


template<class K, class T, class Offset>
void gen_grid_kern_r2049(
    K *keys_d, T *vals_d, K *keysB_d, T *valsB_d,
    const Offset *seg_begins_d, const Offset *seg_ends_d,
    const int *bins_d, const int *bin_counter_d, const int max_segsize,
    cudaStream_t stream)
{
    const int workloads_per_block = 2048;

    dim3 block_per_grid(1, 1, 1);
    block_per_grid.x = 1024;
    block_per_grid.y = (max_segsize+workloads_per_block-1)/workloads_per_block;

    int threads_per_block = 512;
    kern_block_sort<<<block_per_grid, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        seg_begins_d, seg_ends_d,
        bins_d, bin_counter_d,
        workloads_per_block);

    std::swap(keys_d, keysB_d);
    std::swap(vals_d, valsB_d);
    int cnt_swaps = 1;

    threads_per_block = 128;
    for(int stride = 2048; // unit for already sorted
        stride < max_segsize;
        stride <<= 1)
    {
        kern_block_merge<<<block_per_grid, threads_per_block, 0, stream>>>(
            keys_d, vals_d, keysB_d, valsB_d,
            seg_begins_d, seg_ends_d,
            bins_d, bin_counter_d,
            stride, workloads_per_block);
        std::swap(keys_d, keysB_d);
        std::swap(vals_d, valsB_d);
        cnt_swaps++;
    }
    // std::cout << "cnt_swaps " << cnt_swaps << std::endl;

    if((cnt_swaps&1)) {
        std::swap(keys_d, keysB_d);
        std::swap(vals_d, valsB_d);
    }

    threads_per_block = 128;
    kern_copy<<<block_per_grid, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        seg_begins_d, seg_ends_d,
        bins_d, bin_counter_d,
        workloads_per_block);
}

#endif

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

#ifndef _H_BB_COMPUT_L_COMMON
#define _H_BB_COMPUT_L_COMMON

__device__
int binary_search(
    const int *blk_stat, const int bin_size, const int gid, int const blk_num)
{
    int l = 0;
    int h = bin_size;
    int m;
    int lr, rr;
    while(l < h)
    {
        m = l + (h-l)/2;
        lr = blk_stat[m];
        rr = (m==bin_size)?blk_num:blk_stat[m+1];
        if(lr<=gid && gid<rr)
        {
            return m;
        } else if(gid < lr)
        {
            h = m;
        } else
        {
            l = m+1;
        }
    }
    return m;
}

__device__ inline
int upper_power_of_two(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}

__device__ inline
int log2(int u)
{
    int s, t;
    t = (u > 0xffff) << 4; u >>= t;
    s = (u > 0xff  ) << 3; u >>= s, t |= s;
    s = (u > 0xf   ) << 2; u >>= s, t |= s;
    s = (u > 0x3   ) << 1; u >>= s, t |= s;
    return (t | (u >> 1));
}

__global__
void kern_get_num_blk_init(
    int *max_segsize, const int *segs, const int *bin, int *blk_stat,
    const int n, const int bin_size, const int length, const int workloads_per_block)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < bin_size)
    {
        const int seg_id = bin[gid];
        const int seg_size = ((seg_id==length-1)?n:segs[seg_id+1])-segs[seg_id];
        blk_stat[gid] = (seg_size+workloads_per_block-1)/workloads_per_block;
        atomicMax(max_segsize, seg_size);
    }
}

__global__
void kern_get_init_pos(
    const int *blk_stat, int *blk_innerid, int *blk_seg_start,
    const int blk_num, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < blk_num)
    {
        int found = binary_search(blk_stat, bin_size, gid, blk_num);
        blk_innerid[gid] = gid - blk_stat[found];
        blk_seg_start[gid] = found;
    }
}

__global__
void kern_get_num_blk(
    const int *segs, const int *bin, int *blk_stat,
    const int n, const int bin_size, const int length, const int workloads_per_block)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < bin_size)
    {
        int seg_size = ((bin[gid]==length-1)?n:segs[bin[gid]+1])-segs[bin[gid]];
        blk_stat[gid] = (seg_size+workloads_per_block-1)/workloads_per_block;
    }
}

#endif

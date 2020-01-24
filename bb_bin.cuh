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

#ifndef _H_BB_BIN
#define _H_BB_BIN

#include "bb_segsort_common.cuh"

#define SEGBIN_NUM 13



template<class T>
__global__
void exclusive_sum(T * in, T * out, int n)
{
    const int lane = threadIdx.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    T data = (tid > 0 && tid < n) ? in[tid-1] : 0;

    for(int i = 1; i < 32; i *= 2) {
        T other = __shfl_up_sync(0xFFFFFFFF, data, i);
        if(lane > i)
            data += other;
    }

    if(tid < n) {
        out[tid] = data;
    }
}



template<class T>
__device__
void warp_exclusive_sum(T * in, T * out, int n)
{
    const int lane = threadIdx.x & 31;

    T data = (lane > 0 && lane < n) ? in[lane-1] : 0;

    for(int i = 1; i < 32; i *= 2) {
        T other = __shfl_up_sync(0xFFFFFFFF, data, i);
        if(lane > i)
            data += other;
    }

    if(lane < n) {
        out[lane] = data;
    }
}



__global__
void bb_bin_histo(int *d_bin_counter, const int *d_segs, int num_segs, int num_keys)
{
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int local_histo[SEGBIN_NUM + 1];
    if (tid < SEGBIN_NUM + 1)
        local_histo[tid] = 0;
    __syncthreads();

    if (gid < num_segs)
    {
        const int size = ((gid==num_segs-1)?num_keys:d_segs[gid+1]) - d_segs[gid];

        if (size <= 1)
            atomicAdd((int *)&local_histo[0 ], 1);
        if (1  < size && size <= 2 )
            atomicAdd((int *)&local_histo[1 ], 1);
        if (2  < size && size <= 4 )
            atomicAdd((int *)&local_histo[2 ], 1);
        if (4  < size && size <= 8 )
            atomicAdd((int *)&local_histo[3 ], 1);
        if (8  < size && size <= 16)
            atomicAdd((int *)&local_histo[4 ], 1);
        if (16 < size && size <= 32)
            atomicAdd((int *)&local_histo[5 ], 1);
        if (32 < size && size <= 64)
            atomicAdd((int *)&local_histo[6 ], 1);
        if (64 < size && size <= 128)
            atomicAdd((int *)&local_histo[7 ], 1);
        if (128 < size && size <= 256)
            atomicAdd((int *)&local_histo[8 ], 1);
        if (256 < size && size <= 512)
            atomicAdd((int *)&local_histo[9 ], 1);
        if (512 < size && size <= 1024)
            atomicAdd((int *)&local_histo[10], 1);
        if (1024 < size && size <= 2048)
            atomicAdd((int *)&local_histo[11], 1);
        if (2048 < size) {
            // atomicAdd((int *)&local_histo[12], 1);
            atomicMax((int *)&local_histo[13], size);
        }
    }
    __syncthreads();

    if(tid < 32) {
        warp_exclusive_sum(local_histo, local_histo, SEGBIN_NUM);

        if (tid < SEGBIN_NUM)
            atomicAdd((int *)&d_bin_counter[tid], local_histo[tid]);
        if (tid == SEGBIN_NUM)
            atomicMax((int *)&d_bin_counter[tid], local_histo[tid]);
    }
}



__global__
void bb_bin_group(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, int num_segs, int num_keys)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < num_segs)
    {
        const int size = ((gid==num_segs-1)?num_keys:d_segs[gid+1]) - d_segs[gid];
        int position;
        if (size <= 1)
            position = atomicAdd((int *)&d_bin_counter[0 ], 1);
        else if (size <= 2)
            position = atomicAdd((int *)&d_bin_counter[1 ], 1);
        else if (size <= 4)
            position = atomicAdd((int *)&d_bin_counter[2 ], 1);
        else if (size <= 8)
            position = atomicAdd((int *)&d_bin_counter[3 ], 1);
        else if (size <= 16)
            position = atomicAdd((int *)&d_bin_counter[4 ], 1);
        else if (size <= 32)
            position = atomicAdd((int *)&d_bin_counter[5 ], 1);
        else if (size <= 64)
            position = atomicAdd((int *)&d_bin_counter[6 ], 1);
        else if (size <= 128)
            position = atomicAdd((int *)&d_bin_counter[7 ], 1);
        else if (size <= 256)
            position = atomicAdd((int *)&d_bin_counter[8 ], 1);
        else if (size <= 512)
            position = atomicAdd((int *)&d_bin_counter[9 ], 1);
        else if (size <= 1024)
            position = atomicAdd((int *)&d_bin_counter[10], 1);
        else if (size <= 2048)
            position = atomicAdd((int *)&d_bin_counter[11], 1);
        else
            position = atomicAdd((int *)&d_bin_counter[12], 1);
        d_bin_segs_id[position] = gid;
    }
}



void bb_bin(
    int *d_bin_segs_id, int *d_bin_counter, const int *d_segs,
    const int num_segs, const int num_keys, int *h_bin_counter,
    cudaStream_t stream, cudaEvent_t event)
{
    cudaMemsetAsync(d_bin_counter, 0, (SEGBIN_NUM+1) * sizeof(int), stream);

    const int num_threads = 256;
    const int num_blocks = ceil((double)num_segs/(double)num_threads);

    bb_bin_histo<<< num_blocks, num_threads, 0, stream >>>(d_bin_counter, d_segs, num_segs, num_keys);

    // show_d(d_bin_counter, SEGBIN_NUM, "d_bin_counter:\n");

    // exclusive_sum<<< 1, 32, 0, stream >>>(d_bin_counter, d_bin_counter, SEGBIN_NUM);

    // show_d(d_bin_counter, SEGBIN_NUM, "d_bin_counter:\n");

    cudaMemcpyAsync(h_bin_counter, d_bin_counter, (SEGBIN_NUM+1)*sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaEventRecord(event, stream);

    // group segment IDs (that belong to the same bin) together
    bb_bin_group<<< num_blocks, num_threads, 0, stream >>>(d_bin_segs_id, d_bin_counter, d_segs, num_segs, num_keys);

    // show_d(d_bin_segs_id, num_segs, "d_bin_segs_id:\n");

    // wait for h_bin_counter copy to host
    cudaEventSynchronize(event);
}

#endif

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
#include <limits>

#include "bb_bin.cuh"
#include "bb_dispatch_keys.cuh"
#include "bb_segsort_common.cuh"


template<class K>
class bb_segsort_keys {
public:
    bb_segsort_keys(
        K *d_keys, K *d_keysB,
        const int *d_segs,
        int *d_bin_segs_id, int *d_bin_counter,
        cudaStream_t stream)
    :
        d_keys_{d_keys},
        d_keysB_{d_keysB},
        d_segs_{d_segs},
        d_bin_segs_id_{d_bin_segs_id},
        d_bin_counter_{d_bin_counter},
        static_num_segs_{false}
    {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

        // sort small segments
        dispatch_kernels(
            d_keys_, d_keysB_,
            d_segs_, d_bin_segs_id_, d_bin_counter_,
            stream);

        cudaStreamEndCapture(stream, &graph_);
        cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0);
    }

    bb_segsort_keys(
        K *d_keys, K *d_keysB,
        const int *d_segs, int num_segs, int max_segsize,
        int *d_bin_segs_id, int *d_bin_counter,
        cudaStream_t stream)
    :
        d_keys_{d_keys},
        d_keysB_{d_keysB},
        d_segs_{d_segs},
        d_bin_segs_id_{d_bin_segs_id},
        d_bin_counter_{d_bin_counter},
        static_num_segs_{true}
    {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

        bb_bin(d_segs_, num_segs,
            d_bin_segs_id_, d_bin_counter_,
            stream);

        dispatch_kernels(
            d_keys_, d_keysB_,
            d_segs_, d_bin_segs_id_, d_bin_counter_,
            stream);

        gen_grid_kern_r2049(
            d_keys_, d_keysB_,
            d_segs_, d_bin_segs_id_, d_bin_counter_+11, max_segsize,
            stream);

        cudaStreamEndCapture(stream, &graph_);
        cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0);
    }

    void run(int num_segs, int max_segsize, cudaStream_t stream) const
    {
        if(static_num_segs_ == true) {
            std::cerr << "error: called wrong run function\n";
            return;
        }

        bb_bin(d_segs_, num_segs,
            d_bin_segs_id_, d_bin_counter_,
            stream);

        cudaGraphLaunch(instance_, stream);

        gen_grid_kern_r2049(
            d_keys_, d_keysB_,
            d_segs_, d_bin_segs_id_, d_bin_counter_+11, max_segsize,
            stream);
    }

    void run(cudaStream_t stream) const
    {
        if(static_num_segs_ == false) {
            std::cerr << "error: called wrong run function\n";
            return;
        }

        cudaGraphLaunch(instance_, stream);
    }

private:
    K *d_keys_;
    K *d_keysB_;
    const int *d_segs_;
    int *d_bin_segs_id_;
    int *d_bin_counter_;

    bool static_num_segs_;
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
};


template<class K>
void bb_segsort_run(
    K *d_keys, K *d_keysB,
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
        d_keys, d_keysB,
        d_segs, d_bin_segs_id, d_bin_counter,
        stream);

    // sort large segments
    gen_grid_kern_r2049(
        d_keys, d_keysB,
        d_segs, d_bin_segs_id, d_bin_counter+11, max_segsize,
        stream);
}


template<class K>
int bb_segsort(
    K * & d_keys, const int num_elements,
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

    K *d_keysB;
    cuda_err = cudaMalloc((void **)&d_keysB, num_elements * sizeof(K));
    CUDA_CHECK(cuda_err, "alloc d_keysB");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // bb_segsort_run(
    //     d_keys, d_keysB,
    //     d_segs, num_segs, max_segsize,
    //     d_bin_segs_id, d_bin_counter,
    //     stream);

    // bb_segsort_keys<K> sorter(
    //     d_keys, d_keysB,
    //     d_segs, num_segs, max_segsize,
    //     d_bin_segs_id, d_bin_counter,
    //     stream);

    // sorter.run(stream);

    bb_segsort_keys<K> sorter(
        d_keys, d_keysB,
        d_segs,
        d_bin_segs_id, d_bin_counter,
        stream);

    sorter.run(num_segs, max_segsize, stream);

    cudaStreamSynchronize(stream);

    std::swap(d_keys, d_keysB);

    cuda_err = cudaFree(d_bin_counter);
    CUDA_CHECK(cuda_err, "free d_bin_counter");
    cuda_err = cudaFree(d_bin_segs_id);
    CUDA_CHECK(cuda_err, "free d_bin_segs_id");
    cuda_err = cudaFree(d_keysB);
    CUDA_CHECK(cuda_err, "free keysB");

    cudaStreamDestroy(stream);
    return 1;
}

#endif

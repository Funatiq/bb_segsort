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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include "bb_segsort.cuh"

using std::vector;
using std::pair;

using index_t = int;
using seg_t = int;

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }

template<class K, class T>
void gold_segsort(vector<K> &key, vector<T> &val, index_t n, const vector<seg_t> &seg, index_t m);

int show_mem_usage();

int main(int argc, char **argv)
{
    using key_t = int;
    using val_t = double;

    cudaError_t err;
    // index_t n = 400 000 000;
    index_t n = 1UL << 24;
    vector<key_t> key(n, 0);
    vector<val_t> val(n, 0.0);
    for(auto &k: key)
        k = rand()%(n-1-0+1)+0;
    for(auto &v: val)
        v = (val_t)(rand()%(n-1-0+1)+0);
    seg_t max_seg_sz = 10000;
    seg_t min_seg_sz = 0;
    vector<seg_t> seg;
    seg_t off = 0;
    seg.push_back(off); // must have a zero
    while(off < n)
    {
        seg.push_back(off);
        seg_t sz = rand()%(max_seg_sz-min_seg_sz+1)+min_seg_sz;
        off = seg.back()+sz;
    }
    index_t m = seg.size();
    printf("synthesized segments # %d (max_size: %d, min_size: %d)\n", m, max_seg_sz, min_seg_sz);

    // cout << "key:\n"; for(auto k: key) cout << k << ", "; cout << endl;
    // cout << "val:\n"; for(auto v: val) cout << v << ", "; cout << endl;
    // cout << "seg:\n"; for(auto s: seg) cout << s << ", "; cout << endl;

    key_t *key_d;
    val_t *val_d;
    key_t *seg_d;
    err = cudaMalloc((void**)&key_d, sizeof(key_t)*n);
    CUDA_CHECK(err, "alloc key_d");
    err = cudaMalloc((void**)&val_d, sizeof(val_t)*n);
    CUDA_CHECK(err, "alloc val_d");
    err = cudaMalloc((void**)&seg_d, sizeof(key_t)*n);
    CUDA_CHECK(err, "alloc seg_d");

    err = cudaMemcpy(key_d, &key[0], sizeof(key_t)*n, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "copy to key_d");
    err = cudaMemcpy(val_d, &val[0], sizeof(val_t)*n, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "copy to val_d");
    err = cudaMemcpy(seg_d, &seg[0], sizeof(key_t)*m, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "copy to seg_d");

    show_mem_usage();

    gold_segsort(key, val, n, seg, m);

    // cout << "key:\n"; for(auto k: key) cout << k << ", "; cout << endl;
    // cout << "val:\n"; for(auto v: val) cout << v << ", "; cout << endl;

    // for(int i = 0; i < 3; i++) // test repeated execution
    bb_segsort(key_d, val_d, n, seg_d, m);

    vector<key_t> key_h(n, 0);
    vector<val_t> val_h(n, 0.0);
    err = cudaMemcpy(&key_h[0], key_d, sizeof(key_t)*n, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "copy from key_d");
    err = cudaMemcpy(&val_h[0], val_d, sizeof(val_t)*n, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "copy from val_d");

    // cout << "key_h:\n"; for(auto k: key_h) cout << k << ", "; cout << endl;
    // cout << "val_h:\n"; for(auto v: val_h) cout << v << ", "; cout << endl;

    index_t cnt = 0;
    for(index_t i = 0; i < n; i++)
        if(key[i] != key_h[i]) cnt++;
    if(cnt != 0) printf("[NOT PASSED] checking keys: #err = %i (%4.2f%% #nnz)\n", cnt, 100.0*(double)cnt/n);
    else printf("[PASSED] checking keys\n");
    cnt = 0;
    for(index_t i = 0; i < n; i++)
        if(val[i] != val_h[i]) cnt++;
    if(cnt != 0) printf("[NOT PASSED] checking vals: #err = %i (%4.2f%% #nnz)\n", cnt, 100.0*(double)cnt/n);
    else printf("[PASSED] checking vals\n");

    err = cudaFree(key_d);
    CUDA_CHECK(err, "free key_d");
    err = cudaFree(val_d);
    CUDA_CHECK(err, "free val_d");
    err = cudaFree(seg_d);
    CUDA_CHECK(err, "free seg_d");
}


template<class K, class T>
void gold_segsort(vector<K> &key, vector<T> &val, index_t n, const vector<seg_t> &seg, index_t m)
{
    vector<pair<K,T>> pairs;
    for(index_t i = 0; i < n; i++)
    {
        pairs.push_back({key[i], val[i]});
    }

    for(index_t i = 0; i < m; i++)
    {
        seg_t st = seg[i];
        seg_t ed = (i<m-1)?seg[i+1]:n;
        stable_sort(pairs.begin()+st, pairs.begin()+ed, [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});
        // sort(pairs.begin()+st, pairs.begin()+ed, [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});
    }

    for(index_t i = 0; i < n; i++)
    {
        key[i] = pairs[i].first;
        val[i] = pairs[i].second;
    }
}

int show_mem_usage()
{
    cudaError_t err;
     // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    err = cudaMemGetInfo(&free_byte, &total_byte);
    CUDA_CHECK(err, "check memory info.");
    size_t used_byte  = total_byte - free_byte;
    printf("GPU memory usage: used = %4.2lf MB, free = %4.2lf MB, total = %4.2lf MB\n",
        used_byte/1024.0/1024.0, free_byte/1024.0/1024.0, total_byte/1024.0/1024.0);
    return cudaSuccess;
}

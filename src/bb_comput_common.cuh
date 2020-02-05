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

#ifndef _H_BB_COMPUT_COMMON
#define _H_BB_COMPUT_COMMON

template<class T>
__device__ inline
void swap(T &a, T &b) noexcept {
    T tmp = a;
    a = b;
    b = tmp;
}


template<class K>
__device__
int find_kth3(K* a, int aCount, K* b, int bCount, int diag)
{
    int begin = max(0, diag - bCount);
    int end = min(diag, aCount);

    while(begin < end) {
        int mid = (begin + end)>> 1;
        K aKey = a[mid];
        K bKey = b[diag - 1 - mid];
        bool pred = aKey <= bKey;
        if(pred) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

#endif

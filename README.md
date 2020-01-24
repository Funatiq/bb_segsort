# bb_segsort (segmented sort): Fast Segmented Sort on GPUs

This repository provides a fast segmented sort on NVIDIA GPUs. The library contains many parallel kernels for different types of segments. In particular, the kernels for solving short/medium segments are automatically generated to efficiently utilize registers in GPUs. More details about the kernels and code generation can be found in our paper (below).

## Original Work

* [Original GitHub repository](https://github.com/vtsynergy/bb_segsort)
* Contact Email: kaixihou@vt.edu

## Improvements in this fork

* Added key only version
* Asynchronous execution using a single CUDA stream inside bb_segsort_run
* No temporary memory allocation inside bb_segsort_run
* Reduced memory overhead
* Two dimensional kernel grid to avoid index calculations
* Avoiding boundaries check by using one-past-the-end offset
* No dependency on Thrust

## Interface differences

* This version expects a one-past-the-end offset at the end of segments array

## Usage

You can make changes to the Makefile accordingly. Especially, you need to change the ARCH according to your GPU platform. For example, if you are using the P100, you should update ARCH to 61. The main.cu contains an example of how to use it.

The following shows how to run the example codes.

```[Bash]
$ make
```

After compilation, run the executable as: 

```[Bash]
$ ./main.out
```

To use the segmented sort (**bb_segsort**), you need to include the `bb_segsort.cuh` (key-value) or `bb_segsort_keys.cuh` (key only).
Use `bb_segsort(...)` if you don't care about memory allocation or asynchronous execution, or use `bb_segsort_run(...)` and provide your own memory allocation and stream.

 Note, bb_segsort utilizes an unstable sorting network as the building block; thus, equivalent elements are not guaranteed to keep the original relative order. We plan to provide a version to support stable sort in the future.

## License

Please refer to the included LICENSE file.

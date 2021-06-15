Performance of static vs dynamic [CUDA] based PageRank ([pull], [CSR], [scaled-fill]).

This experiment was for comparing the performance between:
1. Find **static** pagerank of updated graph using [nvGraph][pr-nvgraph].
2. Find **dyanmic** pagerank of updated graph using [nvGraph][pr-nvgraph].
3. Find **static** pagerank of updated graph using [CUDA].
4. Find **dynamic** pagerank of updated graph using [CUDA].

Each approach was attempted on a number of graphs, running each with multiple
batch sizes (`1`, `5`, `10`, `50`, ...). Each batch size was run with 5
different updates to graph, and each specific update was run 5 times for each
approach to get a good time measure. **Levelwise** pagerank is the
[STIC-D algorithm], without **ICD** optimizations (using single-thread).
On average, skipping unchanged components is **barely faster** than not
skipping.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at ["graphs"] (for small ones), and
the [SuiteSparse Matrix Collection].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# ...
#
# Loading graph soc-LiveJournal1.mtx ...
# order: 4847572 size: 68993774 {}
# [00168.551 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
#
# # Batch size 1e+0
# order: 4847571.4 size: 68993774 {} [00168.449 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847571.4 size: 68993774 {} [00158.919 ms; 051 iters.] [3.2012e-6 err.] pagerankCuda [static]
# order: 4847571.4 size: 68993774 {} [00014.478 ms; 000 iters.] [2.5781e-7 err.] pagerankNvgraph [dynamic]
# order: 4847571.4 size: 68993774 {} [00003.223 ms; 001 iters.] [3.0940e-7 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+0
# order: 4847572.2 size: 68993778 {} [00168.358 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847572.2 size: 68993778 {} [00158.904 ms; 051 iters.] [3.1609e-6 err.] pagerankCuda [static]
# order: 4847572.2 size: 68993778 {} [00020.177 ms; 000 iters.] [3.0107e-7 err.] pagerankNvgraph [dynamic]
# order: 4847572.2 size: 68993778 {} [00003.229 ms; 001 iters.] [7.3041e-7 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+1
# order: 4847572.6 size: 68993783 {} [00168.505 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847572.6 size: 68993783 {} [00158.939 ms; 051 iters.] [3.1840e-6 err.] pagerankCuda [static]
# order: 4847572.6 size: 68993783 {} [00029.471 ms; 000 iters.] [3.5690e-7 err.] pagerankNvgraph [dynamic]
# order: 4847572.6 size: 68993783 {} [00005.714 ms; 002 iters.] [1.2775e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+1
# order: 4847579.6 size: 68993823 {} [00168.357 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847579.6 size: 68993823 {} [00158.850 ms; 051 iters.] [3.1943e-6 err.] pagerankCuda [static]
# order: 4847579.6 size: 68993823 {} [00040.351 ms; 000 iters.] [4.1340e-7 err.] pagerankNvgraph [dynamic]
# order: 4847579.6 size: 68993823 {} [00016.352 ms; 005 iters.] [1.5645e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+2
# order: 4847593.4 size: 68993873 {} [00168.375 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847593.4 size: 68993873 {} [00158.924 ms; 051 iters.] [3.2172e-6 err.] pagerankCuda [static]
# order: 4847593.4 size: 68993873 {} [00045.479 ms; 000 iters.] [3.8000e-7 err.] pagerankNvgraph [dynamic]
# order: 4847593.4 size: 68993873 {} [00020.080 ms; 006 iters.] [2.0526e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+2
# order: 4847664 size: 68994273 {} [00168.223 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847664 size: 68994273 {} [00158.758 ms; 051 iters.] [3.1916e-6 err.] pagerankCuda [static]
# order: 4847664 size: 68994273 {} [00061.011 ms; 000 iters.] [3.5308e-7 err.] pagerankNvgraph [dynamic]
# order: 4847664 size: 68994273 {} [00031.181 ms; 010 iters.] [2.4921e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+3
# order: 4847751.4 size: 68994773 {} [00168.302 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4847751.4 size: 68994773 {} [00158.856 ms; 051 iters.] [3.2357e-6 err.] pagerankCuda [static]
# order: 4847751.4 size: 68994773 {} [00071.313 ms; 000 iters.] [3.8162e-7 err.] pagerankNvgraph [dynamic]
# order: 4847751.4 size: 68994773 {} [00040.616 ms; 013 iters.] [3.0226e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+3
# order: 4848488.2 size: 68998773 {} [00168.424 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4848488.2 size: 68998773 {} [00158.896 ms; 051 iters.] [3.2321e-6 err.] pagerankCuda [static]
# order: 4848488.2 size: 68998773 {} [00097.210 ms; 000 iters.] [3.4122e-7 err.] pagerankNvgraph [dynamic]
# order: 4848488.2 size: 68998773 {} [00053.127 ms; 017 iters.] [3.3493e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+4
# order: 4849358.8 size: 69003772.8 {} [00168.413 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4849358.8 size: 69003772.8 {} [00158.831 ms; 051 iters.] [3.1586e-6 err.] pagerankCuda [static]
# order: 4849358.8 size: 69003772.8 {} [00100.402 ms; 000 iters.] [3.7737e-7 err.] pagerankNvgraph [dynamic]
# order: 4849358.8 size: 69003772.8 {} [00059.900 ms; 019 iters.] [3.7515e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+4
# order: 4856615.8 size: 69043772.8 {} [00168.724 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4856615.8 size: 69043772.8 {} [00157.668 ms; 051 iters.] [3.3104e-6 err.] pagerankCuda [static]
# order: 4856615.8 size: 69043772.8 {} [00115.448 ms; 000 iters.] [3.4372e-7 err.] pagerankNvgraph [dynamic]
# order: 4856615.8 size: 69043772.8 {} [00083.845 ms; 027 iters.] [3.8729e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+5
# order: 4865420 size: 69093773 {} [00169.227 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4865420 size: 69093773 {} [00157.476 ms; 051 iters.] [3.3977e-6 err.] pagerankCuda [static]
# order: 4865420 size: 69093773 {} [00120.910 ms; 000 iters.] [3.5480e-7 err.] pagerankNvgraph [dynamic]
# order: 4865420 size: 69093773 {} [00094.737 ms; 030 iters.] [4.1878e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+5
# order: 4930552 size: 69493772 {} [00169.894 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4930552 size: 69493772 {} [00158.407 ms; 050 iters.] [3.5375e-6 err.] pagerankCuda [static]
# order: 4930552 size: 69493772 {} [00140.122 ms; 000 iters.] [3.3353e-7 err.] pagerankNvgraph [dynamic]
# order: 4930552 size: 69493772 {} [00120.348 ms; 038 iters.] [4.0407e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+6
# order: 4999025.6 size: 69993770 {} [00172.167 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 4999025.6 size: 69993770 {} [00158.961 ms; 049 iters.] [3.8532e-6 err.] pagerankCuda [static]
# order: 4999025.6 size: 69993770 {} [00141.913 ms; 000 iters.] [4.2386e-7 err.] pagerankNvgraph [dynamic]
# order: 4999025.6 size: 69993770 {} [00132.405 ms; 041 iters.] [3.7711e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+6
# order: 5258439.4 size: 73993760.2 {} [00193.943 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 5258439.4 size: 73993760.2 {} [00172.771 ms; 046 iters.] [3.8700e-6 err.] pagerankCuda [static]
# order: 5258439.4 size: 73993760.2 {} [00172.954 ms; 000 iters.] [3.0559e-7 err.] pagerankNvgraph [dynamic]
# order: 5258439.4 size: 73993760.2 {} [00161.416 ms; 043 iters.] [3.7434e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+7
# order: 5321105.6 size: 78993745 {} [00213.113 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 5321105.6 size: 78993745 {} [00189.342 ms; 043 iters.] [3.4230e-6 err.] pagerankCuda [static]
# order: 5321105.6 size: 78993745 {} [00191.422 ms; 000 iters.] [2.9675e-7 err.] pagerankNvgraph [dynamic]
# order: 5321105.6 size: 78993745 {} [00180.495 ms; 041 iters.] [3.4062e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+7
# order: 5332329 size: 118993642.2 {} [00318.800 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# order: 5332329 size: 118993642.2 {} [00300.499 ms; 032 iters.] [2.0933e-6 err.] pagerankCuda [static]
# order: 5332329 size: 118993642.2 {} [00322.030 ms; 000 iters.] [1.3913e-7 err.] pagerankNvgraph [dynamic]
# order: 5332329 size: 118993642.2 {} [00281.682 ms; 030 iters.] [2.7472e-6 err.] pagerankCuda [dynamic]
#
# ...
```

[![](https://i.imgur.com/kdiENBk.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/68DVPzP.jpg)](https://www.youtube.com/watch?v=SoiKp2oSUl0)

[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
[nvGraph]: https://github.com/rapidsai/nvgraph
["graphs"]: https://github.com/puzzlef/graphs
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[scaled-fill]: https://github.com/puzzlef/pagerank-dynamic-adjust-ranks
[charts]: https://photos.app.goo.gl/7zTbHBXV6uh7FGyd8
[sheets]: https://docs.google.com/spreadsheets/d/1TPFX5al0-rlSde0xr7zlfCHNYEqxXSfS6P8QIa2dDsA/edit?usp=sharing

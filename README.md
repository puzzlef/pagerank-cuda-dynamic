Design of CUDA-based Parallel Dynamic [PageRank algorithm] for measuring importance.

PageRank is an algorithm utilized for evaluating the importance of nodes within a network. It assigns numerical scores to nodes based on their link structures. The algorithm is grounded on the notion that nodes with a higher quantity of high-quality inbound links should possess greater significance and consequently, higher ranks. Initially designed for ranking web pages in search engine results, this metric has diversified into numerous domains. Its applications span urban planning, video object tracking, traffic flow prediction, dynamic asset valuation, protein target identification, software system characterization, and the identification of pivotal species for environmental health.

With the proliferation of interconnected data, there has been a notable surge in interest surrounding parallel PageRank computation. Consequently, numerous implementations of parallel PageRank for multicore CPUs have emerged. However, multicore CPUs are constrained by limited memory bandwidth, rendering them ill-suited for graph algorithms like PageRank, which exhibit a low computation-to-communication ratio. In contrast, GPUs offer remarkably high-bandwidth memory, closely integrated with thousands of lightweight cores equipped with user-managed caches. Moreover, GPU hardware is adept at seamlessly switching between running threads at negligible cost to facilitate latency hiding during memory access. When graph algorithms are appropriately optimized for GPU architecture, they demonstrate superior performance compared to parallel CPU-based implementations.

In recent years, considerable research endeavors have been directed towards devising efficient parallel implementations of PageRank tailored for GPUs. These implementations, known as Static PageRank, are designed to calculate ranks from scratch for a given graph, under the assumption that the graph remains static throughout the computation. This paper introduces our GPU implementation of Static PageRank, employing a synchronous pull-based computation method devoid of atomics. Notably, the implementation employs a strategy of partitioning and processing low and high indegree vertices separately using distinct kernels. Furthermore, it circumvents the computation of global teleport rank contribution stemming from dead ends (vertices lacking outgoing edges) by ensuring the input graph contains no such dead ends. To the best of our knowledge, our implementation represents the most efficient approach for parallel PageRank computation on GPUs.

Real-world graphs often exhibit dynamic characteristics, undergoing frequent edge updates. Recomputing PageRank scores for vertices upon each update, known as Static PageRank, can be resource-intensive. To mitigate this, a strategy involves initiating PageRank computation from previous vertex ranks, thereby minimizing the iterations required for convergence. This strategy, referred to as the *Naive-dynamic (ND)* approach, aims to optimize runtime. However, for further optimization, it's crucial to recalculate ranks solely for potentially affected vertices. One common approach, known as the *Dynamic Traversal (DT)* approach, involves identifying reachable vertices from updated graph regions and processing only those. However, marking all reachable vertices as affected, even for minor changes, may lead to unnecessary computation, especially in dense graph regions. To address these concerns, our previous work introduced the incrementally expanding *Dynamic Frontier (DF)* and *Dynamic Frontier with Pruning (DF-P)* approaches. These approaches process only a subset of vertices likely to change ranks and were originally implemented as parallel multicore algorithms. In this paper, we present our GPU implementation of DF-P PageRank, which is based on the Static PageRank method. This implementation features partitioning between low and high outdegree vertices for incremental expansion of affected vertices using two additional kernels.

<br>

Below we illustrate the runtime of Hornet, Gunrock, and our Static PageRank on the GPU, for each graph in the dataset. On the `sk-2005` graph, our Static PageRank computes the ranks of vertices with an iteration tolerance `𝜏` of `10^−10` in `4.2 seconds`, achieving a processing rate of `471 million edges/s`.

[![](https://i.imgur.com/1yDypa3.png)][sheets-o1]

Next, we show the speedup of Our Static PageRank with respect to Hornet and Gunrock. Our Static PageRank is on average `31×` faster than Hornet, and `5.9×` times faster than Gunrock. This speedup is particularly high on the `webbase-2001` graph and road networks with Hornet, and on the `indochina-2004` graph with Gunrock. Further, our GPU implementation of Static PageRank is on average `24×` times faster than our parallel multicore implementation of Static PageRank.

[![](https://i.imgur.com/mhLfKiM.png)][sheets-o1]

Finally, we plot the average time taken by our GPU implementation of Static, Naive-dynamic (ND), Dynamic Traversal (DT), Dynamic Frontier (DF), and Dynamic Frontier with Pruning (DF-P) PageRank on 5 real-world dynamic graphs, with batch updates of size `10^-5|Eᴛ|` to `10^-3|Eᴛ|`. The labels indicate the speedup of each approach with respect to Static PageRank. DF PageRank is, on average, `1.4×` faster than Static PageRank with batch updates of size `10^-5|Eᴛ|`. In contrast, DF-P PageRank is, on average, `3.6×`, `2.0×`, and `1.3×` faster than Static PageRank on batch updates of size `10^-5|Eᴛ|` to `10^-3|Eᴛ|`.

[![](https://i.imgur.com/HlZTmCZ.png)][sheets-o2]

Refer to our technical report for more details: \
[Efficient GPU Implementation of Static and Incrementally Expanding DF-P PageRank for Dynamic Graphs][report].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.

[PageRank algorithm]: https://www.cis.upenn.edu/~mkearns/teaching/NetworkedLife/pagerank.pdf
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[Prof. Hemalatha Eedi]: https://jntuhceh.ac.in/faculty_details/5/dept/369
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[sheets-o1]: https://docs.google.com/spreadsheets/d/1bcjDJ0KNU_2kmIdLuF_eJOZfMxY75K1uUoDhXkt-mAw/edit?usp=sharing
[sheets-o2]: https://docs.google.com/spreadsheets/d/12VGIq2Q8XJH40jlIFaFi6gCA1B2dnslqzwHDQTRCiWc/edit?usp=sharing
[report]: https://arxiv.org/abs/2404.08299

<br>
<br>


### Code structure

The code structure of our GPU implementation of Dynamic Frontier with Pruning (DF-P) PageRank is as follows:

```bash
- inc/_algorithm.hxx: Algorithm utility functions
- inc/_bitset.hxx: Bitset manipulation functions
- inc/_cmath.hxx: Math functions
- inc/_ctypes.hxx: Data type utility functions
- inc/_cuda.hxx: CUDA utility functions
- inc/_debug.hxx: Debugging macros (LOG, ASSERT, ...)
- inc/_iostream.hxx: Input/output stream functions
- inc/_iterator.hxx: Iterator utility functions
- inc/_main.hxx: Main program header
- inc/_mpi.hxx: MPI (Message Passing Interface) utility functions
- inc/_openmp.hxx: OpenMP utility functions
- inc/_queue.hxx: Queue utility functions
- inc/_random.hxx: Random number generation functions
- inc/_string.hxx: String utility functions
- inc/_utility.hxx: Runtime measurement functions
- inc/_vector.hxx: Vector utility functions
- inc/batch.hxx: Batch update generation functions
- inc/bfs.hxx: Breadth-first search algorithms
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/dfs.hxx: Depth-first search algorithms
- inc/duplicate.hxx: Graph duplicating functions
- inc/Graph.hxx: Graph data structure functions
- inc/main.hxx: Main header
- inc/mtx.hxx: Graph file reading functions
- inc/pagerank.hxx: PageRank algorithms
- inc/pagerankPrune.hxx: Dynamic Frontier with Pruning PageRank algorithms
- inc/pagerankCuda.hxx: CUDA-based PageRank algorithms
- inc/properties.hxx: Graph Property functions
- inc/selfLoop.hxx: Graph Self-looping functions
- inc/symmetricize.hxx: Graph Symmetricization functions
- inc/transpose.hxx: Graph transpose functions
- inc/update.hxx: Update functions
- main.cxx: Experimentation code
- process.js: Node.js script for processing output logs
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [The PageRank Citation Ranking: Bringing Order to the Web; Larry Page et al. (1998)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)

<br>
<br>


[![](https://i.imgur.com/3abceEx.png)](https://www.youtube.com/watch?v=yqO7wVBTuLw&pp)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
![](https://ga-beacon.deno.dev/G-KD28SG54JQ:hbAybl6nQFOtmVxW4if3xw/github.com/puzzlef/pagerank-cuda-dynamic)

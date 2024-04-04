#pragma once
#include <tuple>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "_main.hxx"
#include "properties.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::tuple;
using std::vector;
using std::unordered_map;
using std::partition;




#pragma region METHODS
#pragma region UPDATE RANKS
/**
 * Calculate total contribution for a vertex [device function].
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param r previous rank of each vertex
 * @param v given vertex
 * @param i start index
 * @param DI index stride
 * @returns Σ r[u]/deg[u] | u ∈ x.in(v)[i..DI..]
 */
template <class O, class K, class V>
inline V __device__ pagerankCalculateContribCud(const O *xtoff, const K *xtdat, const K *xtedg, const V *r, K v, size_t i, size_t DI) {
  size_t EO = xtoff[v];
  size_t EN = xtoff[v+1] - xtoff[v];
  V a = V();
  for (; i<EN; i+=DI) {
    K u = xtedg[EO+i];
    K d = xtdat[u];
    a  += r[u]/d;
  }
  return a;
}


/**
 * Mark out-neighbors of a vertex as affected [device function].
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param u given vertex
 * @param i start index
 * @param DI index stride
 */
template <class O, class K, class F>
inline void __device__ pagerankMarkNeighborsCudU(F *vaff, const O *xoff, const K *xedg, K u, size_t i, size_t DI) {
  size_t EO = xoff[u];
  size_t EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    vaff[v] = F(1);
  }
}


/**
 * Update ranks for vertices in a graph, using thread-per-vertex approach [kernel].
 * @tparam PARTITION is partitioning enabled?
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam PRUNE dynamic frontier with pruning?
 * @param a current rank of each vertex (updated)
 * @param naff neighbor affected flags (updated)
 * @param vaff vertex affected flags (updated)
 * @param r previous rank of each vertex
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param xtpar partitioned vertices of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <bool PARTITION=false, bool DYNAMIC=false, bool FRONTIER=false, bool PRUNE=false, class O, class K, class V, class F>
void __global__ pagerankUpdateRanksThreadCukU(V *a, F *naff, F *vaff, const V *r, const O *xtoff, const K *xtdat, const K *xtedg, const K *xtpar, K NB, K NE, V C0, V P, V D, V C) {
  DEFINE_CUDA(t, b, B, G);
  for (K i=NB+B*b+t; i<NE; i+=G*B) {
    K v = PARTITION? xtpar[i] : i;
    size_t EN = xtoff[v+1] - xtoff[v];
    if (!PARTITION && EN>64) continue;  // Skip high-degree vertices
    if (DYNAMIC && !vaff[v]) continue;  // Skip unaffected vertices
    // Update rank for vertex v.
    K d  = xtdat[v];
    V rv = r[v];
    V cv = pagerankCalculateContribCud(xtoff, xtdat, xtedg, r, v, 0, 1);
    V av = PRUNE? (C0 + P * (cv - rv/d)) * (1/(1-(P/d))) : C0 + P * cv;
    a[v] = av;
    // Incremental update of affected vertices.
    const auto ev = abs(av - rv);
    if (FRONTIER && PRUNE && ev/max(rv, av) <= C) vaff[v] = F();
    if (FRONTIER && ev/max(rv, av) > D)           naff[v] = F(1);
  }
}


/**
 * Update ranks for vertices in a graph, using thread-per-vertex approach.
 * @tparam PARTITION is partitioning enabled?
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam PRUNE dynamic frontier with pruning?
 * @param a current rank of each vertex (updated)
 * @param naff neighbor affected flags (updated)
 * @param vaff vertex affected flags (updated)
 * @param r previous rank of each vertex
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param xtpar partitioned vertices of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <bool PARTITION=false, bool DYNAMIC=false, bool FRONTIER=false, bool PRUNE=false, class O, class K, class V, class F>
inline void pagerankUpdateRanksThreadCuU(V *a, F *naff, F *vaff, const V *r, const O *xtoff, const K *xtdat, const K *xtedg, const K *xtpar, K NB, K NE, V C0, V P, V D, V C) {
  const int B = blockSizeCu(NE-NB, 512);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  pagerankUpdateRanksThreadCukU<PARTITION, DYNAMIC, FRONTIER, PRUNE><<<G, B>>>(a, naff, vaff, r, xtoff, xtdat, xtedg, xtpar, NB, NE, C0, P, D, C);
}


/**
 * Update ranks for vertices in a graph, using block-per-vertex approach [kernel].
 * @tparam PARTITION is partitioning enabled?
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam PRUNE dynamic frontier with pruning?
 * @param a current rank of each vertex (updated)
 * @param naff neighbor affected flags (updated)
 * @param vaff vertex affected flags (updated)
 * @param r previous rank of each vertex
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param xtpar partitioned vertices of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <bool PARTITION=false, bool DYNAMIC=false, bool FRONTIER=false, bool PRUNE=false, int CACHE=BLOCK_LIMIT_REDUCE_CUDA, class O, class K, class V, class F>
void __global__ pagerankUpdateRanksBlockCukU(V *a, F *naff, F *vaff, const V *r, const O *xtoff, const K *xtdat, const K *xtedg, const K *xtpar, K NB, K NE, V C0, V P, V D, V C) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ V cache[CACHE];
  for (K i=NB+b; i<NE; i+=G) {
    K v = PARTITION? xtpar[i] : i;
    size_t EN = xtoff[v+1] - xtoff[v];
    if (!PARTITION && EN<=64) continue;  // Skip low-degree vertices
    if (DYNAMIC && !vaff[v])  continue;  // Skip unaffected vertices
    // Update rank for vertex v.
    K d  = xtdat[v];
    V rv = r[v];
    cache[t] = pagerankCalculateContribCud(xtoff, xtdat, xtedg, r, v, t, B);
    __syncthreads();
    sumValuesBlockReduceCudU(cache, B, t);
    if (t!=0) continue;
    V cv = cache[0];
    V av = PRUNE? (C0 + P * (cv - rv/d)) * (1/(1-(P/d))) : C0 + P * cv;
    V ev = abs(av - rv);
    a[v] = av;
    if (FRONTIER && PRUNE && ev/max(rv, av) <= C) vaff[v] = F();
    if (FRONTIER && ev/max(rv, av) > D)           naff[v] = F(1);
  }
}


/**
 * Update ranks for vertices in a graph, using block-per-vertex approach.
 * @tparam PARTITION is partitioning enabled?
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam PRUNE dynamic frontier with pruning?
 * @param a current rank of each vertex (updated)
 * @param naff neighbor affected flags (updated)
 * @param vaff vertex affected flags (updated)
 * @param r previous rank of each vertex
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param xtpar partitioned vertices of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <bool PARTITION=false, bool DYNAMIC=false, bool FRONTIER=false, bool PRUNE=false, class O, class K, class V, class F>
inline void pagerankUpdateRanksBlockCuU(V *a, F *naff, F *vaff, const V *r, const O *xtoff, const K *xtdat, const K *xtedg, const K *xtpar, K NB, K NE, V C0, V P, V D, V C) {
  const int B = blockSizeCu<true>(NE-NB, BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  pagerankUpdateRanksBlockCukU<PARTITION, DYNAMIC, FRONTIER, PRUNE><<<G, B>>>(a, naff, vaff, r, xtoff, xtdat, xtedg, xtpar, NB, NE, C0, P, D, C);
}
#pragma endregion




#pragma region INITIALIZE RANKS
/**
 * Intitialize ranks before PageRank iterations [kernel].
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param N number of vertices in the graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class V>
void __global__ pagerankInitializeRanksCukW(V *a, V *r, K N, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K v=NB+B*b+t; v<NE; v+=G*B) {
    r[v] = V(1)/N;
    a[v] = V(1)/N;
  }
}


/**
 * Intitialize ranks before PageRank iterations.
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param N number of vertices in the graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class V>
inline void pagerankInitializeRanksCuW(V *a, V *r, K N, K NB, K NE) {
  const int B = blockSizeCu(N,   BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (N, B, GRID_LIMIT_MAP_CUDA);
  pagerankInitializeRanksCukW<<<G, B>>>(a, r, N, NB, NE);
}
#pragma endregion




#pragma region MARK AFFECTED DYNAMIC FRONTIER
/**
 * Find affected vertices and neighbors due to a batch update with Dynamic Frontier approach [kernel].
 * @param naff neighbor affected flags (updated)
 * @param vaff vertex affected flags (updated)
 * @param delu source vertices of edge deletions
 * @param delv target vertices of edge deletions
 * @param insu source vertices of edge insertions
 * @param ND number of edge deletions
 * @param NI number of edge insertions
 */
template <class K, class F>
void __global__ pagerankAffectedFrontierPartialCukU(F *naff, F *vaff, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI) {
  DEFINE_CUDA(t, b, B, G);
  for (size_t i=B*b+t; i<ND+NI; i+=G*B) {
    K u = i<ND? delu[i] : insu[i-ND];
    K v = i<ND? delv[i] : K();
    naff[u] = F(1);
    if (i<ND) vaff[v] = F(1);
  }
}


/**
 * Find affected vertices and neighbors due to a batch update with Dynamic Frontier approach.
 * @param naff neighbor affected flags (updated)
 * @param vaff vertex affected flags (updated)
 * @param delu source vertices of edge deletions
 * @param delv target vertices of edge deletions
 * @param insu source vertices of edge insertions
 * @param ND number of edge deletions
 * @param NI number of edge insertions
 */
template <class K, class F>
inline void pagerankAffectedFrontierPartialCuU(F *naff, F *vaff, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI) {
  const int B = blockSizeCu(ND+NI, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (ND+NI, B, GRID_LIMIT_MAP_CUDA);
  pagerankAffectedFrontierPartialCukU<<<G, B>>>(naff, vaff, delu, delv, insu, ND, NI);
}


/**
 * Find affected vertices due to current set of affected vertices, using thread-per-vertex approach [kernel].
 * @tparam PARTITION is partitioning enabled?
 * @param vaff vertex affected flags (updated)
 * @param naff neighbor affected flags
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xpar partitioned vertices of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <bool PARTITION=false, class F, class O, class K>
void __global__ pagerankAffectedExtendThreadCukU(F *vaff, const F *naff, const O *xoff, const K *xedg, const K *xpar, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K i=NB+B*b+t; i<NE; i+=G*B) {
    K u = PARTITION? xpar[i] : i;
    size_t EN = xoff[u+1] - xoff[u];
    if (!PARTITION && EN>64) continue;
    if (!naff[u]) continue;
    pagerankMarkNeighborsCudU(vaff, xoff, xedg, u, 0, 1);
  }
}


/**
 * Find affected vertices due to current set of affected vertices, using thread-per-vertex approach.
 * @tparam PARTITION is partitioning enabled?
 * @param vaff vertex affected flags (updated)
 * @param naff neighbor affected flags
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xpar partitioned vertices of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <bool PARTITION=false, class F, class O, class K>
inline void pagerankAffectedExtendThreadCuU(F *vaff, const F *naff, const O *xoff, const K *xedg, const K *xpar, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  pagerankAffectedExtendThreadCukU<PARTITION><<<G, B>>>(vaff, naff, xoff, xedg, xpar, NB, NE);
}


/**
 * Find affected vertices due to current set of affected vertices, using block-per-vertex approach [kernel].
 * @tparam PARTITION is partitioning enabled?
 * @param vaff vertex affected flags (updated)
 * @param naff neighbor affected flags
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xpar partitioned vertices of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <bool PARTITION=false, class F, class O, class K>
void __global__ pagerankAffectedExtendBlockCukU(F *vaff, const F *naff, const O *xoff, const K *xedg, const K *xpar, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K i=NB+b; i<NE; i+=G) {
    K u = PARTITION? xpar[i] : i;
    size_t EN = xoff[u+1] - xoff[u];
    if (!PARTITION && EN<=64) continue;
    if (!naff[u]) continue;
    pagerankMarkNeighborsCudU(vaff, xoff, xedg, u, t, B);
  }
}


/**
 * Find affected vertices due to current set of affected vertices, using block-per-vertex approach.
 * @tparam PARTITION is partitioning enabled?
 * @param vaff vertex affected flags (updated)
 * @param naff neighbor affected flags
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xpar partitioned vertices of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <bool PARTITION=false, class F, class O, class K>
inline void pagerankAffectedExtendBlockCuU(F *vaff, const F *naff, const O *xoff, const K *xedg, const K *xpar, K NB, K NE) {
  const int B = blockSizeCu<true>(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  pagerankAffectedExtendBlockCukU<PARTITION><<<G, B>>>(vaff, naff, xoff, xedg, xpar, NB, NE);
}


/**
 * Find affected vertices due to a batch update with Dynamic Frontier approach.
 * @tparam PARTITION is partitioning enabled?
 * @param vaff vertex affected flags (output)
 * @param naff neighbor affected flags (scratch)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xpar partitioned vertices of original graph
 * @param delu source vertices of edge deletions
 * @param delv target vertices of edge deletions
 * @param insu source vertices of edge insertions
 * @param ND number of edge deletions
 * @param NI number of edge insertions
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param NP partition point of original graph
 */
template <bool PARTITION=false, class F, class O, class K>
inline void pagerankAffectedFrontierCuW(F *vaff, F *naff, const O *xoff, const K *xedg, const K *xpar, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI, K NB, K NE, K NP) {
  fillValueCuW(vaff+NB, NE-NB, F());
  fillValueCuW(naff+NB, NE-NB, F());
  pagerankAffectedFrontierPartialCuU(naff, vaff, delu, delv, insu, ND, NI);
  pagerankAffectedExtendThreadCuU<PARTITION>(vaff, naff, xoff, xedg, xpar, NB, PARTITION? NP : NE);
  pagerankAffectedExtendBlockCuU <PARTITION>(vaff, naff, xoff, xedg, xpar, PARTITION? NP : NB, NE);
}
#pragma endregion




#pragma region MARK AFFECTED DYNAMIC TRAVERSAL
/**
 * Find affected vertices due to a batch update with Dynamic Traversal approach.
 * @tparam PARTITION is partitioning enabled?
 * @param vaff vertex affected flags (output)
 * @param naff neighbor affected flags (scratch)
 * @param buff buffer for affected flags (scratch)
 * @param bufs buffer for temporary values (scratch)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xpar partitioned vertices of original graph
 * @param delu source vertices of edge deletions
 * @param delv target vertices of edge deletions
 * @param insu source vertices of edge insertions
 * @param ND number of edge deletions
 * @param NI number of edge insertions
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param NP partition point of original graph
 */
template <bool PARTITION=false, class F, class O, class K>
inline void pagerankAffectedTraversalCuW(F *vaff, F *naff, F *buff, uint64_cu* bufs, const O *xoff, const K *xedg, const K *xpar, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI, K NB, K NE, K NP) {
  uint64_cu count = 0, countNew = 0;
  pagerankAffectedFrontierCuW<PARTITION>(vaff, naff, xoff, xedg, xpar, delu, delv, insu, ND, NI, NB, NE, NP);
  copyValuesCuW(naff, vaff, NE-NB);
  countValuesInplaceCuW(bufs, vaff, NE-NB, F(1));
  TRY_CUDA( cudaMemcpy(&count, bufs, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
  while (true) {
    fillValueCuW(buff, NE-NB, F());
    pagerankAffectedExtendThreadCuU<PARTITION>(buff, naff, xoff, xedg, xpar, NB, PARTITION? NP : NE);
    pagerankAffectedExtendBlockCuU <PARTITION>(buff, naff, xoff, xedg, xpar, PARTITION? NP : NB, NE);
    bitwiseOrCuW(vaff, vaff, buff, NE-NB);
    swap(naff, buff);
    countValuesInplaceCuW(bufs, vaff, NE-NB, F(1));
    TRY_CUDA( cudaMemcpy(&countNew, bufs, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
    if (countNew==count) break;
    count = countNew;
  }
}
#pragma endregion




#pragma region PARTITION
/**
 * Partition vertices into low-degree and high-degree sets.
 * @param ks vertex keys (updated)
 * @param xt transposed graph
 * @returns number of low-degree vertices
 */
template <class H, class K>
inline K pagerankPartitionVerticesCudaU(vector<K>& ks, const H& xt) {
  K SWITCH_DEGREE = 64;  // Switch to block-per-vertex approach if degree >= SWITCH_DEGREE
  K SWITCH_LIMIT  = 64;  // Avoid switching if number of vertices < SWITCH_LIMIT
  size_t N = ks.size();
  auto  kb = ks.begin(), ke = ks.end();
  auto  ft = [&](K v) { return xt.degree(v) < SWITCH_DEGREE; };
  partition(kb, ke, ft);
  size_t n = count_if(kb, ke, ft);
  if (n   < SWITCH_LIMIT) n = 0;
  if (N-n < SWITCH_LIMIT) n = N;
  return K(n);
}


/**
 * Test if degree of each vertex in a graph is less than or equal to a given value [kernel].
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
void __global__ testDegreeLessOrEqualCukW(K *a, const O *xoff, K N, K d) {
  DEFINE_CUDA(t, b, B, G);
  for (K v=B*b+t; v<N; v+=G*B) {
    size_t EN = xoff[v+1] - xoff[v];
    a[v] = EN <= d;
  }
}


/**
 * Test if degree of each vertex in a graph is less than or equal to a given value.
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
inline void testDegreeLessOrEqualCuW(K *a, const O *xoff, K N, K d) {
  const int B = blockSizeCu(N, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (N, B, GRID_LIMIT_MAP_CUDA);
  testDegreeLessOrEqualCukW<<<G, B>>>(a, xoff, N, d);
}


/**
 * Test if degree of each vertex in a graph is greater than a given value [kernel].
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
void __global__ testDegreeGreaterCukW(K *a, const O *xoff, K N, K d) {
  DEFINE_CUDA(t, b, B, G);
  for (K v=B*b+t; v<N; v+=G*B) {
    size_t EN = xoff[v+1] - xoff[v];
    a[v] = EN > d;
  }
}


/**
 * Test if degree of each vertex in a graph is greater than a given value.
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
inline void testDegreeGreaterCuW(K *a, const O *xoff, K N, K d) {
  const int B = blockSizeCu(N, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (N, B, GRID_LIMIT_MAP_CUDA);
  testDegreeGreaterCukW<<<G, B>>>(a, xoff, N, d);
}


/**
 * Populate vertices in an array if its degree is less than or equal to a given value [kernel].
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param xidx index of vertices in result array
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
void __global__ fillVerticesLessOrEqualCukW(K *a, const O *xoff, const K *xidx, K N, K d) {
  DEFINE_CUDA(t, b, B, G);
  for (K v=B*b+t; v<N; v+=G*B) {
    size_t EN = xoff[v+1] - xoff[v];
    if (EN <= d) a[xidx[v]] = v;
  }
}


/**
 * Populate vertices in an array if its degree is less than or equal to a given value.
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param xidx index of vertices in result array
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
inline void fillVerticesLessOrEqualCuW(K *a, const O *xoff, const K *xidx, K N, K d) {
  const int B = blockSizeCu(N, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (N, B, GRID_LIMIT_MAP_CUDA);
  fillVerticesLessOrEqualCukW<<<G, B>>>(a, xoff, xidx, N, d);
}


/**
 * Populate vertices in an array if its degree is greater than a given value [kernel].
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param xidx index of vertices in result array
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
void __global__ fillVerticesGreaterCukW(K *a, const O *xoff, const K *xidx, K N, K d) {
  DEFINE_CUDA(t, b, B, G);
  for (K v=B*b+t; v<N; v+=G*B) {
    size_t EN = xoff[v+1] - xoff[v];
    if (EN > d) a[xidx[v]] = v;
  }
}


/**
 * Populate vertices in an array if its degree is greater than a given value.
 * @param a result array (output)
 * @param xoff offsets of original graph
 * @param xidx index of vertices in result array
 * @param N number of vertices in the graph
 * @param d degree to test against
 */
template <class O, class K>
inline void fillVerticesGreaterCuW(K *a, const O *xoff, const K *xidx, K N, K d) {
  const int B = blockSizeCu(N, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (N, B, GRID_LIMIT_MAP_CUDA);
  fillVerticesGreaterCukW<<<G, B>>>(a, xoff, xidx, N, d);
}


/**
 * Partition vertices into low-degree and high-degree sets.
 * @param xpar partitioned vertices (output)
 * @param bufk buffer for temporary values (scratch, size N+1)
 * @param bufkx buffer for temporary values (scratch, size N)
 * @param xoff offsets of original graph
 * @param N number of vertices in the graph
 * @returns partition point of original graph
 */
template <class O, class K>
inline K pagerankPartitionVerticesCuW(K *xpar, K *bufk, K *bufkx, const O *xoff, K N) {
  K switchDegree = 64;
  testDegreeLessOrEqualCuW(bufk, xoff, N, switchDegree);
  fillValueCuW(bufk+N, 1, K());
  exclusiveScanCubW(bufk, bufkx, bufk, N+1, N);
  fillVerticesLessOrEqualCuW(xpar, xoff, bufk, N, switchDegree);
  K NP = K();
  TRY_CUDA( cudaMemcpy(&NP, bufk+N, sizeof(K), cudaMemcpyDeviceToHost) );
  testDegreeGreaterCuW(bufk, xoff, N, switchDegree);
  exclusiveScanCubW(bufk, bufkx, bufk, N, N);
  fillVerticesGreaterCuW(xpar+NP, xoff, bufk, N, switchDegree);
  return NP;
}
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform PageRank iterations upon a graph.
 * @tparam PARTITION is partitioning enabled?
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam PRUNE dynamic frontier with pruning?
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param naff neighbor affected flags (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufv buffer for temporary values (scratch, size R)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xpar partitioned vertices of original graph
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param xtpar partitioned vertices of transposed graph
 * @param N number of vertices in the graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param NP partition point of original graph
 * @param NQ partition point of transposed graph
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param D frontier tolerance
 * @param C prune tolerance
 * @returns number of iterations performed
 */
template <bool PARTITION=false, bool DYNAMIC=false, bool FRONTIER=false, bool PRUNE=false, class O, class K, class V, class F>
inline int pagerankLoopCuU(V *a, V *r, F *naff, F *vaff, V *bufv, const O *xoff, const K *xedg, const K *xpar, const O *xtoff, const K *xtdat, const K *xtedg, const K *xtpar, K N, K NB, K NE, K NP, K NQ, V P, V E, int L, V D, V C) {
  int l = 0;
  V  el = V();
  V  C0 = (1-P)/N;
  while (l<L) {
    if (FRONTIER) fillValueCuW(naff, N, F());
    pagerankUpdateRanksThreadCuU<PARTITION, DYNAMIC, FRONTIER, PRUNE>(a, naff, vaff, r, xtoff, xtdat, xtedg, xtpar, NB, PARTITION? NQ : NE, C0, P, D, C);
    pagerankUpdateRanksBlockCuU <PARTITION, DYNAMIC, FRONTIER, PRUNE>(a, naff, vaff, r, xtoff, xtdat, xtedg, xtpar, PARTITION? NQ : NB, NE, C0, P, D, C); ++l;
    liNormDeltaInplaceCuW(bufv, a, r, N);  // Compare previous and current ranks
    TRY_CUDA( cudaMemcpy(&el, bufv, sizeof(V), cudaMemcpyDeviceToHost) );
    if (FRONTIER) pagerankAffectedExtendThreadCuU<PARTITION>(vaff, naff, xoff, xedg, xpar, NB, PARTITION? NP : NE);
    if (FRONTIER) pagerankAffectedExtendBlockCuU <PARTITION>(vaff, naff, xoff, xedg, xpar, PARTITION? NP : NB, NE);
    swap(a, r);  // Final ranks in (r)
    if (el<E) break;
  }
  return l;
}
#pragma endregion




#pragma region ENVIROMENT SETUP
/**
 * Setup environment and find the rank of each vertex in a graph.
 * @tparam PARTITION is partitioning enabled?
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam PRUNE dynamic frontier with pruning?
 * @tparam FLAG type of vertex affected flags
 * @param x original graph
 * @param xt transposed graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fm function to mark affected vertices
 * @returns pagerank result
 */
template <bool PARTITION=false, bool DYNAMIC=false, bool FRONTIER=false, bool PRUNE=false, class FLAG=char, class G, class H, class K, class V, class FM>
inline PagerankResult<V> pagerankInvokeCuda(const G& x, const H& xt, const vector<tuple<K, K>> deletions, const vector<tuple<K, K>> insertions, const vector<V> *q, const PagerankOptions<V>& o, FM fm) {
  using  O = uint32_t;
  using  F = FLAG;
  size_t S = xt.span();
  size_t N = xt.order();
  size_t M = xt.size();
  size_t ND = deletions.size();
  size_t NI = insertions.size();
  V   P  = o.damping;
  V   E  = o.tolerance;
  V   D  = o.frontierTolerance;
  V   C  = o.pruneTolerance;
  int L  = o.maxIterations, l = 0;
  int R  = reduceSizeCu(N);
  vector<K> xoff, xedg;
  vector<K> xtoff(N+1), xtedg(M), xtdat(N);
  vector<V> r(S), rc(N), qc(N);
  vector<K> delu(ND), delv(ND), insu(NI);
  if (DYNAMIC) xoff.resize(N+1);
  if (DYNAMIC) xedg.resize(M);
  O *xoffD  = nullptr;
  K *xedgD  = nullptr;
  K *xparD  = nullptr;
  O *xtoffD = nullptr;
  K *xtedgD = nullptr;
  K *xtdatD = nullptr;
  K *xtparD = nullptr;
  V *aD     = nullptr;
  V *rD     = nullptr;
  F *naffD  = nullptr;
  F *vaffD  = nullptr;
  K *deluD  = nullptr;
  K *delvD  = nullptr;
  K *insuD  = nullptr;
  K *bufkD  = nullptr;
  K *bufkxD = nullptr;
  V *bufvD  = nullptr;
  F *buffD  = nullptr;
  uint64_cu* bufsD = nullptr;
  K NB = K(), NE = K(N);
  K NP = K(), NQ = K();
  // Obtain data for CSR.
  vector<K> ks = vertexKeys(xt);
  if (DYNAMIC) csrCreateOffsetsW (xoff,  x,  ks);
  if (DYNAMIC) csrCreateEdgeKeysW(xedg,  x,  ks);
  csrCreateOffsetsW (xtoff, xt, ks);
  csrCreateEdgeKeysW(xtedg, xt, ks);
  csrCreateVertexValuesW(xtdat, xt, ks);
  // Obtain initial ranks.
  if (q) gatherValuesW(qc, *q, ks);
  // Obtain batch update data.
  unordered_map<K, K> ksMap;
  for (K i=0; i<ks.size(); ++i)
    ksMap[ks[i]] = K(i);
  for (size_t i=0; i<ND; ++i) {
    delu[i] = ksMap[get<0>(deletions[i])];
    delv[i] = ksMap[get<1>(deletions[i])];
  }
  for (size_t i=0; i<NI; ++i) {
    insu[i] = ksMap[get<0>(insertions[i])];
  }
  ksMap.clear();
  // Allocate device memory.
  TRY_CUDA( cudaSetDeviceFlags(cudaDeviceMapHost) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&xoffD,  (N+1) * sizeof(O)) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&xedgD,   M    * sizeof(K)) );
  if (DYNAMIC && PARTITION) TRY_CUDA( cudaMalloc(&xparD,   N * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xtoffD, (N+1) * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&xtedgD,  M    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xtdatD,  N    * sizeof(K)) );
  if (PARTITION) TRY_CUDA( cudaMalloc(&xtparD,  N * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&aD,    N * sizeof(V)) );
  TRY_CUDA( cudaMalloc(&rD,    N * sizeof(V)) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&naffD, N * sizeof(F)) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&vaffD, N * sizeof(F)) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&deluD, ND * sizeof(K)) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&delvD, ND * sizeof(K)) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&insuD, NI * sizeof(K)) );
  if (PARTITION) TRY_CUDA( cudaMalloc(&bufkD,  (N+1) * sizeof(K)) );
  if (PARTITION) TRY_CUDA( cudaMalloc(&bufkxD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&bufvD, R * sizeof(V)) );
  TRY_CUDA( cudaMalloc(&bufsD, R * sizeof(uint64_cu)) );
  if (DYNAMIC && !FRONTIER) TRY_CUDA( cudaMalloc(&buffD,  N * sizeof(F)) );
  // Copy data to device.
  if (DYNAMIC) TRY_CUDA( cudaMemcpy(xoffD,  xoff .data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
  if (DYNAMIC) TRY_CUDA( cudaMemcpy(xedgD,  xedg .data(),  M    * sizeof(K), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xtoffD, xtoff.data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xtedgD, xtedg.data(),  M    * sizeof(K), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xtdatD, xtdat.data(),  N    * sizeof(K), cudaMemcpyHostToDevice) );
  if (DYNAMIC) TRY_CUDA( cudaMemcpy(deluD, delu.data(), ND * sizeof(K), cudaMemcpyHostToDevice) );
  if (DYNAMIC) TRY_CUDA( cudaMemcpy(delvD, delv.data(), ND * sizeof(K), cudaMemcpyHostToDevice) );
  if (DYNAMIC) TRY_CUDA( cudaMemcpy(insuD, insu.data(), NI * sizeof(K), cudaMemcpyHostToDevice) );
  // Perform PageRank algorithm on device.
  float ti = 0, tm = 0, tc = 0;
  float t  = measureDurationMarked([&](auto mark) {
    // Partition vertices into low-degree and high-degree sets.
    if (q) TRY_CUDA( cudaMemcpy(rD, qc.data(), N * sizeof(V), cudaMemcpyHostToDevice) );
    ti += mark([&]() {
      if (PARTITION) NQ = pagerankPartitionVerticesCuW(xtparD, bufkD, bufkxD, xtoffD, K(N));
      if (DYNAMIC && PARTITION) NP = pagerankPartitionVerticesCuW(xparD, bufkD, bufkxD, xoffD, K(N));
    });
    // Setup initial ranks.
    ti += mark([&]() {
      if (q) copyValuesCuW(aD, rD, N);
      else   pagerankInitializeRanksCuW(aD, rD, K(N), NB, NE);
    });
    // Mark initial affected vertices.
    if (DYNAMIC) tm += mark([&]() { fm(vaffD, naffD, buffD, bufsD, xoffD, xedgD, xparD, deluD, delvD, insuD, ND, NI, NB, NE, NP); });
    // Perform PageRank iterations.
    tc += mark([&]() { l = pagerankLoopCuU<PARTITION, DYNAMIC, FRONTIER, PRUNE>(aD, rD, naffD, vaffD, bufvD, xoffD, xedgD, xparD, xtoffD, xtdatD, xtedgD, xtparD, K(N), NB, NE, NP, NQ, P, E, L, D, C); });
  }, o.repeat);
  // Obtain final ranks.
  TRY_CUDA( cudaMemcpy(rc.data(), rD, N * sizeof(V), cudaMemcpyDeviceToHost) );
  scatterValuesW(r, rc, ks);
  // Free device memory.
  if (DYNAMIC) TRY_CUDA( cudaFree(xoffD) );
  if (DYNAMIC) TRY_CUDA( cudaFree(xedgD) );
  if (DYNAMIC && PARTITION) TRY_CUDA( cudaFree(xparD) );
  TRY_CUDA( cudaFree(xtoffD) );
  TRY_CUDA( cudaFree(xtedgD) );
  TRY_CUDA( cudaFree(xtdatD) );
  if (PARTITION) TRY_CUDA( cudaFree(xtparD) );
  TRY_CUDA( cudaFree(aD) );
  TRY_CUDA( cudaFree(rD) );
  if (DYNAMIC) TRY_CUDA( cudaFree(naffD) );
  if (DYNAMIC) TRY_CUDA( cudaFree(vaffD) );
  if (DYNAMIC) TRY_CUDA( cudaFree(deluD) );
  if (DYNAMIC) TRY_CUDA( cudaFree(delvD) );
  if (DYNAMIC) TRY_CUDA( cudaFree(insuD) );
  if (PARTITION) TRY_CUDA( cudaFree(bufkD) );
  if (PARTITION) TRY_CUDA( cudaFree(bufkxD) );
  TRY_CUDA( cudaFree(bufvD) );
  TRY_CUDA( cudaFree(bufsD) );
  if (DYNAMIC && !FRONTIER) TRY_CUDA( cudaFree(buffD) );
  return {r, l, t, ti/o.repeat, tm/o.repeat, tc/o.repeat};
}
#pragma endregion




#pragma region STATIC
/**
 * Find the rank of each vertex in a static graph with Static approach.
 * @tparam PARTITION is partitioning enabled?
 * @param x original graph
 * @param xt transposed graph
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool PARTITION=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankStaticCuda(const G& x, const H& xt, const PagerankOptions<V>& o) {
  using K = typename G::key_type;
  using O = uint32_t;
  using F = FLAG;
  if (xt.empty()) return {};
  vector<tuple<K, K>> deletions;
  vector<tuple<K, K>> insertions;
  vector<V> *q = nullptr;
  auto fm = [&](F *vaff, F *naff, F *buff, uint64_cu* bufs, const O *xoff, const K *xedg, const K *xpar, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI, K NB, K NE, K NP) {};
  return pagerankInvokeCuda<PARTITION, false, false, false, FLAG>(x, xt, deletions, insertions, q, o, fm);
}
#pragma endregion




#pragma region NAIVE-DYNAMIC
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @tparam PARTITION is partitioning enabled?
 * @param x original graph
 * @param xt transposed graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool PARTITION=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankNaiveDynamicCuda(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename G::key_type;
  using O = uint32_t;
  using F = FLAG;
  if (xt.empty()) return {};
  vector<tuple<K, K>> deletions;
  vector<tuple<K, K>> insertions;
  auto fm = [&](F *vaff, F *naff, F *buff, uint64_cu* bufs, const O *xoff, const K *xedg, const K *xpar, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI, K NB, K NE, K NP) {};
  return pagerankInvokeCuda<PARTITION, false, false, false, FLAG>(x, xt, deletions, insertions, q, o, fm);
}
#pragma endregion




#pragma region DYNAMIC TRAVERSAL
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Traversal approach.
 * @tparam PARTITION is partitioning enabled?
 * @param x original graph
 * @param xt transpose of original graph
 * @param y updated graph
 * @param yt transpose of updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool PARTITION=false, class FLAG=char, class G, class H, class K, class V>
inline PagerankResult<V> pagerankDynamicTraversalCuda(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  using O = uint32_t;
  using F = FLAG;
  if (xt.empty()) return {};
  auto fm = [&](F *vaff, F *naff, F *buff, uint64_cu* bufs, const O *xoff, const K *xedg, const K *xpar, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI, K NB, K NE, K NP) {
    if (ND>0 || NI>0) pagerankAffectedTraversalCuW<PARTITION>(vaff, naff, buff, bufs, xoff, xedg, xpar, delu, delv, insu, ND, NI, NB, NE, NP);
  };
  return pagerankInvokeCuda<PARTITION, true, false, false, FLAG>(y, yt, deletions, insertions, q, o, fm);
}
#pragma endregion




#pragma region DYNAMIC FRONTIER
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Frontier approach.
 * @tparam PARTITION is partitioning enabled?
 * @param x original graph
 * @param xt transpose of original graph
 * @param y updated graph
 * @param yt transpose of updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool PARTITION=false, class FLAG=char, class G, class H, class K, class V>
inline PagerankResult<V> pagerankDynamicFrontierCuda(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  using O = uint32_t;
  using F = FLAG;
  if (xt.empty()) return {};
  auto fm = [&](F *vaff, F *naff, F *buff, uint64_cu* bufs, const O *xoff, const K *xedg, const K *xpar, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI, K NB, K NE, K NP) {
    if (ND>0 || NI>0) pagerankAffectedFrontierCuW<PARTITION>(vaff, naff, xoff, xedg, xpar, delu, delv, insu, ND, NI, NB, NE, NP);
  };
  return pagerankInvokeCuda<PARTITION, true, true, false, FLAG>(y, yt, deletions, insertions, q, o, fm);
}
#pragma endregion




#pragma region DYNAMIC FRONTIER WITH PRUNING
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Frontier with Pruning approach.
 * @tparam PARTITION is partitioning enabled?
 * @param x original graph
 * @param xt transpose of original graph
 * @param y updated graph
 * @param yt transpose of updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool PARTITION=false, class FLAG=char, class G, class H, class K, class V>
inline PagerankResult<V> pagerankPruneDynamicFrontierCuda(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  using O = uint32_t;
  using F = FLAG;
  if (xt.empty()) return {};
  auto fm = [&](F *vaff, F *naff, F *buff, uint64_cu* bufs, const O *xoff, const K *xedg, const K *xpar, const K *delu, const K *delv, const K *insu, size_t ND, size_t NI, K NB, K NE, K NP) {
    if (ND>0 || NI>0) pagerankAffectedFrontierCuW<PARTITION>(vaff, naff, xoff, xedg, xpar, delu, delv, insu, ND, NI, NB, NE, NP);
  };
  return pagerankInvokeCuda<PARTITION, true, true, true, FLAG>(y, yt, deletions, insertions, q, o, fm);
}
#pragma endregion
#pragma endregion

#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "properties.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::vector;
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
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam ASYNC asynchronous computation?
 * @param a current rank of each vertex (updated)
 * @param vaff vertex affected flags (updated)
 * @param updt vertices under update flag (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param E iteration tolerance [10^-10]
 * @param D frontier tolerance
 */
template <bool DYNAMIC=false, bool FRONTIER=false, bool ASYNC=false, class O, class K, class V, class F>
void __global__ pagerankUpdateRanksThreadCukU(V *a, F *vaff, F *updt, const O *xoff, const K *xedg, const O *xtoff, const K *xtdat, const K *xtedg, K NB, K NE, const V *r, V C0, V P, V E, V D) {
  DEFINE_CUDA(t, b, B, G);
  V et = V();
  for (K v=NB+B*b+t; v<NE; v+=G*B) {
    if (DYNAMIC && !vaff[v]) continue;  // Skip unaffected vertices
    // Update rank for vertex v.
    V rv = r[v];
    V cv = pagerankCalculateContribCud(xtoff, xtdat, xtedg, r, v, 0, 1);
    V av = C0 + P * cv;
    a[v] = av;
    if (ASYNC) et = max(et, abs(av - rv));
    if (!FRONTIER || abs(av - rv) <= D) continue;
    // Mark neighbors as affected, if rank change significant.
    pagerankMarkNeighborsCudU(vaff, xoff, xedg, v, 0, 1);
  }
  if (ASYNC && et>E && !updt[0]) updt[0] = F(1);
}


/**
 * Update ranks for vertices in a graph, using thread-per-vertex approach.
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam ASYNC asynchronous computation?
 * @param a current rank of each vertex (updated)
 * @param vaff vertex affected flags (updated)
 * @param updt vertices under update flag (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param E iteration tolerance [10^-10]
 * @param D frontier tolerance
 */
template <bool DYNAMIC=false, bool FRONTIER=false, bool ASYNC=false, class O, class K, class V, class F>
inline void pagerankUpdateRanksThreadCuU(V *a, F *vaff, F *updt, const O *xoff, const K *xedg, const O *xtoff, const K *xtdat, const K *xtedg, K NB, K NE, const V *r, V C0, V P, V E, V D) {
  const int B = blockSizeCu(NE-NB, 512);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  pagerankUpdateRanksThreadCukU<DYNAMIC, FRONTIER, ASYNC><<<G, B>>>(a, vaff, updt, xoff, xedg, xtoff, xtdat, xtedg, NB, NE, r, C0, P, E, D);
}


/**
 * Update ranks for vertices in a graph, using block-per-vertex approach [kernel].
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam ASYNC asynchronous computation?
 * @tparam CACHE size of shared memory cache
 * @param a current rank of each vertex (updated)
 * @param vaff vertex affected flags (updated)
 * @param updt vertices under update flag (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param E iteration tolerance [10^-10]
 * @param D frontier tolerance
 */
template <bool DYNAMIC=false, bool FRONTIER=false, bool ASYNC=false, int CACHE=BLOCK_LIMIT_REDUCE_CUDA, class O, class K, class V, class F>
void __global__ pagerankUpdateRanksBlockCukU(V *a, F *vaff, F *updt, const O *xoff, const K *xedg, const O *xtoff, const K *xtdat, const K *xtedg, K NB, K NE, const V *r, V C0, V P, V E, V D) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ V cache[CACHE];
  V eb = V();
  for (K v=NB+b; v<NE; v+=G) {
    if (DYNAMIC && !vaff[v]) continue;  // Skip unaffected vertices
    // Update rank for vertex v.
    V rv = r[v];
    cache[t] = pagerankCalculateContribCud(xtoff, xtdat, xtedg, r, v, t, B);
    __syncthreads();
    sumValuesBlockReduceCudU(cache, B, t);
    V cv = cache[0];
    V av = C0 + P * cv;
    if (t==0) a[v] = av;
    if (ASYNC && t==0) eb = max(eb, abs(av - rv));
    if (!FRONTIER || abs(av - rv) <= D) continue;
    // Mark neighbors as affected, if rank change significant.
    pagerankMarkNeighborsCudU(vaff, xoff, xedg, v, t, B);
  }
  if (ASYNC && t==0 && eb>E && !updt[0]) updt[0] = F(1);
}


/**
 * Update ranks for vertices in a graph, using block-per-vertex approach.
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam ASYNC asynchronous computation?
 * @param a current rank of each vertex (updated)
 * @param vaff vertex affected flags (updated)
 * @param updt vertices under update flag (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param E iteration tolerance [10^-10]
 * @param D frontier tolerance
 */
template <bool DYNAMIC=false, bool FRONTIER=false, bool ASYNC=false, class O, class K, class V, class F>
inline void pagerankUpdateRanksBlockCuU(V *a, F *vaff, F *updt, const O *xoff, const K *xedg, const O *xtoff, const K *xtdat, const K *xtedg, K NB, K NE, const V *r, V C0, V P, V E, V D) {
  const int B = blockSizeCu<true>(NE-NB, BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  pagerankUpdateRanksBlockCukU<DYNAMIC, FRONTIER, ASYNC><<<G, B>>>(a, vaff, updt, xoff, xedg, xtoff, xtdat, xtedg, NB, NE, r, C0, P, E, D);
}
#pragma endregion




#pragma region INITIALIZE RANKS
/**
 * Intitialize ranks before PageRank iterations [kernel].
 * @tparam ASYNC asynchronous computation?
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param N number of vertices in the graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <bool ASYNC=false, class K, class V>
void __global__ pagerankInitializeRanksCukW(V *a, V *r, K N, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K v=NB+B*b+t; v<NE; v+=G*B) {
    r[v] = V(1)/N;
    if (!ASYNC) a[v] = V(1)/N;
  }
}


/**
 * Intitialize ranks before PageRank iterations.
 * @tparam ASYNC asynchronous computation?
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param N number of vertices in the graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <bool ASYNC=false, class K, class V>
inline void pagerankInitializeRanksCuW(V *a, V *r, K N, K NB, K NE) {
  const int B = blockSizeCu(N,   BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (N, B, GRID_LIMIT_MAP_CUDA);
  pagerankInitializeRanksCukW<<<G, B>>>(a, r, N, NB, NE);
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
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform PageRank iterations upon a graph.
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam ASYNC asynchronous computation?
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param vaff vertex affected flags (updated)
 * @param bufv buffer for temporary values
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xtoff offsets of transposed graph
 * @param xtdat vertex data of transposed graph (out-degrees)
 * @param xtedg edge keys of transposed graph
 * @param N number of vertices in the graph
 * @param NB begin vertex (inclusive)
 * @param NM middle vertex
 * @param NE end vertex (exclusive)
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param D frontier tolerance
 * @returns number of iterations performed
 */
template <bool DYNAMIC=false, bool FRONTIER=false, bool ASYNC=false, class O, class K, class V, class F>
inline int pagerankLoopCuU(V *a, V *r, F *vaff, V *bufv, const O *xoff, const K *xedg, const O *xtoff, const K *xtdat, const K *xtedg, K N, K NB, K NM, K NE, V P, V E, int L, V D) {
  int l = 0;
  V  el = V();
  V  C0 = (1-P)/N;
  F *updt  = (F*) bufv;
  F  updtH = F();
  while (l<L) {
    if ( ASYNC) fillValueCuW(updt, 1, F());  // Reset vertices under update flag
    pagerankUpdateRanksThreadCuU<DYNAMIC, FRONTIER, ASYNC>(a, vaff, updt, xoff, xedg, xtoff, xtdat, xtedg, NB, NM, r, C0, P, E, D);
    pagerankUpdateRanksBlockCuU <DYNAMIC, FRONTIER, ASYNC>(a, vaff, updt, xoff, xedg, xtoff, xtdat, xtedg, NM, NE, r, C0, P, E, D); ++l;
    if ( ASYNC) TRY_CUDA( cudaMemcpy(&updtH, updt, sizeof(F), cudaMemcpyDeviceToHost) );
    if (!ASYNC) liNormDeltaInplaceCuW(bufv, a, r, N);  // Compare previous and current ranks
    if (!ASYNC) TRY_CUDA( cudaMemcpy(&el, bufv, sizeof(V), cudaMemcpyDeviceToHost) );
    if (!ASYNC) swap(a, r);  // Final ranks in (r)
    if (ASYNC) { if (!updtH) break; }  // Check convergence
    else       { if (el<E)   break; }
  }
  return l;
}
#pragma endregion




#pragma region ENVIROMENT SETUP
/**
 * Setup environment and find the rank of each vertex in a graph.
 * @tparam DYNAMIC is a dynamic algorithm?
 * @tparam FRONTIER dynamic frontier approach?
 * @tparam ASYNC asynchronous computation?
 * @tparam FLAG type of vertex affected flags
 * @param x original graph
 * @param xt transposed graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fm function to mark affected vertices
 * @returns pagerank result
 */
template <bool DYNAMIC=false, bool FRONTIER=false, bool ASYNC=false, class FLAG=char, class G, class H, class V, class FM>
inline PagerankResult<V> pagerankInvokeCuda(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FM fm) {
  using  K = typename H::key_type;
  using  O = uint32_t;
  using  F = FLAG;
  size_t S = xt.span();
  size_t N = xt.order();
  size_t M = xt.size();
  V   P  = o.damping;
  V   E  = o.tolerance;
  V   D  = o.frontierTolerance;
  int L  = o.maxIterations, l = 0;
  int R  = reduceSizeCu(N);
  vector<K> xoff, xedg;
  vector<K> xtoff(N+1), xtedg(M), xtdat(N);
  vector<V> r(S), rc(N), qc(N);
  vector<F> vaff, vaffc;
  if (FRONTIER) xoff.resize(N+1);
  if (FRONTIER) xedg.resize(M);
  if (DYNAMIC) vaff.resize(S);
  if (DYNAMIC) vaffc.resize(N);
  O *xoffD  = nullptr;
  K *xedgD  = nullptr;
  O *xtoffD = nullptr;
  K *xtedgD = nullptr;
  K *xtdatD = nullptr;
  V *aD     = nullptr;
  V *rD     = nullptr;
  F *vaffD  = nullptr;
  V *bufvD  = nullptr;
  // Partition vertices into low-degree and high-degree sets.
  vector<K> ks = vertexKeys(xt);
  K NL = pagerankPartitionVerticesCudaU(ks, xt);
  // Obtain data for CSR.
  if (FRONTIER) csrCreateOffsetsW (xoff,  x,  ks);
  if (FRONTIER) csrCreateEdgeKeysW(xedg,  x,  ks);
  csrCreateOffsetsW (xtoff, xt, ks);
  csrCreateEdgeKeysW(xtedg, xt, ks);
  csrCreateVertexValuesW(xtdat, xt, ks);
  // Obtain initial ranks.
  if (q) gatherValuesW(qc, *q, ks);
  // Allocate device memory.
  TRY_CUDA( cudaSetDeviceFlags(cudaDeviceMapHost) );
  if (FRONTIER) TRY_CUDA( cudaMalloc(&xoffD,  (N+1) * sizeof(O)) );
  if (FRONTIER) TRY_CUDA( cudaMalloc(&xedgD,   M    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xtoffD, (N+1) * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&xtedgD,  M    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xtdatD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&aD,    N * sizeof(V)) );
  TRY_CUDA( cudaMalloc(&rD,    N * sizeof(V)) );
  if (DYNAMIC) TRY_CUDA( cudaMalloc(&vaffD, N * sizeof(F)) );
  TRY_CUDA( cudaMalloc(&bufvD, R * sizeof(V)) );
  // Copy data to device.
  if (FRONTIER) TRY_CUDA( cudaMemcpy(xoffD,  xoff .data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
  if (FRONTIER) TRY_CUDA( cudaMemcpy(xedgD,  xedg .data(),  M    * sizeof(K), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xtoffD, xtoff.data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xtedgD, xtedg.data(),  M    * sizeof(K), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xtdatD, xtdat.data(),  N    * sizeof(K), cudaMemcpyHostToDevice) );
  // Perform PageRank algorithm on device.
  float t = measureDurationMarked([&](auto mark) {
    // Setup initial ranks.
    if (q) TRY_CUDA( cudaMemcpy(rD, qc.data(), N   * sizeof(V), cudaMemcpyHostToDevice) );
    if (q && !ASYNC) copyValuesCuW(aD, rD, N);
    else   pagerankInitializeRanksCuW<ASYNC>(aD, rD, K(N), K(0), K(N));
    TRY_CUDA( cudaDeviceSynchronize() );
    // Mark initial affected vertices.
    if (DYNAMIC) mark([&]() { fm(vaff); });
    if (DYNAMIC) gatherValuesW(vaffc, vaff, ks);
    if (DYNAMIC) TRY_CUDA( cudaMemcpy(vaffD, vaffc.data(), N * sizeof(F), cudaMemcpyHostToDevice) );
    // Perform PageRank iterations.
    mark([&]() { l = pagerankLoopCuU<DYNAMIC, FRONTIER, ASYNC>(ASYNC? rD : aD, rD, vaffD, bufvD, xoffD, xedgD, xtoffD, xtdatD, xtedgD, K(N), K(0), K(NL), K(N), P, E, L, D); });
  }, o.repeat);
  // Obtain final ranks.
  TRY_CUDA( cudaMemcpy(rc.data(), rD, N * sizeof(V), cudaMemcpyDeviceToHost) );
  scatterValuesW(r, rc, ks);
  // Free device memory.
  if (FRONTIER) TRY_CUDA( cudaFree(xoffD) );
  if (FRONTIER) TRY_CUDA( cudaFree(xedgD) );
  TRY_CUDA( cudaFree(xtoffD) );
  TRY_CUDA( cudaFree(xtedgD) );
  TRY_CUDA( cudaFree(xtdatD) );
  TRY_CUDA( cudaFree(aD) );
  TRY_CUDA( cudaFree(rD) );
  if (DYNAMIC) TRY_CUDA( cudaFree(vaffD) );
  TRY_CUDA( cudaFree(bufvD) );
  return {r, l, t};
}
#pragma endregion




#pragma region STATIC
/**
 * Find the rank of each vertex in a static graph with Static approach.
 * @param x original graph
 * @param xt transposed graph
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankStaticCuda(const G& x, const H& xt, const PagerankOptions<V>& o) {
  if  (xt.empty()) return {};
  vector<V> *q = nullptr;
  auto fm = [&](auto& vaff) {};
  return pagerankInvokeCuda<false, false, ASYNC, FLAG>(x, xt, q, o, fm);
}
#pragma endregion




#pragma region NAIVE-DYNAMIC
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @param x original graph
 * @param xt transposed graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankNaiveDynamicCuda(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  if  (xt.empty()) return {};
  auto fm = [&](auto& vaff) {};
  return pagerankInvokeCuda<false, false, ASYNC, FLAG>(x, xt, q, o, fm);
}
#pragma endregion




#pragma region DYNAMIC TRAVERSAL
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Traversal approach.
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
template <bool ASYNC=false, class FLAG=char, class G, class H, class K, class V>
inline PagerankResult<V> pagerankDynamicTraversalCuda(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  if (xt.empty()) return {};
  auto fm = [&](auto& vaff) { pagerankAffectedTraversalOmpW(vaff, x, y, deletions, insertions); };
  return pagerankInvokeCuda<true, false, ASYNC, FLAG>(y, yt, q, o, fm);
}
#pragma endregion




#pragma region DYNAMIC FRONTIER
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Frontier approach.
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
template <bool ASYNC=false, class FLAG=char, class G, class H, class K, class V>
inline PagerankResult<V> pagerankDynamicFrontierCuda(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  if (xt.empty()) return {};
  auto fm = [&](auto& vaff) { pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions); };
  return pagerankInvokeCuda<true, true, ASYNC, FLAG>(y, yt, q, o, fm);
}
#pragma endregion
#pragma endregion

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of PageRank values. */
#define TYPE double
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_BATCH
/** Number of times to repeat each batch. */
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion




#pragma region METHODS
#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x input graph
 * @param xt transpose of the input graph with degree
 * @param fstream input file stream
 * @param rows number of rows/vetices in the graph
 * @param size number of lines/edges (temporal) in the graph
 * @param batchFraction fraction of edges to use in each batch
 */
template <class G, class H>
inline void runExperiment(G& x, H& xt, istream& fstream, size_t rows, size_t size, double batchFraction) {
  using  K = typename G::key_type;
  using  V = TYPE;
  int repeat     = REPEAT_METHOD;
  int numThreads = MAX_THREADS;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog  = [&](const auto& ans, const auto& ref, const char *technique, int numThreads, double deletionsf, double insertionsf, int batchIndex) {
    auto err = l1NormDeltaOmp(ans.ranks, ref.ranks);
    printf(
      "{-%.3e/+%.3e batchf, %03d batchi, %03d threads} -> {%09.1fms, %09.1fms init, %09.1fms mark, %09.1fms comp, %03d iter, %.2e err} %s\n",
      deletionsf, insertionsf, batchIndex, numThreads,
      ans.time, ans.initializationTime, ans.markingTime, ans.computationTime, ans.iterations, err, technique
    );
  };
  vector<tuple<K, K>> deletions;
  vector<tuple<K, K>> insertions;
  // Get ranks of vertices on original graph (static).
  auto r0  = pagerankStaticOmp(xt, PagerankOptions<V>(1, 1e-100));
  auto R10 = r0.ranks;
  auto R12 = r0.ranks;
  auto R14 = r0.ranks;
  auto R20 = r0.ranks;
  auto R22 = r0.ranks;
  auto R24 = r0.ranks;
  auto R30 = r0.ranks;
  auto R31 = r0.ranks;
  auto R32 = r0.ranks;
  auto R33 = r0.ranks;
  auto R34 = r0.ranks;
  auto R35 = r0.ranks;
  // Get ranks of vertices on updated graph (dynamic).
  for (int batchIndex=0; batchIndex<BATCH_LENGTH; ++batchIndex) {
    auto y = duplicate(x);
    insertions.clear();
    auto fb = [&](auto u, auto v, auto w) { insertions.push_back({u, v}); };
    readTemporalDo(fstream, false, false, rows, size_t(batchFraction * size), fb);
    tidyBatchUpdateU(deletions, insertions, y);
    applyBatchUpdateOmpU(y, deletions, insertions);
    auto yt = transposeWithDegreeOmp(y);
    LOG(""); print(y); printf(" (insertions=%zu)\n", insertions.size());
    auto s0 = pagerankStaticOmp(yt, PagerankOptions<V>(1, 1e-100));
    // Find multi-threaded OpenMP-based Static PageRank (synchronous, no dead ends).
    {
      auto a0 = pagerankStaticOmp(yt, PagerankOptions<V>(repeat));
      glog(a0, s0, "pagerankStaticOmp", numThreads, 0.0, batchFraction, batchIndex);
      auto c0 = pagerankStaticCuda<false>(y, yt, PagerankOptions<V>(repeat));
      glog(c0, s0, "pagerankStaticCuda", numThreads, 0.0, batchFraction, batchIndex);
      auto e0 = pagerankStaticCuda<true> (y, yt, PagerankOptions<V>(repeat));
      glog(e0, s0, "pagerankStaticCudaPartition", numThreads, 0.0, batchFraction, batchIndex);
    }
    // Find multi-threaded OpenMP-based Naive-dynamic PageRank (synchronous, no dead ends).
    {
      auto a1 = pagerankNaiveDynamicOmp<true>(yt, &R10, {repeat});
      glog(a1, s0, "pagerankNaiveDynamicOmp", numThreads, 0.0, batchFraction, batchIndex);
      auto c1 = pagerankNaiveDynamicCuda<false>(y, yt, &R12, {repeat});
      glog(c1, s0, "pagerankNaiveDynamicCuda", numThreads, 0.0, batchFraction, batchIndex);
      auto e1 = pagerankNaiveDynamicCuda<true> (y, yt, &R14, {repeat});
      glog(e1, s0, "pagerankNaiveDynamicCudaPartition", numThreads, 0.0, batchFraction, batchIndex);
      copyValuesOmpW(R10, a1.ranks);
      copyValuesOmpW(R12, c1.ranks);
      copyValuesOmpW(R14, e1.ranks);
    }
    // Find multi-threaded OpenMP-based Frontier-based Dynamic PageRank (synchronous, no dead ends).
    {
      auto a3 = pagerankDynamicFrontierOmp<true>(x, xt, y, yt, deletions, insertions, &R30, {repeat});
      glog(a3, s0, "pagerankDynamicFrontierOmp", numThreads, 0.0, batchFraction, batchIndex);
      auto c3 = pagerankDynamicFrontierCuda<false>(x, xt, y, yt, deletions, insertions, &R32, {repeat});
      glog(c3, s0, "pagerankDynamicFrontierCuda", numThreads, 0.0, batchFraction, batchIndex);
      auto e3 = pagerankDynamicFrontierCuda<true> (x, xt, y, yt, deletions, insertions, &R34, {repeat});
      glog(e3, s0, "pagerankDynamicFrontierCudaPartition", numThreads, 0.0, batchFraction, batchIndex);
      copyValuesOmpW(R30, a3.ranks);
      copyValuesOmpW(R32, c3.ranks);
      copyValuesOmpW(R34, e3.ranks);
    }
    {
      auto b3 = pagerankPruneDynamicFrontierOmp<true>(x, xt, y, yt, deletions, insertions, &R31, {repeat});
      glog(b3, s0, "pagerankPruneDynamicFrontierOmp", numThreads, 0.0, batchFraction, batchIndex);
      auto d3 = pagerankPruneDynamicFrontierCuda<false>(x, xt, y, yt, deletions, insertions, &R33, {repeat});
      glog(d3, s0, "pagerankPruneDynamicFrontierCuda", numThreads, 0.0, batchFraction, batchIndex);
      auto f3 = pagerankPruneDynamicFrontierCuda<true> (x, xt, y, yt, deletions, insertions, &R35, {repeat});
      glog(f3, s0, "pagerankPruneDynamicFrontierCudaPartition", numThreads, 0.0, batchFraction, batchIndex);
      copyValuesOmpW(R31, b3.ranks);
      copyValuesOmpW(R33, d3.ranks);
      copyValuesOmpW(R35, f3.ranks);
    }
    // Find multi-threaded OpenMP-based Dynamic Traversal PageRank (synchronous, no dead ends).
    {
      auto a2 = pagerankDynamicTraversalOmp<true>(x, xt, y, yt, deletions, insertions, &R20, {repeat});
      glog(a2, s0, "pagerankDynamicTraversalOmp", numThreads, 0.0, batchFraction, batchIndex);
      auto c2 = pagerankDynamicTraversalCuda<false>(x, xt, y, yt, deletions, insertions, &R22, {repeat});
      glog(c2, s0, "pagerankDynamicTraversalCuda", numThreads, 0.0, batchFraction, batchIndex);
      auto e2 = pagerankDynamicTraversalCuda<true> (x, xt, y, yt, deletions, insertions, &R24, {repeat});
      glog(e2, s0, "pagerankDynamicTraversalCudaPartition", numThreads, 0.0, batchFraction, batchIndex);
      copyValuesOmpW(R20, a2.ranks);
      copyValuesOmpW(R22, c2.ranks);
      copyValuesOmpW(R24, e2.ranks);
    }
    swap(x, y);
    swap(xt, yt);
  }
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  char *file = argv[1];
  size_t rows = strtoull(argv[2], nullptr, 10);
  size_t size = strtoull(argv[3], nullptr, 10);
  double batchFraction = strtod(argv[5], nullptr);
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<uint32_t> x;
  ifstream fstream(file);
  readTemporalOmpW(x, fstream, false, false, rows, size_t(0.90 * size)); LOG(""); print(x); printf(" (90%%)\n");
  auto fl = [](auto u) { return true; };
  x = addSelfLoopsOmp(x, None(), fl);  LOG(""); print(x);  printf(" (selfLoopAllVertices)\n");
  auto xt = transposeWithDegreeOmp(x); LOG(""); print(xt); printf(" (transposeWithDegree)\n");
  runExperiment(x, xt, fstream, rows, size, batchFraction);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion

#include <random>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "src/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of PageRank values. */
#define TYPE double
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 32
#endif
#ifndef REPEAT_BATCH
/** Number of times to repeat each batch. */
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 1
#endif
#pragma endregion




#pragma region METHODS
#pragma region EXPERIMENTAL SETUP
/**
 * Add random edges to a graph.
 * @param a input graph (updated)
 * @param rnd random number generator
 * @param batchSize number of edges to add
 * @param i start vertex id
 * @param n number of allowed vertex ids
 * @returns edge insertions {source, target}
 */
template <class G, class R>
inline auto addRandomEdges(G& a, R& rnd, size_t batchSize, size_t i, size_t n) {
  using K = typename G::key_type;
  int retries = 5;
  vector<tuple<K, K>> insertions;
  auto fe = [&](auto u, auto v, auto w) {
    a.addEdge(u, v);
    insertions.push_back(make_tuple(u, v));
  };
  for (size_t l=0; l<batchSize; ++l)
    retry([&]() { return addRandomEdge(a, rnd, i, n, None(), fe); }, retries);
  updateOmpU(a);
  return insertions;
}


/**
 * Remove random edges from a graph.
 * @param a input graph (updated)
 * @param rnd random number generator
 * @param batchSize number of edges to remove
 * @param i start vertex id
 * @param n number of allowed vertex ids
 * @returns edge deletions {source, target}
 */
template <class G, class R>
inline auto removeRandomEdges(G& a, R& rnd, size_t batchSize, size_t i, size_t n) {
  using K = typename G::key_type;
  int retries = 5;
  vector<tuple<K, K>> deletions;
  auto fe = [&](auto u, auto v) {
    a.removeEdge(u, v);
    deletions.push_back(make_tuple(u, v));
  };
  for (size_t l=0; l<batchSize; ++l)
    retry([&]() { return removeRandomEdge(a, rnd, i, n, fe); }, retries);
  updateOmpU(a);
  return deletions;
}


/**
 * Run a function on each batch update, with a specified range of batch sizes.
 * @param x original graph
 * @param rnd random number generator
 * @param fn function to run on each batch update
 */
template <class G, class R, class F>
inline void runBatches(const G& x, R& rnd, F fn) {
  auto fl = [](auto u) { return true; };
  double d = BATCH_DELETIONS_BEGIN;
  double i = BATCH_INSERTIONS_BEGIN;
  while (true) {
    for (int r=0; r<REPEAT_BATCH; ++r) {
      auto y  = duplicate(x);
      auto deletions  = removeRandomEdges(y, rnd, size_t(d * x.size() + 0.5), 1, x.span()-1);
      auto insertions = addRandomEdges   (y, rnd, size_t(i * x.size() + 0.5), 1, x.span()-1);
      selfLoopOmpU(y, None(), fl);
      auto yt = transposeWithDegreeOmp(y);
      fn(y, yt, d, deletions, i, insertions);
    }
    if (d>=BATCH_DELETIONS_END && i>=BATCH_INSERTIONS_END) break;
    d BATCH_DELETIONS_STEP;
    i BATCH_INSERTIONS_STEP;
    d = min(d, double(BATCH_DELETIONS_END));
    i = min(i, double(BATCH_INSERTIONS_END));
  }
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x original graph
 * @param xt transposed graph with degree
 */
template <class G, class H>
inline void runExperiment(const G& x, const H& xt) {
  using  K = typename G::key_type;
  using  V = TYPE;
  vector<V> *init = nullptr;
  random_device dev;
  default_random_engine rnd(dev());
  int repeat = REPEAT_METHOD;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog  = [&](const auto& ans, const auto& ref, const char *technique, auto deletionsf, auto insertionsf) {
    auto err = liNormOmp(ans.ranks, ref.ranks);
    printf(
      "{-%.3e/+%.3e batchf} -> {%09.1fms, %03d iter, %.2e err} %s\n",
      deletionsf, insertionsf, ans.time, ans.iterations, err, technique
    );
  };
  // Get ranks of vertices on original graph (static).
  auto r0   = pagerankBasicOmp(xt, init, {1, 1e-100});
  // Get ranks of vertices on updated graph (dynamic).
  runBatches(x, rnd, [&](const auto& y, const auto& yt, double deletionsf, const auto& deletions, double insertionsf, const auto& insertions) {
    auto flog = [&](const auto& ans, const auto& ref, const char *technique) {
      glog(ans, ref, technique, deletionsf, insertionsf);
    };
    auto s0 = pagerankBasicOmp(yt, init, {1, 1e-100});
    // Find multi-threaded OpenMP-based Static PageRank (synchronous, no dead ends).
    auto a0 = pagerankBasicOmp(yt, init, {repeat});
    flog(a0, s0, "pagerankBasicOmp");
    auto b0 = pagerankContribOmp(yt, init, {repeat});
    flog(b0, s0, "pagerankContribOmp");
    // Find multi-threaded OpenMP-based Naive-dynamic PageRank (synchronous, no dead ends).
    auto a1 = pagerankBasicOmp(yt, &r0.ranks, {repeat});
    flog(a1, s0, "pagerankBasicNaiveDynamicOmp");
    auto b1 = pagerankContribOmp(yt, &r0.ranks, {repeat});
    flog(b1, s0, "pagerankContribNaiveDynamicOmp");
    // Find multi-threaded OpenMP-based Frontier-based Dynamic PageRank (synchronous, no dead ends).
    auto a2 = pagerankBasicDynamicFrontierOmp(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat});
    flog(a2, s0, "pagerankBasicDynamicFrontierOmp");
    auto b2 = pagerankContribDynamicFrontierOmp(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat});
    flog(b2, s0, "pagerankContribDynamicFrontierOmp");
  });
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  char *file = argv[1];
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  OutDiGraph<uint32_t> x;
  readMtxOmpW(x, file); LOG(""); println(x);
  auto fl = [](auto u) { return true; };
  x = selfLoopOmp(x, None(), fl);      LOG(""); print(x);  printf(" (selfLoopAllVertices)\n");
  auto xt = transposeWithDegreeOmp(x); LOG(""); print(xt); printf(" (transposeWithDegree)\n");
  runExperiment(x, xt);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion

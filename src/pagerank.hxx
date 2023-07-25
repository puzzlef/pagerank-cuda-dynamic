#pragma once
#include <utility>
#include <chrono>
#include <random>
#include <atomic>
#include <tuple>
#include <vector>
#include <cmath>
#include <algorithm>
#include "_main.hxx"
#include "dfs.hxx"

#ifdef OPENMP
#include <omp.h>
#endif

using std::random_device;
using std::default_random_engine;
using std::chrono::system_clock;
using std::tuple;
using std::vector;
using std::atomic;
using std::get;
using std::move;
using std::abs;
using std::max;




// PAGERANK OPTIONS
// ----------------

enum NormFunction {
  L0_NORM = 0,
  L1_NORM = 1,
  L2_NORM = 2,
  LI_NORM = 3
};


template <class V>
struct PagerankOptions {
  int repeat;
  int toleranceNorm;
  V   tolerance;
  V   damping;
  int maxIterations;

  PagerankOptions(int repeat=1, int toleranceNorm=LI_NORM, V tolerance=1e-10, V damping=0.85, int maxIterations=500) :
  repeat(repeat), toleranceNorm(toleranceNorm), tolerance(tolerance), damping(damping), maxIterations(maxIterations) {}
};




// PAGERANK RESULT
// ---------------

template <class V>
struct PagerankResult {
  vector<V> ranks;
  int   iterations;
  float time;

  PagerankResult() :
  ranks(), iterations(0), time(0) {}

  PagerankResult(vector<V>&& ranks, int iterations=0, float time=0) :
  ranks(ranks), iterations(iterations), time(time) {}

  PagerankResult(vector<V>& ranks, int iterations=0, float time=0) :
  ranks(move(ranks)), iterations(iterations), time(time) {}
};




// PAGERANK TELEPORT
// -----------------
// For teleport contribution from vertices (inc. dead ends).

/**
 * Find total teleport contribution from each vertex (inc. dead ends).
 * @param xt transpose of original graph
 * @param r rank of each vertex
 * @param P damping factor [0.85]
 * @returns common teleport rank contribution to each vertex
 */
template <class H, class V>
inline V pagerankTeleport(const H& xt, const vector<V>& r, V P) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  V a = (1-P)/N;
  xt.forEachVertex([&](auto u, auto d) {
    if (d==0) a += P * r[u]/N;
  });
  return a;
}


#ifdef OPENMP
template <class H, class V>
inline V pagerankTeleportOmp(const H& xt, const vector<V>& r, V P) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V a = (1-P)/N;
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (K u=0; u<S; ++u) {
    if (!xt.hasVertex(u)) continue;
    K   d = xt.vertexValue(u);
    if (d==0) a += P * r[u]/N;
  }
  return a;
}
#endif




// PAGERANK CALCULATE
// ------------------
// For rank calculation from in-edges.

/**
 * Calculate rank for a given vertex.
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param v given vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @returns change between previous and current rank value
 */
template <class H, class K, class V>
inline V pagerankCalculateRank(vector<V>& a, const H& xt, const vector<V>& r, K v, V C0, V P) {
  V av = C0;
  V rv = r[v];
  xt.forEachEdgeKey(v, [&](auto u) {
    K d = xt.vertexValue(u);
    av += P * r[u]/d;
  });
  a[v] = av;
  return abs(av - rv);
}


/**
 * Calculate ranks for vertices in a graph.
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param fa is vertex affected? (vertex)
 * @param fr called if vertex rank changes (vertex, delta)
 */
template <class H, class V, class FA, class FR>
inline void pagerankCalculateRanks(vector<V>& a, const H& xt, const vector<V>& r, V C0, V P, V E, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v) || !fa(v)) continue;
    V ev = pagerankCalculateRank(a, xt, r, v, C0, P);
    fr(v, ev);
  }
}


#ifdef OPENMP
template <class H, class V, class FA, class FR>
inline void pagerankCalculateRanksOmp(vector<V>& a, const H& xt, const vector<V>& r, V C0, V P, V E, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v) || !fa(v)) continue;
    V ev = pagerankCalculateRank(a, xt, r, v, C0, P);
    fr(v, ev);
  }
}
#endif




// PAGERANK ERROR
// --------------
// For convergence check.

/**
 * Get the error between two rank vectors.
 * @param x first rank vector
 * @param y second rank vector
 * @param EF error function (L1/L2/LI)
 * @returns error between the two rank vectors
 */
template <class V>
inline V pagerankError(const vector<V>& x, const vector<V>& y, int EF) {
  switch (EF) {
    case 1:  return l1Norm(x, y);
    case 2:  return l2Norm(x, y);
    default: return liNorm(x, y);
  }
}


#ifdef OPENMP
template <class V>
inline V pagerankErrorOmp(const vector<V>& x, const vector<V>& y, int EF) {
  switch (EF) {
    case 1:  return l1NormOmp(x, y);
    case 2:  return l2NormOmp(x, y);
    default: return liNormOmp(x, y);
  }
}
#endif




// PAGERANK AFFECTED (FRONTIER)
// ----------------------------

/**
 * Find affected vertices due to a batch update.
 * @param vis affected flags (output)
 * @param x original graph
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 */
template <class B, class G, class K>
inline void pagerankAffectedFrontierW(vector<B>& vis, const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
  for (const auto& [u, v] : deletions)
    x.forEachEdgeKey(u, [&](auto v) { vis[v] = B(1); });
  for (const auto& [u, v] : insertions)
    y.forEachEdgeKey(u, [&](auto v) { vis[v] = B(1); });
}


#ifdef OPENMP
template <class B, class G, class K>
inline void pagerankAffectedFrontierOmpW(vector<B>& vis, const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
  size_t D = deletions.size();
  size_t I = insertions.size();
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<D; ++i) {
    K u = get<0>(deletions[i]);
    x.forEachEdgeKey(u, [&](auto v) { vis[v] = B(1); });
  }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<I; ++i) {
    K u = get<0>(insertions[i]);
    y.forEachEdgeKey(u, [&](auto v) { vis[v] = B(1); });
  }
}
#endif




// PAGERANK-SEQ
// ------------
// For single-threaded (sequential) PageRank implementation.

/**
 * Find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class H, class V, class FL>
PagerankResult<V> pagerankSeq(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  int EF = o.toleranceNorm;
  vector<V> a(S), r(S);
  float t = measureDuration([&]() {
    if (q) copyValuesW(r, *q);
    else   fillValueU (r, V(1)/N);
    if (!ASYNC) copyValuesW(a, r);
    l = fl(ASYNC? r : a, r, xt, P, E, L, EF);
  }, o.repeat);
  return {r, l, t};
}




// PAGERANK-OMP
// ------------
// For multi-threaded OpenMP-based PageRank implementation.

#ifdef OPENMP
/**
 * Find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class H, class V, class FL>
PagerankResult<V> pagerankOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  int EF = o.toleranceNorm;
  vector<V> a(S), r(S);
  float t = measureDuration([&]() {
    if (q) copyValuesOmpW(r, *q);
    else   fillValueOmpU (r, V(1)/N);
    if (!ASYNC) copyValuesOmpW(a, r);
    l = fl(ASYNC? r : a, r, xt, P, E, L, EF);
  }, o.repeat);
  return {r, l, t};
}
#endif

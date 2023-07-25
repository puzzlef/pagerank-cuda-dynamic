#pragma once
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include "_main.hxx"
#include "dfs.hxx"
#ifdef OPENMP
#include <omp.h>
#endif

using std::tuple;
using std::vector;
using std::get;
using std::move;
using std::swap;
using std::abs;
using std::max;




#pragma region TYPES
/**
 * Options for PageRank algorithm.
 * @tparam V rank value type
 */
template <class V>
struct PagerankOptions {
  #pragma region DATA
  /** Number of times to repeat the algorithm [1]. */
  int repeat;
  /** Tolerance for convergence [10^-10]. */
  V   tolerance;
  /** Damping factor [0.85]. */
  V   damping;
  /** Maximum number of iterations [500]. */
  int maxIterations;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define a set of PageRank options.
   * @param repeat number of times to repeat the algorithm [1]
   * @param tolerance tolerance for convergence [10^-10]
   * @param damping damping factor [0.85]
   * @param maxIterations maximum number of iterations [500]
   */
  PagerankOptions(int repeat=1, V tolerance=1e-10, V damping=0.85, int maxIterations=500) :
  repeat(repeat), tolerance(tolerance), damping(damping), maxIterations(maxIterations) {}
  #pragma endregion
};




/**
 * Result of PageRank algorithm.
 * @tparam V rank value type
 */
template <class V>
struct PagerankResult {
  #pragma region DATA
  /** Rank of each vertex. */
  vector<V> ranks;
  /** Number of iterations performed. */
  int   iterations;
  /** Average time taken to perform the algorithm. */
  float time;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define empty PageRank result.
   */
  PagerankResult() :
  ranks(), iterations(0), time(0) {}

  /**
   * Define a PageRank result.
   * @param ranks rank of each vertex
   * @param iterations number of iterations performed
   * @param time average time taken to perform the algorithm
   */
  PagerankResult(vector<V>&& ranks, int iterations=0, float time=0) :
  ranks(ranks), iterations(iterations), time(time) {}

  /**
   * Define a PageRank result.
   * @param ranks rank of each vertex (moved)
   * @param iterations number of iterations performed
   * @param time average time taken to perform the algorithm
   */
  PagerankResult(vector<V>& ranks, int iterations=0, float time=0) :
  ranks(move(ranks)), iterations(iterations), time(time) {}
  #pragma endregion
};
#pragma endregion




#pragma region METHODS
#pragma region CALCULATE RANKS
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
 * @param fa is vertex affected? (vertex)
 * @param fr called with vertex rank change (vertex, delta)
 */
template <class H, class V, class FA, class FR>
inline void pagerankCalculateRanks(vector<V>& a, const H& xt, const vector<V>& r, V C0, V P, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v) || !fa(v)) continue;
    V ev = pagerankCalculateRank(a, xt, r, v, C0, P);
    fr(v, ev);
  }
}


#ifdef OPENMP
/**
 * Calculate ranks for vertices in a graph (using OpenMP).
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param fa is vertex affected? (vertex)
 * @param fr called with vertex rank change (vertex, delta)
 */
template <class H, class V, class FA, class FR>
inline void pagerankCalculateRanksOmp(vector<V>& a, const H& xt, const vector<V>& r, V C0, V P, FA fa, FR fr) {
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
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup environment and find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class H, class V, class FL>
PagerankResult<V> pagerankInvoke(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), r(S);
  float t = measureDuration([&]() {
    if (q) copyValuesW(r, *q);
    else   fillValueU (r, V(1)/N);
    if (!ASYNC) copyValuesW(a, r);
    l = fl(ASYNC? r : a, r, xt, P, E, L);
  }, o.repeat);
  return {r, l, t};
}


#ifdef OPENMP
/**
 * Setup environment and find the rank of each vertex in a graph (using OpenMP).
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class H, class V, class FL>
PagerankResult<V> pagerankInvokeOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), r(S);
  float t = measureDuration([&]() {
    if (q) copyValuesOmpW(r, *q);
    else   fillValueOmpU (r, V(1)/N);
    if (!ASYNC) copyValuesOmpW(a, r);
    l = fl(ASYNC? r : a, r, xt, P, E, L);
  }, o.repeat);
  return {r, l, t};
}
#endif
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform PageRank iterations upon a graph.
 * @param a current rank of each vertex (updated)
 * @param r previous rank of each vertex (updated)
 * @param xt transpose of original graph
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param fa is vertex affected? (vertex)
 * @param fr called if vertex rank changes (vertex, delta)
 * @returns iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FR>
inline int pagerankBasicLoop(vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  while (l<L) {
    V C0 = (1-P)/N;
    pagerankCalculateRanks(a, xt, r, C0, P, E, fa, fr); ++l;  // update ranks of vertices
    V el = liNorm(a, r);     // compare previous and current ranks
    if (!ASYNC) swap(a, r);  // final ranks in (r)
    if (el<E) break;         // check tolerance
  }
  return l;
}


#ifdef OPENMP
/**
 * Perform PageRank iterations upon a graph (using OpenMP).
 * @param a current rank of each vertex (updated)
 * @param r previous rank of each vertex (updated)
 * @param xt transpose of original graph
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param fa is vertex affected? (vertex)
 * @param fr called if vertex rank changes (vertex, delta)
 * @returns iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FR>
inline int pagerankBasicLoopOmp(vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  while (l<L) {
    V C0 = (1-P)/N;
    pagerankCalculateRanksOmp(a, xt, r, C0, P, E, fa, fr); ++l;  // update ranks of vertices
    V el = liNormOmp(a, r);  // compare previous and current ranks
    if (!ASYNC) swap(a, r);  // final ranks in (r)
    if (el<E) break;         // check tolerance
  }
  return l;
}
#endif
#pragma endregion




#pragma region STATIC/NAIVE-DYNAMIC
/**
 * Find the rank of each vertex in a static graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankBasic(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  return pagerankInvoke<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankBasicLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a static graph (using OpenMP).
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankBasicOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  return pagerankInvokeOmp<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankBasicLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}
#endif
#pragma endregion




#pragma region DYNAMIC FRONTIER
/**
 * Find affected vertices due to a batch update with the Dynamic Frontier approach.
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
/**
 * Find affected vertices due to a batch update with the Dynamic Frontier approach (using OpenMP).
 * @param vis affected flags (output)
 * @param x original graph
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 */
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


/**
 * Find the rank of each vertex in a dynamic graph with the Dynamic Frontier approach.
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
inline PagerankResult<V> pagerankBasicDynamicFrontier(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // see adjust-tolerance
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankSeq<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierW(vaff, x, y, deletions, insertions);
    return pagerankBasicSeqLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with the Dynamic Frontier approach (using OpenMP).
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
inline PagerankResult<V> pagerankBasicDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // see adjust-tolerance
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankOmp<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions);
    return pagerankBasicOmpLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}
#endif
#pragma endregion
#pragma endregion

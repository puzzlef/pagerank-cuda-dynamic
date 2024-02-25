#pragma once
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include "_main.hxx"

using std::tuple;
using std::vector;
using std::get;
using std::move;
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
  /** Tolerance for marking neighbors of a vertex as affected [10^-15]. */
  V   frontierTolerance;
  /** Tolerance for pruning an affected vertex [10^-15]. */
  V   pruneTolerance;
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
   * @param frontierTolerance tolerance for marking neighbors of a vertex as affected [10^-15]
   * @param pruneTolerance tolerance for pruning an affected vertex [10^-15]
   * @param damping damping factor [0.85]
   * @param maxIterations maximum number of iterations [500]
   */
  PagerankOptions(int repeat=1, V tolerance=1e-10, V frontierTolerance=1e-13, V pruneTolerance=1e-13, V damping=0.85, int maxIterations=500) :
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
  /** Average time taken to initialize ranks. */
  float initializationTime;
  /** Average time taken to mark affected vertices. */
  float markingTime;
  /** Average time taken to compute ranks. */
  float computationTime;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define empty PageRank result.
   */
  PagerankResult() :
  ranks(), iterations(0), time(0), initializationTime(0), markingTime(0), computationTime(0) {}

  /**
   * Define a PageRank result.
   * @param ranks rank of each vertex
   * @param iterations number of iterations performed
   * @param time average time taken to perform the algorithm
   * @param initializationTime average time taken to initialize ranks
   * @param markingTime average time taken to mark affected vertices
   * @param computationTime average time taken to compute ranks
   */
  PagerankResult(vector<V>&& ranks, int iterations=0, float time=0, float initializationTime=0, float markingTime=0, float computationTime=0) :
  ranks(ranks), iterations(iterations), time(time), initializationTime(initializationTime), markingTime(markingTime), computationTime(computationTime) {}

  /**
   * Define a PageRank result.
   * @param ranks rank of each vertex (moved)
   * @param iterations number of iterations performed
   * @param time average time taken to perform the algorithm
   * @param initializationTime average time taken to initialize ranks
   * @param markingTime average time taken to mark affected vertices
   * @param computationTime average time taken to compute ranks
   */
  PagerankResult(vector<V>& ranks, int iterations=0, float time=0, float initializationTime=0, float markingTime=0, float computationTime=0) :
  ranks(move(ranks)), iterations(iterations), time(time), initializationTime(initializationTime), markingTime(markingTime), computationTime(computationTime) {}
  #pragma endregion
};
#pragma endregion




#pragma region METHODS
#pragma region UPDATE RANKS
/**
 * Update rank for a given vertex.
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param v given vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @returns change between previous and current rank value
 */
template <class H, class K, class V>
inline V pagerankUpdateRank(vector<V>& a, const H& xt, const vector<V>& r, K v, V C0, V P) {
  V cv = V();
  V rv = r[v];
  xt.forEachEdgeKey(v, [&](auto u) {
    K d = xt.vertexValue(u);
    cv += r[u]/d;
  });
  V av = C0 + P * cv;
  a[v] = av;
  return abs(av - rv);
}


/**
 * Update ranks for vertices in a graph.
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param fa is vertex affected? (vertex)
 * @param fr called with vertex rank change (vertex, delta)
 */
template <class H, class V, class FA, class FR>
inline void pagerankUpdateRanks(vector<V>& a, const H& xt, const vector<V>& r, V C0, V P, FA fa, FR fr) {
  xt.forEachVertexKey([&](auto v) {
    if (!fa(v)) return;
    V ev = pagerankUpdateRank(a, xt, r, v, C0, P);
    fr(v, ev);
  });
}


#ifdef OPENMP
/**
 * Update ranks for vertices in a graph (using OpenMP).
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param fa is vertex affected? (vertex)
 * @param fr called with vertex rank change (vertex, delta)
 */
template <class H, class V, class FA, class FR>
inline void pagerankUpdateRanksOmp(vector<V>& a, const H& xt, const vector<V>& r, V C0, V P, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v) || !fa(v)) continue;
    V ev = pagerankUpdateRank(a, xt, r, v, C0, P);
    fr(v, ev);
  }
}
#endif
#pragma endregion




#pragma region INITIALIZE RANKS
/**
 * Intitialize ranks before PageRank iterations.
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param xt transpose of original graph
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeRanks(vector<V>& a, vector<V>& r, const H& xt) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  for (K v=0; v<S; ++v) {
    r[v] = xt.hasVertex(v)? V(1)/N : V();
    if (!ASYNC) a[v] = r[v];
  }
}


#ifdef OPENMP
/**
 * Intitialize ranks before PageRank iterations (using OpenMP).
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param xt transpose of original graph
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeRanksOmp(vector<V>& a, vector<V>& r, const H& xt) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    r[v] = xt.hasVertex(v)? V(1)/N : V();
    if (!ASYNC) a[v] = r[v];
  }
}
#endif


/**
 * Intitialize ranks before PageRank iterations from given ranks.
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param xt transpose of original graph
 * @param q initial ranks
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeRanksFrom(vector<V>& a, vector<V>& r, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  for (K v=0; v<S; ++v) {
    r[v] = q[v];
    if (!ASYNC) a[v] = q[v];
  }
}


#ifdef OPENMP
/**
 * Intitialize ranks before PageRank iterations from given ranks (using OpenMP).
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param xt transpose of original graph
 * @param q initial ranks
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeRanksFromOmp(vector<V>& a, vector<V>& r, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    r[v] = q[v];
    if (!ASYNC) a[v] = q[v];
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
template <bool ASYNC=false, class H, class V, class FL, class FI>
inline PagerankResult<V> pagerankInvoke(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl, FI fi) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> r(S), a;
  if (!ASYNC) a.resize(S);
  float ti = 0;
  float t  = measureDuration([&]() {
    // Intitialize rank of each vertex.
    ti += measureDuration([&]() { fi(a, r); });
    l = fl(ASYNC? r : a, r, xt, P, E, L);
  }, o.repeat);
  return {r, l, t, ti/o.repeat};
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
template <bool ASYNC=false, class H, class V, class FL, class FI>
inline PagerankResult<V> pagerankInvokeOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl, FI fi) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> r(S), a;
  if (!ASYNC) a.resize(S);
  float ti = 0;
  float t  = measureDuration([&]() {
    // Intitialize rank of each vertex.
    ti += measureDuration([&]() { fi(a, r); });
    l = fl(ASYNC? r : a, r, xt, P, E, L);
  }, o.repeat);
  return {r, l, t, ti/o.repeat};
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
 * @returns number of iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FR>
inline int pagerankLoop(vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  V  C0 = (1-P)/N;
  while (l<L) {
    pagerankUpdateRanks(a, xt, r, C0, P, fa, fr); ++l;  // Update ranks of vertices
    V el = liNormDelta(a, r);  // Compare previous and current ranks
    if (!ASYNC) swap(a, r);    // Final ranks in (r)
    if (el<E) break;           // Check tolerance
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
 * @returns number of iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FR>
inline int pagerankLoopOmp(vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  V  C0 = (1-P)/N;
  while (l<L) {
    pagerankUpdateRanksOmp(a, xt, r, C0, P, fa, fr); ++l;  // Update ranks of vertices
    V el = liNormDeltaOmp(a, r);  // Compare previous and current ranks
    if (!ASYNC) swap(a, r);       // Final ranks in (r)
    if (el<E) break;              // Check tolerance
  }
  return l;
}
#endif
#pragma endregion




#pragma region STATIC
/**
 * Find the rank of each vertex in a dynamic graph with Static approach.
 * @param xt transpose of original graph
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankStatic(const H& xt, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  vector<V> *q = nullptr;
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanks<ASYNC>(a, r, xt); };
  return pagerankInvoke<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  }, fi);
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with Static approach.
 * @param xt transpose of original graph
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankStaticOmp(const H& xt, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  vector<V> *q = nullptr;
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanksOmp<ASYNC>(a, r, xt); };
  return pagerankInvokeOmp<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  }, fi);
}
#endif
#pragma endregion




#pragma region NAIVE-DYNAMIC
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankNaiveDynamic(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanksFrom<ASYNC>(a, r, xt, *q); };
  return pagerankInvoke<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  }, fi);
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankNaiveDynamicOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanksFromOmp<ASYNC>(a, r, xt, *q); };
  return pagerankInvokeOmp<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  }, fi);
}
#endif
#pragma endregion




#pragma region DYNAMIC TRAVERSAL
/**
 * Find affected vertices due to a batch update with Dynamic Traversal approach.
 * @param vis affected flags (output)
 * @param x original graph
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 */
template <class B, class G, class K>
inline void pagerankAffectedTraversalW(vector<B>& vis, const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
  auto ft = [](auto u, auto d) { return true; };
  auto fp = [](auto u, auto d) { };
  for (const auto& [u, v] : deletions)
    x.forEachEdgeKey(u, [&](auto v) { bfsVisitedForEachU(vis, y, v, ft, fp); });
  for (const auto& [u, v] : insertions)
    y.forEachEdgeKey(u, [&](auto v) { bfsVisitedForEachU(vis, y, v, ft, fp); });
}


#ifdef OPENMP
/**
 * Find affected vertices due to a batch update with Dynamic Traversal approach (using OpenMP).
 * @param vis affected flags (output)
 * @param x original graph
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 */
template <class B, class G, class K>
inline void pagerankAffectedTraversalOmpW(vector<B>& vis, const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
  auto  ft = [](auto u, auto d) { return true; };
  auto  fp = [](auto u, auto d) { };
  size_t D = deletions.size();
  size_t I = insertions.size();
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<D; ++i) {
    K u = get<0>(deletions[i]);
    x.forEachEdgeKey(u, [&](auto v) { bfsVisitedForEachU(vis, y, v, ft, fp); });
  }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<I; ++i) {
    K u = get<0>(insertions[i]);
    y.forEachEdgeKey(u, [&](auto v) { bfsVisitedForEachU(vis, y, v, ft, fp); });
  }
}
#endif
#pragma endregion




#pragma region DYNAMIC FRONTIER
/**
 * Find affected vertices due to a batch update with Dynamic Frontier approach.
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
 * Find affected vertices due to a batch update with Dynamic Frontier approach (using OpenMP).
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
inline PagerankResult<V> pagerankDynamicFrontier(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = o.frontierTolerance;
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanksFrom<ASYNC>(a, r, xt, *q); };
  return pagerankInvoke<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierW(vaff, x, y, deletions, insertions);
    return pagerankLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  }, fi);
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Frontier approach (using OpenMP).
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
inline PagerankResult<V> pagerankDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = o.frontierTolerance;
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanksFromOmp<ASYNC>(a, r, xt, *q); };
  return pagerankInvokeOmp<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions);
    return pagerankLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  }, fi);
}
#endif
#pragma endregion
#pragma endregion

#pragma once
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include "_main.hxx"

using std::tuple;
using std::vector;
using std::get;
using std::abs;
using std::max;




#pragma region METHODS
#pragma region UPDATE RANKS
/**
 * Update rank for a given vertex.
 * @param a current rank of each vertex (updated)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param v given vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @returns previous rank of given vertex
 */
template <class H, class K, class V>
inline V pagerankPruneUpdateRank(vector<V>& a, const H& xt, const vector<V>& r, K v, V C0, V P) {
  V av = V();
  V rv = r[v];
  xt.forEachEdgeKey(v, [&](auto u) {
    K d = xt.vertexValue(u);
    av += r[u]/d;
  });
  K d  = xt.vertexValue(v);
  // Handle dangling vertices.
  a[v] = (C0 + P * (av - rv/d)) * (1/(1-(P/d)));  // av = C0 + P * av;
  return rv;
}


/**
 * Update ranks for vertices in a graph.
 * @param a current rank of each vertex (updated)
 * @param vaff affected vertices (updated)
 * @param x original graph
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <class G, class H, class V, class B>
inline void pagerankPruneUpdateRanks(vector<V>& a, vector<B>& vaff, const G& x, const H& xt, const vector<V>& r, V C0, V P, V D, V C) {
  xt.forEachVertexKey([&](auto u) {
    if (!vaff[u]) { a[u] = r[u]; return; }
    V ru = pagerankPruneUpdateRank(a, xt, r, u, C0, P);
    const auto au = a[u];
    const auto eu = abs(ru - au);
    if (eu/max(ru, au) <= C) vaff[u] = B(0);
    if (eu/max(ru, au) <= D) return;
    x.forEachEdgeKey(u, [&](auto v) { if (v!=u && !vaff[v]) vaff[v] = B(1); });
  });
}


/**
 * Update ranks for vertices in a graph.
 * @param a current rank of each vertex (updated)
 * @param vaff affected vertices (updated)
 * @param x original graph
 * @param xt transpose of original graph
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <class G, class H, class V, class B>
inline V pagerankPruneUpdateRanksAsync(vector<V>& a, vector<B>& vaff, const G& x, const H& xt, V C0, V P, V D, V C) {
  V el = V();
  xt.forEachVertexKey([&](auto u) {
    if (!vaff[u]) return;
    V ru = pagerankPruneUpdateRank(a, xt, a, u, C0, P);
    const auto au = a[u];
    const auto eu = abs(ru - au);
    el = max(el, eu);
    if (eu/max(ru, au) <= C) vaff[u] = B(0);
    if (eu/max(ru, au) <= D) return;
    x.forEachEdgeKey(u, [&](auto v) { if (v!=u && !vaff[v]) vaff[v] = B(1); });
  });
  return el;
}


#ifdef OPENMP
/**
 * Update ranks for vertices in a graph (using OpenMP).
 * @param a current rank of each vertex (updated)
 * @param vaff affected vertices (updated)
 * @param x original graph
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <class G, class H, class V, class B>
inline void pagerankPruneUpdateRanksOmp(vector<V>& a, vector<B>& vaff, const G& x, const H& xt, const vector<V>& r, V C0, V P, V D, V C) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K u=0; u<S; ++u) {
    if (!xt.hasVertex(u)) continue;
    if (!vaff[u]) { a[u] = r[u]; continue; }
    V ru = pagerankPruneUpdateRank(a, xt, r, u, C0, P);
    const auto au = a[u];
    const auto eu = abs(ru - au);
    if (eu/max(ru, au) <= C) vaff[u] = B(0);
    if (eu/max(ru, au) <= D) continue;
    x.forEachEdgeKey(u, [&](auto v) { if (v!=u && !vaff[v]) vaff[v] = B(1); });
  }
}


/**
 * Update ranks for vertices in a graph (using OpenMP).
 * @param a current rank of each vertex (updated)
 * @param vaff affected vertices (updated)
 * @param x original graph
 * @param xt transpose of original graph
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 * @param C prune tolerance
 */
template <class G, class H, class V, class B>
inline V pagerankPruneUpdateRanksAsyncOmp(vector<V>& a, vector<B>& vaff, const G& x, const H& xt, V C0, V P, V D, V C) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  V el = V();
  #pragma omp parallel for schedule(dynamic, 2048) reduction(max:el)
  for (K u=0; u<S; ++u) {
    if (!xt.hasVertex(u)) continue;
    if (!vaff[u]) continue;
    V ru = pagerankPruneUpdateRank(a, xt, a, u, C0, P);
    const auto au = a[u];
    const auto eu = abs(ru - au);
    el = max(el, eu);
    if (eu/max(ru, au) <= C) vaff[u] = B(0);
    if (eu/max(ru, au) <= D) continue;
    x.forEachEdgeKey(u, [&](auto v) { if (v!=u && !vaff[v]) vaff[v] = B(1); });
  }
  return el;
}
#endif
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the PageRank algorithm.
 * @param x original graph
 * @param xt transpose of original graph
 * @param o pagerank options
 * @param fi initializing rank of each vertex (a, r)
 * @param fm marking affected vertices (vaff)
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V, class FI, class FM>
inline PagerankResult<V> pagerankPruneInvoke(const G& x, const H& xt, const PagerankOptions<V>& o, FI fi, FM fm) {
  using  K = typename H::key_type;
  using  B = FLAG;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  V   D  = o.frontierTolerance;
  V   C  = o.pruneTolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> r(S), a;
  vector<B> vaff(S);
  if (!ASYNC)  a.resize(S);
  float ti = 0, tm = 0, tc = 0;
  float t  = measureDuration([&]() {
    // Initialize rank of each vertex.
    ti += measureDuration([&]() { fi(a, r); });
    // Mark affected vertices.
    tm += measureDuration([&]() { fm(vaff); });
    // Compute ranks.
    tc += measureDuration([&]() {
      const V C0 = (1-P)/N;
      for (l=0; l<L;) {
        if (ASYNC) {
          V el = pagerankPruneUpdateRanksAsync(r, vaff, x, xt, C0, P, D, C); ++l;  // Update ranks of vertices
          if (el<E) break;   // Check tolerance
        }
        else {
          pagerankPruneUpdateRanks(a, vaff, x, xt, r, C0, P, D, C); ++l;  // Update ranks of vertices
          V el = liNormDelta(a, r);  // Compare previous and current ranks
          swap(a, r);        // Final ranks in (r)
          if (el<E) break;   // Check tolerance
        }
      }
    });
  }, o.repeat);
  return {r, l, t, ti/o.repeat, tm/o.repeat, tc/o.repeat};
}


#ifdef OPENMP
/**
 * Setup and perform the PageRank algorithm (using OpenMP).
 * @param x original graph
 * @param xt transpose of original graph
 * @param o pagerank options
 * @param fi initializing rank of each vertex (a, r)
 * @param fm marking affected vertices (vaff)
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V, class FI, class FM>
inline PagerankResult<V> pagerankPruneInvokeOmp(const G& x, const H& xt, const PagerankOptions<V>& o, FI fi, FM fm) {
  using  K = typename H::key_type;
  using  B = FLAG;
  size_t S = xt.span();
  V   P  = o.damping;
  V   E  = o.tolerance;
  V   D  = o.frontierTolerance;
  V   C  = o.pruneTolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> r(S), a;
  vector<B> vaff(S);
  if (!ASYNC)  a.resize(S);
  float ti = 0, tm = 0, tc = 0;
  float t  = measureDuration([&]() {
    // Initialize rank of each vertex.
    ti += measureDuration([&]() { fi(a, r); });
    // Mark affected vertices.
    tm += measureDuration([&]() { fm(vaff); });
    // Compute ranks.
    tc += measureDuration([&]() {
      const V C0 = (1-P)/S;
      for (l=0; l<L;) {
        if (ASYNC) {
          V el = pagerankPruneUpdateRanksAsyncOmp(r, vaff, x, xt, C0, P, D, C); ++l;  // Update ranks of vertices
          if (el<E) break;   // Check tolerance
        }
        else {
          pagerankPruneUpdateRanksOmp(a, vaff, x, xt, r, C0, P, D, C); ++l;  // Update ranks of vertices
          V el = liNormDeltaOmp(a, r);  // Compare previous and current ranks
          swap(a, r);        // Final ranks in (r)
          if (el<E) break;   // Check tolerance
        }
      }
    });
  }, o.repeat);
  return {r, l, t, ti/o.repeat, tm/o.repeat, tc/o.repeat};
}
#endif
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
inline PagerankResult<V> pagerankPruneDynamicFrontier(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  if (xt.empty()) return {};
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanksFrom<ASYNC>(a, r, xt, *q); };
  auto fm = [&](auto& vaff) { pagerankAffectedFrontierW(vaff, x, y, deletions, insertions); };
  return pagerankPruneInvoke<ASYNC, FLAG>(y, yt, o, fi, fm);
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
inline PagerankResult<V> pagerankPruneDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  if (xt.empty()) return {};
  auto fi = [&](auto& a, auto& r) { pagerankInitializeRanksFromOmp<ASYNC>(a, r, xt, *q); };
  auto fm = [&](auto& vaff) { pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions); };
  return pagerankPruneInvokeOmp<ASYNC, FLAG>(y, yt, o, fi, fm);
}
#endif
#pragma endregion
#pragma endregion

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
  V av = V();
  V rv = r[v];
  xt.forEachEdgeKey(v, [&](auto u) {
    K d = xt.vertexValue(u);
    av += r[u]/d;
  });
  av   = C0 + P*av;
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
  xt.forEachVertexKey([&](auto v) {
    if (!fa(v)) return;
    V ev = pagerankCalculateRank(a, xt, r, v, C0, P);
    fr(v, ev);
  });
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




#pragma region CALCULATE RANKS (CONTRIB)
/**
 * Calculate contribution for a given vertex.
 * @param a current contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param f contribution factor for each vertex
 * @param c previous contribution of each vertex
 * @param v given vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @returns change between previous and current contribution value
 */
template <class H, class K, class V>
inline V pagerankCalculateContribution(vector<V>& a, const H& xt, const vector<V>& f, const vector<V>& c, K v, V C0, V P) {
  V av = V();
  V cv = c[v];
  xt.forEachEdgeKey(v, [&](auto u) { av += c[u]; });
  av   = (C0 + P*av) * f[v];
  a[v] = av;
  return abs(av - cv);
}


/**
 * Calculate contributions for vertices in a graph.
 * @param a current contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param f contribution factor for each vertex
 * @param c previous contribution of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param fa is vertex affected? (vertex)
 * @param fr called with vertex rank change (vertex, delta)
 */
template <class H, class V, class FA, class FR>
inline void pagerankCalculateContributions(vector<V>& a, const H& xt, const vector<V>& f, const vector<V>& c, V C0, V P, FA fa, FR fr) {
  xt.forEachVertexKey([&](auto v) {
    if (!fa(v)) return;
    V ev = pagerankCalculateContribution(a, xt, f, c, v, C0, P);
    fr(v, ev / f[v]);
  });
}


#ifdef OPENMP
/**
 * Calculate contributions for vertices in a graph (using OpenMP).
 * @param a current contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param f contribution factor for each vertex
 * @param c previous contribution of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param fa is vertex affected? (vertex)
 * @param fr called with vertex rank change (vertex, delta)
 */
template <class H, class V, class FA, class FR>
inline void pagerankCalculateContributionsOmp(vector<V>& a, const H& xt, const vector<V>& f, const vector<V>& c, V C0, V P, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v) || !fa(v)) continue;
    V ev = pagerankCalculateContribution(a, xt, f, c, v, C0, P);
    fr(v, ev / f[v]);
  }
}
#endif


/**
 * Calculate the rank of each vertex from its contribution.
 * @param a rank of each vertex (output)
 * @param xt transpose of original graph
 * @param f contribution factor for each vertex
 * @param c contribution of each vertex
 */
template <class H, class V>
inline void pagerankObtainRanks(vector<V>& a, const H& xt, const vector<V>& f, const vector<V>& c) {
  xt.forEachVertexKey([&](auto v) { a[v] = c[v] / f[v]; });
}


#ifdef OPENMP
/**
 * Calculate the rank of each vertex from its contribution (using OpenMP).
 * @param a rank of each vertex (output)
 * @param xt transpose of original graph
 * @param f contribution factor for each vertex
 * @param c contribution of each vertex
 */
template <class H, class V>
inline void pagerankObtainRanksOmp(vector<V>& a, const H& xt, const vector<V>& f, const vector<V>& c) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v)) continue;
    a[v] = c[v] / f[v];
  }
}
#endif
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Intitialize ranks before PageRank iterations.
 * @param a current rank of each vertex (output)
 * @param r previous rank of each vertex (output)
 * @param xt transpose of original graph
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankBasicInitialize(vector<V>& a, vector<V>& r, const H& xt) {
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
inline void pagerankBasicInitializeOmp(vector<V>& a, vector<V>& r, const H& xt) {
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
inline void pagerankBasicInitializeFrom(vector<V>& a, vector<V>& r, const H& xt, const vector<V>& q) {
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
inline void pagerankBasicInitializeFromOmp(vector<V>& a, vector<V>& r, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    r[v] = q[v];
    if (!ASYNC) a[v] = q[v];
  }
}
#endif


/**
 * Setup environment and find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V, class FL>
PagerankResult<V> pagerankBasicInvoke(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), r(S);
  float t = measureDuration([&]() {
    if (q) pagerankBasicInitializeFrom<ASYNC>(a, r, xt, *q);
    else   pagerankBasicInitialize    <ASYNC>(a, r, xt);
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
template <bool ASYNC=false, class H, class V, class FL>
PagerankResult<V> pagerankBasicInvokeOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), r(S);
  float t = measureDuration([&]() {
    if (q) pagerankBasicInitializeFromOmp<ASYNC>(a, r, xt, *q);
    else   pagerankBasicInitializeOmp    <ASYNC>(a, r, xt);
    l = fl(ASYNC? r : a, r, xt, P, E, L);
  }, o.repeat);
  return {r, l, t};
}
#endif
#pragma endregion




#pragma region ENVIRONMENT SETUP (CONTRIB)
/**
 * Intitialize contributions before PageRank iterations.
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param f contribution factor for each vertex (output)
 * @param xt transpose of original graph
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankContribInitialize(vector<V>& a, vector<V>& c, vector<V>& f, const H& xt) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  for (K v=0; v<S; ++v) {
    K d  = xt.vertexValue(v);
    f[v] = d? V(1)/d : V(1);
    c[v] = xt.hasVertex(v)? f[v] / N : V();
    if (!ASYNC) a[v] = c[v];
  }
}


#ifdef OPENMP
/**
 * Intitialize contributions before PageRank iterations (using OpenMP).
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param f contribution factor for each vertex (output)
 * @param xt transpose of original graph
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankContribInitializeOmp(vector<V>& a, vector<V>& c, vector<V>& f, const H& xt) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    K d  = xt.vertexValue(v);
    f[v] = d? V(1)/d : V(1);
    c[v] = xt.hasVertex(v)? f[v] / N : V();
    if (!ASYNC) a[v] = c[v];
  }
}
#endif


/**
 * Intitialize contributions before PageRank iterations from given ranks.
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param f contribution factor for each vertex (output)
 * @param xt transpose of original graph
 * @param q initial ranks
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankContribInitializeFrom(vector<V>& a, vector<V>& c, vector<V>& f, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  for (K v=0; v<S; ++v) {
    K d  = xt.vertexValue(v);
    f[v] = d? V(1)/d : V(1);
    c[v] = q[v] * f[v];
    if (!ASYNC) a[v] = c[v];
  }
}


#ifdef OPENMP
/**
 * Intitialize contributions before PageRank iterations from given ranks (using OpenMP).
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param f contribution factor for each vertex (output)
 * @param xt transpose of original graph
 * @param q initial ranks
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankContribInitializeFromOmp(vector<V>& a, vector<V>& c, vector<V>& f, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    K d  = xt.vertexValue(v);
    f[v] = d? V(1)/d : V(1);
    c[v] = q[v] * f[v];
    if (!ASYNC) a[v] = c[v];
  }
}
#endif


/**
 * Setup environment and find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V, class FL>
PagerankResult<V> pagerankContribInvoke(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), c(S), f(S);
  float t = measureDuration([&]() {
    if (q) pagerankContribInitializeFrom<ASYNC>(a, c, f, xt, *q);
    else   pagerankContribInitialize    <ASYNC>(a, c, f, xt);
    l = fl(ASYNC? c : a, c, xt, f, P, E, L);
    pagerankObtainRanks(a, xt, f, c);
  }, o.repeat);
  return {a, l, t};
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
template <bool ASYNC=false, class H, class V, class FL>
PagerankResult<V> pagerankContribInvokeOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), c(S), f(S);
  float t = measureDuration([&]() {
    if (q) pagerankContribInitializeFromOmp<ASYNC>(a, c, f, xt, *q);
    else   pagerankContribInitializeOmp    <ASYNC>(a, c, f, xt);
    l = fl(ASYNC? c : a, c, xt, f, P, E, L);
    pagerankObtainRanksOmp(a, xt, f, c);
  }, o.repeat);
  return {a, l, t};
}
#endif
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Get the net change in rank between two iterations.
 * @param a current rank of each vertex
 * @param r previous rank of each vertex
 * @returns L∞-norm of the difference between the two vectors
 */
template <class V>
inline V pagerankBasicError(const vector<V>& a, const vector<V>& r) {
  return liNorm(a, r);
}


#ifdef OPENMP
/**
 * Get the net change in rank between two iterations (using OpenMP).
 * @param a current rank of each vertex
 * @param r previous rank of each vertex
 * @returns L∞-norm of the difference between the two vectors
 */
template <class V>
inline V pagerankBasicErrorOmp(const vector<V>& a, const vector<V>& r) {
  return liNormOmp(a, r);
}
#endif


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
    pagerankCalculateRanks(a, xt, r, C0, P, fa, fr); ++l;  // update ranks of vertices
    V el = pagerankBasicError(a, r);  // compare previous and current ranks
    if (!ASYNC) swap(a, r);           // final ranks in (r)
    if (el<E) break;                  // check tolerance
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
    pagerankCalculateRanksOmp(a, xt, r, C0, P, fa, fr); ++l;  // update ranks of vertices
    V el = pagerankBasicErrorOmp(a, r);  // compare previous and current ranks
    if (!ASYNC) swap(a, r);              // final ranks in (r)
    if (el<E) break;                     // check tolerance
  }
  return l;
}
#endif
#pragma endregion




#pragma region COMPUTATION LOOP (CONTRIB)
/**
 * Get the net change in rank between two iterations.
 * @param a current contribution of each vertex
 * @param c previous contribution of each vertex
 * @param f contribution factor for each vertex
 * @returns L∞-norm of the difference between the two vectors
 */
template <class V>
inline V pagerankContribError(const vector<V>& a, const vector<V>& c, const vector<V>& f) {
  V e = V();
  size_t S = a.size();
  for (size_t v=0; v<S; ++v)
    e = max(e, abs(a[v] - c[v]) / f[v]);
  return e;
}


#ifdef OPENMP
/**
 * Get the net change in rank between two iterations (using OpenMP).
 * @param a current contribution of each vertex
 * @param c previous contribution of each vertex
 * @param f contribution factor for each vertex
 * @returns L∞-norm of the difference between the two vectors
 */
template <class V>
inline V pagerankContribErrorOmp(const vector<V>& a, const vector<V>& c, const vector<V>& f) {
  V e = V();
  size_t S = a.size();
  #pragma omp parallel for schedule(auto) reduction(max:e)
  for (size_t v=0; v<S; ++v)
    e = max(e, abs(a[v] - c[v]) / f[v]);
  return e;
}
#endif


/**
 * Perform PageRank iterations upon a graph.
 * @param a current contribution of each vertex (updated)
 * @param c previous contribution of each vertex (updated)
 * @param xt transpose of original graph
 * @param f contribution factor for each vertex
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param fa is vertex affected? (vertex)
 * @param fr called if vertex rank changes (vertex, delta)
 * @returns iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FR>
inline int pagerankContribLoop(vector<V>& a, vector<V>& c, const H& xt, const vector<V>& f, V P, V E, int L, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  while (l<L) {
    V C0 = (1-P)/N;
    pagerankCalculateContributions(a, xt, f, c, C0, P, fa, fr); ++l;  // update contributions of vertices
    V el = pagerankContribError(a, c, f);  // compare previous and current contributions
    if (!ASYNC) swap(a, c);                // final contributions in (c)
    if (el<E) break;                       // check tolerance
  }
  return l;
}


#ifdef OPENMP
/**
 * Perform PageRank iterations upon a graph (using OpenMP).
 * @param a current contribution of each vertex (updated)
 * @param c previous contribution of each vertex (updated)
 * @param xt transpose of original graph
 * @param f contribution factor for each vertex
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param fa is vertex affected? (vertex)
 * @param fr called if vertex rank changes (vertex, delta)
 * @returns iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FR>
inline int pagerankContribLoopOmp(vector<V>& a, vector<V>& c, const H& xt, const vector<V>& f, V P, V E, int L, FA fa, FR fr) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  while (l<L) {
    V C0 = (1-P)/N;
    pagerankCalculateContributionsOmp(a, xt, f, c, C0, P, fa, fr); ++l;  // update contributions of vertices
    V el = pagerankContribErrorOmp(a, c, f);  // compare previous and current contributions
    if (!ASYNC) swap(a, c);                   // final contributions in (c)
    if (el<E) break;                          // check tolerance
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
  return pagerankBasicInvoke<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
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
  return pagerankBasicInvokeOmp<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankBasicLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}
#endif
#pragma endregion




#pragma region STATIC/NAIVE-DYNAMIC (CONTRIB)
/**
 * Find the rank of each vertex in a static graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankContrib(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  return pagerankContribInvoke<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& c, const H& xt, const vector<V>& f, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankContribLoop<ASYNC>(a, c, xt, f, P, E, L, fa, fr);
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
inline PagerankResult<V> pagerankContribOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  return pagerankContribInvokeOmp<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& c, const H& xt, const vector<V>& f, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fr = [](K u, V eu) {};
    return pagerankContribLoopOmp<ASYNC>(a, c, xt, f, P, E, L, fa, fr);
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
  return pagerankBasicInvoke<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierW(vaff, x, y, deletions, insertions);
    return pagerankBasicLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
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
  return pagerankBasicInvokeOmp<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions);
    return pagerankBasicLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}
#endif
#pragma endregion




#pragma region DYNAMIC FRONTIER (CONTRIB)
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
inline PagerankResult<V> pagerankContribDynamicFrontier(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // see adjust-tolerance
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankContribInvoke<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& c, const H& xt, const vector<V>& f, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierW(vaff, x, y, deletions, insertions);
    return pagerankContribLoop<ASYNC>(a, c, xt, f, P, E, L, fa, fr);
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
inline PagerankResult<V> pagerankContribDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // see adjust-tolerance
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankContribInvokeOmp<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& c, const H& xt, const vector<V>& f, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) { if (eu>D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions);
    return pagerankContribLoopOmp<ASYNC>(a, c, xt, f, P, E, L, fa, fr);
  });
}
#endif
#pragma endregion
#pragma endregion

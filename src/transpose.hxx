#pragma once
#include "Graph.hxx"
#include "update.hxx"




// TRANSPOSE
// ---------

template <class H, class G>
inline void transposeW(H& a, const G& x) {
  a.reserve(x.span());
  x.forEachVertex([&](auto u, auto d) { a.addVertex(u, d); });
  x.forEachVertexKey([&](auto u) {
    x.forEachEdge(u, [&](auto v, auto w) { a.addEdge(v, u, w); });
  });
  a.update();
}
template <class G>
inline auto transpose(const G& x) {
  G a; transposeW(a, x);
  return a;
}


#ifdef OPENMP
template <class H, class G>
inline void transposeOmpW(H& a, const G& x) {
  a.reserve(x.span());
  x.forEachVertex([&](auto u, auto d) { a.addVertex(u, d); });
  #pragma omp parallel
  {
    x.forEachVertexKey([&](auto u) {
      x.forEachEdge(u, [&](auto v, auto w) { addEdgeOmpU(a, v, u, w); });
    });
  }
  updateOmpU(a);
}
template <class G>
inline auto transposeOmp(const G& x) {
  G a; transposeOmpW(a, x);
  return a;
}
#endif



// TRANSPOSE WITH DEGREE
// ---------------------

template <class H, class G>
inline void transposeWithDegreeW(H& a, const G& x) {
  a.reserve(x.span());
  x.forEachVertexKey([&](auto u) { a.addVertex(u, x.degree(u)); });
  x.forEachVertexKey([&](auto u) {
    x.forEachEdge(u, [&](auto v, auto w) { a.addEdge(v, u, w); });
  });
  a.update();
}
template <class G>
inline auto transposeWithDegree(const G& x) {
  using K = typename G::key_type;
  using H = decltype(retype(x, K(), K()));
  H a; transposeWithDegreeW(a, x);
  return a;
}


#ifdef OPENMP
template <class H, class G>
inline void transposeWithDegreeOmpW(H& a, const G& x) {
  a.reserve(x.span());
  x.forEachVertexKey([&](auto u) { a.addVertex(u, x.degree(u)); });
  #pragma omp parallel
  {
    x.forEachVertexKey([&](auto u) {
      x.forEachEdge(u, [&](auto v, auto w) { addEdgeOmpU(a, v, u, w); });
    });
  }
  updateOmpU(a);
}
template <class G>
inline auto transposeWithDegreeOmp(const G& x) {
  using K = typename G::key_type;
  using H = decltype(retype(x, K(), K()));
  H a; transposeWithDegreeOmpW(a, x);
  return a;
}
#endif

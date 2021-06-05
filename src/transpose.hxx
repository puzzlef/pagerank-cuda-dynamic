#pragma once
#include "DiGraph.hxx"




// TRANSPOSE
// ---------

template <class H, class G>
void transpose(H& a, const G& x) {
  for (int u : x.vertices())
    a.addVertex(u, x.vertexData(u));
  for (int u : x.vertices()) {
    for (int v : x.edges(u))
      a.addEdge(v, u, x.edgeData(u, v));
  }
}

template <class G>
auto transpose(const G& x) {
  G a; transpose(a, x);
  return a;
}




// TRANSPOSE-WITH-DEGREE
// ---------------------

template <class H, class G>
void transposeWithDegree(H& a, const G& x) {
  for (int u : x.vertices())
    a.addVertex(u, x.degree(u));
  for (int u : x.vertices()) {
    for (int v : x.edges(u))
      a.addEdge(v, u, x.edgeData(u, v));
  }
}

template <class G>
auto transposeWithDegree(const G& x) {
  using E = typename G::TEdge;
  DiGraph<int, E> a; transposeWithDegree(a, x);
  return a;
}

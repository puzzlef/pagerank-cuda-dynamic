#pragma once




template <class H, class G>
void copy(H& a, const G& x) {
  for (int u : x.vertices())
    a.addVertex(u, x.vertexData(u));
  for (int u : x.vertices()) {
    for (int v : x.edges(u))
      a.addEdge(u, v, x.edgeData(u, v));
  }
}

template <class G>
auto copy(const G& x) {
  G a; copy(a, x);
  return a;
}

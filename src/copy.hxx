#pragma once




// COPY
// ----

template <class H, class G, class FV, class FE>
void copyTo(H& a, const G& x, FV fv, FE fe) {
  // add vertices
  for (int u : x.vertices())
    if (fv(u)) a.addVertex(u, x.vertexData(u));
  // add edges
  for (int u : x.vertices()) {
    if (fv(u)) for (int v : x.edges(u)) {
      if (fv(v) && fe(u, v)) a.addEdge(u, v, x.edgeData(u, v));
    }
  }
}

template <class H, class G, class FV>
void copyTo(H& a, const G& x, FV fv) {
  copyTo(a, x, fv, [](int u, int v) { return true; });
}

template <class H, class G>
void copyTo(H& a, const G& x) {
  copyTo(a, x, [](int u) { return true; });
}

template <class G, class FV, class FE>
auto copy(const G& x, FV fv, FE fe) {
  G a; copyTo(a, x, fv, fe);
  return a;
}

template <class G, class FV>
auto copy(const G& x, FV fv) {
  G a; copyTo(a, x, fv);
  return a;
}

template <class G>
auto copy(const G& x) {
  G a; copyTo(a, x);
  return a;
}

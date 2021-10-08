#pragma once
#include <vector>
#include <iterator>
#include <algorithm>
#include "_main.hxx"

using std::vector;
using std::transform;
using std::back_inserter;




// EDGES
// -----

template <class G, class F, class D>
auto edges(const G& x, int u, F fm, D fp) {
  vector<int> a;
  append(a, x.edges(u));
  auto ie = a.end(), ib = a.begin();
  fp(ib, ie); transform(ib, ie, ib, fm);
  return a;
}

template <class G, class F>
auto edges(const G& x, int u, F fm) {
  return edges(x, u, fm, [](auto ib, auto ie) {});
}

template <class G>
auto edges(const G& x, int u) {
  return edges(x, u, [](int v) { return v; });
}




// EDGE
// ----

template <class G, class F>
auto edge(const G& x, int u, F fm) {
  for (int v : x.edges(u))
    return fm(v);
  return -1;
}

template <class G>
auto edge(const G& x, int u) {
  return edge(x, u, [](int v) { return v; });
}




// EDGE-DATA
// ---------

template <class G, class J, class F, class D>
auto edgeData(const G& x, J&& ks, F fm, D fp) {
  using E = decltype(fm(0, 0));
  vector<E> a;
  vector<int> b;
  for (int u : ks) {
    b.clear(); append(b, x.edges(u));
    auto ie = b.end(), ib = b.begin();
    fp(ib, ie); transform(ib, ie, back_inserter(a), [&](int v) { return fm(u, v); });
  }
  return a;
}

template <class G, class J, class F>
auto edgeData(const G& x, J&& ks, F fm) {
  return edgeData(x, ks, fm, [](auto ib, auto ie) {});
}

template <class G, class J>
auto edgeData(const G& x, J&& ks) {
  return edgeData(x, ks, [&](int u, int v) { return x.edgeData(u, v); });
}

template <class G>
auto edgeData(const G& x) {
  return edgeData(x, x.vertices());
}

#pragma once
#include <vector>
#include "_main.hxx"

using std::vector;




// SOURCE-OFFSETS
// --------------

template <class G, class J>
auto sourceOffsets(const G& x, J&& ks) {
  int i = 0;
  vector<int> a;
  a.reserve(x.order()+1);
  for (auto u : ks) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  return a;
}

template <class G>
auto sourceOffsets(const G& x) {
  return sourceOffsets(x, x.vertices());
}




// DESTINATION-INDICES
// -------------------

template <class G, class J>
auto destinationIndices(const G& x, J&& ks) {
  vector<int> a;
  auto ids = indices(ks);
  for (int u : ks) {
    for (int v : x.edges(u))
      a.push_back(ids[v]);
    // sort(a.end()-x.degree(u), a.end());
  }
  return a;
}

template <class G>
auto destinationIndices(const G& x) {
  return destinationIndices(x, x.vertices());
}

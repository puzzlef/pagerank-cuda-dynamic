#pragma once
#include <random>

using std::uniform_real_distribution;




// ADD-RANDOM-EDGE
// ---------------

template <class G, class R>
void addRandomEdge(G& a, R& rnd, int span) {
  uniform_real_distribution<> dis(0.0, 1.0);
  int u = int(dis(rnd) * span);
  int v = int(dis(rnd) * span);
  a.addEdge(u, v);
}


template <class G, class R>
void addRandomEdgeByDegree(G& a, R& rnd, int span) {
  uniform_real_distribution<> dis(0.0, 1.0);
  double deg = a.size() / a.span();
  int un = int(dis(rnd) * deg * span);
  int vn = int(dis(rnd) * deg * span);
  int u = -1, v = -1, n = 0;
  for (int w : a.vertices()) {
    if (un<0 && un > n+a.degree(w)) u = w;
    if (vn<0 && vn > n+a.degree(w)) v = w;
    if (un>0 && vn>=0) break;
    n += a.degree(w);
  }
  if (u<0) u = int(un/deg);
  if (v<0) v = int(vn/deg);
  a.addEdge(u, v);
}

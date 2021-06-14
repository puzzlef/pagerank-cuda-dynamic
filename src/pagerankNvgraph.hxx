#pragma once
#include <vector>
#include <type_traits>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "pagerank.hxx"

using std::vector;
using std::is_same;




// Find pagerank accelerated using nvGraph.
// @param xt transpose graph, with vertex-data=out-degree
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class T=float>
PagerankResult<T> pagerankNvgraph(const G& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  T   p = o.damping;
  T   E = o.tolerance;
  int L = o.maxIterations;
  int N = xt.order();
  nvgraphHandle_t     h;
  nvgraphGraphDescr_t g;
  struct nvgraphCSCTopology32I_st csc;
  cudaDataType_t type = is_same<T, float>::value? CUDA_R_32F : CUDA_R_64F;
  vector<cudaDataType_t> vtype {type, type};
  vector<cudaDataType_t> etype {type};
  vector<T> ranks(N);
  if (N==0) return {ranks};
  auto ks    = vertices(xt);
  auto vfrom = sourceOffsets(xt);
  auto efrom = destinationIndices(xt);
  auto vdata = vertexData(xt, ks, [&](int v) { return xt.vertexData(v)==0? T(1) : T(); });
  auto edata = edgeData(xt, ks, [&](int v, int u) { return T(1)/xt.vertexData(u); });
  if (q) ranks = compressContainer(xt, *q);

  TRY_NVGRAPH( nvgraphCreate(&h) );
  TRY_NVGRAPH( nvgraphCreateGraphDescr(h, &g) );

  csc.nvertices = xt.order();
  csc.nedges    = xt.size();
  csc.destination_offsets = vfrom.data();
  csc.source_indices      = efrom.data();
  TRY_NVGRAPH( nvgraphSetGraphStructure(h, g, &csc, NVGRAPH_CSC_32) );

  TRY_NVGRAPH( nvgraphAllocateVertexData(h, g, vtype.size(), vtype.data()) );
  TRY_NVGRAPH( nvgraphAllocateEdgeData  (h, g, etype.size(), etype.data()) );
  TRY_NVGRAPH( nvgraphSetVertexData(h, g, vdata.data(), 0) );
  TRY_NVGRAPH( nvgraphSetEdgeData  (h, g, edata.data(), 0) );

  float t = measureDurationMarked([&](auto mark) {
    if (q) TRY_NVGRAPH( nvgraphSetVertexData(h, g, ranks.data(), 1) );
    mark([&]() { TRY_NVGRAPH( nvgraphPagerank(h, g, 0, &p, 0, !!q, 1, E, L) ); });
  }, o.repeat);
  TRY_NVGRAPH( nvgraphGetVertexData(h, g, ranks.data(), 1) );

  TRY_NVGRAPH( nvgraphDestroyGraphDescr(h, g) );
  TRY_NVGRAPH( nvgraphDestroy(h) );
  return {decompressContainer(xt, ranks, ks), 0, t};
}

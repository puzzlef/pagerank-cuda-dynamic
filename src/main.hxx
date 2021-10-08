#pragma once
#include "_main.hxx"
#include "DiGraph.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "snap.hxx"
#include "copy.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankCuda.hxx"

#ifndef NVGRAPH_DISABLE
#include "pagerankNvgraph.hxx"
#else
#define pagerankNvgraph pagerankCuda
#endif

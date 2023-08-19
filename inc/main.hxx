#pragma once
#define BUILD  0  // 5 == BUILD_TRACE
#define OPENMP 1
#define CUDA   1
#include "_main.hxx"
#include "Graph.hxx"
#include "mtx.hxx"
#include "csr.hxx"
#include "properties.hxx"
#include "duplicate.hxx"
#include "transpose.hxx"
#include "symmetricize.hxx"
#include "selfLoop.hxx"
#include "dfs.hxx"
#include "bfs.hxx"
#include "batch.hxx"
#include "pagerank.hxx"
#include "pagerankCuda.hxx"

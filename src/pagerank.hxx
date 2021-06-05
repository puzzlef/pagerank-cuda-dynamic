#pragma once
#include <vector>
#include <utility>

using std::vector;
using std::move;




// *-DIM (CUDA)
// ------------
// Good launch config.

// For pagerank cuda block-per-vertex
#define BLOCK_DIM_B 64
#define GRID_DIM_B  4096

// For pagerank cuda thread-per-vertex
#define BLOCK_DIM_T 64
#define GRID_DIM_T  4096




template <class T>
struct PagerankOptions {
  int repeat;
  bool sortVertices;
  T   damping;
  T   tolerance;
  int maxIterations;

  PagerankOptions(int repeat=1, bool sortVertices=false, T damping=0.85, T tolerance=1e-6, int maxIterations=500) :
  repeat(repeat), sortVertices(sortVertices), damping(damping), tolerance(tolerance), maxIterations(maxIterations) {}
};


template <class T>
struct PagerankResult {
  vector<T> ranks;
  int   iterations;
  float time;

  PagerankResult(vector<T>&& ranks, int iterations=0, float time=0) :
  ranks(ranks), iterations(iterations), time(time) {}

  PagerankResult(vector<T>& ranks, int iterations=0, float time=0) :
  ranks(move(ranks)), iterations(iterations), time(time) {}
};

#pragma once
#include <cstdlib>
#include <cstdio>
#include <nvgraph.h>

using std::exit;
using std::fprintf;




#ifndef TRY_NVGRAPH
void tryNvgraph(nvgraphStatus_t err, const char* exp, const char* func, int line, const char* file) {
  if (err == NVGRAPH_STATUS_SUCCESS) return;
  fprintf(stderr,
    "ERROR: nvGraph-%d\n"
    "  in expression %s\n"
    "  at %s:%d in %s\n",
    err, exp, func, line, file);
  exit(err);
}

#define TRY_NVGRAPH(exp) tryNvgraph(exp, #exp, __func__, __LINE__, __FILE__)
#endif

// This code is part of the project "Theoretically Efficient Parallel Graph
// Algorithms Can Be Fast and Scalable", presented at Symposium on Parallelism
// in Algorithms and Architectures, 2018.
// Copyright (c) 2018 Laxman Dhulipala, Guy Blelloch, and Julian Shun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all  copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Usage:
// numactl -i all ./DensestSubgraph -rounds 3 -s -m com-orkut.ungraph.txt_SJ
// flags:
//   required:
//     -s : indicates that the graph is symmetric
//   optional:
//     -m : indicate that the graph should be mmap'd
//     -c : indicate that the graph is compressed
//     -rounds : the number of times to run the algorithm

#include "DensestSubgraph.h"

namespace gbbs {
namespace {

template <class Graph>
double DensestSubgraph_runner(Graph& G, commandLine P) {
  size_t seed = P.getOptionIntValue("-seed", 1);
  size_t iter = P.getOptionIntValue("-iter", 5);
  double cutoff = P.getOptionDoubleValue("-cutoff", 1);
  int option_run = P.getOptionIntValue("-option", 0);
  double approx_kcore_base = P.getOptionDoubleValue("-approx_kcore_base", 1.0);
  std::cout << "### Application: DensestSubgraph" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << num_workers() << std::endl;
  std::cout << "### n: " << G.n << std::endl;
  std::cout << "### m: " << G.m << std::endl;
  std::cout << "### Params: -seed = " << seed << std::endl;
  std::cout << "### Params: -iter = " << iter << std::endl;
  std::cout << "### Params: -cutoff = " << cutoff << std::endl;
  std::cout << "### Params: -option = " << option_run << std::endl;
  std::cout << "### Params: -approx_kcore_base = " << approx_kcore_base << std::endl;

  std::cout << "### ------------------------------------" << std::endl;
  assert(P.getOption("-s"));

  timer t;
  t.start();

  GreedyPlusPlusDensestSubgraph(G, seed, iter, cutoff, option_run, approx_kcore_base);
  double tt = t.stop();

  //std::cout << "### Running Time: " << tt << std::endl;
  return tt;
}

}  // namespace
}  // namespace gbbs

generate_symmetric_main(gbbs::DensestSubgraph_runner, false);

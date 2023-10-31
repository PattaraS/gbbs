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
// numactl -i all ./KCore -rounds 3 -s -m com-orkut.ungraph.txt_SJ
// flags:
//   required:
//     -s : indicates that the graph is symmetric
//   optional:
//     -m : indicate that the graph should be mmap'd
//     -c : indicate that the graph is compressed
//     -rounds : the number of times to run the algorithm
//     -fa : run the fetch-and-add implementation of k-core
//     -nb : the number of buckets to use in the bucketing implementation

#include "KCore.h"
#include "gbbs/gbbs.h"
#include "gbbs/vertex.h"
#include "gbbs/bridge.h"
#include "gbbs/graph_mutation.h"

namespace gbbs {
template <class Graph>
double KCore_runner(Graph& G, commandLine P) {
  size_t num_buckets = P.getOptionLongValue("-nb", 16);
  bool fa = P.getOption("-fa");
  std::cout << "### Application: KCore" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << num_workers() << std::endl;
  std::cout << "### n: " << G.n << std::endl;
  std::cout << "### m: " << G.m << std::endl;
  std::cout << "### Params: -nb (num_buckets) = " << num_buckets
            << " -fa (use fetch_and_add) = " << fa << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  if (num_buckets != static_cast<size_t>((1 << parlay::log2_up(num_buckets)))) {
    std::cout << "Number of buckets must be a power of two."
              << "\n";
    exit(-1);
  }
  assert(P.getOption("-s"));

  // runs the fetch-and-add based implementation if set.
  timer t;
  t.start();
  double base = P.getOptionDoubleValue("-base", 1.05);

  auto cores = ApproxKCore(G, num_buckets, base);
  auto max_core = parlay::reduce_max(cores);
  using W = gbbs::empty;
  auto predicate = [&](const uintE& u, const uintE& v, const W& wgh) -> bool {
      return (cores[u] >= max_core) && (cores[v] >= max_core);
  };

  auto newN = parlay::count_if(cores, [&](auto i){return i >= max_core;});
  std::cout << "### n: " << newN << std::endl;
  
  auto PG = filterGraph(G, predicate);
  std::cout << "### SubGraph core: " << max_core/2 << std::endl;
  std::cout << "### m: " << PG.m << std::endl;
  std::cout << "### density: " << 0.5*PG.m/newN << std::endl;

  double tt = t.stop();

  std::cout << "### Running Time: " << tt << std::endl;

  return tt;

}
}  // namespace gbbs

generate_symmetric_main(gbbs::KCore_runner, false);

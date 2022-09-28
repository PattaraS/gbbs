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
#include "gbbs/semiasym/graph_filter.h"

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
  auto cores = (fa) ? KCore_FA(G, num_buckets) : KCore(G, num_buckets);
  auto max_core = parlay::reduce_max(cores);
  using W = gbbs::empty;
  auto predicate = [&](const uintE& u, const uintE& v, const W& wgh) -> bool {
      return (cores[u] >= max_core/2 ) && (cores[v] >= max_core/2);
  };
  auto PG = sage::filter_graph(G, predicate);
  std::cout << "### SubGraph core: " << max_core/2 << std::endl;
  std::cout << "### n: " << PG.n << std::endl;
  std::cout << "### m: " << PG.m << std::endl;
  for (size_t u = 0 ; u < PG.n ; ++u) {
    if (PG.get_vertex(u).out_degree()) {
        auto edges = parlay::sequence<std::pair<uintE, uintE>>(PG.get_vertex(u).out_degree());
        size_t k = 0;
        auto map_f = [&](const uintE& u, const uintE& v, const W& wgh) {
                edges[k++] = std::make_pair(u, v);
        };

        PG.get_vertex(u).out_neighbors().map(map_f, false);
        for (size_t i = 0; i < edges.size(); i++) {
                std::cout << edges[i].first << " " << edges[i].second << std::endl;
        }
    }
  }

  double tt = t.stop();

  std::cout << "### Running Time: " << tt << std::endl;

  return tt;
}
}  // namespace gbbs

generate_symmetric_main(gbbs::KCore_runner, false);

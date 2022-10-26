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

#pragma once

#include "benchmarks/KCore/JulienneDBS17/KCore.h"
#include "gbbs/gbbs.h"

namespace gbbs {

// Implements a parallel version of Charikar's 2-appx that runs in O(m+n)
// expected work and O(\rho\log n) depth w.h.p.
template <class Graph>
double GreedyPlusPlusDensestSubgraph(Graph& G, size_t seed = 0, size_t T = 1) {

  using W = typename Graph::weight_type;
  typedef symmetric_graph<gbbs::symmetric_vertex, W> sym_graph;

  // deg_ord = degeneracy_order(GA)
  // ## Now, density check for graph after removing each vertex, in the
  // peeling-order.
  // Let S = stores 2*#edges to vertices > in degeneracy order. Note that 2* is
  //         needed since higher-ordered vertices don't have the edge to us.
  //
  // S = scan(S, fl_inplace | fl_reverse) ## reverse scan
  // density w/o vertex_i = S[i] / (n - i)
  // Compute the max over all v.


  double max_density = 0.0;

  auto cores = KCore(G, 16);
  auto max_core = parlay::reduce_max(cores);
  auto predicate = [&](const uintE& u, const uintE& v, const W& wgh) -> bool {
      return (cores[u] >= max_core/2) && (cores[v] >= max_core/2);
  };
  std::unique_ptr<sym_graph> GA = std::make_unique<sym_graph>(inducedSubgraph(G, predicate));

  std::cout << "### New m: " << GA->m << std::endl;
  std::cout << "### Max Core/2: " << max_core/2 << std::endl;

  size_t n = GA->n;
  auto D = sequence<uintE>::from_function(
      n, [&](size_t i) { return GA->get_vertex(i).out_degree();
      //return floor(pow(1.05, floor(log(GA.get_vertex(i).out_degree()/log(1.05)))));
  });

  auto first = true;

  auto rnd = parlay::random(seed);

  while (--T > 0) {
    auto degeneracy_order = DegeneracyOrderWithLoad(*GA, D, 16, rnd);
    auto vtx_to_position = sequence<uintE>(n);

    parallel_for(0, n, [&](size_t i) {
      uintE v = degeneracy_order.A[i];
      vtx_to_position[v] = i;
    });

    auto density_above = sequence<size_t>(n);

    parallel_for(0, n, 1, [&](size_t i) {
      uintE pos_u = vtx_to_position[i];
      auto vtx_f = [&](const uintE& u, const uintE& v, const W& wgh) {
        uintE pos_v = vtx_to_position[v];
        return pos_u < pos_v;
      };
      density_above[pos_u] = 2 * GA->get_vertex(i).out_neighbors().count(vtx_f);
    });

    parallel_for(0, n, 1, [&](size_t i) {
      uintE pos_u = vtx_to_position[i];
      D[i] = D[i] + density_above[pos_u] / 2;
    });

    auto density_rev =
        parlay::make_slice(density_above.rbegin(), density_above.rend());
    size_t total_edges = parlay::scan_inplace(density_rev);
    if (total_edges != GA->m) {
      std::cout << "Assert failed: total_edges should be " << GA->m
                << " but is: " << total_edges << std::endl;
      exit(0);
    }

    auto density_seq = parlay::delayed_seq<double>(n, [&](size_t i) {
      size_t dens;
      size_t rem;
      if (i == 0) {
        dens = GA->m;
      } else {
        dens = density_above[i - 1];
      }
      rem = n - i;
      return static_cast<double>(dens) / static_cast<double>(rem);
    });
    max_density = std::max(max_density,parlay::reduce_max(density_seq));
    std::cout << "### Density of 2-Densest Subgraph is: " << max_density / 2
              << std::endl;

    std::cout << "### " << T << " remaining rounds" << std::endl;


    //if (first && GA->m > 10e6) {
    if (first && max_density > (max_core/2) * 1.5) {
        auto cores2 = KCore(*GA, 16);
        auto km = (uintE) ceil(max_density/2);
        auto predicate2 = [&cores2, km](const uintE& u, const uintE& v, const W& wgh) -> bool {
            return (cores2[u] >= km && cores2[v] >= km);
        };
        std::unique_ptr<sym_graph> GA2 = std::make_unique<sym_graph>(inducedSubgraph(*GA, predicate2));
        first = false;
        GA = std::move(GA2);
        std::cout << "new number of vertices: " << GA->n << ", new edges: " << GA->m << std::endl;
        n = GA->n;
        D = sequence<uintE>::from_function(
            n, [&](size_t i) { return GA->get_vertex(i).out_degree();
        });

    }

  }
  return max_density;
}

}  // namespace gbbs

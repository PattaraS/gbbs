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

// Implements a parallel version of Greedy++ that runs a parallel version of
// Charikar's 2-approx algorithm for several iterations given as a parameter
// into the problem as T, each run is O(m+n) expected work and O(\rho\log n) depth w.h.p.
// for a total of O(T(m + n)) expected work and O(T\rho\log n) depth w.h.p.
//
// option_run parameters:
//  0: approximate k-core first round, ceil of approximate density second run
//  1: exact k-core first round, ceil of approximate density second run
//  2: outputs running time of algorithm when run on ceil of approximate density
//  3: outputs running time of algorithm when only (max k)/2 core
//  4: outputs running time of parallel algorithm after *no* preprocessing done for cores
template <class Graph>
double GreedyPlusPlusDensestSubgraph(Graph& G, size_t seed = 0, size_t T = 1, double cutoff_mult = 1.0,
        int option_run = 0, double approx_kcore_base = 1.0) {
  timer densest_timer;
  auto num_iters = T;

  using W = typename Graph::weight_type;
  typedef symmetric_graph<gbbs::symmetric_vertex, W> sym_graph;

  double max_density = 0.0;
  auto total_densest_time = 0.0;

  std::unique_ptr<sym_graph> GA;
  uintE max_core = 0;
  if (option_run != 4) {
    if (option_run != 2 && option_run != 3)
        densest_timer.start();
    sequence<uintE> cores;
    if (option_run == 0) {
        cores = ApproxKCore(G, 16, approx_kcore_base);
    } else {
        cores = KCore(G, 16);
    }

    max_core = parlay::reduce_max(cores);
    std::cout << "Max core number is: " << max_core << std::endl;

    auto predicate = [&](const uintE& u, const uintE& v, const W& wgh) -> bool {
        uintE threshold = ceil(max_core/(2 * approx_kcore_base));

        return (cores[u] >= threshold) && (cores[v] >= threshold);
    };
    GA = std::make_unique<sym_graph>(inducedSubgraph(G, predicate));

    if (option_run != 2 && option_run != 3)
        total_densest_time += densest_timer.stop();

    std::cout << "Pruned graph (n,m) = (" << GA->n << "," <<GA->m << ")" << std::endl;

    if (option_run != 2)
        densest_timer.start();

    size_t n = GA->n;

    auto D = sequence<uintE>::from_function(
        n, [&](size_t i) { return GA->get_vertex(i).out_degree();
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
            D[i] = D[i] + density_above[pos_u] / 2;
        });

        auto density_rev =
            parlay::make_slice(density_above.rbegin(), density_above.rend());
        size_t total_edges = parlay::scan_inplace(density_rev);
        /*if (total_edges != GA->m) {
            std::cout << "Assert failed: total_edges should be " << GA->m
                << " but is: " << total_edges << std::endl;
            exit(0);
        }*/

        auto density_seq = parlay::delayed_seq<double>(n, [&](size_t i) {
            size_t dens;
            size_t rem;
            if (i == 0) {
                dens = total_edges;
            } else {
                dens = density_above[i - 1];
            }
            rem = n - i;
            return static_cast<double>(dens) / static_cast<double>(rem);
        });
        max_density = std::max(max_density,parlay::reduce_max(density_seq));
        std::cout << "### Density of current Densest Subgraph is: " << max_density / 2.0
                << std::endl;

        std::cout << "### " << T << " remaining rounds" << std::endl;


        if ((option_run < 3) && first && max_density/2.0 > (max_core/2) * cutoff_mult) {

            auto cores2 = KCore(*GA, 16);
            auto km = (uintE) ceil(max_density/2);
            auto predicate2 = [&cores2, km](const uintE& u, const uintE& v, const W& wgh) -> bool {
                return (cores2[u] >= km && cores2[v] >= km);
            };
            std::unique_ptr<sym_graph> GA2 = std::make_unique<sym_graph>(inducedSubgraph(*GA, predicate2));
            first = false;
            GA = std::move(GA2);
            n = GA->n;
            D = sequence<uintE>::from_function(
                n, [&](size_t i) { return GA->get_vertex(i).out_degree();
            });

            std::cout << GA->n << " " << GA->m << std::endl;
            if (option_run == 2)
                densest_timer.start();
        }
    }
    total_densest_time += densest_timer.stop();
  } else {
    densest_timer.start();
    auto n = G.n;
    auto D = sequence<uintE>::from_function(
        n, [&](size_t i) { return G.get_vertex(i).out_degree();
    });
    auto rnd = parlay::random(seed);

    while (--T > 0) {
        auto degeneracy_order = DegeneracyOrderWithLoad(G, D, 16, rnd);
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
            density_above[pos_u] = 2 * G.get_vertex(i).out_neighbors().count(vtx_f);
            D[i] = D[i] + density_above[pos_u] / 2;
        });

        auto density_rev =
            parlay::make_slice(density_above.rbegin(), density_above.rend());
        size_t total_edges = parlay::scan_inplace(density_rev);
        /*if (total_edges != G.m) {
            std::cout << "Assert failed: total_edges should be " << G.m
                << " but is: " << total_edges << std::endl;
            exit(0);
        }*/

        auto density_seq = parlay::delayed_seq<double>(n, [&](size_t i) {
            size_t dens;
            size_t rem;
            if (i == 0) {
                dens = total_edges;
            } else {
                dens = density_above[i - 1];
            }
            rem = n - i;
            return static_cast<double>(dens) / static_cast<double>(rem);
        });
        max_density = std::max(max_density,parlay::reduce_max(density_seq));
        std::cout << "### Density of current Densest Subgraph is: " << max_density / 2.0
                << std::endl;

        std::cout << "### " << T << " remaining rounds" << std::endl;
    }

    total_densest_time += densest_timer.stop();
  }

  std::cout << "### Total core time: " << total_densest_time << std::endl;
  std::cout << "### Avg core time: " << total_densest_time / num_iters << std::endl;
  return max_density;
}

}  // namespace gbbs

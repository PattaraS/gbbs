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
#include "benchmarks/Connectivity/BFSCC/Connectivity.h"
#include "gbbs/gbbs.h"

namespace gbbs {

template <class Graph>
uintE find_delta (Graph& G) {
    auto degs = sequence<uintE>::from_function(G.n,
        [&](size_t i) {
            return G.get_vertex(i).out_degree();
        });
    return *parlay::max_element(degs);
};


// Implements a parallel version of Greedy++ that runs a parallel version of
// Charikar's 2-approx algorithm for several iterations given as a parameter
// into the problem as T, each run is O(m+n) expected work and O(\rho\log n) depth w.h.p.
// for a total of O(T(m + n)) expected work and O(T\rho\log n) depth w.h.p.
//
// option_run parameters:
//  0: approximate k-core first round, ceil of approximate density second run
//  1: exact k-core first round, ceil of approximate density second run
//  2: outputs running time of algorithm when run on ceil of approximate density THIS OPTION IS OBSOLETED
//  TODO: remove unused options
//  3: outputs running time of algorithm when only (max k)/2 core
//  4: outputs running time of parallel algorithm after *no* preprocessing done for cores
template <class Graph>
double GreedyPlusPlusDensestSubgraph(Graph& G, size_t seed = 0, size_t T = 1, double cutoff_mult = 1.0,
        int option_run = 0, double approx_kcore_base = 1.05, bool use_sorting = false, bool obtain_dsg = false) {
  timer densest_timer;
  auto num_iters = T;

  using pii = typename std::pair<uintE, uintE>;
  using W = typename Graph::weight_type;
  typedef symmetric_graph<gbbs::symmetric_vertex, W> sym_graph;

  double max_density = 0.0;
  auto total_densest_time = 0.0;

  size_t max_width = 0;

  size_t n = G.n;
  size_t m = G.m;

  // Set floating-point output precision
  std::cout << std::setprecision(15) << std::fixed;

  std::cout << "# Initial Delta(G): " << find_delta(G) << std::endl;

  std::unique_ptr<sym_graph> GA, DSG;
  uintE max_core = 0;
  if (option_run != 4) {
    if (option_run != 2 && option_run != 3)
        densest_timer.start();
    sequence<uintE> cores;
    if (option_run == 0) {
        cores = ApproxKCore(G, 16, approx_kcore_base);
    } else {
        approx_kcore_base = 1;
        cores = KCore(G, 16);
    }

    max_core = parlay::reduce_max(cores);
    std::cout << "Max core number is: " << max_core << std::endl;

    // TODO: To implement a better subgraph function we will use a prefix array.
    // More precisely, we sort vertices by their core numbers in desc order.
    // Then to get a k-core, we find the largest index such that core(v)>=k.
    // We then shrink graphs into those vertices.
    // To implement this, we also have to think about how to shrink edges.
    // Notice that, after renumbering, we can decide in O(1) if we want to keep an edge.
    // We can also sort edges by min(core(u),core(v)) desc. This way, we also need prefix set of edges.
    // %%%%%%%%
    auto curN = n, curM = m ;

    sequence<pii> vertices_with_core_number = sequence<pii>::from_function(n,
            [&cores](size_t i) ->pii {
                return {i,cores[i]};
            });
    auto obtain_core = [&](Graph& G, uintE k) {

      // sort vertices by core numbers
      integer_sort_inplace(vertices_with_core_number, [&](const pii& p) {
          return -p.second; // using -p.second to sort in the descending order.
          });
      
      // map vertices ids
      sequence<uintE> new_vertex_ids = sequence<uintE>(n);
      parallel_for(0, n, [&](size_t i) {
          new_vertex_ids[vertices_with_core_number[i].first] = i;
          });

      // correct core numbers
      parallel_for(0,n, [&](size_t i){
          cores[i] = vertices_with_core_number[i].second;
          });

      // map edges 
      auto edges = G.edges();
      parallel_for(0,m, [&](size_t i) {
          std::get<0>(edges[i]) = new_vertex_ids[std::get<0>(edges[i])];
          std::get<1>(edges[i]) = new_vertex_ids[std::get<1>(edges[i])];
          });

      auto shell = parlay::find_if_not(vertices_with_core_number, [&](const pii& p) {return p.second >= k;});
      vertices_with_core_number.pop_tail(shell);
      curN = vertices_with_core_number.size();

      edges = filter(edges, [&](const std::tuple<uintE, uintE, W> &e) -> bool {
          return (std::get<0>(e) < curN) && (std::get<1>(e) < curN);
          });

      integer_sort_inplace(edges, [&](const std::tuple<uintE,uintE, W>&e)  {
          return curN*std::get<0>(e) + std::get<1>(e);
      });

      curM = edges.size();
      return sym_graph_from_edges(edges, curN, true);
    };

    auto obtain_core_sym = [&](sym_graph& G, uintE k) {

      // sort vertices by core numbers
      integer_sort_inplace(vertices_with_core_number, [&](const pii& p) {
          return -p.second; // using -p.second to sort in the descending order.
          });
      
      // map vertices ids
      sequence<uintE> new_vertex_ids = sequence<uintE>(curN);
      parallel_for(0, curN, [&](size_t i) {
          new_vertex_ids[vertices_with_core_number[i].first] = i;
          });

      // correct core numbers
      parallel_for(0,curN, [&](size_t i){
          cores[i] = vertices_with_core_number[i].second;
          });

      // map edges 
      auto edges = G.edges();
      parallel_for(0,curM, [&](size_t i) {
          std::get<0>(edges[i]) = new_vertex_ids[std::get<0>(edges[i])];
          std::get<1>(edges[i]) = new_vertex_ids[std::get<1>(edges[i])];
          });

      auto shell = parlay::find_if_not(vertices_with_core_number, [&](const pii& p) {return p.second >= k;});
      vertices_with_core_number.pop_tail(shell);
      curN = vertices_with_core_number.size();

      edges = filter(edges, [&](const std::tuple<uintE, uintE, W> &e) -> bool {
          return (std::get<0>(e) < curN) && (std::get<1>(e) < curN);
          });

      integer_sort_inplace(edges, [&](const std::tuple<uintE,uintE, W>&e)  {
          return curN*std::get<0>(e) + std::get<1>(e);
      });

      curM = edges.size();
      return sym_graph_from_edges(edges, curN, true);
    };

    auto shrink_graph = [&](sym_graph& G, uintE k) {
      auto shell = parlay::find_if_not(vertices_with_core_number, [&](const pii& p) {return p.second >= k;});
      vertices_with_core_number.pop_tail(shell);

      if (vertices_with_core_number.size() == curN) return;

      curN = vertices_with_core_number.size();

      G.shrinkGraph(curN);

    };
    // %%%%%%%%

    uintE core_threshold = ceil(max_core/(2));
    if (option_run == 0) {
      auto GG = obtain_core(G, ceil(max_core/(2*approx_kcore_base)));
      cores = KCore(GG, 16);
      max_core = parlay::reduce_max(cores);
      core_threshold = ceil(max_core/(2));
      vertices_with_core_number = sequence<pii>::from_function(curN,
            [&cores](size_t i) ->pii {
                return {i,cores[i]};
            });
      GA = std::make_unique<sym_graph>(obtain_core_sym(GG, core_threshold));

    } else {
      // This might be needed as we are converting Graph& to sym_graph
      GA = std::make_unique<sym_graph>(obtain_core(G, core_threshold));
    }
    

    std::cout << "# k/2-core Delta(G): " << find_delta(*GA) << std::endl;

    if (option_run != 2)
        total_densest_time += densest_timer.stop();

    std::cout << "Pruned graph (n,m) = (" << GA->n << "," <<GA->m << ")" << std::endl;

    std::cout << "### Initialization Time: " << total_densest_time << std::endl;

    if (option_run != 2)
        densest_timer.start();

    n = GA->n;

    auto D = sequence<uintE>::from_function(
        n, [&](size_t i) { return GA->get_vertex(i).out_degree();
    });

    auto load_pairs = sequence<pii>::from_function(
            n, [&D](size_t i) { return std::make_pair(D[i],i);});
    auto get_key = [&] (const pii& p) { return p.first; };

    auto first_sort = true;

    auto rnd = parlay::random(seed);

    auto vtx_to_position = sequence<uintE>(n);
    size_t round_ctr= 0;
    while (round_ctr++ < T) {
        
        size_t round_width = 0;

        if (use_sorting) {
            auto order = integer_sort(load_pairs, get_key);
            if (first_sort) { // reset load after the first round
                first_sort = false;
                load_pairs = sequence<pii>::from_function(
                        n, [&D](size_t i) { return std::make_pair(0,i);});
            }
            parallel_for(0, n, [&](size_t i) {
                vtx_to_position[order[i].second] = i;
            });
        } else {
            auto degeneracy_order = DegeneracyOrderWithLoad(*GA, D, 16, rnd);
            parallel_for(0, n, [&](size_t i) {
                uintE v = degeneracy_order.A[i];
                vtx_to_position[v] = i;
            });
        }

        auto density_above = sequence<size_t>(n);

        parallel_for(0, n, 1, [&](size_t i) {
            uintE pos_u = vtx_to_position[i];
            auto vtx_f = [&](const uintE& u, const uintE& v, const W& wgh) {
                uintE pos_v = vtx_to_position[v];
                return pos_u < pos_v;
            };
            density_above[pos_u] = 2 * GA->get_vertex(i).out_neighbors().count(vtx_f);
            D[i] = D[i] + density_above[pos_u] / 2;
            load_pairs[i].first = load_pairs[i].first + density_above[pos_u] / 2;
            //round_width = std::max(round_width, density_above[pos_u]);
        });
        round_width = parlay::reduce_max(density_above);
        max_width = std::max(max_width, round_width);

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

        auto max_it = parlay::max_element(density_seq, [&] (const double& a, const double& b) {
            return a<b;
            });

        std::cout << "# ROUND " << round_ctr <<  "/" << T <<" Densest Subgraph is: " << (*max_it) /2.0 << std::endl;

        //if (obtain_dsg && ((*max_it) > max_density) ) {
          //size_t pos_max = max_it - density_seq.begin();
          //auto predicate_DSG = [&](const uintE& u, const uintE& v, const W& wgh) -> bool {
            ////uintE threshold = ceil(max_core/2);
            //return vtx_to_position[u] >= pos_max && vtx_to_position[v] >= pos_max && u!= v;
            ////return (cores[u] >= core_threshold) && (cores[v] >= core_threshold);
          //};
          //auto induced_subgraph_with_mapping = inducedSubgraph(*GA, predicate_DSG, false);
          //DSG= std::make_unique<sym_graph>(std::get<0>(induced_subgraph_with_mapping));
        //}

        max_density = std::max(max_density,parlay::reduce_max(density_seq));
        auto round_time = densest_timer.stop();
        std::cout << "### Density of current Densest Subgraph is: " << max_density / 2.0
                << std::endl;
        std::cout << "### Empirical width so far (round/max) is: " << round_width / 2 << "/" << max_width / 2 << std::endl;
        
        std::cout << "### " << T - round_ctr << " remaining rounds" << std::endl;
        std::cout << "### MWU iteration time: " << round_time << std::endl;
        total_densest_time += round_time;
        std::cout << "### Cumulative time: " << total_densest_time << std::endl;
        densest_timer.start();

        if ((option_run < 3) && max_density/2.0 > core_threshold * cutoff_mult) {
            auto km = (uintE) ceil(max_density/2);
            core_threshold = km;

            shrink_graph(*GA, core_threshold);

            n = GA->n;
            if (use_sorting) {
                load_pairs.pop_tail(load_pairs.begin()+n);
            } else {
                D.pop_tail(D.begin()+n);
            }
            vtx_to_position = sequence<uintE>(n);

            std::cout << "Pruned graph (n,m) = (" << GA->n << "," <<GA->m << ")" << std::endl;
            //std::cout << "# " << core_threshold<< "-core Delta(G): " << find_delta(*GA) << std::endl;
        }
    }
    total_densest_time += densest_timer.stop();
  } else {
    densest_timer.start();
    auto n = G.n;
    auto D = sequence<uintE>::from_function(
        n, [&](size_t i) { return G.get_vertex(i).out_degree();
    });

    auto load_pairs = sequence<pii>::from_function(
            n, [&D](size_t i) { return std::make_pair(D[i],i);});
    auto get_key = [&] (const pii& p) { return p.first; };

    auto rnd = parlay::random(seed);

    auto first_sort = true;

    auto vtx_to_position = sequence<uintE>(n);

    size_t round_ctr= 0;
    while (round_ctr++ < T) {
        size_t round_width = 0;

        if (use_sorting) {
            auto order = integer_sort(load_pairs, get_key);
            if (first_sort) {
                first_sort = false;
                load_pairs = sequence<pii>::from_function(
                        n, [&D](size_t i) { return std::make_pair(0,i);});
            }
            parallel_for(0, n, [&](size_t i) {
                vtx_to_position[order[i].second] = i;
            });
        } else {
            auto degeneracy_order = DegeneracyOrderWithLoad(G, D, 16, rnd);
            parallel_for(0, n, [&](size_t i) {
                uintE v = degeneracy_order.A[i];
                vtx_to_position[v] = i;
            });
        }

        auto density_above = sequence<size_t>(n);

        parallel_for(0, n, 1, [&](size_t i) {
            uintE pos_u = vtx_to_position[i];
            auto vtx_f = [&](const uintE& u, const uintE& v, const W& wgh) {
                uintE pos_v = vtx_to_position[v];
                return pos_u < pos_v;
            };
            density_above[pos_u] = 2 * G.get_vertex(i).out_neighbors().count(vtx_f);
            D[i] = D[i] + density_above[pos_u] / 2;
            load_pairs[i].first = load_pairs[i].first + density_above[pos_u];
        });

        round_width = parlay::reduce_max(density_above);
        max_width = std::max(max_width, round_width);

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

        auto max_it = parlay::max_element(density_seq, [&] (const double& a, const double& b) {
            return a<b;
            });

        std::cout << "# ROUND " << round_ctr <<  "/" << T <<" Densest Subgraph is: " << (*max_it) /2.0 << std::endl;

        max_density = std::max(max_density,parlay::reduce_max(density_seq));
        auto round_time = densest_timer.stop(); // 42e-05
        std::cout << "### Density of current Densest Subgraph is: " << max_density / 2.0
                << std::endl;
        std::cout << "### Empirical width so far (round/max) is: " << round_width / 2 << "/" << max_width / 2 << std::endl;

        std::cout << "### " << T-round_ctr << " remaining rounds" << std::endl;
        //sprintf(buff,"%.6lf", round_time);
        //auto formatted_time = std::string(buff);
        //std::cout << "### MWU iteration time: " << formatted_time << std::endl;
        std::cout << "### MWU iteration time: " << round_time << std::endl;
        total_densest_time += round_time;
        std::cout << "### Cumulative time: " << total_densest_time << std::endl;
        densest_timer.start();
    }

  }

  std::cout << "### Total core time: " << total_densest_time << std::endl;
  std::cout << "### Avg core time: " << total_densest_time / num_iters << std::endl;
  //if (obtain_dsg && DSG) {
    //std::cout << "DESNSEST SUBGRAPH nm: " << DSG->n << " " << DSG->m/2 << " " << (1.0*DSG->m/DSG->n/2) <<  std::endl;
    //auto parents = gbbs::bfs_cc::CC(*DSG);
    //auto unique_parents = unique(parents);
    //std::cout << "CC counts: " << unique_parents.size() << std::endl;
  //}
  return max_density;
}

}  // namespace gbbs

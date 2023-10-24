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
//  5: approx k-core first round an following rounds (running coreness only once)
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

  auto update_time = [&]() {
    auto time_since_last_start = densest_timer.stop();
    total_densest_time += time_since_last_start;
    densest_timer.start();
    return time_since_last_start;
  };

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
    if (option_run == 0 || option_run == 5) {
        cores = ApproxKCore(G, 16, approx_kcore_base);
    } else {
        approx_kcore_base = 1;
        cores = KCore(G, 16);
    }

    std::cout << "k-core-decomposition time: " << update_time() << std::endl;

    max_core = parlay::reduce_max(cores);
    std::cout << "Max core number is: " << max_core << std::endl;

    auto curN = n, curM = m ;
    
    sequence<pii> vertices_with_core_number = sequence<pii>::from_function(n,
            [&cores](size_t i) ->pii {
                return {i,cores[i]};
            });
    auto obtain_core = [&](Graph& G, uintE k) {
      // sort vertices by core numbers
      integer_sort_inplace(vertices_with_core_number, [&](const pii& p) {
          return ~p.second; // using ~p.second to sort in the descending order.
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

      
      auto shell = parlay::find_if_not(vertices_with_core_number, [&](const pii& p) {return p.second >= k;});
      vertices_with_core_number.pop_tail(shell);
      auto nextN = vertices_with_core_number.size();

      // map edges 
      auto pred = [&](const uintE& u, const uintE&v, const W& w) {
        return new_vertex_ids[u] < nextN && new_vertex_ids[v] < nextN;
      };
      std::cout <<"new n " << nextN << std::endl;

      auto degs = sequence<size_t>::from_function(curN, [&](size_t i) {
          return G.get_vertex(i).out_neighbors().count(pred);

      });
      size_t sum_degs = parlay::scan_inplace(make_slice(degs));
      std::cout <<"new M " << sum_degs << std::endl;

      auto edges = sequence<std::tuple<uintE,uintE,W>>(sum_degs);
      parallel_for(0,curN, [&](size_t i)  {
        if (new_vertex_ids[i] >= nextN) return;
        size_t k = degs[i];
        auto map_f = [&](const uintE& u, const uintE& v, const W& wgh) {
          if (pred(u,v,wgh))
            edges[k++] = std::make_tuple(new_vertex_ids[u], new_vertex_ids[v], wgh);
        };
        G.get_vertex(i).out_neighbors().map(map_f, false);
      });

      integer_sort_inplace(edges, [&](const std::tuple<uintE,uintE, W>&e)  {
          return nextN*std::get<0>(e) + std::get<1>(e);
      });

      curN = nextN;
      curM = edges.size();
      return sym_graph_from_edges(edges, curN, true);
    };

    auto obtain_core_sym = [&](sym_graph& G, uintE k) {
      // sort vertices by core numbers
      integer_sort_inplace(vertices_with_core_number, [&](const pii& p) {
          return ~p.second; // using -p.second to sort in the descending order.
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

      auto shell = parlay::find_if_not(vertices_with_core_number, [&](const pii& p) {return p.second >= k;});
      if (shell != vertices_with_core_number.end()) {
        vertices_with_core_number.pop_tail(shell);
      }
      auto nextN = vertices_with_core_number.size();

      auto pred = [&](const uintE& u, const uintE&v, const W& w) {
        return new_vertex_ids[u] < nextN && new_vertex_ids[v] < nextN;
      };
      auto degs = sequence<size_t>::from_function(curN, [&](size_t i) {
          return G.get_vertex(i).out_neighbors().count(pred);
      });
      size_t sum_degs = parlay::scan_inplace(make_slice(degs));
      auto edges = sequence<std::tuple<uintE,uintE,W>>(sum_degs);
      parallel_for(0,curN, [&](size_t i)  {
        if (new_vertex_ids[i] >= nextN) return;
        size_t k = degs[i];
        auto map_f = [&](const uintE& u, const uintE& v, const W& wgh) {
          if (pred(u,v,wgh))
            edges[k++] = std::make_tuple(new_vertex_ids[u], new_vertex_ids[v], wgh);
        };
        G.get_vertex(i).out_neighbors().map(map_f, false);
      });

      integer_sort_inplace(edges, [&](const std::tuple<uintE,uintE, W>&e)  {
          return nextN*std::get<0>(e) + std::get<1>(e);
      });

      curN = nextN;
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
      std::cout<< "shrink graph before second-k-core-decomposition time: " << update_time() << std::endl;
      cores = KCore(GG, 16);

      std::cout<< "second-k-core-decomposition time: " << update_time() << std::endl;
      max_core = parlay::reduce_max(cores);
      core_threshold = ceil(max_core/(2));
      vertices_with_core_number = sequence<pii>::from_function(curN,
            [&cores](size_t i) ->pii {
                return {i,cores[i]};
            });
      GA = std::make_unique<sym_graph>(obtain_core_sym(GG, core_threshold));

    } else if (option_run == 5 ) {
      GA = std::make_unique<sym_graph>(obtain_core(G, ceil(max_core/(2*approx_kcore_base))));
      std::cout<< "shrink graph time: " << update_time() << std::endl;
    } else {
      // This might be needed as we are converting Graph& to sym_graph
      //std::cout << "DEBUG: call obtain_core() on original graph "<< std::endl;
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

        //std::cout << "# ROUND " << round_ctr <<  "/" << T <<" Densest Subgraph is: " << (*max_it) /2.0 << std::endl;

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
        auto round_time = update_time();

        //auto round_time = densest_timer.stop();
        //std::cout << "### Density of current Densest Subgraph is: " << max_density / 2.0
                //<< std::endl;
        //std::cout << "### Empirical width so far (round/max) is: " << round_width / 2 << "/" << max_width / 2 << std::endl;
        
        //std::cout << "### " << T - round_ctr << " remaining rounds" << std::endl;
        //std::cout << "### MWU iteration time: " << round_time << std::endl;
        //total_densest_time += round_time;
        //std::cout << "### Cumulative time: " << total_densest_time << std::endl;
        //densest_timer.start();
        
        auto prune_time = 0.0;
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

            //std::cout << "### Pruned graph (n,m) = (" << GA->n << "," <<GA->m << ")" << std::endl;
            //std::cout << "# " << core_threshold<< "-core Delta(G): " << find_delta(*GA) << std::endl;
            prune_time = densest_timer.stop();
            total_densest_time += prune_time;
        } else if (option_run == 5) {
            auto km = (uintE) ceil(max_density/(2*approx_kcore_base));
            core_threshold = km;
          
            shrink_graph(*GA, core_threshold);

            n = GA->n;
            if (use_sorting) {
                load_pairs.pop_tail(load_pairs.begin()+n);
            } else {
                D.pop_tail(D.begin()+n);
            }
            vtx_to_position = sequence<uintE>(n);

            //std::cout << "### Pruned graph (n,m) = (" << GA->n << "," <<GA->m << ")" << std::endl;
            //std::cout << "# " << core_threshold<< "-core Delta(G): " << find_delta(*GA) << std::endl;
            prune_time = densest_timer.stop();
            total_densest_time += prune_time;
        }
        std::cout << "### summary: " 
          << option_run << " "
          << approx_kcore_base << " "
          << (use_sorting? "Sort" :"Peel") << " "
          << round_ctr << " "
          << round_time << " "
          << prune_time << " "
          << total_densest_time << " "
          << max_density / 2.0 << " "
          << GA->n << " "
          << GA->m << " "
          << (round_width/2) << " "
          << max_width 
          << std::endl;
        densest_timer.start();
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

        //std::cout << "# ROUND " << round_ctr <<  "/" << T <<" Densest Subgraph is: " << (*max_it) /2.0 << std::endl;

        max_density = std::max(max_density,parlay::reduce_max(density_seq));

        auto round_time = update_time(); // 42e-05

        //auto round_time = densest_timer.stop(); // 42e-05
        //std::cout << "### Density of current Densest Subgraph is: " << max_density / 2.0
                //<< std::endl;
        //std::cout << "### Empirical width so far (round/max) is: " << round_width / 2 << "/" << max_width / 2 << std::endl;

        //std::cout << "### " << T-round_ctr << " remaining rounds" << std::endl;
        ////sprintf(buff,"%.6lf", round_time);
        ////auto formatted_time = std::string(buff);
        ////std::cout << "### MWU iteration time: " << formatted_time << std::endl;
        //std::cout << "### MWU iteration time: " << round_time << std::endl;
        //total_densest_time += round_time;
        //std::cout << "### Cumulative time: " << total_densest_time << std::endl;
        //
        std::cout << "### summary: " 
          << option_run << " "
          << approx_kcore_base << " "
          << (use_sorting? "Sort" :"Peel") << " "
          << round_ctr << " "
          << round_time << " "
          << 0 << " "
          << total_densest_time << " "
          << max_density / 2.0 << " "
          << G.n << " "
          << G.m << " "
          << (round_width/2) << " "
          << max_width 
          << std::endl;
        densest_timer.start();
    }

  }

  std::cout << "### Total runtime: " << total_densest_time << std::endl;
  std::cout << "### Avg time per iteration: " << total_densest_time / num_iters << std::endl;
  //if (obtain_dsg && DSG) {
    //std::cout << "DESNSEST SUBGRAPH nm: " << DSG->n << " " << DSG->m/2 << " " << (1.0*DSG->m/DSG->n/2) <<  std::endl;
    //auto parents = gbbs::bfs_cc::CC(*DSG);
    //auto unique_parents = unique(parents);
    //std::cout << "CC counts: " << unique_parents.size() << std::endl;
  //}
  return max_density;
}

}  // namespace gbbs

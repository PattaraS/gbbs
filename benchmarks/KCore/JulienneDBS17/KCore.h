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

#include <set>
#include <queue>
#include <vector>

#include "gbbs/gbbs.h"
#include "gbbs/julienne.h"


namespace gbbs {

template <class Graph>
inline sequence<uintE> KCore(Graph& G, size_t num_buckets = 16) {
  const size_t n = G.n;
  auto D = sequence<uintE>::from_function(
      n, [&](size_t i) { return G.get_vertex(i).out_degree(); });

  auto em = hist_table<uintE, uintE>(std::make_tuple(UINT_E_MAX, 0),
                                     (size_t)G.m / 50);
  auto b = make_vertex_buckets(n, D, increasing, num_buckets);
  timer bt;

  size_t finished = 0, rho = 0, k_max = 0;
  while (finished != n) {
    bt.start();
    auto bkt = b.next_bucket();
    bt.stop();
    auto active = vertexSubset(n, std::move(bkt.identifiers));
    uintE k = bkt.id;
    finished += active.size();
    k_max = std::max(k_max, bkt.id);

    auto apply_f = [&](const std::tuple<uintE, uintE>& p)
        -> const std::optional<std::tuple<uintE, uintE> > {
          uintE v = std::get<0>(p), edgesRemoved = std::get<1>(p);
          uintE deg = D[v];
          if (deg > k) {
            uintE new_deg = std::max(deg - edgesRemoved, k);
            D[v] = new_deg;
            return wrap(v, b.get_bucket(new_deg));
          }
          return std::nullopt;
        };

    auto cond_f = [](const uintE& u) { return true; };
    vertexSubsetData<uintE> moved =
        nghCount(G, active, cond_f, apply_f, em, no_dense);

    bt.start();
    b.update_buckets(moved);
    bt.stop();
    rho++;
  }
  std::cout << "### rho = " << rho << " k_{max} = " << k_max << "\n";
  debug(bt.next("bucket time"););
  return D;
}

/*template <class Graph>
inline auto HalfKCoreSubgraph(Graph& G)
  using W = typename Graph::weight_type;
  auto cores = KCore(G, 16);
  auto max_core = parlay::reduce_max(cores);
  auto predicate = [&](const uintE& u, const uintE& v, const W& wgh) -> bool {
      return (cores[u] >= max_core/2) && (cores[v] >= max_core/2);
  };
  auto GA = filterGraph(G, predicate);
  return GA;
}*/

template <class W>
struct kcore_fetch_add {
  uintE* er;
  uintE* D;
  uintE k;
  kcore_fetch_add(uintE* _er, uintE* _D, uintE _k) : er(_er), D(_D), k(_k) {}
  inline std::optional<uintE> update(const uintE& s, const uintE& d,
                                     const W& w) {
    er[d]++;
    if (er[d] == 1) {
      return std::optional<uintE>((uintE)0);
    }
    return std::nullopt;
  }
  inline std::optional<uintE> updateAtomic(const uintE& s, const uintE& d,
                                           const W& wgh) {
    if (gbbs::fetch_and_add(&er[d], (uintE)1) == 1) {
      return std::optional<uintE>((uintE)0);
    }
    return std::nullopt;
  }
  inline bool cond(uintE d) { return D[d] > k; }
};

template <class Graph>
inline sequence<uintE> KCore_FA(Graph& G, size_t num_buckets = 16) {
  using W = typename Graph::weight_type;
  const size_t n = G.n;
  auto D = sequence<uintE>::from_function(
      n, [&](size_t i) { return G.get_vertex(i).out_degree(); });
  auto ER = sequence<uintE>::from_function(n, [&](size_t i) { return 0; });

  auto b = make_vertex_buckets(n, D, increasing, num_buckets);

  size_t finished = 0;
  size_t rho = 0;
  size_t k_max = 0;
  while (finished != n) {
    auto bkt = b.next_bucket();
    auto active = vertexSubset(n, std::move(bkt.identifiers));
    uintE k = bkt.id;
    finished += active.size();
    k_max = std::max(k_max, bkt.id);

    auto apply_f = [&](const uintE v, uintE& bkt_to_modify) -> void {
      uintE deg = D[v];
      uintE edgesRemoved = ER[v];
      ER[v] = 0;
      uintE new_deg = std::max(deg - edgesRemoved, k);
      D[v] = new_deg;
      bkt_to_modify = b.get_bucket(deg, new_deg);
    };

    auto moved = edgeMapData<uintE>(
        G, active, kcore_fetch_add<W>(ER.begin(), D.begin(), k));
    vertexMap(moved, apply_f);

    if (moved.dense()) {
      b.update_buckets(moved.get_fn_repr(), n);
    } else {
      b.update_buckets(moved.get_fn_repr(), moved.size());
    }
    rho++;
  }
  std::cout << "### rho = " << rho << " k_{max} = " << k_max << "\n";
  return D;
}

template <class Graph>
inline gbbs::dyn_arr<uintE> DegeneracyOrder(Graph& G, size_t num_buckets = 16) {
  const size_t n = G.n;
  auto D = sequence<uintE>::from_function(
      n, [&](size_t i) { return G.get_vertex(i).out_degree(); });

  auto em = EdgeMap<uintE, Graph>(G, std::make_tuple(UINT_E_MAX, 0),
                                  (size_t)G.m / 50);
  auto b = make_vertex_buckets(n, D, increasing, num_buckets);
  timer bt;

  auto degeneracy_order = gbbs::dyn_arr<uintE>(n);

  size_t finished = 0, rho = 0, k_max = 0;
  while (finished < n) {
    bt.start();
    auto bkt = b.next_bucket();
    bt.stop();
    auto active = vertexSubset(n, std::move(bkt.identifiers));
    uintE k = bkt.id;
    finished += active.size();
    k_max = std::max(k_max, bkt.id);

    //std::cout << "generating order" << finished << std::endl;
    auto active_seq = parlay::delayed_seq<uintE>(
        active.size(), [&](uintE i) { return active.s[i]; });
    degeneracy_order.copyIn(active_seq, active.size());

    auto apply_f = [&](const std::tuple<uintE, uintE>& p)
        -> const std::optional<std::tuple<uintE, uintE> > {
          uintE v = std::get<0>(p), edgesRemoved = std::get<1>(p);
          uintE deg = D[v];
          if (deg > k) {
            uintE new_deg = std::max(deg - edgesRemoved, k);
            D[v] = new_deg;
            return wrap(v, b.get_bucket(new_deg));
          }
          return std::nullopt;
        };

    vertexSubsetData<uintE> moved =
        em.template edgeMapCount_sparse<uintE>(active, apply_f);
    bt.start();
    if (moved.dense()) {
      b.update_buckets(moved.get_fn_repr(), n);
    } else {
      b.update_buckets(moved.get_fn_repr(), moved.size());
    }

    bt.stop();
    rho++;
  }
  std::cout << "### rho = " << rho << " k_{max} = " << k_max << "\n";
  debug(bt.next("bucket time"););
  return degeneracy_order;
}

template <class Graph>
inline sequence<uintE> SeqDegeneracyOrderWithLoad(Graph& G, sequence<std::pair<uintE, uintE>> D) {
    auto degeneracy_order = sequence<uintE>(D.size());
    auto finished = 0;
    using pi = std::pair<uintE, uintE>;
    using W = typename Graph::weight_type;
    std::priority_queue<pi, std::vector<pi>, std::greater<pi>> pq;
    std::set<uintE> verts;

    for (size_t i = 0; i < D.size(); i++) {
        pq.push(std::make_pair(D[i].second, D[i].first));
    }

    while (finished < D.size()) {
        auto vert_pair = pq.top();
        pq.pop();

        if (verts.find(vert_pair.second) == verts.end()) {
            verts.insert(vert_pair.second);
            degeneracy_order[finished] = vert_pair.second;
            auto moved = sequence<std::pair<uintE, uintE>>(G.get_vertex(vert_pair.second).out_degree());
            auto map_f = [&](const uintE& u, const uintE& v, const W& wgh, const uintE& nghind) {
                if (verts.find(v) == verts.end()) {
                    D[v].second -= 1;
                }
                moved[nghind] = std::make_pair(v, D[v].second);
            };
            G.get_vertex(vert_pair.second).out_neighbors().map_with_index(map_f, false);

            for (int j = 0; j < moved.size(); j++) {
                pq.push(std::make_pair(moved[j].second, moved[j].first));
            }
            finished++;
        }
    }

    std::set<uintE> new_verts;

    for (size_t i = 0; i < degeneracy_order.size(); i++) {
        new_verts.insert(degeneracy_order[i]);
    }

    return degeneracy_order;
}

template <class Graph>
inline gbbs::dyn_arr<uintE> DegeneracyOrderWithLoad(Graph& G, sequence<uintE> D,size_t num_buckets = 16, parlay::random& rnd =parlay::random()) {
  const size_t n = G.n;

  auto em = EdgeMap<uintE, Graph>(G, std::make_tuple(UINT_E_MAX, 0),
                                  (size_t)G.m / 50);
  auto b = make_vertex_buckets(n, D, increasing, num_buckets);
  timer bt;

  auto degeneracy_order = gbbs::dyn_arr<uintE>(n);

  size_t finished = 0, rho = 0, k_max = 0;
  while (finished != n) {
    bt.start();
    auto bkt = b.next_bucket();
    bt.stop();
    auto active = vertexSubset(n, std::move(bkt.identifiers));
    uintE k = bkt.id;
    finished += active.size();
    k_max = std::max(k_max, bkt.id);

    auto active_seq = parlay::delayed_seq<uintE>(
        active.size(), [&](size_t i) { return active.s[i]; });
    degeneracy_order.copyIn(active_seq, active_seq.size());

    //auto shifting = rand() % active.size();;
    //auto active_rotate = parlay::rotate(active_seq, shifting);
    //auto active_shuffle = parlay::random_shuffle(active_seq, rnd);
    //rnd = rnd.next();

    /*for (size_t i = 0; i < active_shuffle.size(); i++) {
        std::cout << "Bucket element: " << active_shuffle[i] << std::endl;
    }*/
    //std::cout << "active size() = " << active.size() << std::endl;
    //std::cout << "active_shuffle size() = " << active_shuffle.size() << std::endl;
    //degeneracy_order.copyIn(active_shuffle, active_shuffle.size());

    //degeneracy_order.copyIn(active_rotate, active.size());

    auto apply_f = [&](const std::tuple<uintE, uintE>& p)
        -> const std::optional<std::tuple<uintE, uintE> > {
          uintE v = std::get<0>(p), edgesRemoved = std::get<1>(p);
          uintE deg = D[v];
          if (deg > k) {
            // store actual degree for densest subgraph # and peeling degree
            uintE new_deg =
                std::max(k,
                        //(uintE) floor(pow(1.05,floor(log(deg - edgesRemoved)/log(1.05)))));
                        deg-edgesRemoved);
            uintE original = std::max(deg - edgesRemoved, k);
            D[v] = new_deg;
            return wrap(v, b.get_bucket(new_deg));
          }
          return std::nullopt;
        };

    vertexSubsetData<uintE> moved =
        em.template edgeMapCount_sparse<uintE>(active, apply_f);

    bt.start();
    if (moved.dense()) {
      b.update_buckets(moved.get_fn_repr(), n);
    } else {
      b.update_buckets(moved.get_fn_repr(), moved.size());
    }

    bt.stop();
    rho++;
  }
  std::cout << "### rho = " << rho << " k_{max} = " << k_max << "\n";
  debug(bt.next("bucket time"););
  return degeneracy_order;
}
}  // namespace gbbs

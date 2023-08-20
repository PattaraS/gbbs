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
  bool use_sorting = P.getOption("-use_sorting");
  bool obtain_dsg = P.getOption("-obtain_dsg");
  double approx_kcore_base = P.getOptionDoubleValue("-approx_kcore_base", 1.05);
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

  GreedyPlusPlusDensestSubgraph(
      G, 
      seed, 
      iter, 
      cutoff, 
      option_run, 
      approx_kcore_base, 
      use_sorting,
      obtain_dsg
          );
  double tt = t.stop();

  //std::cout << "### Running Time: " << tt << std::endl;
  return tt;
}

}  // namespace
}  // namespace gbbs

generate_symmetric_main(gbbs::DensestSubgraph_runner, false);

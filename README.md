# Practical Parallel Algorithms for Near-Optimal Densest Subgraphs on Massive Graphs


This repository contains code for our parallel densest subgraph algorithms.
Our code uses the framework of the [Graph-Based Benchmark Suite (GBBS)](https://github.com/ParAlg/gbbs).
The `benchmark/ApproximateDensestSubgraph/ours/` directory contains our code as well as our build/make files.
<!--The `benchmark/ApproximateDensestSubgraph/ours/` directory ([Ours Subdirectory](./benchmark/ApproximateDensestSubgraph/ours.README.md))-->
<!--contains relevant information to our algorithms as well as how to run the experiments.-->
For more information regarding GBBS, please visit [their repository](https://github.com/ParAlg/gbbs) for more information.


## Updating Submodules
GBBS requires the [parlaylib](https://github.com/cmuparlay/parlaylib) library as a submodule. Please run the below command to update this submodule:

```
$ git submodule update --init
```

## Build and run our codes
We provide a `makefile` in our subdirectory. Hence, the following commands should compile our programs.

```
$ cd benchmark/ApproximateDensestSubgraph/ours
$ make
```

Following is a sample command to run our algorithm on a dataset.

```
$ PARLAY_NUM_THREADS=<thread> benchmark/ApproximateDensestSubgraph/ours/DensestSubgraph -s  -m [-c]  -rounds <r> -iter <num_iter> -option <option>  -approx_kcore_base <kcore_base> <graph>
```

Notice that, there are many parameters in our command. The following is their description
```
<thread>: number of thread used to run the experiment
-c: this is to specified that the input graph is compressed.
<r>: number of rounds we want to run our algorithm.
<num_iter>: number of iterations
<option>: 0 -> use approx-k-core, 1 -> use exact-k-core, 4-> do not use core (vanilla greedy++)
<kcore_base>: specify the scaling factor for vertices' degrees.
<graph>: an absolute path to graph in gbbs format.
```

## Datasets
You can download the graphs we used in our experiments using the following link:
```
wget https://storage.googleapis.com/densest-subgraph-data/<GRAPH_NAME>

```

The following graphs in gbbs format are available for download:

brain
dblp
orkut
wiki
livejournal
youtube
stackoverflow
hepph

Note that we utilize [gbbs format](https://github.com/ParAlg/gbbs#input-formats) to store graphs.

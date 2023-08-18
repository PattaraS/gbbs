licenses(["notice"])

cc_library(
    name = "DensestSubgraph",
    hdrs = ["DensestSubgraph.h"],
    deps = [
        "//benchmarks/KCore/JulienneDBS17:KCore",
        "//gbbs",
    ],
)

cc_binary(
    name = "DensestSubgraph_main",
    srcs = ["DensestSubgraph.cc"],
    deps = [":DensestSubgraph"],
)

cc_library(
    name = "DensestSubgraphLarge",
    hdrs = ["DensestSubgraphLarge.h"],
    deps = [
        "//benchmarks/KCore/JulienneDBS17:KCore",
        "//gbbs",
    ],
)

cc_binary(
    name = "DensestSubgraphLarge_main",
    srcs = ["DensestSubgraphLarge.cc"],
    deps = [":DensestSubgraphLarge"],
)


package(
    default_visibility = ["//visibility:public"],
)

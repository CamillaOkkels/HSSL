This repository contains the code for the HNSWhssl, HNSWkruskal and HNSWmst algorithms presented in [the paper](https://link.springer.com/chapter/10.1007/978-3-032-06069-3_19?fbclid=IwY2xjawNTD-5leHRuA2FlbQIxMABicmlkETB3SmpYbW1Oa05lVTNYRDlLAR4-bZwl1lfu_SfbDis6F1kOr21S5bZBoAw-Ttl99jKcGXkSQxf6LU4f2Yp0vQ_aem_Ts4he6Ug4XJa_tEmEFT6Cg):

> Camilla Birch Okkels, Erik Thordsen, Martin Aumüller, Arthur Zimek and Erich Schubert:
Approximate Single-Linkage Clustering Using Graph-based Indexes: MST-based Approaches and Incremental Searchers. SISAP 2025

The benchmark framework can be found at [singleLinkage-benchmark](https://github.com/CamillaOkkels/singleLinkage-benchmark/tree/main)

# How To
## Installation

Assuming Rust and python are installed, the rust files can be compiled as follows:
- Navigate to the HNSWhsslRust folder.
- In the terminal, run:

```bash
maturin develop -r
```

# Acknowledgements

The repo is build on the framework from:
>  [GraphIndexBaselines](https://github.com/eth42/GraphIndexBaselines)

Likewise the code in heaps.py is provided by Erik Thordsen

The code in this repo uses functionalities from:
>  [GraphIndexAPI](https://github.com/eth42/GraphIndexAPI)

Please ensure this repo is downloaded to the same directory in order for all dependencies to work proporly.



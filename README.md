# IAN - Iterative Adaptive Neighborhoods

IAN is an algorithm for estimating a data graph from data points or pairwise distances. It can be used in manifold applications, such as dimensionality reduction, geodesic inference, and local dimensionality estimation. For more information, please refer to the paper:

> Dyballa, L., Zucker, S. W. (2023), "IAN: Iterative Adaptive Neighborhoods for manifold learning and dimensionality estimation", _Neural Computation_, 35 (3): 453-524. https://doi.org/10.1162/neco_a_01566 Preprint: https://arxiv.org/abs/2208.09123

## Installation

### Dependencies
IAN requires:
- Python 3
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- CVXPY

Optional: the CVXPY optimization library is needed when using the `'l1'` objective function (default); it supports a number of different solvers. Although its pre-installed solvers (e.g., ECOS and SCS) will work fine with moderate sized datasets, using a commercial solver usually leads to considerably faster convergence. IAN has been tested with the GUROBI solver (https://www.cvxpy.org/install/#install-with-gurobi-support), for which free academic licenses and evaluation trials are available.

### Instructions


From the command line run:

```
pip install git+https://github.com/dyballa/IAN
```

You will need to have `git` installed for this command to work. I strongly recommend creating a fresh virtual environment for installing IAN.

## Documentation

You will find short demos in the notebooks available in the [`examples`](/examples) folder. I am currently adding docs for all available functions. Feel free to contact me by email if you have any questions.
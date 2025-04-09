# VeryDiff Polynomial Experiments

Experimental Evaluation for Equivalence Verification of Polynomial Neural Networks.

# Repository Structure

- `datasets`: Contains datasets used for training the verified neural networks (if directory is missing, it will be auto-extracted from `datasets.tar.gz` once calling `using VeryDiffPolyExperiments` in Julia)
- `networks`: Neural networks used for verification and polynomial approximation in the experiments
- `results`: Results of the experiments
- `src`: Code to run the experiments
    - `generate_nns`: Code to run experiments for synthesis of polynomial neural networks
    - `verify_nns`: Code to run experiments for verifying equivalence of standard and polynomial neural networks

# Setup

Before running experiments, the Julia environment has to be set up.

First, clone the git repository and start Julia
```bash
> git clone https://github.com/phK3/VeryDiffPolyExperiments
> cd VeryDiffPolyExperiments
> julia
```

Within Julia, we can set up the environment via the built-in package manager.
```julia
julia> ]  # to enter the package manager
pkg> activate .  # to activate the environment in the current directory
(VeryDiffPolyExperiments) pkg> instantiate  # to download the necessary dependencies
julia>  # first type backspace to leave the package manager
```

Once the environment is set up, we can run the experiments via (set `test_run=true` if you don't want to run all experiments but just a quick test)
```julia
julia> using VeryDiffPolyExperiments
julia> generate_mnist_nets_verified_bounds(test_run=true)
```

## Ensure Single-Threaded Execution

Although our Julia implementation should only use a single thread by default, calls to external libraries might use multiple threads.
To ensure single-threaded execution, use
```bash
taskset -c 0 julia
```
to start Julia and pin it to processor $0$.






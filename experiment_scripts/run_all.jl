
using VeryDiffPolyExperiments, VeryDiff

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

VeryDiffPolyExperiments.run_mnist_experiment(degrees=20:20:100, n_threads=Threads.nthreads())
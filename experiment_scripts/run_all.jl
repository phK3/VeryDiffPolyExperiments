
using VeryDiffPolyExperiments, VeryDiff

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

VeryDiffPolyExperiments.warmup()

VeryDiffPolyExperiments.run_mnist_experiment(degrees=20:20:100, n_threads=Threads.nthreads())

VeryDiffPolyExperiments.run_collins_experiment(degrees=20:20:160, n_threads=Threads.nthreads())

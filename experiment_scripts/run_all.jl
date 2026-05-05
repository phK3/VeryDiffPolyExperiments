
using VeryDiffPolyExperiments, VeryDiff

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

VeryDiffPolyExperiments.warmup()

VeryDiffPolyExperiments.run_mnist_experiment(degrees=20:20:100, n_threads=Threads.nthreads())

VeryDiffPolyExperiments.run_collins_experiment(degrees=20:20:160, n_threads=Threads.nthreads())

# VeryDiffPolyExperiments.generate_cifar_large_scale()
# VeryDiffPolyExperiments.generate_collins_large_scale(degree=119)
# VeryDiffPolyExperiments.generate_heloc_large_scale(degree=27)
# VeryDiffPolyExperiments.generate_nn4sys_large_scale(degree=59)
# VeryDiffPolyExperiments.generate_cer_large_scale(degree=27)
# VeryDiffPolyExperiments.generate_mnist_large_scale(degree=119)

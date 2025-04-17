
"""
Compute verified bounds on the difference between the original network and the polynomially approximated network.

args:
    logfile_generated_filter - only use logfiles from /results/mnist that contain this string
    use_approximation_domain - if true, use the approximation domain of the polynomials for bounds refinement
    test_run - if true, run a test with a small number of inputs and degrees
"""
function verify_eps_equivalence_mnist(logfile_generated_filter::String ; use_approximation_domain=true, test_run=false)
    println("### Verifying epsilon equivalence MNIST ###")
    test_run && println("[INFO] TESTRUN ...")
    
    logfiles_generated = readdir(string(@__DIR__, "/../../results/mnist"), join=true)
    logfiles_generated = filter(x -> contains(x, logfile_generated_filter), logfiles_generated)

    z = Zonotope(zeros(784), ones(784))
    verify_epsilon_equivalence(logfiles_generated, z; use_approximation_domain=use_approximation_domain, test_run=test_run)
end


function warmup_eps_equivalence_mnist()
    println("[INFO] Warm-up to trigger precompilation ...")
    verify_eps_equivalence_mnist("mnist_256x4_1e4_2025-04-08T21:48:01.008", use_approximation_domain=true, test_run=true)
end


function verify_eps_equivalence_sample_mnist(logfile_generated_filter::String; start_sample=1, n_sample=10, radii=nothing, 
                                             use_approximation_domain=true, only_degrees=nothing, test_run=false)
    println("### Verifying epsilon equivalence MNIST ###")
    test_run && println("[INFO] TESTRUN ...")
    n_sample = test_run ? 1 : n_sample
    radii = test_run ? [0.05] : radii
    radii = isnothing(radii) ? [0.01, 0.03, 0.05] : radii
    !isnothing(only_degrees) && println("[INFO] only considering degrees ", only_degrees)
    println("[INFO] start_sample = ", start_sample, ", n_sample = ", n_sample)
    println("[INFO] turn off warnings for almost zero leading coeffs ...")
    # TODO: why is this still happening?
    saved_val = VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[]
    VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false
    
    logfiles_generated = readdir(string(@__DIR__, "/../../results/mnist"), join=true)
    logfiles_generated = filter(x -> contains(x, logfile_generated_filter), logfiles_generated)

    logfiles_generated = test_run ? [logfiles_generated[1]] : logfiles_generated

    println("[INFO] loading MNIST data ...")
    X_test, y_test = load_mnist_data()
    xs = X_test[start_sample:start_sample+n_sample-1]

    logfile_eps_equiv = string(@__DIR__, "/../../results/mnist/mnist_eps_equiv_samples_", now(), ".csv")

    verify_epsilon_equivalence_sample_csv(logfiles_generated, logfile_eps_equiv, xs, radii, l=0., u=1., 
                                          use_approximation_domain=use_approximation_domain, test_run=test_run,
                                          only_degrees=only_degrees, start_sample=start_sample)

    VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = saved_val
end
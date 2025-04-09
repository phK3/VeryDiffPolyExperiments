
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
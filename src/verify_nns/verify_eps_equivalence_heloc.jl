
"""
Compute verified bounds on the difference between the original network and the polynomially approximated network.

args:
    logfile_generated_filter - only use logfiles from /results/mnist that contain this string
    use_approximation_domain - if true, use the approximation domain of the polynomials for bounds refinement
    test_run - if true, run a test with a small number of inputs and degrees
"""
function verify_eps_equivalence_heloc(logfile_generated_filter::String ; use_approximation_domain=true, test_run=false)
    println("### Verifying epsilon equivalence HELOC ###")
    test_run && println("[INFO] TESTRUN ...")
    
    logfiles_generated = readdir(string(@__DIR__, "/../../results/heloc"), join=true)
    logfiles_generated = filter(x -> contains(x, logfile_generated_filter), logfiles_generated)

    z = Zonotope(zeros(23), ones(23))
    verify_epsilon_equivalence(logfiles_generated, z; use_approximation_domain=use_approximation_domain, test_run=test_run)
end


function warmup_eps_equivalence_heloc()
    println("[INFO] Warm-up to trigger precompilation ...")
    verify_eps_equivalence_heloc("heloc_verified_bounds_heloc_2e5_2025-04-11T09:43:00.999.jld2", use_approximation_domain=true, test_run=true)
end


"""
Compute the name of the logfile for the epsilon equivalence verification from the generated bounds logfile.

The logfile generated from the generate_poly_networks function is in the format:
    <path>/<name>_<date>.jld2
where <name> is the name of the network and <date> is the date when the logfile was created.
"""
function get_eps_equiv_logfile_name_from_generated_bounds_logfile(logfile_generated_nns; samples=false)
    filename = basename(logfile_generated_nns)
    @assert endswith(filename, ".jld2") "logfile_generated_nns must be a .jld2 file (got $(filename))"
    filename_no_ext = join(split(filename, ".")[1:end-1], ".")
    date_string = split(filename_no_ext, "_")[end]
    try 
        DateTime(date_string)
    catch e
        println("Error parsing date string $(date_string) from filename $(filename)")
        println("Make sure the filename is in the format <name>_<date>.jld2")
        rethrow(e)
    end
    filename_no_ext = join(split(filename_no_ext, "_")[1:end-1], "_")
    outname = string(dirname(logfile_generated_nns), "/", filename_no_ext, "_eps_equiv_", ifelse(samples, "samples_", ""), now(), ".jld2")  
    return outname
end


"""
Compute verified bounds for nn_poly(x) - nn(x) for all inputs in z and all outputs of the network.

args:
    nn_poly - polynomial network
    nn - neural network
    z - zonotope input set

kwargs:
    use_approximation_domain - if true, use the approximation domain of the polynomial approximations 
                                    in nn_poly to tighten pre-activation bounds 
                               if false, only use the current zonotope bounds
                               (default: true)

returns:
    bounds_diff - (n_out x 2)-array holding lower and upper bounds for the difference of each output neuron 
"""
function verify_epsilon_equivalence(nn_poly::Network, nn::Network, z::Zonotope; use_approximation_domain=true)
    nn_diff = GeminiNetwork(nn_poly, nn)

    ∂z = Zonotope(zero(z.G), zero(z.c), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)

    if use_approximation_domain
        dom = extract_approximation_domain(nn_poly)
        ẑΔ = nn_diff(zΔ, PropState(true), dom, nothing)
    else
        ẑΔ = nn_diff(zΔ, PropState(true))
    end

    bounds_diff = zono_bounds(ẑΔ.∂Z)
    return bounds_diff
end


function verify_epsilon_equivalence(logfile_generated_nns::String, logfile_eps_equiv::String, z::Zonotope; use_approximation_domain=true, test_run=false)
    println("\n$(logfile_generated_nns) -> $(logfile_eps_equiv)")
    nn, nn_polys, degrees = extract_networks_and_degrees_from_generated_log(logfile_generated_nns)

    if test_run
        nn_polys = nn_polys[[5, 14]]
        degrees = degrees[[5, 14]]
    end

    ∂bounds = Float64[]
    ∂bounds_all = []
    times = Float64[]
    for (degree, nn_poly) in zip(degrees, nn_polys)
        t = @elapsed bounds = verify_epsilon_equivalence(nn_poly, nn, z; use_approximation_domain=use_approximation_domain)
        ∂bound = maximum(abs.(bounds))
        push!(∂bounds, ∂bound)
        push!(∂bounds_all, bounds)
        push!(times, t)
        println("degree = ", degree, " verified bound = ", ∂bound, " (", t, "s)")
    end

    jldsave(logfile_eps_equiv; degrees, ∂bounds, times, ∂bounds_all)
end


function verify_epsilon_equivalence(logfiles_generated::AbstractVector, z::Zonotope; use_approximation_domain=true, test_run=false)
    println("### Verifying epsilon equivalence ###")
    println("use_approximation_domain: $(use_approximation_domain)")
    println("for logfiles: ")
    for logfile_generated in logfiles_generated
        println("  $(logfile_generated)")
    end

    logfiles_eps_equiv = []
    for logfile_generated in logfiles_generated
        logfile_eps_equiv = get_eps_equiv_logfile_name_from_generated_bounds_logfile(logfile_generated)
        push!(logfiles_eps_equiv, logfile_eps_equiv)
    end

    println("storing logfiles at:")
    for logfile_eps_equiv in logfiles_eps_equiv
        println("  $(logfile_eps_equiv)")
    end

    for (logfile_generated, logfile_eps_equiv) in zip(logfiles_generated, logfiles_eps_equiv)
        verify_epsilon_equivalence(logfile_generated, logfile_eps_equiv, z; use_approximation_domain=use_approximation_domain, test_run=test_run)
    end
end



function verify_epsilon_equivalence_sample(logfile_generated_nns::String, logfile_eps_equiv::String, xs::AbstractVector, radii::AbstractVector; 
                                           l=0., u=1., use_approximation_domain=true, test_run=false)
    println("\n$(logfile_generated_nns) -> $(logfile_eps_equiv)")
    nn, nn_polys, degrees = extract_networks_and_degrees_from_generated_log(logfile_generated_nns)

    if test_run
        nn_polys = nn_polys[[5, 14]]
        degrees = degrees[[5, 14]]
    end

    logs = []
    for r in radii
        println("## radius = ", r)

        for (degree, nn_poly) in zip(degrees, nn_polys)
            println("# degree = ", degree)

            for (i, x) in enumerate(xs)
                lb = clamp.(x .- r, l, u)
                ub = clamp.(x .+ r, l, u)

                z = Zonotope(lb, ub)

                t = @elapsed bounds = verify_epsilon_equivalence(nn_poly, nn, z; use_approximation_domain=use_approximation_domain)
                ∂bound = maximum(abs.(bounds))

                push!(logs, (r, degree, ∂bound, bounds, t))

                println("\t", i, ": verified bound = ", ∂bound, " (", t, "s)")
            end

            println("----------------------------")
        end
    end

    jldsave(logfile_eps_equiv; logs)
end


function verify_epsilon_equivalence_sample(logfiles_generated::AbstractVector, xs::AbstractVector, radii::AbstractVector; 
                                           l=0., u=1., use_approximation_domain=true, test_run=false)
    println("### Verifying epsilon equivalence for ", length(xs), " samples ###")
    println("radii = ", radii)
    println("clamping to [", l, ", ", u, "]")
    println("use_approximation_domain: $(use_approximation_domain)")
    println("for logfiles: ")
    for logfile_generated in logfiles_generated
        println("  $(logfile_generated)")
    end

    logfiles_eps_equiv = []
    for logfile_generated in logfiles_generated
        logfile_eps_equiv = get_eps_equiv_logfile_name_from_generated_bounds_logfile(logfile_generated, samples=true)
        push!(logfiles_eps_equiv, logfile_eps_equiv)
    end

    println("storing logfiles at:")
    for logfile_eps_equiv in logfiles_eps_equiv
        println("  $(logfile_eps_equiv)")
    end

    for (logfile_generated, logfile_eps_equiv) in zip(logfiles_generated, logfiles_eps_equiv)
        verify_epsilon_equivalence_sample(logfile_generated, logfile_eps_equiv, xs, radii; l=l, u=u, use_approximation_domain=use_approximation_domain, test_run=test_run)
    end
end


function verify_epsilon_equivalence_sample_csv(logfiles_generated::AbstractVector, logfile_eps_equiv::String, xs::AbstractVector, radii::AbstractVector; 
    l=0., u=1., use_approximation_domain=true, test_run=false)

    for (i, x) in enumerate(xs)
        println("### Sample ", i)
        for r in radii
            println("## radius = ", r)

            for logfile_generated_nns in logfiles_generated
                nn, nn_polys, degrees = extract_networks_and_degrees_from_generated_log(logfile_generated_nns)
                net_name = basename(map_logfile2network(logfile_generated_nns))
                println("# net_name = ", net_name)
        
                if test_run
                    nn_polys = nn_polys[[5, 14]]
                    degrees = degrees[[5, 14]]
                end

                for (degree, nn_poly) in zip(degrees, nn_polys)
                    println("# degree = ", degree)

                    
                    lb = clamp.(x .- r, l, u)
                    ub = clamp.(x .+ r, l, u)

                    z = Zonotope(lb, ub)

                    t = @elapsed bounds = verify_epsilon_equivalence(nn_poly, nn, z; use_approximation_domain=use_approximation_domain)
                    ∂bound = maximum(abs.(bounds))

                    open(logfile_eps_equiv, "a") do f 
                        # netname, degree, sample no., radius, bound, time
                        println(f, string(net_name, ", ", degree, ", ", i, ", ", r, ", ", ∂bound, ", ", t))
                    end

                    println("\t", i, ": verified bound = ", ∂bound, " (", t, "s)")
                end
            end
        end

        println("----------------------------")
    end

end

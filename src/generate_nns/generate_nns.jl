

"""
Given a network and a zonotopic input set, sample random inputs and return the minimal and maximal activation and preactivation values.

args:
    net - network to get bounds from
    z - zonotopic input set

kwargs:
    n_inputs - number of random inputs to sample

returns:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
"""
function get_empirical_bounds(net::Network, z::Zonotope; n_inputs=1000)
    bounds = []

    for i in 1:n_inputs
        x = VeryDiff.random_point(z)
        ys = VeryDiff.intermediate_activations(net, x)

        if length(bounds) == 0
            bounds = [[y y] for y in ys]
        else
            for j in 1:length(bounds)
                bounds[j][:,1] .= min.(bounds[j][:,1], ys[j])
                bounds[j][:,2] .= max.(bounds[j][:,2], ys[j])
            end
        end
    end

    return bounds 
end


"""
Given a network and a set of inputs, return the minimal and maximal activation and preactivation values.

args:
    net - network to get bounds from
    X_in - set of inputs (vector of vectors)

returns:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
"""
function get_empirical_bounds(net::Network, X_in::AbstractVector)
    bounds = []

    for x in X_in
        ys = VeryDiff.intermediate_activations(net, x)

        if length(bounds) == 0
            bounds = [[y y] for y in ys]
        else
            for j in 1:length(bounds)
                bounds[j][:,1] .= min.(bounds[j][:,1], ys[j])
                bounds[j][:,2] .= max.(bounds[j][:,2], ys[j])
            end
        end
    end

    return bounds 
end


"""
Given an array of lower and upper bounds for each neuron, widen the bounds by a given factor.

If you have lower and upper bounds l, u for a neuron, the new bounds are given by

    l_new = center - factor * radius
    u_new = center + factor * radius

where center = 0.5 * (l + u) and radius = 0.5 * (u - l).

args:
    bnds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
           after that layer was applied
    factor - factor to widen the bounds by

returns:
    bnd_new - list of (n_neurons x 2)-array for each layer holding widened lower and upper bounds for each neuron 
              after that layer was applied
"""
function widen_bounds(bnds::AbstractVector, factor::N) where N<:Number
    bnd_new = deepcopy(bnds)
    for i in 1:length(bnds) 
        bnd = bnds[i]
        radius = 0.5 .* (bnd[:,2] .- bnd[:,1])
        center = 0.5 .* (bnd[:,1]  .+ bnd[:,2])

        bnd_new[i][:,1] .= center .- factor .* radius
        bnd_new[i][:,2] .= center .+ factor .* radius
    end 

    return bnd_new
end


"""
Given a network and a zonotopic input set, generate a polynomial approximation of the network.

args:
    net - network to approximate
    z - zonotopic input set
    degree - degree of the polynomial approximation
    max_fun - function to compute the maximum error 
    mae_fun - function to compute the mean absolute error
    mse_fun - function to compute the mean squared error
    acc_fun - function to compute the accuracy

kwargs:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
    X_test - set of inputs to test the polynomial approximation on
    y_test - set of outputs to test the polynomial approximation on (logits)
    y_labels - labels for the outputs
    empirical - if true, use empirical bounds instead of zonotopic bounds
    cheby - if true, use Chebyshev polynomials instead of monomials
    verbosity - verbosity level
    max_iter - maximum number of iterations of the Remez algorithm for polynomial approximation
    widen_factor - factor to widen empirical the bounds by
"""
function generate_poly_network(net::Network, z::Zonotope, degree, max_fun, mae_fun, mse_fun, acc_fun; 
                              bounds=nothing, X_test=nothing, y_test=nothing, y_labels=nothing, 
                              empirical=false, cheby=true, verbosity=0, max_iter=20, widen_factor=1.0)
    if empirical
        bounds = isnothing(bounds) ? get_empirical_bounds(net, z) : bounds
        bounds = widen_bounds(bounds, widen_factor)
        nn_poly = approximate_polynomial(net, bounds,degree)
    else
        nn_poly = approximate_polynomial_iterative(net, z, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter)
    end

    if !isnothing(X_test) && !isnothing(y_test)
        ŷ = [nn_poly(x) for x in X_test]
        ŷ = hcat(ŷ...)'
        max_err = max_fun(y_test, ŷ)
        mae = mae_fun(y_test, ŷ)
        mse = mse_fun(y_test, ŷ)
        # need to add one as labels are 0 indexed
        acc = acc_fun(y_labels .+ 1, ŷ)
        return nn_poly, max_err, mae, mse, acc
    else
        return nn_poly
    end   
end



function generate_poly_networks(net_paths::AbstractVector, z::Zonotope, degrees::AbstractVector, log_file_prefix, 
                                max_fun, mae_fun, mse_fun, acc_fun; bounds=nothing, X_test=nothing, y_labels=nothing, 
                                empirical=false, cheby=true, verbosity=0, max_iter=20, widen_factor=1.0)
    @assert !isnothing(log_file_prefix) "log_file_prefix must be set to a valid path (got $(log_file_prefix))"
    @assert !isnothing(y_labels) "y_labels must be set to a valid vector (got $(y_labels))"

    println("### Generating polynomial networks ###")
    println("\tParameters:")
    println("\t- net_paths: ")
    for net_path in net_paths
        println("\t\t- ", net_path)
    end
    println("\t- degrees: ", degrees)
    println("\t- empirical: ", empirical)

    log_file_names = []
    for net_path in net_paths
        log_file_name = string(log_file_prefix, "_", split(basename(net_path), ".")[1], "_", now(), ".jld2")
        push!(log_file_names, log_file_name)
    end
    println("\t- log_file_names: ")
    for log_file_name in log_file_names
        println("\t\t- ", log_file_name)
    end

    for (net_path, log_file_name) in zip(net_paths, log_file_names)
        println("\n## net_path: ", net_path)

        net = load_network(net_path)

        ŷ = [net(x) for x in X_test]
        ŷ = hcat(ŷ...)'
        original_acc = acc_fun(y_labels .+ 1, ŷ)
        println("\tOriginal accuracy: ", original_acc)

        max_errs = []
        maes = []
        mses = []
        accs = []
        times = []
        nets = []
        for degree in degrees
            t = @elapsed nn_poly, max_err, mae, mse, acc = generate_poly_network(net, z, degree, max_fun, mae_fun, mse_fun, acc_fun; 
                                                                        bounds=bounds, X_test=X_test, y_test=ŷ, y_labels=y_labels, 
                                                                        empirical=empirical, cheby=cheby, verbosity=verbosity, max_iter=max_iter,
                                                                        widen_factor=widen_factor)
            
            println("degree = ", degree, " max_err = ", max_err, " mae = ", mae, " mse = ", mse, " acc = ", acc, " (", t, "s)")
            push!(max_errs, max_err)
            push!(maes, mae)
            push!(mses, mse)
            push!(accs, acc)
            push!(times, t)
            push!(nets, nn_poly)
        end

        jldsave(log_file_name; max_errs, maes, mses, accs, times, nets)
    end   
end
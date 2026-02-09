

const GELU_LOG_FILE_PREFIX = string(@__DIR__, "/../../results/gelu/generate")

"""
Compute maximum error, mean absolute, mean squared error and accuracy for polynomially approximated NNs.

args:
    onnx_path - path to the model to be approximated
    degrees   - vector (or range) of polynomial degrees to try for approximation 
    X_test    - dataset inputs to use for evaluation
    y_labels  - dataset ground truth to use for evaluation

kwargs:
    max_polys_per_layer - currently either Inf (separate polynomial for each neuron) or 1 (single polynomial for a whole layer) is supported
    max_iter - maximum number of Remez iterations 
    verbosity - 
    mode - :sampling -> use sampled bounds over X_test, :sampling01 -> use samples over [0, 1]ⁿ, :abcrown -> use α-CROWN for verified bounds, :zono -> zonotopes for verified bounds 
"""
function compute_approximation_stats(onnx_path, degrees, X_test, y_labels; max_polys_per_layer=1, max_iter=0, verbosity=0, mode=:zono)
    onnx_model = load_onnx_model(onnx_path)
    nn = to_layered_model(onnx_model)

    ŷ = [nn(x) for x in X_test]
    ŷ = hcat(ŷ...)'
    original_acc = acc_fun(y_labels, ŷ)
    println("Original accuracy: $original_acc")

    X_rand = nothing
    if mode == :sampling01 
        X_rand = [rand(784) for i in 1:60000];
        ŷ = [nn(x) for x in X_rand]
        ŷ = hcat(ŷ...)'
    end

    max_errs_single = []
    maes_single = []
    mses_single = []
    accs_single = []

    for d in degrees
        max_err, mae, mse, acc = Inf, Inf, Inf, 0.
        try 
            if mode == :sampling
                nn_poly = approximate_polynomial_iterative_sampling(nn, X_test, d, max_polys_per_layer=max_polys_per_layer, verbosity=verbosity, max_iter=max_iter)
            elseif mode == :sampling01
                nn_poly = approximate_polynomial_iterative_sampling(nn, X_rand, d, max_polys_per_layer=max_polys_per_layer, verbosity=verbosity, max_iter=max_iter)
            elseif mode == :abcrown 
                nn_poly = approximate_polynomial_abcrown(onnx_path, d, max_polys_per_layer=max_polys_per_layer, max_iter=max_iter, verbosity=verbosity);
            elseif mode == :zono
                nn_poly = approximate_polynomial_iterative(nn, Zonotope(zeros(784), ones(784)), d, max_polys_per_layer=max_polys_per_layer, verbosity=verbosity, max_iter=max_iter)
            else 
                @assert false "unknown mode $mode"
            end

            X = mode == :sampling01 ? X_rand : X_test

            max_err, mae, mse, acc, _ = evaluate_network(nn_poly, X, ŷ, y_labels, max_fun, mae_fun, mse_fun, acc_fun)
        catch e 
            @warn "Got exception $e !!!"
            println("Setting error metrics to worst case!")
        end 

        push!(max_errs_single, max_err)
        push!(maes_single, mae)
        push!(mses_single, mse)
        push!(accs_single, acc)
        println("Degree $d: max err = $max_err, mae = $mae, mse = $mse, acc = $acc")
    end

    return max_errs_single, maes_single, mses_single, accs_single
end


function generate_gelu_nets(onnx_paths, degrees; max_polys_per_layer=Inf, max_iter=20)
    println("### Starting GeLU Experiments ###")
    println("\t- degrees: ", degrees)
    log_file_names = []
    for net_path in onnx_paths
        log_file_name = string(GELU_LOG_FILE_PREFIX, "_", split(basename(net_path), ".")[1], "_", now(), ".jld2")
        push!(log_file_names, log_file_name)
    end
    println("\t- log_file_names: ")
    for log_file_name in log_file_names
        println("\t\t- ", log_file_name)
    end

    X_test, y_labels = load_mnist_data()
    y_labels = y_labels .+ 1;  # convert to 1-based indexing


    for (onnx_path, logfile_name) in zip(onnx_paths, log_file_names)
        result_dict = Dict()
        net_name = split(basename(onnx_path), ".")[1]

        @info "Using :abcrown for network $(net_name)"
        mode = :abcrown
        max_errs, maes, mses, accs = compute_approximation_stats(onnx_path, degrees, X_test, y_labels; max_polys_per_layer=max_polys_per_layer, max_iter=max_iter, mode=mode)
        result_dict["$(net_name)_abcrown"] = (max_errs=max_errs, maes=maes, mses=mses, accs=accs)

        @info "Using :sampling for network $(net_name)"
        mode = :sampling
        max_errs, maes, mses, accs = compute_approximation_stats(onnx_path, degrees, X_test, y_labels; max_polys_per_layer=max_polys_per_layer, max_iter=max_iter, mode=mode)
        result_dict["$(net_name)_sampling"] = (max_errs=max_errs, maes=maes, mses=mses, accs=accs)

        jldsave(logfile_name; result_dict)
    end
end


function generate_gelu_nets()
    VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

    onnx_prefix = string(@__DIR__, "/../../networks/mnist/")
    onnx_paths = [
        string(onnx_prefix, "mnist_gelu_256x4_1e4.onnx"), 
        string(onnx_prefix, "mnist/mnist_256x4_1e4.onnx"), 
        string(onnx_prefix, "mnist_scaled_gelu_256x4_1e4.onnx")
    ]

    degrees = (1:19) ∪ (20:5:100)

    generate_gelu_nets(onnx_paths, degrees)
end
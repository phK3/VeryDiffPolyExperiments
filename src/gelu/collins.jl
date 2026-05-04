

COLLINS_FEATURES_PATH = joinpath(@__DIR__, "..", "..", "datasets", "collins_X20_test.csv")
COLLINS_LABELS_PATH   = joinpath(@__DIR__, "..", "..", "datasets", "collins_y20_test.csv")


function load_collins_data()
    f_collins_features = CSV.File(COLLINS_FEATURES_PATH, header=false)
    f_collins_labels   = CSV.File(COLLINS_LABELS_PATH, header=false)
    
    X_test = [reshape(Float64.([x for x in f_collins_features[i]]), 20, 20, 1, 1) for i in 1:size(f_collins_features, 1)]
    y_test = [Float64.([x for x in f_collins_labels[i]])[1] for i in 1:size(f_collins_labels, 1)]
    return X_test, y_test
end


function get_input_bounds_collins(X)
    # get maximum value over all time steps for each sensor
    X_mat = stack(X)[:,:,1,:,:]
    lb = minimum(X_mat, dims=[2,4])
    ub = maximum(X_mat, dims=[2,4])

    println("Input Bounds:")
    println("Lower bound: ", vec(lb))
    println("Upper bound: ", vec(ub))

    # repeat bounds for each time step
    lb = repeat(lb, 1, 20, 1, 1)
    ub = repeat(ub, 1, 20, 1, 1)
    input_bounds = Dict("input" => (lb, ub));
    
    return input_bounds
end


collins_mse = (y, ŷ) -> sum((ŷ .- y).^2) / size(y, 1)


function generate_collins_single(onnx_path, degree)
    if VeryDiff.APPROX_POLY_THREADS[] > 1
        BLAS.set_num_threads(1)
        println("Using ", VeryDiff.APPROX_POLY_THREADS[], " threads for approximation, set BLAS to 1 thread to avoid oversubscription")
    end
    
    X_test, y_test = load_collins_data()
    input_bounds = get_input_bounds_collins(X_test)

    model = load_onnx_model(onnx_path);
    ŷ, mse, t_eval = evaluate_network(model, X_test, y_test, collins_mse)

    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, degree, input_bounds=input_bounds, verbosity=1)
    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))
    model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes));

    ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network(model_poly_dense, X_test, y_test, collins_mse; y_pred=ŷ)
    
    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))

    output_file = string(basename(onnx_path)[1:end-5], "_" , degree, ".json")
    VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "..", "results", "gelu", "collins", output_file))

    println("Summary:")
    println("\tModel: ", basename(onnx_path))
    println("\tDegree: ", degree)
    println("\tBounds: ", input_bounds)
    println("\tMSE: ", mse)
    println("\tMSE (poly): ", mse_poly)
    println("\tSampled Error: ", ϵ_sample)
    println("\tVerified Error Bound: ", ϵ)
    println("\tApproximation Time: ", t_approx)
    println("\tEvaluation Time: ", t_eval)
    println("\tVerification Time: ", t_verify)
    println("\tExported JSON: ", output_file)
end


function generate_collins(onnx_path, degrees; max_polys_per_layer=Inf, method=:acrown, tol=1e-9)
    generate_networks(onnx_path, degrees, load_collins_data, get_input_bounds_collins, collins_mse, max_polys_per_layer=max_polys_per_layer, method=method, tol=tol)
end


function run_collins_experiment(;degrees=20:20:160, n_threads=Threads.nthreads(), tol=1e-9)
    VeryDiff.APPROX_POLY_THREADS[] = n_threads
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_small_window_20_gelu_1e-3l1_kernel_size.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_small_window_20_1e-3l1_kernel_size.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_window_20_gelu_1e-3l1_kernel_size.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_window_20_1e-3l1_kernel_size.onnx")
    ]

    for onnx_path in onnx_paths
        if !contains(basename(onnx_path), "small") 
            # already have results for small ones
            generate_collins(onnx_path, degrees, max_polys_per_layer=Inf, method=:zono, tol=tol)
        end

        # because we already ran that
        if !contains(basename(onnx_path), "gelu")
            generate_collins(onnx_path, degrees, max_polys_per_layer=1, tol=tol)
            generate_collins(onnx_path, degrees, max_polys_per_layer=Inf, tol=tol)
        end
    end
end


function generate_collins_large_scale(;degree=119)
    onnx_paths = [
        # joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_small_window_20_gelu_1e-3l1_kernel_size.onnx"),
        # joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_small_window_20_1e-3l1_kernel_size.onnx"),
        # joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_window_20_gelu_1e-3l1_kernel_size.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_window_20_1e-3l1_kernel_size.onnx")
    ]

    for onnx_path in onnx_paths
        generate_collins_single(onnx_path, degree)
    end
end


function sample_collins_output_ranges()
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_small_window_20_gelu_1e-3l1_kernel_size.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_small_window_20_1e-3l1_kernel_size.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_window_20_gelu_1e-3l1_kernel_size.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "collins", "NN_rul_window_20_1e-3l1_kernel_size.onnx")
    ]

    for onnx_path in onnx_paths
        sample_output_ranges(onnx_path, load_collins_data, collins_mse)
    end
end
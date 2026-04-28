

CER_FEATURES_PATH = joinpath(@__DIR__, "..", "..", "datasets", "cer_x.csv")
CER_LABELS_PATH   = joinpath(@__DIR__, "..", "..", "datasets", "cer_y.csv")


function load_cer_data()
    f_cer_features = CSV.File(CER_FEATURES_PATH, header=false)
    f_cer_labels   = CSV.File(CER_LABELS_PATH, header=false)
    
    X_test = [reshape(Float64.([x for x in f_cer_features[i]]), 51, 1) for i in 1:size(f_cer_features, 1)]
    y_test = [Float64.([x for x in f_cer_labels[i]])[1] for i in 1:size(f_cer_labels, 1)]
    return X_test, y_test
end


function get_input_bounds_cer(X)
    # get maximum value over all time steps for each sensor
    X_mat = stack(X)[:,1,:]
    lb = minimum(X_mat, dims=2)
    ub = maximum(X_mat, dims=2)

    println("Input Bounds:")
    println("Lower bound: ", vec(lb))
    println("Upper bound: ", vec(ub))

    input_bounds = Dict("input" => (lb, ub));    
    return input_bounds
end


# TODO: replace by weighted mse
cer_mse = (y, ŷ) -> sum((ŷ .- y).^2) / size(y, 1)


function generate_cer_single(onnx_path, degree)
    if VeryDiff.APPROX_POLY_THREADS[] > 1
        BLAS.set_num_threads(1)
        println("Using ", VeryDiff.APPROX_POLY_THREADS[], " threads for approximation, set BLAS to 1 thread to avoid oversubscription")
    end
    
    X_test, y_test = load_cer_data()
    input_bounds = get_input_bounds_cer(X_test)

    model = load_onnx_model(onnx_path);
    ŷ, mse, t_eval = evaluate_network(model, X_test, y_test, cer_mse)

    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, degree, input_bounds=input_bounds, verbosity=1)
    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))
    model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes));

    ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network(model_poly_dense, X_test, y_test, cer_mse; y_pred=ŷ)
    
    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))

    output_file = string(basename(onnx_path)[1:end-5], "_" , degree, ".json")
    VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "..", "results", "gelu", "cer", output_file))

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


function generate_cer(onnx_path, degrees; max_polys_per_layer=Inf)
    generate_networks(onnx_path, degrees, load_cer_data, get_input_bounds_cer, cer_mse, max_polys_per_layer=max_polys_per_layer)
end


function run_cer_experiment(;degrees=20:20:160, n_threads=Threads.nthreads())
    VeryDiff.APPROX_POLY_THREADS[] = n_threads
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "cer", "cer_nusr10_GeLU_arch1x64_val4.06e-01.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "cer", "cer_nusr10_ReLU_arch1x64_val4.30e-01.onnx")
    ]

    for onnx_path in onnx_paths
        generate_cer(onnx_path, degrees, max_polys_per_layer=1)
        generate_cer(onnx_path, degrees, max_polys_per_layer=Inf)
    end
end


function generate_cer_large_scale(;degree=27)
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "cer", "cer_nusr10_GeLU_arch1x64_val4.06e-01.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "cer", "cer_nusr10_ReLU_arch1x64_val4.30e-01.onnx")
    ]
    for onnx_path in onnx_paths
        generate_cer(onnx_path, [degree], max_polys_per_layer=Inf)
    end 
end

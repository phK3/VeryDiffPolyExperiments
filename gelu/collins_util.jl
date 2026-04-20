

COLLINS_FEATURES_PATH = joinpath(@__DIR__, "..", "datasets", "collins_X20_test.csv")
COLLINS_LABELS_PATH   = joinpath(@__DIR__, "..", "datasets", "collins_y20_test.csv")


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


function evaluate_network_collins(model, X, y; y_pred=nothing)
    input_name = first(keys(model.input_shapes))
    input_shape = model.input_shapes[input_name]

    if length(input_shape) == 1
        # flattened model
        t_eval = @elapsed  ŷ = [VNNLib.compute_output(model, vec(X[i])) for i in 1:size(X, 1)];
    else
        t_eval = @elapsed  ŷ = [VNNLib.compute_output(model, X[i]) for i in 1:size(X, 1)];
    end

    ŷ = hcat(ŷ...)'
    mse = sum((ŷ .- y).^2) / size(y, 1)
    println("MSE: ", mse)
    println("Evaluation Time: ", t_eval)

    if !isnothing(y_pred)
        ϵ_sample = maximum(abs.(y_pred - ŷ))
        println("Sampled Error: ", ϵ_sample)
        return mse, t_eval, ϵ_sample
    end

    return ŷ, mse, t_eval
end


function generate_collins_single(onnx_path, degree)
    X_test, y_test = load_collins_data()
    input_bounds = get_input_bounds_collins(X_test)

    model = load_onnx_model(onnx_path);
    ŷ, mse, t_eval = evaluate_network_collins(model, X_test, y_test)

    model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, degree, input_bounds=input_bounds, verbosity=1)
    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))
    model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes));

    ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network_collins(model_poly_dense, X_test, y_test; y_pred=ŷ)
    
    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))

    output_file = string(basename(model_path)[1:end-5], "_" , degree, ".json")
    VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "results", "gelu", "collins", output_file))

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


function generate_collins(onnx_path, degrees)
    println("Using ", min(Threads.nthreads(), VeryDiff.approx_poly_threads[]), " threads for approximation")

    logfile = string(basename(onnx_path)[1:end-5], "_results_", now(), ".jld2")
    println("Logging results to ", logfile)

    X_test, y_test = load_collins_data()
    input_bounds = get_input_bounds_collins(X_test)

    model = load_onnx_model(onnx_path)

    ŷ, mse, t_eval = evaluate_network_collins(model, X_test, y_test)
    println("Base model MSE: ", mse)

    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))

    ϵs = []
    ϵ_samples = []
    mse_polys = []
    t_approxs = []
    t_verifies = []
    t_evals = []
    # stores by reference, so if ϵs is modified, result_dict is also updated when saved
    result_dict = Dict(
            "degrees" => degrees,
            "ϵs" => ϵs,
            "ϵ_samples" => ϵ_samples,
            "mse_polys" => mse_polys,
            "t_approxs" => t_approxs,
            "t_verifies" => t_verifies,
            "t_evals" => t_evals,
            "threads" => Threads.nthreads(),
            "approx_poly_threads" => VeryDiff.approx_poly_threads[]
        )
    for d in degrees
        println("Approximating with degree ", d, "...")
        model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, d, input_bounds=input_bounds, verbosity=1)
        model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes))

        ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network_collins(model_poly_dense, X_test, y_test; y_pred=ŷ)
        println("Polynomial MSE: ", mse_poly)

        t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))
        println("Verified Error Bound: ", ϵ)

        push!(ϵs, ϵ)
        push!(ϵ_samples, ϵ_sample)
        push!(mse_polys, mse_poly)
        push!(t_approxs, t_approx)
        push!(t_verifies, t_verify)
        push!(t_evals, t_eval)

        # save result dict in every iteration to avoid losing results in case of crashes
        jldsave(logfile; result_dict)

        println("Degree: ", d, " ϵ: ", ϵ, ", ϵ_sample: ", ϵ_sample, ", mse_poly: ", mse_poly, ", t_approx: ", t_approx, " t_verify: ", t_verify, " t_eval: ", t_eval)
    end

end
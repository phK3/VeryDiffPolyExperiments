

NN4SYS_DATA_PATH = joinpath(@__DIR__, "..", "..", "datasets", "nn4sys_spec_all.txt")

function load_nn4sys_data()
    f_nn4sys_data = CSV.File(NN4SYS_DATA_PATH, header=false)
    X_lb = [Float64.([x for x in f_nn4sys_data[i]][1]) for i in 1:size(f_nn4sys_data, 1)]
    X_ub = [Float64.([x for x in f_nn4sys_data[i]][2]) for i in 1:size(f_nn4sys_data, 1)]
    y_lb = [Float64.([x for x in f_nn4sys_data[i]][3]) for i in 1:size(f_nn4sys_data, 1)]
    y_ub = [Float64.([x for x in f_nn4sys_data[i]][4]) for i in 1:size(f_nn4sys_data, 1)]

    X_test = [reshape([0.5 * (lb + ub)], 1, 1) for (lb, ub) in zip(X_lb, X_ub)]
    y_test = [0.5 * (lb + ub) for (lb, ub) in zip(y_lb, y_ub)]

    return X_test, y_test
end


function get_input_bounds_nn4sys(X)
    lb = reshape([0.], 1, 1)
    ub = reshape([1.], 1, 1)
    input_bounds = Dict("input" => (lb, ub));
    
    return input_bounds
end


nn4sys_mae = (y, ŷ) -> sum(abs.(ŷ .- y)) / size(y, 1)


function generate_nn4sys_single(onnx_path, degree)
    if VeryDiff.APPROX_POLY_THREADS[] > 1
        BLAS.set_num_threads(1)
        println("Using ", VeryDiff.APPROX_POLY_THREADS[], " threads for approximation, set BLAS to 1 thread to avoid oversubscription")
    end
    
    X_test, y_test = load_nn4sys_data()
    input_bounds = get_input_bounds_nn4sys(X_test)

    model = load_onnx_model(onnx_path);
    ŷ, mse, t_eval = evaluate_network(model, X_test, y_test, nn4sys_mae)

    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, degree, input_bounds=input_bounds, verbosity=1)
    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))
    model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes));

    ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network(model_poly_dense, X_test, y_test, nn4sys_mae; y_pred=ŷ)
    
    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))

    output_file = string(basename(onnx_path)[1:end-5], "_" , degree, ".json")
    VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "..", "results", "gelu", "nn4sys", output_file))

    println("Summary:")
    println("\tModel: ", basename(onnx_path))
    println("\tDegree: ", degree)
    println("\tBounds: ", input_bounds)
    println("\tMAE: ", mse)
    println("\tMAE (poly): ", mse_poly)
    println("\tSampled Error: ", ϵ_sample)
    println("\tVerified Error Bound: ", ϵ)
    println("\tApproximation Time: ", t_approx)
    println("\tEvaluation Time: ", t_eval)
    println("\tVerification Time: ", t_verify)
    println("\tExported JSON: ", output_file)
end


function generate_nn4sys(onnx_path, degrees)
    generate_networks(onnx_path, degrees, load_nn4sys_data, get_input_bounds_nn4sys, nn4sys_mae)
end


function run_nn4sys_experiment(;n_threads=Threads.nthreads())
    VeryDiff.APPROX_POLY_THREADS[] = n_threads
    onnx_path = joinpath(@__DIR__, "..", "..", "networks", "nn4sys", "lindex_gelu_5e-7l1.onnx")
    degrees = 20:20:100
    generate_nn4sys(onnx_path, degrees)
end
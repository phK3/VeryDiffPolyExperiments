

HELOC_DATA_PATH = joinpath(@__DIR__, "..", "..", "datasets", "heloc_dataset.csv")

σ(x) = 1 / (1 + exp(-x))

"""
Accuracy function for binary classification tasks.

args:
    ŷ - matrix (n_inputs × outputs) of predicted logits
    y - vector of true labels (make sure they are 1 indexed)
"""
function acc_fun_binary(ŷ::AbstractArray, y::AbstractArray{<:Number})
    count(round.(σ.(ŷ)) .== y) / length(y)
end


function load_heloc_data()
    data_min = -9 .* ones(23)
    # different upper bounds than for Zonopoly??? But just took max of data in f_heloc.
    data_max = [93, 803, 383, 383, 79, 19, 19, 100, 83, 9, 8, 104, 19, 100, 24, 66, 66, 232, 471, 32, 23, 18, 100];

    f_heloc = CSV.File(HELOC_DATA_PATH)
    X_test = [Float64.([x for x in f_heloc[i]][2:end]) for i in 1:size(f_heloc, 1)]
    X_test = [(x .- data_min) ./ (data_max .- data_min) for x in X_test]
    y_test = [[x for x in f_heloc[i]][1] for i in 1:size(f_heloc, 1)]
    y_test = [ifelse(y == "Good", 1., 0.) for y in y_test]
    return X_test, y_test
end


function get_input_bounds_heloc(X)
    lb = reshape(zeros(23), 23, 1)
    ub = reshape(ones(23), 23, 1)
    input_bounds = Dict("input" => (lb, ub));
    return input_bounds
end
    
function generate_heloc_single(onnx_path, degree)
    if VeryDiff.APPROX_POLY_THREADS[] > 1
        BLAS.set_num_threads(1)
        println("Using ", VeryDiff.APPROX_POLY_THREADS[], " threads for approximation, set BLAS to 1 thread to avoid oversubscription")
    end
    
    X_test, y_test = load_heloc_data()
    input_bounds = get_input_bounds_heloc(X_test)

    model = load_onnx_model(onnx_path);
    ŷ, mse, t_eval = evaluate_network(model, X_test, y_test, acc_fun_binary)

    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, degree, input_bounds=input_bounds, verbosity=1)
    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))
    model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes));

    ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network(model_poly_dense, X_test, y_test, acc_fun_binary; y_pred=ŷ)
    
    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))

    output_file = string(basename(onnx_path)[1:end-5], "_" , degree, ".json")
    VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "..", "results", "gelu", "heloc", output_file))

    println("Summary:")
    println("\tModel: ", basename(onnx_path))
    println("\tDegree: ", degree)
    println("\tBounds: ", input_bounds)
    println("\tAccuracy: ", mse)
    println("\tAccuracy (poly): ", mse_poly)
    println("\tSampled Error: ", ϵ_sample)
    println("\tVerified Error Bound: ", ϵ)
    println("\tApproximation Time: ", t_approx)
    println("\tEvaluation Time: ", t_eval)
    println("\tVerification Time: ", t_verify)
    println("\tExported JSON: ", output_file)
end


function generate_heloc(onnx_path, degrees; max_polys_per_layer=Inf, method=:acrown)
    generate_networks(onnx_path, degrees, load_heloc_data, get_input_bounds_heloc, acc_fun_binary, max_polys_per_layer=max_polys_per_layer, method=method)
end

function generate_heloc_sampling(onnx_path, degrees; widen_factor=2.)
    model = load_onnx_model(onnx_path)
    X_test, y_test = load_heloc_data()

    sampled_networks = []
    for d in degrees
        println("Approximating with degree ", d, " (sampling) ...")
        nn_sampled = VeryDiff.approximate_polynomial_iterative_sampling(model, X_test, d, widen_factor=widen_factor, verbosity=1, max_polys_per_layer=Inf)
        push!(sampled_networks, nn_sampled)
    end

    logfile = string(basename(onnx_path)[1:end-5], "_sampling_" , now(), ".jld2")
    jldsave(logfile; sampled_networks) 
end


function run_heloc_experiment(;degrees=20:20:100, n_threads=Threads.nthreads())
    VeryDiff.APPROX_POLY_THREADS[] = n_threads
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "heloc", "heloc_2e5.onnx")
        joinpath(@__DIR__, "..", "..", "networks", "heloc", "heloc_2e5_gelu.onnx")
    ]
    for onnx_path in onnx_paths
        generate_heloc(onnx_path, degrees, max_polys_per_layer=1)
        generate_heloc(onnx_path, degrees, max_polys_per_layer=Inf)
        generate_heloc(onnx_path, degrees, max_polys_per_layer=Inf, method=:zono)
    end

    for onnx_path in onnx_paths
        generate_heloc_sampling(onnx_path, degrees; widen_factor=2.)
    end
end


function generate_heloc_large_scale(;degree=27)
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "heloc", "heloc_2e5.onnx")
        joinpath(@__DIR__, "..", "..", "networks", "heloc", "heloc_2e5_gelu.onnx")
    ]
    for onnx_path in onnx_paths
        generate_heloc(onnx_path, [degree], max_polys_per_layer=Inf)
    end 
end
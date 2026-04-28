

MNIST_DATA_PATH = string(@__DIR__, "/../../datasets/mnist_train.csv")
MNIST_TEST_DATA_PATH = string(@__DIR__, "/../../datasets/mnist_test.csv")

"""
Accuracy function for classification tasks.

args:
    ŷ - matrix (n_inputs × outputs) of predicted logits
    y - vector of true labels (make sure they are 1 indexed)
"""
function mnist_acc_fun(ŷ::AbstractArray, y::AbstractArray{<:Number})
    # argmax(ŷ, dims=2) returns a CartesianIndex object, so we need to extract the index of the row
    sum(getindex.(argmax(ŷ, dims=2), 2) .== y .+ 1) / size(y, 1)
end


function load_mnist_data(;test_data=false)
    path = test_data ? MNIST_TEST_DATA_PATH : MNIST_DATA_PATH
    f_mnist = CSV.File(path, header=false)
    X_test = [reshape(Float64.([x for x in f_mnist[i]][2:end]) ./ 255, 28, 28, 1, 1) for i in 1:size(f_mnist, 1)]
    y_test = [[x for x in f_mnist[i]][1] for i in 1:size(f_mnist, 1)]
    return X_test, y_test
end


function get_input_bounds_mnist(X)
    lb = reshape(zeros(784), 28, 28, 1, 1)
    ub = reshape(ones(784), 28, 28, 1, 1)
    input_bounds = Dict("input" => (lb, ub));    
    return input_bounds
end
    
function generate_mnist_single(onnx_path, degree)
    if VeryDiff.APPROX_POLY_THREADS[] > 1
        BLAS.set_num_threads(1)
        println("Using ", VeryDiff.APPROX_POLY_THREADS[], " threads for approximation, set BLAS to 1 thread to avoid oversubscription")
    end
    
    X_test, y_test = load_mnist_data()
    input_bounds = get_input_bounds_mnist(X_test)

    model = load_onnx_model(onnx_path);
    ŷ, mse, t_eval = evaluate_network(model, X_test, y_test, mnist_acc_fun)

    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, degree, input_bounds=input_bounds, verbosity=1)
    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))
    model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes));

    ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network(model_poly_dense, X_test, y_test, mnist_acc_fun; y_pred=ŷ)
    
    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))

    output_file = string(basename(onnx_path)[1:end-5], "_" , degree, ".json")
    VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "..", "results", "gelu", "mnist", output_file))

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


function generate_mnist(onnx_path, degrees; max_polys_per_layer=Inf, method=:acrown)
    generate_networks(onnx_path, degrees, load_mnist_data, get_input_bounds_mnist, mnist_acc_fun, max_polys_per_layer=max_polys_per_layer, method=method)
end

function generate_mnist_sampling(onnx_path, degrees; widen_factor=2.)
    model = load_onnx_model(onnx_path)
    X_test, y_test = load_mnist_data()

    sampled_networks = []
    for d in degrees
        println("Approximating with degree ", d, " (sampling) ...")
        nn_sampled = VeryDiff.approximate_polynomial_iterative_sampling(model, X_test, d, widen_factor=widen_factor, verbosity=1, max_polys_per_layer=Inf)
        push!(sampled_networks, nn_sampled)
    end

    logfile = string(basename(onnx_path)[1:end-5], "_sampling_" , now(), ".jld2")
    jldsave(logfile; sampled_networks) 
end


function run_mnist_experiment(;degrees=20:20:100, n_threads=Threads.nthreads())
    VeryDiff.APPROX_POLY_THREADS[] = n_threads
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "mnist", "mnist_256x4_1e4.onnx")
        joinpath(@__DIR__, "..", "..", "networks", "mnist", "mnist_gelu_256x4_1e4.onnx")
    ]
    for onnx_path in onnx_paths
        generate_mnist(onnx_path, degrees, max_polys_per_layer=1)
        generate_mnist(onnx_path, degrees, max_polys_per_layer=Inf)
        generate_mnist(onnx_path, degrees, max_polys_per_layer=Inf, method=:zono)
    end

    for onnx_path in onnx_paths
        generate_mnist_sampling(onnx_path, degrees; widen_factor=2.)
    end
end


function generate_mnist_large_scale(;degree=119)
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "mnist", "mnist_256x4_1e4.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "mnist", "mnist_gelu_256x4_1e4.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "mnist", "mnist_256x6_1e4.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "mnist", "mnist_gelu_256x6_1e4.onnx")
    ]
    for onnx_path in onnx_paths
        generate_mnist(onnx_path, [degree], max_polys_per_layer=Inf)
    end 
end
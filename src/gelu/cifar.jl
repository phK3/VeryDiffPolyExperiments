

function load_cifar_data()
    dataset = CIFAR10(Tx=Float64, split=:test)

    X_test = [dataset.features[:,:,:,i:i] for i in 1:size(dataset.features, 4)]
    y_test = dataset.targets
    return X_test, y_test
end


function get_input_bounds_cifar(X)
    lb = fill(0., 32, 32, 3, 1)
    ub = fill(1., 32, 32, 3, 1)
    input_bounds = Dict("input" => (lb, ub))
    return input_bounds
end


cifar_acc = (ŷ, y) -> sum(getindex.(argmax(ŷ, dims=2), 2) .== y .+ 1) / size(y, 1)


function generate_cifar_single(onnx_path, degree)
    if VeryDiff.APPROX_POLY_THREADS[] > 1
        BLAS.set_num_threads(1)
        println("Using ", VeryDiff.APPROX_POLY_THREADS[], " threads for approximation, set BLAS to 1 thread to avoid oversubscription")
    end
    
    X_test, y_test = load_cifar_data()
    input_bounds = get_input_bounds_cifar(X_test)

    model = load_onnx_model(onnx_path);
    ŷ, mse, t_eval = evaluate_network(model, X_test, y_test, cifar_acc)

    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, degree, input_bounds=input_bounds, verbosity=1)
    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))
    model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes));

    ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network(model_poly_dense, X_test, y_test, cifar_acc; y_pred=ŷ)
    
    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))

    output_file = string(basename(onnx_path)[1:end-5], "_" , degree, ".json")
    VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "..", "results", "gelu", "cifar", output_file))

    println("Summary:")
    println("\tModel: ", basename(onnx_path))
    println("\tDegree: ", degree)
    println("\tBounds: ", input_bounds)
    println("\tACC: ", mse)
    println("\tACC (poly): ", mse_poly)
    println("\tSampled Error: ", ϵ_sample)
    println("\tVerified Error Bound: ", ϵ)
    println("\tApproximation Time: ", t_approx)
    println("\tEvaluation Time: ", t_eval)
    println("\tVerification Time: ", t_verify)
    println("\tExported JSON: ", output_file)
end


function generate_cifar(onnx_path, degrees; max_polys_per_layer=Inf)
    generate_networks(onnx_path, degrees, load_cifar_data, get_input_bounds_cifar, cifar_acc, max_polys_per_layer=max_polys_per_layer)
end


function run_cifar_experiment(;degrees=20:20:160, n_threads=Threads.nthreads())
    VeryDiff.APPROX_POLY_THREADS[] = n_threads
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "cifar", "best_model_bn_4_0.0001l1.onnx"),
        # joinpath(@__DIR__, "..", "..", "networks", "cifar", "best_model_bn_8_0.0005l1.onnx")
    ]

    for onnx_path in onnx_paths
        generate_cifar(onnx_path, degrees, max_polys_per_layer=Inf)
    end
end


function generate_cifar_large_scale(;degrees=[119, 247])
    onnx_paths = [
        joinpath(@__DIR__, "..", "..", "networks", "cifar", "best_model_bn_4_0.0001l1.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "cifar", "best_model_bn_8_0.0001l1_no_pad.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "cifar", "models_relu", "best_model_relu_bn_4_0.0001l1.onnx"),
        joinpath(@__DIR__, "..", "..", "networks", "cifar", "models_relu", "best_model_relu_bn_8_0.0001l1.onnx")
    ]

    for onnx_path in onnx_paths
        generate_cifar(onnx_path, degrees, max_polys_per_layer=Inf)
    end
end

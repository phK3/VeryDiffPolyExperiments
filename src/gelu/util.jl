


function evaluate_network(model, X, y, metric; y_pred=nothing)
    input_name = first(keys(model.input_shapes))
    input_shape = model.input_shapes[input_name]

    if length(input_shape) == 1
        # flattened model
        t_eval = @elapsed  ŷ = [VNNLib.compute_output(model, vec(X[i])) for i in 1:size(X, 1)];
    else
        t_eval = @elapsed  ŷ = [VNNLib.compute_output(model, X[i]) for i in 1:size(X, 1)];
    end

    ŷ = hcat(ŷ...)'
    mse = metric(ŷ, y)
    println("acc/mse: ", mse)
    println("Evaluation Time: ", t_eval)

    if !isnothing(y_pred)
        ϵ_sample = maximum(abs.(y_pred - ŷ))
        println("Sampled Error: ", ϵ_sample)
        return ŷ, mse, t_eval, ϵ_sample
    end

    return ŷ, mse, t_eval
end

"""
    `generate_networks(onnx_path, degrees, load_data)`

Generates polynomial approximations of the network specified by `onnx_path` for each degree in `degrees`. 
    
Evaluates the original and polynomial models on the test set loaded by `load_data`, computes verified error bounds, and logs results to a JLD2 file.

args:
- `onnx_path`: path to the ONNX model file
- `degrees`: vector of degrees for polynomial approximation
- `load_data`: function that loads the test data, should return `(X_test, y_test)`
- `get_input_bounds`: function that takes `X_test` and returns a dictionary of input bounds for verification
- `metric`: function that computes the performance metric (e.g., accuracy or MSE)

kwargs:
- `max_polys_per_layer`: maximum number of polynomials per layer (default: `Inf`)

"""
function generate_networks(onnx_path, degrees, load_data, get_input_bounds, metric; max_polys_per_layer=Inf)
    println("Generating polynomial networks for ", onnx_path)
    println("Using ", min(Threads.nthreads(), VeryDiff.APPROX_POLY_THREADS[]), " threads for approximation")
    
    logfile = string(basename(onnx_path)[1:end-5], "_results_", now(), ".jld2")
    println("Logging results to ", logfile)

    X_test, y_test = load_data()
    input_bounds = get_input_bounds(X_test)

    model = load_onnx_model(onnx_path)

    ŷ, mse, t_eval = evaluate_network(model, X_test, y_test, metric)
    println("Base model acc/MSE: ", mse)

    model_dense = VNNLib.net2dense(model, Dict(k => rand(v...) for (k, v) in model.input_shapes))

    ϵs = []
    ϵ_samples = []
    mse_polys = []
    t_approxs = []
    t_verifies = []
    t_evals = []
    networks = []
    # stores by reference, so if ϵs is modified, result_dict is also updated when saved
    result_dict = Dict(
            "degrees" => degrees,
            "ϵs" => ϵs,
            "ϵ_samples" => ϵ_samples,
            "mse_polys" => mse_polys,
            "t_approxs" => t_approxs,
            "t_verifies" => t_verifies,
            "t_evals" => t_evals,
            "networks" => networks,
            "threads" => Threads.nthreads(),
            "approx_poly_threads" => VeryDiff.APPROX_POLY_THREADS[]
        )
    for d in degrees
        println("Approximating with degree ", d, "...")
        t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, d, input_bounds=input_bounds, 
                                                                                 max_polys_per_layer=max_polys_per_layer, verbosity=1)
        println("Approximation Time: ", t_approx)

        model_poly_dense = VNNLib.net2dense(model_poly, Dict(k => rand(v...) for (k, v) in model_poly.input_shapes))

        ŷ_poly, mse_poly, t_eval, ϵ_sample = evaluate_network(model_poly_dense, X_test, y_test, metric; y_pred=ŷ)
        println("Polynomial acc/MSE: ", mse_poly)

        t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(input_bounds["input"][1]), vec(input_bounds["input"][2]))
        println("Verified Error Bound: ", ϵ)

        push!(ϵs, ϵ)
        push!(ϵ_samples, ϵ_sample)
        push!(mse_polys, mse_poly)
        push!(t_approxs, t_approx)
        push!(t_verifies, t_verify)
        push!(t_evals, t_eval)
        push!(networks, model_poly)

        # save result dict in every iteration to avoid losing results in case of crashes
        jldsave(logfile; result_dict)

        println("Degree: ", d, " ϵ: ", ϵ, ", ϵ_sample: ", ϵ_sample, ", acc/mse (poly): ", mse_poly, ", t_approx: ", t_approx, " t_verify: ", t_verify, " t_eval: ", t_eval)
    end
end


function warmup(;n_threads=Threads.nthreads())
    VeryDiff.APPROX_POLY_THREADS[] = n_threads
    @info "Running low degree for warm up (precompilation)..."
    onnx_path = joinpath(@__DIR__, "..", "..", "networks", "mnist", "mnist_gelu_256x4_1e4.onnx")
    degrees = [5]
    generate_mnist(onnx_path, degrees)
end
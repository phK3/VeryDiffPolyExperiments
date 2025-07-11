



MNIST_DATA_PATH = string(@__DIR__, "/../../datasets/mnist_train.csv")
MNIST_TEST_DATA_PATH = string(@__DIR__, "/../../datasets/mnist_test.csv")
MNIST_MODEL_PATHS = readdir(string(@__DIR__, "/../../networks/mnist"), join=true)
MNIST_DEGREES = (1:9) ∪ (10:10:200)
MNIST_LOG_FILE_PREFIX = string(@__DIR__, "/../../results/mnist")




max_fun(y, ŷ) = maximum(abs.(y .- ŷ))
mae_fun(y, ŷ) = sum(maximum(abs.(y .- ŷ), dims=2)) / size(y, 1)
mse_fun(y, ŷ) = sum(sum((y .- ŷ).^2, dims=2)) / size(y, 1)

"""
Accuracy function for classification tasks.

args:
    y - vector of true labels (make sure they are 1 indexed)
    ŷ - matrix (n_inputs × outputs) of predicted logits
"""
function acc_fun(y::AbstractArray{<:Number}, ŷ::AbstractArray)
    # argmax(ŷ, dims=2) returns a CartesianIndex object, so we need to extract the index of the row
    sum(getindex.(argmax(ŷ, dims=2), 2) .== y) / size(y, 1)
end


function load_mnist_data(;test_data=false)
    path = test_data ? MNIST_TEST_DATA_PATH : MNIST_DATA_PATH
    f_mnist = CSV.File(path, header=false)
    X_test = [Float64.([x for x in f_mnist[i]][2:end]) ./ 255 for i in 1:size(f_mnist, 1)]
    y_test = [[x for x in f_mnist[i]][1] for i in 1:size(f_mnist, 1)]
    return X_test, y_test
end


function generate_mnist_nets_verified_bounds(;max_iter=20, test_run=false)
    degrees = test_run ? [5, 20] : MNIST_DEGREES
    test_run && println("[INFO] Generating MNIST networks TESTRUN ...")

    if !isdir(MNIST_LOG_FILE_PREFIX)
        println("[INFO] Creating directory for MNIST logs at $(MNIST_LOG_FILE_PREFIX)...")
        mkpath(MNIST_LOG_FILE_PREFIX)
    end
    log_file_prefix = string(MNIST_LOG_FILE_PREFIX, "/mnist_verified_bounds")

    X_test, y_test = load_mnist_data()
    
    z = Zonotope(zeros(784), ones(784))
    # need to add one as labels are they are 0 indexed
    generate_poly_networks(MNIST_MODEL_PATHS, z, degrees, log_file_prefix, 
                                max_fun, mae_fun, mse_fun, acc_fun; X_test=X_test, y_labels=y_test .+ 1, empirical=false, cheby=true, verbosity=0, 
                                max_iter=max_iter, widen_factor=1.0)
end


function generate_mnist_nets_empirical_bounds(widen_factor; test_run=false)
    degrees = test_run ? [5, 20] : MNIST_DEGREES
    test_run && println("[INFO] Generating empirical MNIST networks TESTRUN ...")

    if !isdir(MNIST_LOG_FILE_PREFIX)
        println("[INFO] Creating directory for MNIST logs at $(MNIST_LOG_FILE_PREFIX)...")
        mkpath(MNIST_LOG_FILE_PREFIX)
    end
    widen_string = replace(string(widen_factor), "." => "_")
    log_file_prefix = string(MNIST_LOG_FILE_PREFIX, "/mnist_empirical_bounds_widen_", widen_string, "_")

    X_test, y_test = load_mnist_data()
    
    # method wants a zonotope, so create it even though we don't need it
    z = Zonotope(zeros(784), ones(784))
    # need to add one as labels are they are 0 indexed
    generate_poly_networks(MNIST_MODEL_PATHS, z, degrees, log_file_prefix, 
                                max_fun, mae_fun, mse_fun, acc_fun; X_test=X_test, y_labels=y_test .+ 1, empirical=true, cheby=true, verbosity=0, 
                                widen_factor=widen_factor)
end


function append_mnist_test_evaluation_to_logs(log_filter; verbosity=0)
    logfiles_generated = readdir(string(@__DIR__, "/../../results/mnist"), join=true)
    logfiles_generated = filter(x -> contains(x, log_filter), logfiles_generated)

    X, y = load_mnist_data(test_data=true)

    for logfile in logfiles_generated
        println("[INFO] Appending test data for $logfile")
        # need to add 1 to labels since they are zero indexed
        append_dataset_results_to_log(logfile, X, y .+ 1, "test", max_fun, mae_fun, mse_fun, acc_fun, verbosity=verbosity)
    end
end



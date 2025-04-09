
HELOC_DATA_PATH = string(@__DIR__, "/../../datasets/heloc_dataset.csv")
HELOC_MODEL_PATHS = readdir(string(@__DIR__, "/../../networks/heloc"), join=true)
HELOC_DEGREES = (1:9) ∪ (10:10:200)
HELOC_LOG_FILE_PREFIX = string(@__DIR__, "/../../results/heloc")

σ(x) = 1 / (1 + exp(-x))

"""
Accuracy function for binary classification tasks.

args:
    y - vector of true labels (make sure they are 1 indexed)
    ŷ - matrix (n_inputs × outputs) of predicted logits
"""
function acc_fun_binary(y::AbstractArray{<:Number}, ŷ::AbstractArray)
    count(round.(σ.(ŷ)) .== y) / length(y)
end


function load_heloc_data()
    data_min = -9 .* ones(23)
    data_max = [93, 789, 383, 383, 74, 17, 16, 100, 83, 9, 8, 87, 17, 100, 24, 66, 66, 232, 471, 32, 23, 18, 100];

    f_heloc = CSV.File(HELOC_DATA_PATH)
    X_test = [Float64.([x for x in f_heloc[i]][2:end]) for i in 1:size(f_heloc, 1)]
    X_test = [(x .- data_min) ./ (data_max .- data_min) for x in X_test]
    y_test = [[x for x in f_heloc[i]][1] for i in 1:size(f_heloc, 1)]
    y_test = [ifelse(y == "Good", 1., 0.) for y in y_test]
    return X_test, y_test
end


function generate_heloc_nets_verified_bounds(;test_run=false)
    degrees = test_run ? [5, 20] : HELOC_DEGREES
    test_run && println("[INFO] Generating HELOC networks TESTRUN ...")

    if !isdir(HELOC_LOG_FILE_PREFIX)
        println("[INFO] Creating directory for HELOC logs at $(HELOC_LOG_FILE_PREFIX)...")
        mkpath(HELOC_LOG_FILE_PREFIX)
    end
    log_file_prefix = string(HELOC_LOG_FILE_PREFIX, "/heloc_verified_bounds")

    X_test, y_test = load_heloc_data()
    
    z = Zonotope(zeros(23), ones(23))
    generate_poly_networks(HELOC_MODEL_PATHS, z, degrees, log_file_prefix, 
                                max_fun, mae_fun, mse_fun, acc_fun_binary; X_test=X_test, y_labels=y_test, empirical=false, cheby=true, verbosity=0, 
                                max_iter=20, widen_factor=1.0)
end

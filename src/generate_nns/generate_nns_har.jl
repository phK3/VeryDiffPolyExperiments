
HAR_DATA_PATH = string(@__DIR__, "/../../datasets")
HAR_MODEL_PATHS = readdir(string(@__DIR__, "/../../networks/har"), join=true)
HAR_DEGREES = (1:9) ∪ (10:10:200)
HAR_LOG_FILE_PREFIX = string(@__DIR__, "/../../results/har")


function load_har_data(;test_data=false)
    x_path = test_data ? string(HAR_DATA_PATH, "/HAR_X_test.txt") : string(HAR_DATA_PATH, "/HAR_X_train.txt")
    y_path = test_data ? string(HAR_DATA_PATH, "/HAR_y_test.txt") : string(HAR_DATA_PATH, "/HAR_y_train.txt")

    f_x_har = CSV.File(x_path, header=false, delim=" ", ignorerepeated=true)
    f_y_har = CSV.File(y_path, header=false)

    # data is already normalized to [-1, 1]
    X = [Float64.([x for x in row]) for row in f_x_har]
    y = [x[1] for x in f_y_har]

    return X, y
end


function generate_har_nets_verified_bounds(;test_run=false)
    degrees = test_run ? [5, 20] : HAR_DEGREES
    test_run && println("[INFO] Generating HAR networks TESTRUN ...")

    if !isdir(HAR_LOG_FILE_PREFIX)
        println("[INFO] Creating directory for HAR logs at $(HAR_LOG_FILE_PREFIX)...")
        mkpath(HAR_LOG_FILE_PREFIX)
    end
    log_file_prefix = string(HAR_LOG_FILE_PREFIX, "/har_verified_bounds")

    X_test, y_test = load_har_data()
    
    z = Zonotope(.-ones(561), ones(561))
    # don't need to add one as labels are 1 indexed for HAR
    generate_poly_networks(HAR_MODEL_PATHS, z, degrees, log_file_prefix, 
                                max_fun, mae_fun, mse_fun, acc_fun; X_test=X_test, y_labels=y_test, empirical=false, cheby=true, verbosity=0, 
                                max_iter=20, widen_factor=1.0)
end
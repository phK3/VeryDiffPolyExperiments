
using VeryDiff, VNNLib, JLD2, Plots, CSV

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false 

function load_cifar_data(;test_data=false)
    path = joinpath(@__DIR__, "..", "datasets", "cifar_test.csv")
    f_cifar = CSV.File(path, header=false)
    X_test = [Float64.([x for x in f_cifar[i]][2:end]) ./ 255 for i in 1:size(f_cifar, 1)]
    y_test = [[x for x in f_cifar[i]][1] for i in 1:size(f_cifar, 1)]
    return X_test, y_test
end


model_path = joinpath(@__DIR__, "..", "networks", "cifar", "cifar_conv2_ultra_tiny.onnx")
model = load_onnx_model(model_path)

function run_experiment(model_path, degree)
    model = load_onnx_model(model_path)
    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, degree, verbosity=1)

    input_data = Dict(k => rand(v...) for (k, v) in model_poly.input_shapes)
    model_dense = VNNLib.net2dense(model, input_data)
    model_poly_dense = VNNLib.net2dense(model_poly, input_data)

    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, zeros(3*32*32), ones(3*32*32))
    return ϵ, t_approx, t_verify
end

ϵs = []
t_approxs = []
t_verifies = []
for d in 20:20:100
    ϵ, t_approx, t_verify = run_experiment(model_path, d);
    push!(ϵs, ϵ)
    push!(t_approxs, t_approx)
    push!(t_verifies, t_verify)
    println("Degree: ", d, " ϵ: ", ϵ, " t_approx: ", t_approx, " t_verify: ", t_verify)
end

plot(20:20:100, ϵs, xlabel="Degree", ylabel="ϵ", title="Verified Difference for CIFAR Conv2 Ultra Tiny", mark=:o)
plot(20:20:100, t_approxs, xlabel="Degree", ylabel="Time (s)", title="Approximation Time for CIFAR Conv2 Ultra Tiny", mark=:o)
plot(20:20:100, t_verifies, xlabel="Degree", ylabel="Time (s)", title="Verification Time for CIFAR Conv2 Ultra Tiny", mark=:o)

using VeryDiff, VNNLib, JLD2, Plots, CSV, MLDatasets

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false 

dataset = CIFAR10(Tx=Float64, split=:test)

logfile_name = joinpath(@__DIR__, "..", "results", "gelu", "cifar_conv.jld2")

# model_path = joinpath(@__DIR__, "..", "networks", "cifar", "cifar_conv2_ultra_tiny.onnx")
model_path = joinpath(@__DIR__, "..", "networks", "cifar", "conv_bn_fused.onnx")
model = load_onnx_model(model_path);

# if normalization was *not* included in the model export, then we need to do it here.
# TODO: include normalization in file export. 
#       Can we merge it with the first layer? 
#       ==> cannot merge it due to padding in conv layers!!!
#μ_cifar = reshape([0.4914, 0.4822, 0.4465], 1, 1, 3)
#σ_cifar = reshape([0.2470, 0.2435, 0.2616], 1, 1, 3)
#t_eval = @elapsed ŷ = [VNNLib.compute_output(model, (dataset.features[:,:,:,i:i] .- μ_cifar) ./ σ_cifar) for i in 1:10000]
#ŷ = hcat(ŷ...)'

#acc = sum(getindex.(argmax(ŷ, dims=2), 2) .== dataset.targets .+ 1) / size(dataset.targets, 1)

# if normalization was already included in the model export, then we can just do:
t_eval = @elapsed ŷ = [VNNLib.compute_output(model, dataset.features[:,:,:,i:i]) for i in 1:10000]
ŷ = hcat(ŷ...)'
acc = sum(getindex.(argmax(ŷ, dims=2), 2) .== dataset.targets .+ 1) / size(dataset.targets, 1)

function run_experiment(model_path, degree)
    model = load_onnx_model(model_path)
    t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, degree, verbosity=1)

    input_data = Dict(k => rand(v...) for (k, v) in model_poly.input_shapes)
    model_dense = VNNLib.net2dense(model, input_data)
    model_poly_dense = VNNLib.net2dense(model_poly, input_data)

    # uncomment top line if normalization is not included in model export, otherwise use bottom line
    # t_eval = @elapsed ŷ_poly = [VNNLib.compute_output(model_poly_dense, vec((dataset.features[:,:,:,i:i] .- μ_cifar) ./ σ_cifar)) for i in 1:10000]
    t_eval = @elapsed ŷ_poly = [VNNLib.compute_output(model_poly_dense, vec(dataset.features[:,:,:,i:i])) for i in 1:10000]
    ŷ_poly = hcat(ŷ_poly...)'
    acc_poly = sum(getindex.(argmax(ŷ_poly, dims=2), 2) .== dataset.targets .+ 1) / size(dataset.targets, 1)

    ϵ_sample = maximum(abs.(ŷ_poly - ŷ))

    t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, zeros(3*32*32), ones(3*32*32))
    return ϵ, ϵ_sample, acc_poly, t_approx, t_verify, t_eval
end

ϵs = []
ϵ_samples = []
acc_polys = []
t_approxs = []
t_verifies = []
t_evals = []

for d in 20:20:100
    ϵ, ϵ_sample, acc_poly, t_approx, t_verify, t_eval = run_experiment(model_path, d);
    push!(ϵs, ϵ)
    push!(ϵ_samples, ϵ_sample)
    push!(acc_polys, acc_poly)
    push!(t_approxs, t_approx)
    push!(t_verifies, t_verify)
    push!(t_evals, t_eval)
    println("Degree: ", d, " ϵ: ", ϵ, ", ϵ_sample: ", ϵ_sample, ", acc_poly: ", acc_poly, ", t_approx: ", t_approx, " t_verify: ", t_verify, " t_eval: ", t_eval)
end

result_dict = Dict(
    "degrees" => 20:20:100,
    "ϵs" => ϵs,
    "ϵ_samples" => ϵ_samples,
    "acc_polys" => acc_polys,
    "t_approxs" => t_approxs,
    "t_verifies" => t_verifies,
    "t_evals" => t_evals
)
jldsave(logfile_name; result_dict)

result_dict = load(logfile_name)["result_dict"]

plot(result_dict["degrees"], result_dict["ϵs"], label="Verified Difference",xlabel="Degree", ylabel="ϵ", title="Difference for CIFAR Conv2 Ultra Tiny", markershape=:auto)
plot!(result_dict["degrees"], result_dict["ϵ_samples"], label="Sampled Difference", markershape=:auto)

plot(result_dict["degrees"], result_dict["t_approxs"], xlabel="Degree", ylabel="Time (s)", label="approximation time", title="Approximation Time for CIFAR Conv2 Ultra Tiny", markershape=:auto)
plot(result_dict["degrees"], result_dict["t_verifies"], xlabel="Degree", ylabel="Time (s)", label="verification time", title="Verification Time for CIFAR Conv2 Ultra Tiny", markershape=:auto)
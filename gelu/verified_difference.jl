
using VeryDiff, VNNLib, JLD2, Plots

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false 


model_path = joinpath(@__DIR__, "..", "networks", "mnist", "mnist_gelu_256x4_1e4.onnx")
model = load_onnx_model(model_path)

# we need a concrete input to infer shapes
input_name = first(keys(model.input_shapes))
input_shape = model.input_shapes[input_name]
input_data = Dict(input_name => randn(input_shape...))

model_dense = net2dense(model, input_data, verbosity=2);

model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, 20, tight_gelu=true)
model_poly_dense = net2dense(model_poly, input_data, verbosity=2);

verification_pass(model_poly_dense, model_dense, zeros(784), ones(784))


onnx_names = [
    "mnist_gelu_256x4_1e4.onnx",
    "mnist_gelu_256x6_1e4.onnx",
    "mnist_gelu_256x4_2e5.onnx",
    "mnist_gelu_256x6_2e5.onnx",
    "mnist_gelu_256x4.onnx",
    "mnist_gelu_256x6.onnx"
]

degrees = 5:5:100
# degrees = (1:19) ∪ (20:5:100)
# degrees = [20]

logfile_name = joinpath(@__DIR__, "..", "results", "gelu", "verified_difference.jld2")



result_dict = Dict()
for model_name in onnx_names
    println("")
    println("### Processing model: ", model_name)
    model_path = joinpath(@__DIR__, "..", "networks", "mnist", model_name)

    model = load_onnx_model(model_path)

    # we need a concrete input to infer shapes
    input_name = first(keys(model.input_shapes))
    input_shape = model.input_shapes[input_name]
    input_data = Dict(input_name => randn(input_shape...))

    model_dense = net2dense(model, input_data, verbosity=0)

    ϵs = []
    ts_approx = []
    ts_verify = []
    for d in degrees
        t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, d, tight_gelu=true)
        model_poly_dense = net2dense(model_poly, input_data, verbosity=0);

        t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, zeros(784), ones(784))

        push!(ϵs, ϵ)
        push!(ts_approx, t_approx)
        push!(ts_verify, t_verify)

        println("\tdegree: ", d, ", verified error: ", ϵ, ", t_approx: ", t_approx, ", t_verify: ", t_verify)
    end

    result_dict[model_name] = (degrees=degrees, ts_approx=ts_approx, ts_verify=ts_verify, errors=ϵs)
    jldsave(logfile_name; result_dict)
end


result_dict = load(logfile_name)["result_dict"]

gelu_4layer_1e4 = result_dict["mnist_gelu_256x4_1e4.onnx"]
gelu_6layer_1e4 = result_dict["mnist_gelu_256x6_1e4.onnx"]
gelu_4layer_2e5 = result_dict["mnist_gelu_256x4_2e5.onnx"]
gelu_6layer_2e5 = result_dict["mnist_gelu_256x6_2e5.onnx"]
gelu_4layer = result_dict["mnist_gelu_256x4.onnx"]
plot(gelu_4layer_1e4.degrees, gelu_4layer_1e4.errors, label="4layer 1e-4", yscale=:log10, yticks=[0.01, 0.1, 1., 10, 100, 1000])
plot!(gelu_6layer_1e4.degrees, gelu_6layer_1e4.errors, label="6layer 1e-4")
plot!(gelu_4layer_2e5.degrees, gelu_4layer_2e5.errors, label="4layer 2e-5")
plot!(gelu_6layer_2e5.degrees, gelu_6layer_2e5.errors, label="6layer 2e-5")
plot!(gelu_4layer.degrees, gelu_4layer.errors, label="4layer standard")
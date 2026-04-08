
using VeryDiff, VeryDiffPolyExperiments, VNNLib, JLD2, Plots, CSV, MLDatasets

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

dataset = CIFAR10(Tx=Float64, split=:test)

degree = 50
# model_path = joinpath(@__DIR__, "..", "networks", "cifar", "conv_bn_fused.onnx")  # small model 54% accuracy
# model_path = joinpath(@__DIR__, "..", "networks", "cifar", "best_model_bn_8_0.0001l1.onnx")  # large model 78% accuracy
model_path = joinpath(@__DIR__, "..", "networks", "cifar", "best_model_bn_8_0.0001l1_no_pad.onnx")  # large model 77% accuracy
model = load_onnx_model(model_path);

## Check accuracy of the base model on the test set

t_eval = @elapsed ŷ = [VNNLib.compute_output(model, dataset.features[:,:,:,i:i]) for i in 1:10000]
ŷ = hcat(ŷ...)'
acc = sum(getindex.(argmax(ŷ, dims=2), 2) .== dataset.targets .+ 1) / size(dataset.targets, 1)
println("Accuracy: ", acc)
println("Evaluation Time: ", t_eval)

## Approximate model with polynomial network
t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, degree, verbosity=1)
println("Approximation Time: ", t_approx)

##  convert to dense format for verification and evaluation
input_data = Dict(k => rand(v...) for (k, v) in model_poly.input_shapes)
model_dense = VNNLib.net2dense(model, input_data)
model_poly_dense = VNNLib.net2dense(model_poly, input_data);

## Evaluate polynomial model on test set
t_eval = @elapsed ŷ_poly = [VNNLib.compute_output(model_poly_dense, vec(dataset.features[:,:,:,i:i])) for i in 1:10000]
ŷ_poly = hcat(ŷ_poly...)'

acc_poly = sum(getindex.(argmax(ŷ_poly, dims=2), 2) .== dataset.targets .+ 1) / size(dataset.targets, 1)
ϵ_sample = maximum(abs.(ŷ_poly - ŷ))

println("Polynomial Accuracy: ", acc_poly)
println("Sampled Error: ", ϵ_sample)
println("Evaluation Time: ", t_eval)

## Get verified error bound

t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, zeros(3*32*32), ones(3*32*32))
println("Verified Error Bound: ", ϵ)
println("Verification Time: ", t_verify)

## Export network

output_file = string(basename(model_path)[1:end-5], "_" , degree, ".json")
VeryDiffPolyExperiments.export2json(model_poly, joinpath(@__DIR__, "..", "results", "gelu", output_file))

# just to check that the merged model computes the same output
#layers, _ = VeryDiff.Definitions.sort_network(model_poly)
#layers_merged = VeryDiffPolyExperiments.merge_normalization_into_dense(layers);
#
#model_merged = VNNLib.OnnxNet(layers_merged, model_poly.start_nodes, model_poly.final_nodes, model_poly.input_shapes, model_poly.output_shapes)
#VNNLib.compute_output(model_poly, zeros(32, 32, 3, 1))
#VNNLib.compute_output(model_merged, zeros(32, 32, 3, 1))

## Print summary 

println("Summary:")
println("\tModel: ", basename(model_path))
println("\tDegree: ", degree)
println("\tAccuracy: ", acc)
println("\tPolynomial Accuracy: ", acc_poly)
println("\tSampled Error: ", ϵ_sample)
println("\tVerified Error Bound: ", ϵ)
println("\tApproximation Time: ", t_approx)
println("\tEvaluation Time: ", t_eval)
println("\tVerification Time: ", t_verify)
println("\tExported JSON: ", output_file)

using VeryDiff, VeryDiffPolyExperiments, VNNLib, JLD2, Plots, CSV
VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

HELOC_DATA_PATH = joinpath(@__DIR__, "..", "datasets", "heloc_dataset.csv")
model_path = joinpath(@__DIR__, "..", "networks", "heloc", "heloc_2e5_gelu.onnx")
degree = 50

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


X_test, y_test = load_heloc_data()
model = load_onnx_model(model_path);

## Check accuracy of the base model on the test set

t_eval = @elapsed  ŷ = [VNNLib.compute_output(model, X_test[i]) for i in 1:length(X_test)];
ŷ = hcat(ŷ...)'
acc = acc_fun_binary(y_test, ŷ)
println("Accuracy: ", acc)
println("Evaluation Time: ", t_eval)

## Define input bounds for verification
lb = -9 .* ones(23)
ub = [93, 789, 383, 383, 74, 17, 16, 100, 83, 9, 8, 87, 17, 100, 24, 66, 66, 232, 471, 32, 23, 18, 100];
input_bounds = Dict("input" => (reshape(lb, 23, 1), reshape(ub, 23, 1)));

println("Input Bounds:")
println("Lower bound: ", vec(lb))
println("Upper bound: ", vec(ub))


## Approximate model with polynomial network
t_approx = @elapsed model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, degree, input_bounds=input_bounds, verbosity=1)
println("Approximation Time: ", t_approx)

##  convert to dense format for verification and evaluation
input_data = Dict(k => rand(v...) for (k, v) in model_poly.input_shapes)
model_dense = VNNLib.net2dense(model, input_data)
model_poly_dense = VNNLib.net2dense(model_poly, input_data);

## Evaluate polynomial model on test set
t_eval = @elapsed  ŷ_poly = [VNNLib.compute_output(model_poly_dense, vec(X_test[i])) for i in 1:length(X_test)];
ŷ_poly = hcat(ŷ_poly...)'
acc_poly = acc_fun_binary(y_test, ŷ_poly)
ϵ_sample = maximum(abs.(ŷ_poly - ŷ))

println("Polynomial Accuracy: ", acc_poly)
println("Sampled Error: ", ϵ_sample)
println("Evaluation Time: ", t_eval)

## Get verified error bound

t_verify = @elapsed ϵ = verification_pass(model_poly_dense, model_dense, vec(lb), vec(ub))
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
println("\tBounds: ", input_bounds)
println("\tAccuracy: ", acc)
println("\tPolynomial Accuracy: ", acc_poly)
println("\tSampled Error: ", ϵ_sample)
println("\tVerified Error Bound: ", ϵ)
println("\tApproximation Time: ", t_approx)
println("\tEvaluation Time: ", t_eval)
println("\tVerification Time: ", t_verify)
println("\tExported JSON: ", output_file)
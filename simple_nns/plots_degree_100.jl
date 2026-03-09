
using Plots, JLD2

results_gelu = load(joinpath(@__DIR__, "..", "results/gelu/generate_mnist_gelu_256x4_1e4_2026-02-09T12:30:59.813.jld2"))
results_relu = load(joinpath(@__DIR__, "..", "results/gelu/generate_mnist_256x4_1e4_2026-02-09T13:39:10.112.jld2"))
results_scaled_gelu = load(joinpath(@__DIR__, "..", "results/gelu/generate_mnist_scaled_gelu_256x4_1e4_2026-02-09T13:39:10.112.jld2"))
results_gelu_tight = load(joinpath(@__DIR__, "..", "results/gelu/generate_mnist_gelu_256x4_1e4_2026-02-18T18:15:08.323.jld2"))
results_scaled_gelu_tight = load(joinpath(@__DIR__, "..", "results/gelu/generate_mnist_scaled_gelu_256x4_1e4_2026-02-18T22:00:17.469.jld2"))

results_scaled_gelu2 = load(joinpath(@__DIR__, "..", "results/gelu/generate_mnist_scaled_gelu_256x4_1e4_2026-03-02T13:55:29.091.jld2"))

results_dict = Dict()
results_dict["gelu_256x1_1e4_abcrown"] = results_gelu["result_dict"]["mnist_gelu_256x4_1e4_abcrown"]
results_dict["relu_256x1_1e4_abcrown"] = results_relu["result_dict"]["mnist_256x4_1e4_abcrown"]
results_dict["scaled_gelu_256x1_1e4_abcrown"] = results_scaled_gelu["result_dict"]["mnist_scaled_gelu_256x4_1e4_abcrown"]
results_dict["gelu_256x1_1e4_sampling"] = results_gelu["result_dict"]["mnist_gelu_256x4_1e4_sampling"]
results_dict["relu_256x1_1e4_sampling"] = results_relu["result_dict"]["mnist_256x4_1e4_sampling"]
results_dict["scaled_gelu_256x1_1e4_sampling"] = results_scaled_gelu["result_dict"]["mnist_scaled_gelu_256x4_1e4_sampling"]
results_dict["gelu_256x1_1e4_abcrowntight"] = results_gelu_tight["result_dict"]["mnist_gelu_256x4_1e4_abcrowntight"]
results_dict["scaled_gelu_256x1_1e4_abcrowntight"] = results_scaled_gelu_tight["result_dict"]["mnist_scaled_gelu_256x4_1e4_abcrowntight"]

# after fixing numerical error
results_dict["scaled_gelu_256x1_1e4_abcrown"].max_errs[end-4:end]      .= results_scaled_gelu2["result_dict"]["mnist_scaled_gelu_256x4_1e4_abcrown"].max_errs[end-4:end]
results_dict["scaled_gelu_256x1_1e4_abcrowntight"].max_errs[end-4:end] .= results_scaled_gelu2["result_dict"]["mnist_scaled_gelu_256x4_1e4_abcrowntight"].max_errs[end-4:end]


degrees = (1:19) ∪ (20:5:100)

plot(degrees, results_dict["gelu_256x1_1e4_abcrown"].max_errs, label="gelu abcrown", yscale=:log10, legend=:topright)
plot!(degrees, results_dict["relu_256x1_1e4_abcrown"].max_errs, label="relu abcrown")
plot!(degrees, results_dict["scaled_gelu_256x1_1e4_abcrown"].max_errs, label="scaled gelu abcrown")
plot!(degrees, results_dict["gelu_256x1_1e4_abcrowntight"].max_errs, label="gelu tight")
plot!(degrees, results_dict["scaled_gelu_256x1_1e4_abcrowntight"].max_errs, label="scaled gelu tight")


plot(degrees, results_dict["gelu_256x1_1e4_abcrown"].max_errs, label="gelu abcrown", yscale=:log10, color=1, legend=:inside)
plot!(degrees, results_dict["relu_256x1_1e4_abcrown"].max_errs, label="relu abcrown", color=2)
plot!(degrees, results_dict["scaled_gelu_256x1_1e4_abcrown"].max_errs, label="scaled gelu abcrown", color=3)
plot!(degrees, results_dict["gelu_256x1_1e4_abcrowntight"].max_errs, label="gelu tight", color=4)
plot!(degrees, results_dict["scaled_gelu_256x1_1e4_abcrowntight"].max_errs, label="scaled gelu tight", color=5)

plot!(degrees, results_dict["gelu_256x1_1e4_sampling"].max_errs, label="gelu sampling", color=1, linestyle=:dash)
plot!(degrees, results_dict["relu_256x1_1e4_sampling"].max_errs, label="relu sampling", color=2, linestyle=:dash)
plot!(degrees, results_dict["scaled_gelu_256x1_1e4_sampling"].max_errs, label="scaled gelu sampling", color=3, linestyle=:dash)
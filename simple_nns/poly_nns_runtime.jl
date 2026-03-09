
using VeryDiff, VNNLib, Plots, VeryDiffPolyExperiments, JLD2 
VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false


function generate_json_networks(onnx_path, degrees; max_polys_per_layer=Inf, max_iter=20, verbosity=1, outdir=".")
    onnx_model = load_onnx_model(onnx_path)
    nn = VeryDiff.to_layered_model(onnx_model)

    model_name = split(basename(onnx_path), ".")[1]
    println("Generating networks for model: ", model_name)

    for d in degrees 
        outfile = joinpath(outdir, "mnist_$(model_name)_d_$(d).json")
        println("Generating network with degree $d and saving to $outfile")

        try 
            nn_poly = VeryDiff.approximate_polynomial_abcrown(onnx_path, d, max_polys_per_layer=max_polys_per_layer, max_iter=max_iter, verbosity=verbosity, tight_gelu=true)
            VeryDiffPolyExperiments.export2json(nn_poly, outfile)
        catch e
            @warn "Error generating network with degree $d: $e"
        end
    end
end


function get_enc_evaluation_times(onnx_path, degrees; outdir=".")
    model_name = split(basename(onnx_path), ".")[1]
    poly_paths = [joinpath(outdir, "mnist_$(model_name)_d_$(d).json") for d in degrees]

    hencpath = joinpath(@__DIR__, "../../FormalMethodsForFHENetworks/build/FormalMethods")
    times = Float64[]
    for poly_path in poly_paths
        print("Evaluating network $poly_path using FHE: ")
        cmd = `$(hencpath) --custom $(poly_path) --input random --verbose 1`
        output = read(cmd, String)

        lines = split(output, "\n")
        t_string = split(lines[end-1], " ")[end]
        t_string = replace(t_string, ":" => ".", "s" => "")
        t = parse(Float64, t_string)
        push!(times, t)
        print(t, " seconds\n")
    end
    return times
end


degrees = (1:19) ∪ (20:5:100)
onnx_prefix = string(@__DIR__, "/../networks/mnist/")
onnx_path = string(onnx_prefix, "mnist_gelu_256x4_1e4.onnx")
outdir = joinpath(@__DIR__, "./generated_networks/")

generate_json_networks(onnx_path, degrees, max_polys_per_layer=Inf, max_iter=20, verbosity=1, outdir=outdir)

enc_times = get_enc_evaluation_times(onnx_path, degrees, outdir=outdir)






model_name = split(basename(onnx_path), ".")[1]
poly_paths = [joinpath(outdir, "mnist_$(model_name)_d_$(d).json") for d in degrees]

hencpath = joinpath(@__DIR__, "../../FormalMethodsForFHENetworks/build/FormalMethods")
poly_path = poly_paths[1]
cmd = `$(hencpath) --custom $(poly_path) --input random --verbose 1`
output = read(cmd, String)

lines = split(output, "\n")
t_string = split(lines[end-1], " ")[end]
t_string = replace(t_string, ":" => ".", "s" => "")
t = parse(Float64, t_string)


for i in 1:10
    print("Gonna print a number ... ")
    print(i, " !!!\n")
end


module VeryDiffPolyExperiments

using VeryDiff, VNNLib, LinearAlgebra
using Tar, CodecZlib, JLD2, JSON3, CSV, Dates

import VeryDiff: approximate_polynomial, approximate_polynomial_iterative, approximate_polynomial_abcrown, approximate_polynomial_iterative_sampling,
                 extract_approximation_domain, LayeredModel, to_layered_model

const OXP = VNNLib.OnnxParser


function __init__()
    if !isdir(string(@__DIR__, "/../datasets"))
        println("[INFO] Extracting datasets...")
        open(GzipDecompressorStream, "datasets.tar.gz") do io
            Tar.extract(io, string(@__DIR__, "/../datasets"))
        end
    end
    
end

include("utils.jl")
include("verydiff_patch/patch.jl")
include("fhe_export/json_export.jl")
include("generate_nns/generate_nns.jl")
include("generate_nns/generate_nns_mnist.jl")
include("generate_nns/generate_nns_heloc.jl")
include("generate_nns/generate_nns_har.jl")
include("verify_nns/verify_eps_equivalence.jl")
include("verify_nns/verify_eps_equivalence_mnist.jl")
include("verify_nns/verify_eps_equivalence_heloc.jl")
include("verify_nns/verify_eps_equivalence_har.jl")
include("gelu/generate_nns_gelu.jl")




export generate_mnist_nets_verified_bounds, generate_mnist_nets_empirical_bounds, generate_har_nets_verified_bounds,
       verify_eps_equivalence_mnist, warmup_eps_equivalence_mnist, verify_eps_equivalence_sample_mnist,
        generate_heloc_nets_verified_bounds, verify_eps_equivalence_heloc, warmup_eps_equivalence_heloc,
        verify_eps_equivalence_har, verify_eps_equivalence_sample_har

end # module VeryDiffPolyExperiments

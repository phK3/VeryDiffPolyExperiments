module VeryDiffPolyExperiments

using VeryDiff, VNNLib, LinearAlgebra
using Tar, CodecZlib, JLD2, CSV, Dates

import VeryDiff: approximate_polynomial_iterative, extract_approximation_domain


function __init__()
    if !isdir(string(@__DIR__, "/../datasets"))
        println("[INFO] Extracting datasets...")
        open(GzipDecompressorStream, "datasets.tar.gz") do io
            Tar.extract(io, string(@__DIR__, "/../datasets"))
        end
    end
    
end

include("generate_nns/generate_nns.jl")
include("generate_nns/generate_nns_mnist.jl")
include("generate_nns/generate_nns_heloc.jl")
include("verify_nns/verify_eps_equivalence.jl")
include("verify_nns/verify_eps_equivalence_mnist.jl")


export generate_mnist_nets_verified_bounds, verify_eps_equivalence_mnist, warmup_eps_equivalence_mnist,
        generate_heloc_nets_verified_bounds

end # module VeryDiffPolyExperiments

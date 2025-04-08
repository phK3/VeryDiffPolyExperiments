module VeryDiffPolyExperiments

using VeryDiff, VNNLib, LinearAlgebra
using Tar, CodecZlib, JLD2, CSV, Dates

import VeryDiff: approximate_polynomial_iterative


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


export generate_mnist_nets_verified_bounds

end # module VeryDiffPolyExperiments

# TODO: adjust imports for your setup
using VNNLib.OnnxParser
import VeryDiff: ONNXChebyshevPoly
import VNNLib.OnnxParser: ONNXConv, ONNXBatchNorm, ONNXRelu, ONNXGelu, ONNXFlatten, ONNXLinear, Node
using Random
using VeryDiffPolyExperiments

Random.seed!(42)

# ══════════════════════════════════════════════════════════════════════
# Architecture: 8×8×3 (WHCN) input
#   Conv(3×3, 3→4) → BatchNorm(4) → ReLU
#   → Conv(3×3, 4→2) → GeLU → Flatten
#   → Linear(32→16) → Chebyshev(16, deg=5) → Linear(16→4)
# ══════════════════════════════════════════════════════════════════════

# ── Layer construction ────────────────────────────────────────────────

conv1_w = randn(Float32, 3, 3, 3, 4)
conv1_b = randn(Float32, 4)
conv1   = ONNXConv(["input_0"], ["conv1_out"], "conv1",
                    conv1_w, conv1_b; stride=1, pad=0)

bn1 = ONNXBatchNorm(
    ["conv1_out"], ["bn1_out"], "bn1",
    Float32[0.5, -0.3, 0.1, 0.7],    # μ
    Float32[1.2, 0.8, 1.1, 0.9],     # γ
    Float32[0.1, -0.2, 0.3, -0.1],   # β
    Float32[1.5, 0.8, 1.2, 2.0];     # σ²
    ϵ = 1.0f-5
)

relu1 = ONNXRelu(["bn1_out"], ["relu1_out"], "relu1")

conv2_w = randn(Float32, 3, 3, 4, 2)
conv2_b = randn(Float32, 2)
conv2   = ONNXConv(["relu1_out"], ["conv2_out"], "conv2",
                    conv2_w, conv2_b; stride=1, pad=0)

gelu1    = ONNXGelu(["conv2_out"], ["gelu1_out"], "gelu1", "none")
flatten1 = ONNXFlatten(["gelu1_out"], ["flatten1_out"], "flatten1", 1)

lin1_W  = randn(Float32, 16, 32)
lin1_b  = randn(Float32, 16)
linear1 = ONNXLinear(["flatten1_out"], ["linear1_out"], "linear1",
                      lin1_W, lin1_b)

cheb_coeffs = randn(Float32, 16, 6)           # 16 neurons, degree 5
cheb_l      = fill(-2.0f0, 16)
cheb_u      = fill( 2.0f0, 16)
cheb_ϵ      = zeros(Float32, 16)
cheb1 = ONNXChebyshevPoly(["linear1_out"], ["cheb1_out"], "cheb1",
                           cheb_coeffs, cheb_l, cheb_u, cheb_ϵ)

lin2_W  = randn(Float32, 4, 16)
lin2_b  = randn(Float32, 4)
linear2 = ONNXLinear(["cheb1_out"], ["output_0"], "linear2",
                      lin2_W, lin2_b)

# ── OnnxNet graph ────────────────────────────────────────────────────

layer_names = ["conv1", "bn1", "relu1", "conv2", "gelu1",
               "flatten1", "linear1", "cheb1", "linear2"]
layer_list  = [conv1, bn1, relu1, conv2, gelu1,
               flatten1, linear1, cheb1, linear2]

nodes = Dict{String, Node{String}}(zip(layer_names, layer_list))

output_names = ["conv1_out", "bn1_out", "relu1_out", "conv2_out",
                "gelu1_out", "flatten1_out", "linear1_out",
                "cheb1_out", "output_0"]

output_dict = Dict{String, String}(zip(output_names, layer_names))

node_prevs = Dict{String, Vector{String}}(
    "conv1"    => String[],
    "bn1"      => ["conv1"],
    "relu1"    => ["bn1"],
    "conv2"    => ["relu1"],
    "gelu1"    => ["conv2"],
    "flatten1" => ["gelu1"],
    "linear1"  => ["flatten1"],
    "cheb1"    => ["linear1"],
    "linear2"  => ["cheb1"],
)

node_nexts = Dict{String, Vector{String}}(
    "conv1"    => ["bn1"],
    "bn1"      => ["relu1"],
    "relu1"    => ["conv2"],
    "conv2"    => ["gelu1"],
    "gelu1"    => ["flatten1"],
    "flatten1" => ["linear1"],
    "linear1"  => ["cheb1"],
    "cheb1"    => ["linear2"],
    "linear2"  => String[],
)

input_shapes  = Dict{String, NTuple{4,Int64}}("input_0"  => (8, 8, 3, 1))
output_shapes = Dict{String, NTuple{2,Int64}}("output_0" => (4, 1))

net = OnnxNet(
    ["conv1"],
    ["linear2"],
    nodes, output_dict,
    node_prevs, node_nexts,
    input_shapes, output_shapes,
)

VeryDiffPolyExperiments.export2json(net, "test.json")
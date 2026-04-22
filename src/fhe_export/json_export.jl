

"""
    `convert_to_list_of_tuples(nn_layers)`

Adds description of each layer as a tuple to a list.

Each tuple contains the layer type as a string, the input ids and output ids and the necessary parameters. E.g.:

[("linear", [0], [1], W1, b1), ("chebyshev", [1], [2], coeffs1, l1, u1), ...]

args:
- nn_layers - the list of layers to convert.
"""
function convert_to_list_of_tuples(nn_layers)
    layers = []
    for layer in nn_layers
        if layer isa OXP.ONNXLinear
            push!(layers, ("linear", layer.dense.weight, layer.dense.bias, size(layer.dense.weight)))
        elseif layer isa OXP.ONNXConv
            kernel_size = size(layer.conv.weight)[1:2]
            input_channels = size(layer.conv.weight)[3]
            output_channels = size(layer.conv.weight)[4]
            push!(layers, ("conv", kernel_size, input_channels, output_channels, layer.conv.weight, layer.conv.bias, layer.conv.stride, layer.conv.pad))
        elseif layer isa OXP.ONNXBatchNorm
            # stores scale, bias, mean, variance (not standard-deviation), epsilon for numerical stability and number of channels
            push!(layers, ("batchnorm", layer.batchnorm.γ, layer.batchnorm.β, layer.batchnorm.μ, layer.batchnorm.σ², layer.batchnorm.ϵ, layer.batchnorm.chs))
        elseif layer isa OXP.ONNXReshape
            if sum(layer.shape .!= 1) == 1
                push!(layers, ("flatten"))
            else
                push!(layers, ("reshape", layer.shape))
            end
        elseif layer isa OXP.ONNXFlatten
            push!(layers, ("flatten"))
        elseif layer isa VeryDiff.ONNXChebyshevPoly
            push!(layers, ("chebyshev", layer.coeffs, layer.l, layer.u, size(layer.coeffs)))
        elseif layer isa OXP.ONNXRelu
            push!(layers, ("relu"))
        elseif layer isa OXP.ONNXGelu
            push!(layers, ("gelu"))
        else
            error("Unsupported layer type: $(typeof(layer))")
        end
    end
    return layers
end


"""
Merges normalization of Chebyshev approximation bounds from [l, u] to [-1, 1] into the weights and biases of the linear layer.

args:
- layer_lin - linear layer preceding the chebyshev layer 
- layer_poly - chebyshev layer 

returns:
- tuple (layer_lin_new, layer_poly_new) with normalization merged into linear parameters and [-1, 1] for the chebyshev approximation bounds.
"""
function merge_normalization_into_dense(layer_lin::OXP.ONNXLinear, layer_poly::VeryDiff.ONNXChebyshevPoly{S,N,VN}) where {S,N,VN}
    l, u = layer_poly.l, layer_poly.u

    a = 1 ./ (0.5 .* (u .- l))
    b = -0.5 .* (u .+ l) ./ (0.5 .* (u .- l))

    Ŵ = a .* layer_lin.dense.weight
    b̂ = a .* layer_lin.dense.bias .+ b

    n = length(l)
    return OXP.ONNXLinear(layer_lin.inputs, layer_lin.outputs, layer_lin.name, Ŵ, b̂, double_precision=true), 
           VeryDiff.ONNXChebyshevPoly(layer_poly.inputs, layer_poly.outputs, layer_poly.name, layer_poly.coeffs, -ones(N, n), ones(N, n), layer_poly.ϵ)
end

function merge_normalization_into_dense(layer_lin::OXP.ONNXConv, layer_poly::VeryDiff.ONNXChebyshevPoly)
    # hard to merge bounds into conv layer because every neuron might have different bounds, but conv kernels are applied to all of them.
    # TODO: We can merge, if we only have a single polynomial per layer!
    return layer_lin, layer_poly
end

function merge_normalization_into_dense(layer1, layer2)
    # fallback for all other layers
    return layer1, layer2
end


"""
    `merge_normalization_into_dense(layers)`

Merges normalization of Chebyshev approximation bounds from [l, u] to [-1, 1] into the weights and biases of the preceeding linear layer.
This is done for all chebyshev layers in the network.

Merging is not possible for convolutional layers, because every neuron might have different bounds, but conv kernels are applied to all of them.

args:
- layers - list of layers in the network, must be topologically sorted.

returns:
- new list of layers with normalization merged into linear parameters and [-1, 1] for the chebyshev approximation bounds (where merging was possible).
"""
function merge_normalization_into_dense(layers::Vector{VeryDiff.Definitions.OnnxLayer{S}}) where {S}
    new_layers = Vector{OXP.Node{S}}(undef, length(layers))

    output_map = Dict{Int, Int}() # maps output id to layer that produces that output
    for (i, layer) in enumerate(layers)
        for output_id in layer.output_ids
            output_map[output_id] = i
        end
    end

    for (i, l) in enumerate(layers)
        if l.node isa VeryDiff.ONNXChebyshevPoly
            # find preceding layer
            input_id = l.input_ids[1] # assuming single input for chebyshev layer
            if haskey(output_map, input_id)
                preceding_layer_idx = output_map[input_id]
                preceding_layer = layers[preceding_layer_idx]
                new_lin, new_poly = merge_normalization_into_dense(preceding_layer.node, l.node)
                new_layers[preceding_layer_idx] = new_lin
                new_layers[preceding_layer_idx + 1] = new_poly
            else
                error("No preceding layer found for chebyshev layer with input id: ", input_id)
            end
        else
            # because we assume topologically sorted layers, linear layers may be added here, but will then
            # be overwritten if the following layer is a chebyshev layer. All other layers are not changed.
            new_layers[i] = l.node
        end
    end

    return new_layers
end


function export2json(net::OnnxNet{LayerIdT,NShapeIn,NShapeOut}, outfile) where {LayerIdT,NShapeIn,NShapeOut}
    layers, _ = VeryDiff.Definitions.sort_network(net)
    layers = merge_normalization_into_dense(layers)
    list_of_tuples = convert_to_list_of_tuples(layers)
    s = sprint(JSON3.pretty, list_of_tuples)
    write(outfile, s)
end

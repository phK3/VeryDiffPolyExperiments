

"""
Adds description of each layer in a LayeredModel as a tuple to a list.

Each tuple contains the layer type as a string and the necessary parameters. E.g.:

[("linear", W1, b1), ("chebyshev", coeffs1, l1, u1), ...]

args:
- net - the LayeredModel to convert.
"""
function convert_to_list_of_tuples(net::LayeredModel)
    layers = []
    for layer in net.layers
        if layer isa OXP.ONNXLinear
            push!(layers, ("linear", layer.dense.weight, layer.dense.bias))
        elseif layer isa VeryDiff.ONNXChebyshevPoly
            push!(layers, ("chebyshev", layer.coeffs, layer.l, layer.u))
        elseif layer isa OXP.ONNXRelu
            push!(layers, ("relu",))
        elseif layer isa OXP.ONNXGelu
            push!(layers, ("gelu",))
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
           VeryDiff.ONNXChebyshevPoly(layer_poly.inputs, layer_poly.outputs, layer_poly.name, layer_poly.coeffs, -ones(N, n), ones(N, n))
end


function merge_normalization_into_dense(net::LayeredModel{S}) where S
    layers = net.layers
    new_layers = Vector{OXP.Node{S}}(undef, length(layers))

    for i in 1:2:length(layers) - 1
        layer_lin = layers[i]
        layer_poly = layers[i + 1]
        layer_lin_new, layer_poly_new = merge_normalization_into_dense(layer_lin, layer_poly)
        new_layers[i] = layer_lin_new
        new_layers[i + 1] = layer_poly_new
    end

    if isodd(length(layers))
        new_layers[end] = layers[end]
    end

    return LayeredModel(new_layers)
end


function export2json(net::LayeredModel{S}, outfile) where S
    net = merge_normalization_into_dense(net)
    list_of_tuples = convert_to_list_of_tuples(net)
    s = sprint(JSON3.pretty, list_of_tuples)
    write(outfile, s)
end

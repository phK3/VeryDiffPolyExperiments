
# Would have to load old version of VeryDiff to get approximation via zonotopes.
# Implement the same approximation method here (with some borrowing from the new VeryDiff).
#
# Usage:
# 
#

function propagate_legacy(L::OXP.ONNXLinear, z::VeryDiff.Zonotope)
    W = L.dense.weight
    b = L.dense.bias
    new_c = W * z.c .+ b
    new_Gs = [W * g for g in z.Gs]
    return VeryDiff.Zonotope(new_Gs, new_c, nothing, z.generator_ids, z.owned_generators)
end

function propagate_legacy(L::OXP.ONNXRelu, z::VeryDiff.Zonotope)
    bounds = VeryDiff.zono_bounds(z)
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]

    dim = length(lower)
    crossing = VeryDiff.@simd_bool_expr dim ((lower < 0.0) & (upper > 0.0))
    α = clamp.(upper./(upper.-lower),0.0,1.0)
    λ = ifelse.(crossing, α, ifelse.(lower .>= 0.0, 1.0, 0.0))

    new_gens = count(crossing)
    
    γ = 0.5 .* max.(-λ .* lower,0.0,((-).(1.0,λ)).*upper)  # Computed offset (-λl/2)

    new_c = λ .* z.c .+ crossing.*γ

    new_Gs = Vector{Matrix{Float64}}(undef, length(z.Gs))
    for (idx, g) in enumerate(z.Gs)
        Ĝ = λ .* g
        E = γ .* I(size(g, 1))[:, crossing]
        new_Gs[idx] = [Ĝ E]
    end

    return VeryDiff.Zonotope(new_Gs, new_c, nothing, z.generator_ids, z.owned_generators)
end

function propagate_legacy(L::Union{OXP.ONNXGelu, VeryDiff.ONNXPoly}, z::VeryDiff.Zonotope)
    bounds = VeryDiff.zono_bounds(z)
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]

    λ, β, γ = VeryDiff.Transformers.get_linear_relaxation(L, lower, upper)

    # need new generators for every dimension
    new_gens = size(lower, 1)

    new_c = λ .* z.c .+ β

    new_Gs = Vector{Matrix{Float64}}(undef, length(z.Gs))
    for (idx, g) in enumerate(z.Gs)
        Ĝ = λ .* g
        E = γ .* I(size(g, 1))
        new_Gs[idx] = [Ĝ E]
    end
    
    return VeryDiff.Zonotope(new_Gs, new_c, nothing, z.generator_ids, z.owned_generators)
end

"""
Iteratively approximate each layer in the network by a polynomial of a given degree.

The input ranges for the approximation are verified bounds computed by zonotope propagation.

args:
    net - Network to approximate 
    input_set - input set for which to get bounds 
    degree - degree of polynomial approximation for ReLU layers

kwargs:
    verbosity - verbosity level (0: silent, 1: print first 5 lower and upper bounds)
    cheby - whether to use Chebyshev basis (true) or Monomial basis (false) for polynomial approximation
    max_iter - maximum number of iterations for Remez algorithm
    max_polys_per_layer - maximum number of different polynomials to use per layer
"""
function approximate_polynomial_iterative_zono(model::OnnxNet, input_lb::AbstractVector, input_ub::AbstractVector, degree::Integer; verbosity=0, selection=:contiguous, tol=1e-10, cheby=true, max_iter=20, max_polys_per_layer=Inf)
    @assert (max_polys_per_layer == Inf) || (max_polys_per_layer == 1) "only max_polys_per_layer=1 (one polynomial for all neurons) or Inf (one polynomial for each neuron) supported currently"

    input_center = 0.5 .* (input_lb .+ input_ub)
    input_radius = 0.5 .* (input_ub .- input_lb)
    owned_generator_ids = VeryDiff.SortedVector([1])
    ẑ = VeryDiff.Zonotope([Matrix(Diagonal(input_radius))], input_center, nothing, owned_generator_ids, 1)

    net_layers, io_map = VeryDiff.Definitions.sort_network(model)

    bounds_layer = [input_lb input_ub]
    layers_poly = []
    for (i, l) in enumerate(net_layers)
        bounds_layer = VeryDiff.zono_bounds(ẑ)

        verbosity > 0 && println("--- layer ", l.node.name, " ---")
        verbosity > 0 && println("lower = ", bounds_layer[:,1][1:min(size(bounds_layer, 1), 5)])
        verbosity > 0 && println("upper = ", bounds_layer[:,2][1:min(size(bounds_layer, 1), 5)])
        !all(isfinite.(bounds_layer)) && println("lb non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,1])])
        !all(isfinite.(bounds_layer)) && println("ub non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,2])])

        layer_poly, ϵs = VeryDiff.approximate_polynomial(l.node, bounds_layer, degree, cheby=cheby, selection=selection, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer, tol=tol)
        push!(layers_poly, layer_poly)

        ẑ = propagate_legacy(layer_poly, ẑ)
    end

    model_poly = deepcopy(model)
    for layer_poly in layers_poly
        model_poly.nodes[layer_poly.name] = layer_poly
    end

    return model_poly
end

# some models here have a reshaping layer, we add it here for now 
# TODO: implement in VeryDiff/convert to just linear 
function VeryDiff.approximate_polynomial(L::VNNLib.OnnxParser.ONNXReshape{String}, bounds, degree; cheby=true, verbosity=0, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    return L, 0
end
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:31:45 2023

@author: Edoardo
"""

import warnings
import numpy as np

"""
Object containing the parameters of an affine function
"""
class Affine:
    def __init__(self, weights, biases):
        self.W = weights
        self.b = biases

"""
Create two copies of the input by zeroing the negative and positive entries respectively
Arguments:
    M: input matrix, any numpy array
Returns:
    MatPos: copy of M retaining only the positive elements
    MatNeg: copy of M retaining only the negative elements
"""
def MatrixSignSplit(M):
    
    MatPos = M.copy()
    MatNeg = M.copy()
    MatPos[MatPos < 0] = 0
    MatNeg[MatNeg > 0] = 0
    
    return (MatPos, MatNeg)

"""
Return upper and lower affine bounds on the ReLU activation function
Arguments:
    pot_min: minimum value of the potential(s)
    pot_max: maximum value of the potential(s)
Returns:
    upper_relu: slope and offset of the upper bound(s)
    lower_relu: slope and offset of the lower bound(s)
"""
def ReluBound(pot_min, pot_max):
    
    pot_min = np.atleast_1d(pot_min)
    pot_max = np.atleast_1d(pot_max)
    
    # check validity of the input range
    if (pot_min > pot_max).any():
        raise ValueError("ReluBound(): some potential ranges are empty")
    
    # compute the output range
    relu_max = pot_max * (pot_max >= 0)
    relu_min = pot_min * (pot_min >= 0)
    
    # division by zero can happen, ignore the corresponding warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # compute slope and offset of the upper bounds
        slope = (relu_max - relu_min) / (pot_max - pot_min)
        offset = relu_min - pot_min * slope
    
    # fix degenerate potentials
    degenerate = pot_min == pot_max
    slope[degenerate] = (pot_min >= 0)[degenerate] * 1
    offset[degenerate] = 0
    
    # return upper and lower bounds
    upper_relu = Affine(slope, offset)
    lower_relu = Affine(slope, np.zeros(len(offset)))
    return (lower_relu, upper_relu)

"""
Layer-wise symbolic abstraction propagation
Designed around FastLin-style affine abstractions
"""
class ForwardLayer:
    
    """
    Propagate the domain enclosure at construction time
    Arguments:
        params -> weights and biases of the layer (affine function)
        upper_in -> affine upper bound on the layer inputs
        lower_in -> affine lower bound on the layer inputs
        x_range -> interval ranges of each free variable
        act_func -> activation function with abstraction method
    """
    def __init__(self, params, upper_in, lower_in, x_range, act_func="ReLU"):
        
        # compute affine bounds on the potentials
        WPos, WNeg = MatrixSignSplit(params.W)
        upper_pot = Affine(WPos @ upper_in.W + WNeg @ lower_in.W,
                           WPos @ upper_in.b + WNeg @ lower_in.b + params.b)
        lower_pot = Affine(WNeg @ upper_in.W + WPos @ lower_in.W,
                           WNeg @ upper_in.b + WPos @ lower_in.b + params.b)
        
        # compute concrete bounds on the potentials
        UPos, UNeg = MatrixSignSplit(upper_pot.W)
        LPos, LNeg = MatrixSignSplit(lower_pot.W)
        pot_max = UPos @ x_range[:,1] + UNeg @ x_range[:,0] + upper_pot.b
        pot_min = LNeg @ x_range[:,1] + LPos @ x_range[:,0] + lower_pot.b
        
        if act_func == "ReLU":
            
            # compute element-wise bounds on the activation function
            lower_relu, upper_relu = ReluBound(pot_min, pot_max)
            
            # compute affine bounds on the output
            # assumption: the activation slope is always non-negative
            upper_out = Affine(upper_relu.W[:,np.newaxis] * upper_pot.W,
                               upper_relu.W * upper_pot.b + upper_relu.b)
            lower_out = Affine(lower_relu.W[:,np.newaxis] * lower_pot.W,
                               lower_relu.W * lower_pot.b + lower_relu.b)
        
        # TODO: add support for non-ReLU activations
        # for now assume that they are linear
        else:
            upper_out = upper_pot
            lower_out = lower_pot
        
        # store the following parameters for future use
        self.params = params
        self.pot_max = pot_max
        self.pot_min = pot_min
        self.upper_out = upper_out
        self.lower_out = lower_out

"""
Network-wise symbolic abstraction propagation
Designed around FastLin-style affine abstractions
An additional error input is added after each activation function
"""
class ForwardNetwork:
    
    """
    Compute all reachable activation states (forward pass).
    The network must be feedforward fully-connected and contain
    ReLU activations only, except for the final affine layer.
    Arguments:
        w_list -> list of weight matrices
        b_list -> list of bias vectors
        x_range -> interval ranges of each input variable
        e_range -> interval ranges of each error variable
    """
    def __init__(self, w_list, b_list, x_range, e_range):
        
        # collate the network parameters in a list of affine transformations
        p_list = [Affine(W, b) for W, b in zip(w_list, b_list)]
        
        # compute the number of neurons in each layer
        n_neurons = [len(params.b) for params in p_list]
        n_input = x_range.shape[0]
        n_error = e_range.shape[0]
        
        # check validity of the input dimensions
        if sum(n_neurons[:-1]) != n_error:
            raise ValueError("ActivationStates(): the number of error inputs" +
                             " does not match the number of ReLU neurons" +
                             " in the network")
        if p_list[0].W.shape[1] != n_input:
            raise ValueError("ActivationStates(): the number of network inputs" +
                             " does not match the number of columns of the" +
                             " first weight matrix")
        
        # inject trivial matching bounds to the first layer
        lower_in = Affine(np.eye(n_input), np.zeros(n_input))
        upper_in = lower_in
        xe_range = x_range
        
        # propagate through all ReLU layers
        forward_layer_list = []
        for i, params in enumerate(p_list[:-1]):
            layer = ForwardLayer(params, upper_in, lower_in, xe_range)
            forward_layer_list.append(layer)
            
            # add the contribution of the approximation error to the bounds
            upper_W = np.concatenate([layer.upper_out.W,
                                      np.eye(n_neurons[i])], axis=1)
            lower_W = np.concatenate([layer.lower_out.W,
                                      np.eye(n_neurons[i])], axis=1)
            upper_in = Affine(upper_W, layer.upper_out.b)
            lower_in = Affine(lower_W, layer.lower_out.b)
            
            # extend the input domain with the new errors
            xe_range = np.concatenate([x_range,
                                       e_range[:sum(n_neurons[:i+1]),:]], axis=0)
        
        # propagate through the last affine layer
        layer = ForwardLayer(p_list[-1], upper_in, lower_in, xe_range, act_func=None)
        forward_layer_list.append(layer)
        
        # store the following parameters for future use
        self.forward_layer_list = forward_layer_list
        self.x_range = x_range
        self.e_range = e_range
    
    """
    Return concrete bounds on the potentials of each layer
    """
    def get_concrete_potentials(self):
        
        # extract the jacobian of all ReLU layers
        upper_potentials = [layer.pot_max
                           for layer in self.forward_layer_list]
        lower_potentials = [layer.pot_min
                           for layer in self.forward_layer_list]
        
        return (lower_potentials, upper_potentials)

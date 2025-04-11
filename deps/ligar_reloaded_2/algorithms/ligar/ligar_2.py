# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 22:04:18 2025

@author: Edoardo
"""

import warnings
import numpy as np

from algorithms.jacobian.reachability import Affine, ForwardLayer
from algorithms.remez.piecewise import PiecewisePoly, ReLU
from algorithms.ligar.ligar import deg_1_relu_error, batch_estimation

"""
Compute the approximation error
of the Chebyshev approximation of the ReLU function
"""
def cheby_relu_error(pot_min, pot_max, deg):
    
    # interpolands
    relu_func = ReLU()
    eye_func = PiecewisePoly([np.polynomial.Polynomial([0,1])], [-np.Inf,+np.Inf])
    
    # interpolants
    domain = [pot_min, pot_max]
    relu_cheb = np.polynomial.chebyshev.Chebyshev.interpolate(relu_func, deg, domain=domain)
    eye_cheb = np.polynomial.chebyshev.Chebyshev.interpolate(eye_func, deg, domain=domain)
    
    # all roots
    neg_roots = relu_cheb.deriv().roots().real
    pos_roots = (relu_cheb-eye_cheb).deriv().roots().real
    
    # all candidate extrema
    guesses = np.concatenate([neg_roots, pos_roots, [pot_min, pot_max, 0.0]])
    guesses = np.clip(guesses, pot_min, pot_max)
    
    # max absolute approximation difference
    error = np.max(np.abs(relu_cheb(guesses) - relu_func(guesses)))    
    return error

"""
Modified version of ForwardNetwork in algorithms.jacobian.reachability
that computes the polynomial approximation error on the fly
"""
class ForwardNetwork2:
    
    """
    Compute potential ranges and ReLU approx errors (forward pass).
    The network must be feedforward fully-connected and contain
    ReLU activations only, except for the final affine layer.
    Arguments:
        w_list -> list of weight matrices
        b_list -> list of bias vectors
        x_range -> interval ranges of each input variable
        d_list -> approximation degree for each ReLU and each layer
        mode -> either "approx" or "chebyshev"
    """
    def __init__(self, w_list, b_list, x_range, d_list, mode="approx"):
        
        # collate the network parameters in a list of affine transformations
        p_list = [Affine(W, b) for W, b in zip(w_list, b_list)]
        
        # compute the number of neurons in each layer
        n_neurons = [len(params.b) for params in p_list]
        n_input = x_range.shape[0]
        n_error = sum([len(d) for d in d_list])
        
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
            
            degrees = d_list[i]
            pot_min = layer.pot_min
            pot_max = layer.pot_max
            if mode == "approx":
                e = deg_1_relu_error(pot_min, pot_max)
                e = e / degrees
                e_range = np.column_stack([-e,+e])
            elif mode == "chebyshev":
                e_range = np.zeros([len(degrees),2])
                for i in range(len(degrees)):
                    e = cheby_relu_error(pot_min[i], pot_max[i], degrees[i])
                    e_range[i,:] = [-e,+e]
            else:
                e_range = np.zeros([len(degrees),2])
                warnings.warn("ForwardNetwork2.__init__(): unknown polynomial approx mode, I will continue with no error.")
            
            # extend the input domain with the new errors
            xe_range = np.concatenate([xe_range, e_range], axis=0)
        
        # propagate through the last affine layer
        layer = ForwardLayer(p_list[-1], upper_in, lower_in, xe_range, act_func=None)
        forward_layer_list.append(layer)
        
        # store the following parameters for future use
        self.forward_layer_list = forward_layer_list
        self.x_range = x_range
        self.e_range = xe_range[x_range.shape[0]:,:]
    
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
    
    """
    Return concrete error ranges for the ReLU approximations
    """
    def get_error_ranges(self):
        return self.e_range

"""
Design all ReLU polynomial approximations in one pass
Arguments:
    w_list -> list of weight matrices
    b_list -> list of bias vectors
    x_range -> interval ranges of each input variable
    d_list -> approximation degree for each ReLU and each layer
    mode -> either "approx" or "cheby"
"""
def ligar_design(w_list, b_list, x_range, d_list, mode="approx"):
    
    net = ForwardNetwork2(w_list, b_list, x_range, d_list, mode=mode)
    
    pot_min, pot_max = net.get_concrete_potentials()
    e_ball = np.max(np.abs(net.get_error_ranges()), axis=1)
    
    return pot_min, pot_max, e_ball

"""
Compute an upper bound over the output error
due to the cumulative approximation error at each ReLU
"""
def ligar_output_bound(w_list, b_list, x_list, x_ball, e_ball):
    
    # estimate Lipschitz constants and potential ranges
    lipschitz, _, _ = batch_estimation(w_list,
                                       b_list,
                                       x_list,
                                       x_ball,
                                       e_ball)
    
    # compute the new output bound
    output_bound = lipschitz @ e_ball
    return output_bound

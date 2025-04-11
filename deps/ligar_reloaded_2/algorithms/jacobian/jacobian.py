# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:23:23 2023

@author: Edoardo
"""

import warnings
import numpy as np

from algorithms.jacobian.reachability import Affine, MatrixSignSplit, ForwardNetwork

"""
Return upper and lower affine bounds on the result
of backpropagating gradients through the step function
(gradient of ReLU when its activation state is uncertain)
Arguments:
    out_min: minimum value of the gradient(s)
    out_max: maximum value of the gradient(s)
Returns:
    upper_step: slope and offset of the upper bound(s)
    lower_step: slope and offset of the lower bound(s)
"""
def StepBound(out_min, out_max):
    
    # convert to two-dimensional matrix form (useful in BackwardLayer)
    out_min = np.atleast_2d(out_min)
    out_max = np.atleast_2d(out_max)
    
    # check validity of the input range
    if (out_min > out_max).any():
        raise ValueError("StepBound(): some gradient ranges are empty")
    
    # division by zero can happen, ignore the corresponding warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        # compute the upper bound
        pos_max = out_max * (out_max >= 0)
        pos_min = out_min * (out_min >= 0)
        upper_slope = (pos_max - pos_min) / (out_max - out_min)
        upper_offset = pos_min - out_min * upper_slope
        
        # compute the lower bound
        neg_max = out_max * (out_max <= 0)
        neg_min = out_min * (out_min <= 0)
        lower_slope = (neg_max - neg_min) / (out_max - out_min)
        lower_offset = neg_min - out_min * lower_slope
    
    # fix degenerate gradients
    degenerate = out_min == out_max
    upper_slope[degenerate] = (out_max >= 0)[degenerate] * 1
    lower_slope[degenerate] = (out_min <= 0)[degenerate] * 1
    upper_offset[degenerate] = 0
    lower_offset[degenerate] = 0
    
    # return upper and lower bounds
    upper_step = Affine(upper_slope, upper_offset)
    lower_step = Affine(lower_slope, lower_offset)
    return (lower_step, upper_step)

"""
Layer-wise symbolic Clarke Jacobian propagation
Designed around the algorithm in Shi et al., NeurIPS 2022
"""
class BackwardLayer:
    
    """
    Backpropagate the Clarke Jacobian at construction time
    Arguments:
        params -> weights and biases of the layer (affine function)
        upper_out -> affine upper bound on the Jacobian of the remaining layers
        lower_out -> affine lower bound on the Jacobian of the remaining layers
        act_state -> vector with values 0-inactive, 1-active, 2-both
    """
    def __init__(self, params, upper_out, lower_out, act_state):
        
        # compute concrete bounds on the output
        out_min = lower_out.W + lower_out.b
        out_max = upper_out.W + upper_out.b
        
        # compute symbolic bounds on the ReLU activations
        lower_step, upper_step = StepBound(out_min, out_max)
        
        # bind always-inactive ReLUs to zero (column-wise)
        inactive = act_state == 0
        upper_step.W[:, inactive] = 0
        lower_step.W[:, inactive] = 0
        upper_step.b[:, inactive] = 0
        lower_step.b[:, inactive] = 0
        
        # bind always-active ReLUs to one (column-wise)
        active = act_state == 1
        upper_step.W[:, active] = 1
        lower_step.W[:, active] = 1
        upper_step.b[:, active] = 0
        lower_step.b[:, active] = 0
        
        # propagate through the activations
        upper_pot = Affine(upper_out.W * upper_step.W,
                           upper_out.b * upper_step.W + upper_step.b)
        lower_pot = Affine(lower_out.W * lower_step.W,
                           lower_out.b * lower_step.W + lower_step.b)
        
        # propagate through the weight matrix
        WPos, WNeg = MatrixSignSplit(params.W)
        upper_in = Affine(upper_pot.W @ WPos + lower_pot.W @ WNeg,
                          upper_pot.b @ WPos + lower_pot.b @ WNeg)
        lower_in = Affine(upper_pot.W @ WNeg + lower_pot.W @ WPos,
                          upper_pot.b @ WNeg + lower_pot.b @ WPos)
        
        # store the following parameters for future use
        self.params = params
        self.out_max = out_max
        self.out_min = out_min
        self.upper_in = upper_in
        self.lower_in = lower_in
        self.act_state = act_state

"""
Network-wise computation of the Lipschitz constant
Designed around the algorithm in Shi et al., NeurIPS 2022
An additional error input is added after each activation function
"""
class BackwardNetwork(ForwardNetwork):
    
    """
    Compute all reachable activation states (forward pass)
    and bound the corresponding Clarke Jacobian (backward pass).
    The network must be feedforward fully-connected and contain
    ReLU activations only, except for the final affine layer.
    Arguments:
        w_list -> list of weight matrices
        b_list -> list of bias vectors
        x_range -> interval ranges of each input variable
        e_range -> interval ranges of each error variable
    """
    def __init__(self, w_list, b_list, x_range, e_range):
        
        # execute the forward reachability pass
        ForwardNetwork.__init__(self, w_list, b_list, x_range, e_range)
        
        # extract the activation states of all neurons
        act_state_list = []
        for layer in self.forward_layer_list:
            pot_range = np.column_stack([layer.pot_min, layer.pot_max])
            
            # compute the activation state from the range of the potentials
            active = pot_range[:,0] > 0 # the minimum is positive
            inactive = pot_range[:,1] < 0 # the maximum is negative
            uncertain = np.logical_not(np.logical_or(active, inactive))
            
            # encode the activation state as 0-inactive, 1-active, 2-both
            act_state = (active * 1 + uncertain * 2).astype(int)
            act_state_list.append(act_state)
        
        # trivial matching bounds on the gradient of the last layer
        last_W = self.forward_layer_list[-1].params.W
        lower_out = Affine(last_W, np.zeros(last_W.shape))
        upper_out = lower_out
        
        # backpropagate the gradient through all ReLU layers
        backward_layer_list = []
        for layer, act_state in zip(reversed(self.forward_layer_list[:-1]),
                                    reversed(act_state_list[:-1])):
            
            layer = BackwardLayer(layer.params, upper_out, lower_out, act_state)
            backward_layer_list.append(layer)
            lower_out = layer.lower_in
            upper_out = layer.upper_in
        
        # reorder the layers
        backward_layer_list.reverse()
        
        # store the following parameters for future use
        self.backward_layer_list = backward_layer_list
    
    """
    Return affine bounds on the Clarke Jacobian of each layer
    """
    def get_symbolic_jacobians(self):
        
        # extract the jacobian of all ReLU layers
        upper_jacobians = [layer.upper_in
                           for layer in self.backward_layer_list]
        lower_jacobians = [layer.lower_in
                           for layer in self.backward_layer_list]
        
        # add the weight matrix of the last affine layer
        last_W = self.forward_layer_list[-1].params.W
        last_jacobian = Affine(last_W, np.zeros(last_W.shape))
        upper_jacobians.append(last_jacobian)
        lower_jacobians.append(last_jacobian)
        
        return (lower_jacobians, upper_jacobians)
    
    """
    Return concrete bounds on the Clarke Jacobian of each layer
    """
    def get_concrete_jacobians(self):
        
        lower_symb, upper_symb = self.get_symbolic_jacobians()
        
        lower_jacobians = [symb.W + symb.b for symb in lower_symb]
        upper_jacobians = [symb.W + symb.b for symb in upper_symb]
        
        return (lower_jacobians, upper_jacobians)
    
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:02:24 2023

@author: Edoardo
"""

import warnings
import time

import numpy as np

from algorithms.jacobian.jacobian import BackwardNetwork

"""
Computes Lipschitz constant and potential range for each ReLU neuron
Arguments:
    w_list: minimum value of the gradient(s)
    b_list: maximum value of the gradient(s)
    x_list: list of (concrete) inputs
    x_ball: perturbation distance for each network input
    e_ball: perturbation distance for each error input
"""
def batch_estimation(w_list, b_list, x_list, x_ball, e_ball):
    
    lipschitz = np.zeros(len(e_ball))
    pot_max = np.repeat(np.finfo(float).min, len(e_ball))
    pot_min = np.repeat(np.finfo(float).max, len(e_ball))
    
    for x in x_list:
        
        # perturb the input
        x_range = np.column_stack([x - x_ball, x + x_ball])
        e_range = np.column_stack([-e_ball, +e_ball])
        
        # compute potentials and jacobians
        net = BackwardNetwork(w_list, b_list, x_range, e_range)
        lower_jacs, upper_jacs = net.get_concrete_jacobians()
        lower_pots, upper_pots = net.get_concrete_potentials()
        
        # keep the maximum absolute jacobian coefficient across all outputs
        lower_jacs = [np.atleast_2d(jac) for jac in lower_jacs]
        upper_jacs = [np.atleast_2d(jac) for jac in upper_jacs]
        abs_jacs = np.max(np.maximum(np.abs(np.hstack(lower_jacs[1:])),
                                     np.abs(np.hstack(upper_jacs[1:]))), axis=0)
        lipschitz = np.maximum(lipschitz, abs_jacs)
        
        # keep the maximum and minimum potential of ReLU nodes
        pot_max = np.maximum(pot_max, np.concatenate(upper_pots[:-1]))
        pot_min = np.minimum(pot_min, np.concatenate(lower_pots[:-1]))
    
    return (lipschitz, pot_min, pot_max)

"""
Compute the approximation error
of an affine approximation of the ReLU function
"""
def deg_1_relu_error(pot_min, pot_max):
    
    pot_min = np.atleast_1d(pot_min)
    pot_max = np.atleast_1d(pot_max)
    
    relu_min = pot_min * (pot_min >= 0)
    relu_max = pot_max * (pot_max >= 0)
    
    # suppress division-by-zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        slope = (relu_max - relu_min) / (pot_max - pot_min)
        offset = relu_min - pot_min * slope
        error = offset / 2
    
    # recover division-by-zero entries
    error[pot_max == pot_min] = 0
    
    return error

"""
Simple ReLU function for Chebyshev approximation purposes
"""
def relu(x):
    return np.maximum(x, 0)

"""
Simple identity function for Chebyshev approximation purposes
"""
def identity(x):
    return x

"""
Compute the approximation error
of the Chebyshev approximation of the ReLU function
"""
def cheby_relu_error(pot_min, pot_max, deg):
    
    domain = [pot_min, pot_max]
    relu_cheb = np.polynomial.chebyshev.Chebyshev.interpolate(relu, deg, domain=domain)
    eye_cheb = np.polynomial.chebyshev.Chebyshev.interpolate(identity, deg, domain=domain)
    
    neg_roots = relu_cheb.deriv().roots().real
    pos_roots = (relu_cheb-eye_cheb).deriv().roots().real
    
    guesses = np.concatenate([neg_roots, pos_roots, [pot_min, pot_max, 0.0]])
    
    guesses[guesses <= pot_min] = pot_min
    guesses[guesses >= pot_max] = pot_max
    
    error = np.max(np.abs(relu_cheb(guesses) - relu(guesses)))    
    return error

"""
Project a candidate solution x onto the feasible region
Arguments:
    lipschitz: infinite-norm Lipschitz constant of each neuron
    cost: objective function cost factor of each neuron
    e_ball: perturbation distance for each error input
    d_global: individual global dual variable
"""
def adjust_local_dual(lipschitz, cost, e_ball, d_global):
    
    # compute the minimum positive value of the local dual
    # that would satisfy the constraint x[i] <= e_ball[i]
    d_local = cost / (e_ball * e_ball) - lipschitz * d_global
    d_local[d_local < 0] = 0
    
    return d_local

"""
Project a candidate solution x onto the feasible region
Arguments:
    lipschitz: infinite-norm Lipschitz constant of each neuron
    cost: objective function cost factor of each neuron
    e_out: maximum output error allowed
    d_global: individual global dual variable
    d_local: vector of local dual variables
    max_iter: maximum number of bisection iterations
    min_change: minimum magnitude of relative change
"""
def adjust_global_dual(lipschitz, cost, e_out, d_global, d_local,
                       max_iter=1000, min_change=1e-8):
    
    trajectory = 0
    factor = 2.0
    d_best = -1 # return a negative value if no feasible solution is found
    
    # run the given number of scale-free bisection iterations
    # unless we hit the required precision early
    for i in range(max_iter):
        if factor <= 1.0 + min_change:
            break
        
        # compute the candidate values for x
        x = np.sqrt(cost / (lipschitz * d_global + d_local))
        
        # costraint in the form lipschitz @ x <= e_out
        # active constraint: enlarge d_global
        if lipschitz @ x >= e_out:
            d_global = d_global * factor
            trajectory += 1
        
        # inactive constraint: shrink d_global
        else:
            d_best = d_global # keep track of the best d so far
            d_global = d_global / factor
            trajectory -= 1
        
        # as soon as we start changing trajectory of d_global
        # reduce the scale-free factor
        if abs(trajectory) != i + 1:
            factor = np.sqrt(factor)
    
    # return the best d that satisfies the constraint
    return d_best

"""
Computes the optimal error allocation
Arguments:
    lipschitz: infinite-norm Lipschitz constant of each neuron
    cost: objective function cost factor of each neuron
    e_ball: perturbation distance for each error input
    e_out: maximum output error allowed
    max_iter: maximum number of dual adjustments and bisection iterations
    min_change: minimum magnitude of relative change
    verbose: whether to display progress info
"""
def optimal_allocation(lipschitz, cost, e_ball, e_out,
                       max_iter=1000, min_change=1e-8,
                       verbose=False):
    
    # initialise a (possibly unfeasible) primal solution
    x = e_ball
    value_list = [np.Inf]
    
    # initialise a (possibly unfeasible) dual solution
    d_local = np.zeros(len(e_ball))
    d_global = 1
    
    # each iteration will compute a (tighter) feasible solution
    for i in range(max_iter):
        
        # make sure the global constraint is satisfied
        d_global = adjust_global_dual(lipschitz, cost, e_out,
                                      d_global, d_local,
                                      max_iter=max_iter,
                                      min_change=min_change)
        
        # error warning if invalid value is returned
        if d_global < 0:
            raise ValueError("ligar.optimal_allocation():" +
                             " unable to find a feasible value" +
                             " for the global dual variable;" +
                             " try increasing the max number of iterations.")
        
        # adjust the local constraints accordingly
        d_local = adjust_local_dual(lipschitz, cost, e_ball, d_global)
        
        # suppress division-by-zero warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # turn the dual solution into a primal one
            x = np.sqrt(cost / (lipschitz * d_global + d_local))
            cost_over_x = cost / x
            cost_over_x[cost == 0] = 0 # recover divisions by zero
            value_list.append(np.sum(cost_over_x))
        
        # display progress info
        if verbose:
            print("optimal_allocation(): iteration", i,
                  "value", value_list[-1])
        
        # quit early if we hit the required precision
        progress = value_list[-2] - value_list[-1]
        if progress <= min_change:
            break
        
        break
    
    # return the primal solution
    return (x, value_list[-1])

"""
Main LiGAR iteration
Arguments:
    w_list: minimum value of the gradient(s)
    b_list: maximum value of the gradient(s)
    x_list: list of (concrete) inputs
    x_ball: perturbation distance for each network input
    e_max: maximum perturbation distance for each error input
    e_out: maximum output error allowed
    max_iter: maximum number of LiGAR iterations
    e_prec: tolerance margin to determine constraint activity
    value_prec: absolute convergence threshold
    verbose: whether to display progress info
Returns:
    e_list[best]: optimal error allocation to each ReLU approximation
    pot_min_list[best]: minimum approximation range for each ReLU
    pot_max_list[best]: maximum approximation range for each ReLU
    value_list[best]: value of the associated objective function
"""
def ligar_cycle(w_list, b_list, x_list, x_ball, e_max, e_out,
                max_iter=100, e_prec=1e-8, value_prec=1e-2, verbose=False):
    
    start = time.time()
    
    # neuron-specific convergence bounds
    e_up = e_max
    e_low = np.zeros(len(e_max))
    
    # initialise the domain of validity to the largest possible value
    e_ball = e_max
    
    # initialise a (possibly unfeasible) allocation
    e_list = [e_max]
    pot_min_list = [np.repeat(-np.Inf, len(e_max))]
    pot_max_list = [np.repeat(np.Inf, len(e_max))]
    value_list = [np.Inf]
    
    # always run the maximum number of iterations
    for i in range(max_iter):
        
        # estimate Lipschitz constants and potential ranges
        lipschitz, pot_min, pot_max = batch_estimation(w_list,
                                                       b_list,
                                                       x_list,
                                                       x_ball,
                                                       e_ball)
        pot_min_list.append(pot_min)
        pot_max_list.append(pot_max)
        
        # compute the approximate cost coefficient
        cost = deg_1_relu_error(pot_min, pot_max)
        
        # compute the optimal allocation
        e, value = optimal_allocation(lipschitz, cost, e_ball, e_out)
        e_list.append(e)
        value_list.append(value)
        
        # neuron-specific convergence bounds
        inactive = e < (e_ball - e_prec) # inactive constraints e[i] <= e_ball[i]
        e_up[inactive] = np.minimum(e_up, e_ball)[inactive] # keep track of min ball
        e_low = np.maximum(e_low, e) # keep track of max allocation
        
        # reset the domain of validity:
        # this will (hopefully!) tighten the next batch estimation
        e_ball = (e_up + e_low) / 2
        
        # display progress info
        if verbose:
            print("ligar_cycle(): iter", i,
                  "val", value_list[-1],
                  "time", time.time() - start)
        
        # quit early if the objective function stop changing
        progress = value_list[-2] - value_list[-1]
        if progress <= value_prec:
            break
    
    # return the best error allocation found
    best = np.argmin(value_list)
    result = (e_list[best],
              pot_min_list[best],
              pot_max_list[best],
              value_list[best])
    
    return result

"""
Alternate LiGAR iteration
Arguments:
    w_list: minimum value of the gradient(s)
    b_list: maximum value of the gradient(s)
    x_list: list of (concrete) inputs
    x_ball: perturbation distance for each network input
    d_list: list of polynomial degree approximation (one per ReLU)
    e_guess: initial perturbation distance for each ReLU neuron
    max_iter: maximum number of LiGAR iterations
    e_prec: tolerance margin to determine constraint activity
    value_prec: absolute convergence threshold
    mode: select between
        "approx": uses linear approximation scaled by the required degree
        "chebyshev": computes a chebyshev approximation and exact error
    verbose: whether to display progress info
    include_time: whether to include the total time in the result
Returns:
    e_list[best]: maximum error for each ReLU approximation
    pot_min_list[best]: minimum approximation range for each ReLU
    pot_max_list[best]: maximum approximation range for each ReLU
    value_list[best]: value of the output error bound
    [time]: total time taken to compute the result (if include_time=True)
"""
def ligar_cycle_2(w_list, b_list, x_list, x_ball, d_list,
                  e_guess=0.01, max_iter=100, e_prec=1e-8, value_prec=1e-2,
                  mode="approx", verbose=False, include_time=False):
    
    start = time.time()
    
    # initialise (possibly unsound) approximation errors
    degrees = np.concatenate(d_list)
    e_ball = np.repeat(e_guess, len(degrees))
    
    # keep track of results across all iterations
    e_list = [e_ball]
    pot_min_list = [np.repeat(-np.Inf, len(e_ball))]
    pot_max_list = [np.repeat(np.Inf, len(e_ball))]
    value_list = [np.Inf]
    
    # always run the maximum number of iterations
    for i in range(max_iter):
        
        # estimate Lipschitz constants and potential ranges
        lipschitz, pot_min, pot_max = batch_estimation(w_list,
                                                       b_list,
                                                       x_list,
                                                       x_ball,
                                                       e_ball)
        pot_min_list.append(pot_min)
        pot_max_list.append(pot_max)
        
        # compute the new ReLU approximation errors
        if mode == "approx":
            cost = deg_1_relu_error(pot_min, pot_max)
            e = cost / degrees
        
        elif mode == "chebyshev":
            e = np.zeros(len(degrees))
            for j in range(len(degrees)):
                e[j] = cheby_relu_error(pot_min[j], pot_max[j], degrees[j])
        else:
            print("ligar_cycle_2(): unknown polynomial error mode")
            e = e_ball
        e_list.append(e)
        
        # compute the new output bound
        value = lipschitz @ e
        value_list.append(value)
        
        # quit early if the objective function stop changing
        progress = value_list[-2] - value_list[-1]
        if progress <= value_prec:
            break
        
        t_elapsed = time.time() - start
        # display progress info
        if verbose:
            print("ligar_cycle_2(): iter", i,
                  "val", value_list[-1],
                  "time", t_elapsed)
    
    # return the best error allocation found
    best = np.argmin(value_list)

    if include_time:
        result = (e_list[best],
                  pot_min_list[best],
                  pot_max_list[best],
                  value_list[best],
                  t_elapsed)
    else:
        result = (e_list[best],
                pot_min_list[best],
                pot_max_list[best],
                value_list[best])
    
    return result

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:52:41 2023

@author: Edoardo
"""

import numpy as np

from algorithms.ligar.ligar import batch_estimation, deg_1_relu_error
from algorithms.ligar.ligar import adjust_local_dual, adjust_global_dual
from algorithms.ligar.ligar import optimal_allocation, ligar_cycle

def BatchEstimationTest1(n_samples=50):
    
    # two-layer network with one output
    w_list = [np.array([[1,2], [1,-1]]),
              np.array([[2,-1], [1,-2]]),
              np.array([-1,3])]
    b_list = [np.array([-2,0]), np.array([-1,1]), np.array([-1])]
    
    x_list = [np.random.normal(0,1,2) for _ in range(n_samples)]
    
    lip_0, z_min_0, z_max_0 = batch_estimation(w_list,
                                               b_list,
                                               x_list,
                                               np.zeros(2),
                                               np.zeros(4))
    
    lip_1, z_min_1, z_max_1 = batch_estimation(w_list,
                                               b_list,
                                               x_list,
                                               np.repeat(0.1, 2),
                                               np.repeat(0.1, 4))
    
    # the concrete bounds must be tighter than the perturbed ones
    assert (lip_1 >= lip_0).all()
    assert (z_min_1 <= z_min_0).all()
    assert (z_max_1 >= z_max_0).all()

def BatchEstimationTest2(n_samples=50):
    
    # two-layer network with three outputs
    w_list = [np.array([[1,2], [1,-1]]),
              np.array([[2,-1], [1,-2]]),
              np.array([[-1,3],[0,-2],[1,1]])]
    b_list = [np.array([-2,0]), np.array([-1,1]), np.array([-1,-1,0])]
    
    x_list = [np.random.normal(0,1,2) for _ in range(n_samples)]
    
    lip_0, z_min_0, z_max_0 = batch_estimation(w_list,
                                               b_list,
                                               x_list,
                                               np.zeros(2),
                                               np.zeros(4))
    
    lip_1, z_min_1, z_max_1 = batch_estimation(w_list,
                                               b_list,
                                               x_list,
                                               np.repeat(0.1, 2),
                                               np.repeat(0.1, 4))
    
    # the concrete bounds must be tighter than the perturbed ones
    assert (lip_1 >= lip_0).all()
    assert (z_min_1 <= z_min_0).all()
    assert (z_max_1 >= z_max_0).all()

def Deg1ReluErrorTest1():
    
    # scale-invariant error
    assert np.isclose(deg_1_relu_error(-1, 0) * 2,
                      deg_1_relu_error(-2, 0))
    assert np.isclose(deg_1_relu_error(-1, 0.5) * 2,
                      deg_1_relu_error(-2, 1))
    assert np.isclose(deg_1_relu_error(-1, 1) * 2,
                      deg_1_relu_error(-2, 2))
    assert np.isclose(deg_1_relu_error(-0.5, 1) * 2,
                      deg_1_relu_error(-1, 2))
    assert np.isclose(deg_1_relu_error(-0, 1) * 2,
                      deg_1_relu_error(-0, 2))

def Deg1ReluErrorTest2():
    
    pot_min = np.array([-1,-1,  -1,-0.5,0])
    pot_max = np.array([ 0, 0.5, 1, 1,  1])
    
    error = deg_1_relu_error(pot_min, pot_max)

    assert np.allclose(error, np.array([0, 1/6, 1/4, 1/6, 0]))

def AdjustLocalDualTest():
    
    # challenging feasibility example
    lipschitz = np.array([1.59e+05, 2.37e-01, 1.35e-01, 3.38e-02, 6.50e+02,
                          6.10e+02, 1.23e+00, 4.25e-02, 1.36e-01, 2.21e+01])
    pot_min = np.array([-4.41e-02, -1.03e-02, -1.52e+02, -4.09e+01, -7.24e+06,
                        -6.95e-02, -1.86e+01, -5.60e+05, -3.60e+00, -1.48e+01])
    pot_max = np.array([5.60e+06, 1.12e+06, 8.06e+04, 7.54e+02, 3.08e+01,
                        2.26e+00, 1.41e-01, 3.22e+02, 2.28e-02, 1.36e+01])
    cost = deg_1_relu_error(pot_min, pot_max)
    e_ball = np.repeat(0.1, 10)
    
    # zero global dual
    d_global = 0
    d_local = adjust_local_dual(lipschitz, cost, e_ball, d_global)
    
    assert (d_local >= 0).all()
    assert np.allclose(d_local, cost / (e_ball * e_ball))
    assert (np.sqrt(cost / (lipschitz * d_global + d_local)) <= e_ball).all()
    
    # non-zero global dual
    d_global = 10
    d_local = adjust_local_dual(lipschitz, cost, e_ball, d_global)
    
    assert (d_local >= 0).all()
    assert (d_local >= cost / (e_ball * e_ball) - lipschitz * d_global).all()
    assert (np.sqrt(cost / (lipschitz * d_global + d_local)) <= e_ball).all()

def AdjustGlobalDualTest():
    
    # challenging feasibility example
    lipschitz = np.array([1.59e+05, 2.37e-01, 1.35e-01, 3.38e-02, 6.50e+02,
                          6.10e+02, 1.23e+00, 4.25e-02, 1.36e-01, 2.21e+01])
    pot_min = np.array([-4.41e-02, -1.03e-02, -1.52e+02, -4.09e+01, -7.24e+06,
                        -6.95e-02, -1.86e+01, -5.60e+05, -3.60e+00, -1.48e+01])
    pot_max = np.array([5.60e+06, 1.12e+06, 8.06e+04, 7.54e+02, 3.08e+01,
                        2.26e+00, 1.41e-01, 3.22e+02, 2.28e-02, 1.36e+01])
    cost = deg_1_relu_error(pot_min, pot_max)
    e_out = 0.1
    
    # zero local dual
    d_local = np.zeros(10)
    d_global = adjust_global_dual(lipschitz, cost, e_out, 1, d_local)
    
    assert d_global >= 0
    assert lipschitz @ np.sqrt(cost / (lipschitz * d_global + d_local)) <= e_out
    
    # non-zero local dual
    d_local = np.array([0, 0, 10000, 1000, 0, 0, 0, 10000, 0, 100])
    d_global = adjust_global_dual(lipschitz, cost, e_out, 1, d_local)
    
    assert d_global >= 0
    assert lipschitz @ np.sqrt(cost / (lipschitz * d_global + d_local)) <= e_out

def OptimalAllocationTest1():
    
    # challenging feasibility example
    lipschitz = np.array([1.59e+05, 2.37e-01, 1.35e-01, 3.38e-02, 6.50e+02,
                          6.10e+02, 1.23e+00, 4.25e-02, 1.36e-01, 2.21e+01])
    pot_min = np.array([-4.41e-02, -1.03e-02, -1.52e+02, -4.09e+01, -7.24e+06,
                        -6.95e-02, -1.86e+01, -5.60e+05, -3.60e+00, -1.48e+01])
    pot_max = np.array([5.60e+06, 1.12e+06, 8.06e+04, 7.54e+02, 3.08e+01,
                        2.26e+00, 1.41e-01, 3.22e+02, 2.28e-02, 1.36e+01])
    cost = deg_1_relu_error(pot_min, pot_max)
    e_ball = np.repeat(0.1, 10)
    e_out = 0.1
    
    x, value = optimal_allocation(lipschitz, cost, e_ball, e_out)
    
    assert (x >= 0).all()
    assert (x <= e_ball).all()
    assert lipschitz @ x <= e_out

def optimalAllocationTest2(n_try=10, n_vars=1000):
    
    for _ in range(n_try):
        
        # make the optimization parameters vary wildly
        lipschitz = 1 / (np.random.normal(0, 1, n_vars) ** 10)
        cost = 1 / (np.random.normal(0, 1, n_vars) ** 10)
        e_ball = np.repeat(0.01, n_vars)
        e_out = 0.01
        
        x, value = optimal_allocation(lipschitz, cost, e_ball, e_out)
        
        assert (x >= 0).all()
        assert (x <= e_ball).all()
        assert lipschitz @ x <= e_out

def nn_infer(w_list, b_list, x, e):
    
    # first affine transformation
    y = w_list[0] @ x + b_list[0]
    
    for W, b in zip(w_list[1:], b_list[1:]):
        
        # ReLU activations
        y[y < 0] = 0
        
        # ReLU approximation simulation
        y += e[:len(y)]
        e = e[len(y):]
        
        # next affine transformation
        y = W @ y + b
    
    return y

def LigarCycleTest1(n_samples=50):
    
    # two-layer network with three outputs
    w_list = [np.array([[1,2], [1,-1]]),
              np.array([[2,-1], [1,-2]]),
              np.array([[-1,3],[0,-2],[1,1]])]
    b_list = [np.array([-2,0]), np.array([-1,1]), np.array([-1,-1,0])]
    x_list = [np.random.normal(0,1,2) for _ in range(n_samples)]
    
    x_ball = np.repeat(0.1, 2)
    e_max = np.repeat(0.1, 4)
    e_out = 0.01
    
    e_ligar, _, _, _ = ligar_cycle(w_list, b_list, x_list, x_ball, e_max, e_out)
    
    # compare original and approximated network
    # on each input in the "training" set
    for x in x_list:
        
        e = (2 * np.random.rand(len(e_ligar)) - 1) * e_ligar
        y_approx = nn_infer(w_list, b_list, x, e)
        y_orig = nn_infer(w_list, b_list, x, np.zeros(len(e)))
        
        assert np.max(np.abs(y_orig - y_approx)) <= e_out

def LigarCycleTest2(n_hidden=3, width=10, n_samples=50):
    
    # extract normally-distributed weights, biases and "training" inputs
    # assume inputs, outputs and width are the same
    w_list = [np.random.normal(0, 1, [width,width]) for _ in range(n_hidden+1)]
    b_list = [np.random.normal(0, 1, width) for _ in range(n_hidden+1)]
    x_list = [np.random.normal(0, 1, width) for _ in range(n_samples)]
    
    # non-zero input ball
    x_ball = np.repeat(0.1, width)
    e_max = np.repeat(0.1, width * n_hidden)
    e_out = 0.01
    
    e_ligar, _, _, _ = ligar_cycle(w_list, b_list, x_list, x_ball, e_max, e_out)
    
    # compare original and approximated network
    # on each input in the "training" set
    for x in x_list:
        
        e = (2 * np.random.rand(len(e_ligar)) - 1) * e_ligar
        y_approx = nn_infer(w_list, b_list, x, e)
        y_orig = nn_infer(w_list, b_list, x, np.zeros(len(e)))
        
        assert np.max(np.abs(y_orig - y_approx)) <= e_out

def LigarCycleTest3(n_hidden=3, width=10, n_samples=50):
    
    # extract normally-distributed weights, biases and "training" inputs
    # assume inputs, outputs and width are the same
    w_list = [np.random.normal(0, 1, [width,width]) for _ in range(n_hidden+1)]
    b_list = [np.random.normal(0, 1, width) for _ in range(n_hidden+1)]
    x_list = [np.random.normal(0, 1, width) for _ in range(n_samples)]
    
    # perturbation ball only
    x_ball = np.zeros(width)
    e_max = np.repeat(0.1, width * n_hidden)
    e_out = 0.01
    
    e_ligar, _, _, _ = ligar_cycle(w_list, b_list, x_list, x_ball, e_max, e_out)
    
    # compare original and approximated network
    # on each input in the "training" set
    for x in x_list:
        
        e = (2 * np.random.rand(len(e_ligar)) - 1) * e_ligar
        y_approx = nn_infer(w_list, b_list, x, e)
        y_orig = nn_infer(w_list, b_list, x, np.zeros(len(e)))
        
        assert np.max(np.abs(y_orig - y_approx)) <= e_out

def runTests():
    
    print(">>> testing module ligar.ligar.py")
    
    BatchEstimationTest1()
    BatchEstimationTest2()
    
    Deg1ReluErrorTest1()
    Deg1ReluErrorTest2()
    
    AdjustLocalDualTest()
    AdjustGlobalDualTest()
    
    OptimalAllocationTest1()
    optimalAllocationTest2()
    
    LigarCycleTest1()
    LigarCycleTest2()
    LigarCycleTest3()
    
    print(">>> done!")

if __name__ == "__main__":
    runTests()

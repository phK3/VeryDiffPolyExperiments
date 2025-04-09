# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:43:00 2023

@author: Edoardo
"""

import numpy as np

from algorithms.jacobian.reachability import Affine
from algorithms.jacobian.jacobian import StepBound, BackwardLayer, BackwardNetwork

def StepBoundTest1():
    
    lower, upper = StepBound(1, 2)
    
    assert np.isclose(lower.W, 0)
    assert np.isclose(upper.W, 1)
    assert np.isclose(lower.b, 0)
    assert np.isclose(upper.b, 0)

    
def StepBoundTest2():
    
    lower, upper = StepBound(-4.5, -1)
    
    assert np.isclose(lower.W, 1)
    assert np.isclose(upper.W, 0)
    assert np.isclose(lower.b, 0)
    assert np.isclose(upper.b, 0)

def StepBoundTest3():
    
    lower, upper = StepBound(-0.5, 0.5)
    
    assert np.isclose(lower.W, 0.5)
    assert np.isclose(upper.W, 0.5)
    assert np.isclose(lower.b, -0.25)
    assert np.isclose(upper.b, 0.25)

def StepBoundTest4():
    
    lower, upper = StepBound(-1, 3)
    
    assert np.isclose(lower.W, 0.25)
    assert np.isclose(upper.W, 0.75)
    assert np.isclose(lower.b, -0.75)
    assert np.isclose(upper.b, 0.75)

def StepBoundTest5():
    
    out_min = [1, -4.5, -0.5, -1]
    out_max = [2, -1, 0.5, 3]
    lower, upper = StepBound(out_min, out_max)
    
    assert np.allclose(lower.W, [0, 1, 0.5, 0.25])
    assert np.allclose(upper.W, [1, 0, 0.5, 0.75])
    assert np.allclose(lower.b, [0, 0, -0.25, -0.75])
    assert np.allclose(upper.b, [0, 0, 0.25, 0.75])

def StepBoundTest6():
    
    out_min = np.array([[1, -4.5, -0.5, -1],
                        [-0.5, -1, 1, -4.5]])
    out_max = np.array([[2, -1, 0.5, 3],
                        [0.5, 3, 2, -1]])
    lower, upper = StepBound(out_min, out_max)
    
    assert np.allclose(lower.W, np.array([[0, 1, 0.5, 0.25],
                                          [0.5, 0.25, 0, 1]]))
    assert np.allclose(upper.W, np.array([[1, 0, 0.5, 0.75],
                                          [0.5, 0.75, 1, 0]]))
    assert np.allclose(lower.b, np.array([[0, 0, -0.25, -0.75],
                                          [-0.25, -0.75, 0, 0]]))
    assert np.allclose(upper.b, np.array([[0, 0, 0.25, 0.75],
                                          [0.25, 0.75, 0, 0]]))

def BackwardLayerTest1():
    
    # check correct output bounds 
    params = Affine(np.eye(4), np.zeros(4))
    lower_out = Affine(np.array([[1, -4.5, -0.5, -1],
                                 [-0.5, -1, 1, -4.5]]),
                       np.array([[0, 0.5, -0.5, -2],
                                 [-3, -0.5, 2, -1.5]]))
    upper_out = Affine(np.array([[2, -1, 0.5, 3],
                                 [0.5, 3, 2, -1]]),
                       np.array([[0, 0.5, -0.5, -2],
                                 [-3, -0.5, 2, -1.5]]))
    act_state = np.array([2, 2, 2, 2], dtype=int)
    
    layer = BackwardLayer(params, upper_out, lower_out, act_state)
    
    assert np.allclose(layer.out_min, np.array([[1, -4, -1, -3],
                                                [-3.5, -1.5, 3, -6]]))
    assert np.allclose(layer.out_max, np.array([[2, -0.5, 0, 1],
                                                [-2.5, 2.5, 4, -2.5]]))

def BackwardLayerTest2():
    
    # check correct ReLU bounds (uncertain activations)
    params = Affine(np.eye(4), np.zeros(4))
    lower_out = Affine(np.array([[1, -4.5, -0.5, -1],
                                 [-0.5, -1, 1, -4.5]]),
                       np.zeros([2,4]))
    upper_out = Affine(np.array([[2, -1, 0.5, 3],
                                 [0.5, 3, 2, -1]]),
                       np.zeros([2,4]))
    act_state = np.array([2, 2, 2, 2], dtype=int)
    
    layer = BackwardLayer(params, upper_out, lower_out, act_state)
    
    assert np.allclose(layer.lower_in.W, np.array([[0, -4.5, -0.25, -0.25],
                                                   [-0.25, -0.25, 0, -4.5]]))
    assert np.allclose(layer.upper_in.W, np.array([[2, 0, 0.25, 2.25],
                                                   [0.25, 2.25, 2, 0]]))
    assert np.allclose(layer.lower_in.b, np.array([[0, 0, -0.25, -0.75],
                                                   [-0.25, -0.75, 0, 0]]))
    assert np.allclose(layer.upper_in.b, np.array([[0, 0, 0.25, 0.75],
                                                   [0.25, 0.75, 0, 0]]))

def BackwardLayerTest3():
    
    # check correct ReLU bounds (mixed activations)
    params = Affine(np.eye(4), np.zeros(4))
    lower_out = Affine(np.array([[1, -4.5, -0.5, -1],
                                 [-0.5, -1, 1, -4.5]]),
                       np.zeros([2,4]))
    upper_out = Affine(np.array([[2, -1, 0.5, 3],
                                 [0.5, 3, 2, -1]]),
                       np.zeros([2,4]))
    act_state = np.array([1, 2, 0, 0], dtype=int)
    
    layer = BackwardLayer(params, upper_out, lower_out, act_state)
    
    assert np.allclose(layer.lower_in.W, np.array([[1, -4.5, 0, 0],
                                                   [-0.5, -0.25, 0, 0]]))
    assert np.allclose(layer.upper_in.W, np.array([[2, 0, 0, 0],
                                                   [0.5, 2.25, 0, 0]]))
    assert np.allclose(layer.lower_in.b, np.array([[0, 0, 0, 0],
                                                   [0, -0.75, 0, 0]]))
    assert np.allclose(layer.upper_in.b, np.array([[0, 0, 0, 0],
                                                   [0, 0.75, 0, 0]]))

def BackwardLayerTest4():
    
    # check full layer on a simple example
    params = Affine(np.array([[1,0],
                              [-1,0],
                              [0,-1],
                              [0,1]]),
                    np.ones(4))
    lower_out = Affine(np.array([[1, -4.5, -0.5, -1],
                                 [-0.5, -1, 1, -4.5]]),
                       np.zeros([2,4]))
    upper_out = Affine(np.array([[2, -1, 0.5, 3],
                                 [0.5, 3, 2, -1]]),
                       np.zeros([2,4]))
    act_state = np.array([1, 2, 0, 2], dtype=int)
    
    layer = BackwardLayer(params, upper_out, lower_out, act_state)
    
    assert np.allclose(layer.lower_in.W, np.array([[1, -0.25],
                                                   [-2.75, -4.5]]))
    assert np.allclose(layer.upper_in.W, np.array([[6.5, 2.25],
                                                   [0.75, 0]]))
    assert np.allclose(layer.lower_in.b, np.array([[0, -0.75],
                                                   [-0.75, 0]]))
    assert np.allclose(layer.upper_in.b, np.array([[0, 0.75],
                                                   [0.75, 0]]))

def BackwardNetworkTest1():
    
    # check the backward layers (simple example)
    w_list = [np.eye(2), np.array([1, 2])]
    b_list = [np.zeros(2), np.array([-1])]
    x_range = np.array([[1,2], [-1,1]])
    e_range = np.array([[-0.5,0.5], [1,1.5]])
    
    net = BackwardNetwork(w_list, b_list, x_range, e_range)
    layer_list = net.backward_layer_list
    
    assert len(layer_list) == 1
    assert (layer_list[0].act_state == np.array([1, 2], dtype=int)).all()
    
    assert np.allclose(layer_list[0].lower_in.W, np.array([1, 0]))
    assert np.allclose(layer_list[0].upper_in.W, np.array([1, 2]))
    assert np.allclose(layer_list[0].lower_in.b, np.zeros(2))
    assert np.allclose(layer_list[0].upper_in.b, np.zeros(2))

def BackwardNetworkTest2():
    
    # check the get_symbolic_jacobians() function (simple example)
    w_list = [np.eye(2), np.array([1, 2])]
    b_list = [np.zeros(2), np.array([-1])]
    x_range = np.array([[1,2], [-1,1]])
    e_range = np.array([[-0.5,0.5], [1,1.5]])
    
    net = BackwardNetwork(w_list, b_list, x_range, e_range)
    lower_jacs, upper_jacs = net.get_symbolic_jacobians()
    
    assert np.allclose(lower_jacs[1].W, np.array([1, 2]))
    assert np.allclose(upper_jacs[1].W, np.array([1, 2]))
    assert np.allclose(lower_jacs[1].b, np.zeros(2))
    assert np.allclose(upper_jacs[1].b, np.zeros(2))
    
    assert np.allclose(lower_jacs[0].W, np.array([1, 0]))
    assert np.allclose(upper_jacs[0].W, np.array([1, 2]))
    assert np.allclose(lower_jacs[0].b, np.zeros(2))
    assert np.allclose(upper_jacs[0].b, np.zeros(2))

def BackwardNetworkTest3():
    
    # check the get_symbolic_jacobians() function (deeper network)
    w_list = [np.array([[1,2], [1,-1]]),
              np.array([[2,-1], [1,-2]]),
              np.array([-1,3])]
    b_list = [np.array([-2,0]), np.array([-1,1]), np.array([-1])]
    
    # make the ranges so large that all ReLUs will be in uncertain state
    x_range = np.array([[-3,3], [-4,4]])
    e_range = np.array([[-2,2], [-2,2], [-2,2], [-2,2]])
    
    net = BackwardNetwork(w_list, b_list, x_range, e_range)
    lower_jacs, upper_jacs = net.get_symbolic_jacobians()
    
    assert np.allclose(lower_jacs[2].W, np.array([-1,3]))
    assert np.allclose(upper_jacs[2].W, np.array([-1,3]))
    assert np.allclose(lower_jacs[2].b, np.zeros(2))
    assert np.allclose(upper_jacs[2].b, np.zeros(2))
    
    assert np.allclose(lower_jacs[1].W, np.array([-2,-6]))
    assert np.allclose(upper_jacs[1].W, np.array([3,1]))
    assert np.allclose(lower_jacs[1].b, np.zeros(2))
    assert np.allclose(upper_jacs[1].b, np.zeros(2))
    
    assert np.allclose(lower_jacs[0].W, np.array([-208/35, -61/35]))
    assert np.allclose(upper_jacs[0].W, np.array([68/35, 306/35]))
    assert np.allclose(lower_jacs[0].b, np.array([-72/35, -114/35]))
    assert np.allclose(upper_jacs[0].b, np.array([72/35, 114/35]))

def BackwardNetworkTest4():
    
    # check the get_concrete_jacobians() function (deeper network)
    w_list = [np.array([[1,2], [1,-1]]),
              np.array([[2,-1], [1,-2]]),
              np.array([-1,3])]
    b_list = [np.array([-2,0]), np.array([-1,1]), np.array([-1])]
    
    # make the ranges so large that all ReLUs will be in uncertain state
    x_range = np.array([[-3,3], [-4,4]])
    e_range = np.array([[-2,2], [-2,2], [-2,2], [-2,2]])
    
    net = BackwardNetwork(w_list, b_list, x_range, e_range)
    lower_jacs, upper_jacs = net.get_concrete_jacobians()
    
    assert np.allclose(lower_jacs[2], np.array([-1,3]))
    assert np.allclose(upper_jacs[2], np.array([-1,3]))
    
    assert np.allclose(lower_jacs[1], np.array([-2,-6]))
    assert np.allclose(upper_jacs[1], np.array([3,1]))
    
    assert np.allclose(lower_jacs[0], np.array([-8,-5]))
    assert np.allclose(upper_jacs[0], np.array([4,12]))

def BackwardNetworkTest5(n_try=50):
    
    w_list = [np.array([[1,2], [1,-1]]),
              np.array([[2,-1], [1,-2]]),
              np.array([-1,3])]
    b_list = [np.array([-2,0]), np.array([-1,1]), np.array([-1])]
    
    # extract random degenerate input intervals
    for _ in range(n_try):
        x = np.random.normal(0,1,2)
        e = np.random.normal(0,1,4)
        x_range = np.column_stack([x,x])
        e_range = np.column_stack([e,e])
        
        net = BackwardNetwork(w_list, b_list, x_range, e_range)
        lower_jacs, upper_jacs = net.get_concrete_jacobians()
        
        for low_j, up_j in zip(lower_jacs, upper_jacs):
            assert np.allclose(low_j, up_j)

def BackwardNetworkTest6(n_try=50):
    
    w_list = [np.array([[1,2], [1,-1]]),
              np.array([[2,-1], [1,-2]]),
              np.array([-1,3])]
    b_list = [np.array([-2,0]), np.array([-1,1]), np.array([-1])]
    
    # extract random non-degenerate input intervals
    for _ in range(n_try):
        x = np.random.normal(0,1,2)
        e = np.random.normal(0,1,4)
        x_range = np.column_stack([x,x])
        e_range = np.column_stack([e,e])
        
        net = BackwardNetwork(w_list, b_list, x_range, e_range)
        lower_jacs, upper_jacs = net.get_concrete_jacobians()
        
        for low_j, up_j in zip(lower_jacs, upper_jacs):
            assert (low_j <= up_j).all()

def runTests():
    
    print(">>> testing module jacobian.jacobian.py")
    
    StepBoundTest1()
    StepBoundTest2()
    StepBoundTest3()
    StepBoundTest4()
    StepBoundTest5()
    StepBoundTest6()
    
    BackwardLayerTest1()
    BackwardLayerTest2()
    BackwardLayerTest3()
    BackwardLayerTest4()
    
    BackwardNetworkTest1()
    BackwardNetworkTest2()
    BackwardNetworkTest3()
    BackwardNetworkTest4()
    BackwardNetworkTest5()
    BackwardNetworkTest6()
    
    print(">>> done!")

if __name__ == "__main__":
    runTests()

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:03:38 2023

@author: Edoardo
"""

import numpy as np

from algorithms.jacobian.reachability import Affine, MatrixSignSplit, ReluBound
from algorithms.jacobian.reachability import ForwardLayer, ForwardNetwork

def MatrixSignSplitTest():
    
    M = np.array([[1, 2, -3],
                  [0, 0, 1],
                  [-1, -8, 6]])
    
    mPos, mNeg = MatrixSignSplit(M)
    
    assert (mPos == np.array([[1,2,0],[0,0,1],[0,0,6]])).all(), "Wrong positive matrix"
    assert (mNeg == np.array([[0,0,-3],[0,0,0],[-1,-8,0]])).all(), "Wrong negative matrix"

def ReluBoundTest1():
    
    lower, upper = ReluBound(1, 2)
    
    assert np.isclose(lower.W, 1)
    assert np.isclose(upper.W, 1)
    assert np.isclose(lower.b, 0)
    assert np.isclose(upper.b, 0)

    
def ReluBoundTest2():
    
    lower, upper = ReluBound(-4.5, -1)
    
    assert np.isclose(lower.W, 0)
    assert np.isclose(upper.W, 0)
    assert np.isclose(lower.b, 0)
    assert np.isclose(upper.b, 0)

def ReluBoundTest3():
    
    lower, upper = ReluBound(-0.5, 0.5)
    
    assert np.isclose(lower.W, 0.5)
    assert np.isclose(upper.W, 0.5)
    assert np.isclose(lower.b, 0)
    assert np.isclose(upper.b, 0.25)

def ReluBoundTest4():
    
    lower, upper = ReluBound(-1, 3)
    
    assert np.isclose(lower.W, 0.75)
    assert np.isclose(upper.W, 0.75)
    assert np.isclose(lower.b, 0)
    assert np.isclose(upper.b, 0.75)

def ReluBoundTest5():
    
    pot_min = [1, -4.5, -0.5, -1]
    pot_max = [2, -1, 0.5, 3]
    lower, upper = ReluBound(pot_min, pot_max)
    
    assert np.allclose(lower.W, [1, 0, 0.5, 0.75])
    assert np.allclose(upper.W, [1, 0, 0.5, 0.75])
    assert np.allclose(lower.b, [0, 0, 0, 0])
    assert np.allclose(upper.b, [0, 0, 0.25, 0.75])

def ForwardLayerTest1():
    
    # check correct ReLU bounds (identity weights)
    params = Affine(np.eye(4), np.zeros(4))
    upper_in = Affine(np.eye(4), np.zeros(4))
    lower_in = Affine(np.eye(4), np.zeros(4))
    x_range = np.array([[1,2], [-4.5,-1], [-0.5,0.5], [-1,3]])
    
    layer = ForwardLayer(params, upper_in, lower_in, x_range, act_func="ReLU")
    
    assert np.allclose(layer.lower_out.W, np.diag([1, 0, 0.5, 0.75]))
    assert np.allclose(layer.upper_out.W, np.diag([1, 0, 0.5, 0.75]))
    assert np.allclose(layer.lower_out.b, [0, 0, 0, 0])
    assert np.allclose(layer.upper_out.b, [0, 0, 0.25, 0.75])

def ForwardLayerTest2():
    
    # check correct output potentials (negative weights)
    params = Affine(-np.eye(4), np.zeros(4))
    upper_in = Affine(np.eye(4), np.zeros(4))
    lower_in = Affine(np.eye(4), np.zeros(4))
    x_range = np.array([[-2,-1], [1,4.5], [-0.5,0.5], [-3,1]])
    
    layer = ForwardLayer(params, upper_in, lower_in, x_range, act_func="ReLU")
    
    assert np.allclose(layer.lower_out.W, np.diag([-1, 0, -0.5, -0.75]))
    assert np.allclose(layer.upper_out.W, np.diag([-1, 0, -0.5, -0.75]))
    assert np.allclose(layer.lower_out.b, [0, 0, 0, 0])
    assert np.allclose(layer.upper_out.b, [0, 0, 0.25, 0.75])

def ForwardLayerTest3():
    
    # check correct upper/lower bounds on potential (linear)
    params = Affine(np.array([[1,2,0],
                              [0,-1,2],
                              [2,0,-1]]), [-1,-1,0])
    upper_in = Affine(np.array([[1,2],
                                [0,1],
                                [1,0]]), [1,0,1])
    lower_in = Affine(np.array([[1,1],
                                [0,-1],
                                [1,-1]]), [1,-1,0])
    x_range = np.zeros([2,2])
    
    layer = ForwardLayer(params, upper_in, lower_in, x_range, act_func=None)
    
    assert np.allclose(layer.lower_out.W, np.array([[1,-1],
                                                    [2,-3],
                                                    [1,2]]))
    assert np.allclose(layer.upper_out.W, np.array([[1,4],
                                                    [2,1],
                                                    [1,5]]))
    assert np.allclose(layer.lower_out.b, [-2, -1, 1])
    assert np.allclose(layer.upper_out.b, [0, 2, 2])

def ForwardLayerTest4():
    
    # check correct min/max range on potentials (linear)
    params = Affine(np.array([[1,2,0],
                              [0,-1,2],
                              [2,0,-1]]), [-1,-1,0])
    upper_in = Affine(np.array([[1,2],
                                [0,1],
                                [1,0]]), [1,0,1])
    lower_in = Affine(np.array([[1,1],
                                [0,-1],
                                [1,-1]]), [1,-1,0])
    x_range = np.array([[0,1], [-2,1]])
    
    layer = ForwardLayer(params, upper_in, lower_in, x_range, act_func=None)
    
    assert np.allclose(layer.pot_min, [-3, -4, -3])
    assert np.allclose(layer.pot_max, [5, 5, 8])

def ForwardLayerTest5():
    
    # check full layer on a simple example
    params = Affine(np.array([[1,2,0],
                              [0,-1,2],
                              [2,0,-1]]), [-1,-1,0])
    upper_in = Affine(np.array([[1,2],
                                [0,1],
                                [1,0]]), [1,0,1])
    lower_in = Affine(np.array([[1,1],
                                [0,-1],
                                [1,-1]]), [1,-1,0])
    x_range = np.array([[0,1], [-2,1]])
    
    layer = ForwardLayer(params, upper_in, lower_in, x_range, act_func="ReLU")
    
    # slopes=[5/8,5/9,8/11]
    # offset=[15/8,20/9,24/11]
    assert np.allclose(layer.lower_out.W, np.array([[5/8,-5/8],
                                                    [10/9,-15/9],
                                                    [8/11,16/11]]))
    assert np.allclose(layer.upper_out.W, np.array([[5/8,5/2],
                                                    [10/9,5/9],
                                                    [8/11,40/11]]))
    assert np.allclose(layer.lower_out.b, [-5/4, -5/9, 8/11])
    assert np.allclose(layer.upper_out.b, [15/8, 30/9, 40/11])

def ForwardNetworkTest1():
    
    w_list = [np.eye(2), np.array([1, 2])]
    b_list = [np.zeros(2), np.array([-1])]
    x_range = np.array([[1,2], [-1,1]])
    e_range = np.array([[-0.5,0.5], [1,1.5]])
    
    net = ForwardNetwork(w_list, b_list, x_range, e_range)
    layer_list = net.forward_layer_list
    
    for i, layer in enumerate(layer_list):
        assert np.allclose(layer.params.W, w_list[i])
        assert np.allclose(layer.params.b, b_list[i])
    
    assert np.allclose(layer_list[0].pot_min, np.array([1,-1]))
    assert np.allclose(layer_list[0].pot_max, np.array([2,1]))
    assert np.allclose(layer_list[0].lower_out.W, np.array([[1,0], [0,0.5]]))
    assert np.allclose(layer_list[0].upper_out.W, np.array([[1,0], [0,0.5]]))
    assert np.allclose(layer_list[0].lower_out.b, np.array([0,0]))
    assert np.allclose(layer_list[0].upper_out.b, np.array([0,0.5]))
    
    assert np.allclose(layer_list[1].pot_min, np.array([0.5]))
    assert np.allclose(layer_list[1].pot_max, np.array([6.5]))
    assert np.allclose(layer_list[1].lower_out.W, np.array([1,1,1,2]))
    assert np.allclose(layer_list[1].upper_out.W, np.array([1,1,1,2]))
    assert np.allclose(layer_list[1].lower_out.b, np.array([-1]))
    assert np.allclose(layer_list[1].upper_out.b, np.array([0]))

def ForwardNetworkTest2(n_try=50):
    
    w_list = [np.array([[-1,1], [2,1]]),
              np.array([[-2,1], [-1,-1]]),
              np.array([-1,3])]
    b_list = [np.array([-1,-1]),
              np.array([-1,1]),
              np.array([-2])]
    
    # extract random degenerate input intervals
    for _ in range(n_try):
        x = np.random.normal(0,1,2)
        e = np.random.normal(0,1,4)
        x_range = np.column_stack([x,x])
        e_range = np.column_stack([e,e])
        
        net = ForwardNetwork(w_list, b_list, x_range, e_range)
        
        assert np.allclose(net.forward_layer_list[-1].pot_min,
                           net.forward_layer_list[-1].pot_max)

def ForwardNetworkTest3(n_try=50):
    
    w_list = [np.array([[-1,1], [2,1]]),
              np.array([[-2,1], [-1,-1]]),
              np.array([-1,3])]
    b_list = [np.array([-1,-1]),
              np.array([-1,1]),
              np.array([-2])]
    
    # extract random non-degenerate input intervals
    for _ in range(n_try):
        x_range = np.sort(np.random.normal(0,1,[2,2]), axis=1)
        e_range = np.sort(np.random.normal(0,1,[4,2]), axis=1)

        net = ForwardNetwork(w_list, b_list, x_range, e_range)
        
        assert (net.forward_layer_list[-1].pot_min <=
                net.forward_layer_list[-1].pot_max).all()

def runTests():
    
    print(">>> testing module jacobian.reachability.py")
    
    MatrixSignSplitTest()
    
    ReluBoundTest1()
    ReluBoundTest2()
    ReluBoundTest3()
    ReluBoundTest4()
    ReluBoundTest5()
    
    ForwardLayerTest1()
    ForwardLayerTest2()
    ForwardLayerTest3()
    ForwardLayerTest4()
    ForwardLayerTest5()
    
    ForwardNetworkTest1()
    ForwardNetworkTest2()
    ForwardNetworkTest3()
    
    print(">>> done!")

if __name__ == "__main__":
    runTests()


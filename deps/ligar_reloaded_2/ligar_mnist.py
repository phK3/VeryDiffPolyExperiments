# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:38:41 2025

@author: Edoardo
"""

import onnx
import numpy as np
import matplotlib.pyplot as plt

from algorithms.ligar.ligar_2 import ligar_design, ligar_output_bound

# read model
model = onnx.load('mnist_256x4_2e5.onnx')
params = model.graph.initializer
n_layers = int(len(params) / 2)

# extract parameters
w_list = []
b_list = []
for i in range(n_layers):
    w_list.append(onnx.numpy_helper.to_array(params[2 * i + 0]))
    b_list.append(onnx.numpy_helper.to_array(params[2 * i + 1]))

# choose the polynomial error approximation mode
#mode = "approx"
mode = "chebyshev"

out_b = []

# repeat the analysis for different polynomial degrees
d_scan = np.concatenate([np.arange(1,10), np.arange(10,201,10)])
for degree in d_scan:
    
    # LiGAR allows for custom degrees for each ReLU
    # set them all to the same value :-)
    d_list = [np.repeat(degree, len(b_list[i]))
              for i in range(n_layers-1)]
    
    # PHASE I: design the polynomial network
    
    # entire input space
    x_range = np.column_stack([np.zeros(784), np.ones(784)])
    
    # run a FastLin-style forward pass (parallel linear bounds)
    # we only care about the approximation error at each ReLU
    _, _, e_ball = ligar_design(w_list, b_list, x_range, d_list, mode=mode)
    
    # PHASE II: compute the output error bound
    
    # OPTION I: entire input space
    x_list = [np.repeat(0.5, 784)]
    x_ball = np.repeat(0.5, 784)
    
    # OPTION II: random input balls
    # n_balls = 10
    # ball_radius = 0.05
    # x_list = [np.random.rand(784) for _ in range(n_balls)]
    # x_ball = np.repeat(ball_radius, 784)
    
    # LiGAR-style Lipschitz estimation
    b = ligar_output_bound(w_list, b_list, x_list, x_ball, e_ball)
    out_b.append(b)
    
    # make the anxious user happy
    print("Degree =", degree, "| Output Bound =", b)

# save results on file
filename = "LiGAR_mode_" + mode + ".csv"
np.savetxt(filename, np.column_stack([d_scan, out_b]), delimiter=",")

# plot results
plt.figure()
plt.semilogy(d_scan, out_b, "-k")
plt.grid()
plt.show()

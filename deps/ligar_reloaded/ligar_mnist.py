# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:38:41 2025

@author: Edoardo
"""

import onnx
import numpy as np
import matplotlib.pyplot as plt

from algorithms.ligar.ligar import ligar_cycle_2

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

# OPTION I: entire input space
eps_in = 1.0
x_ball = np.repeat(eps_in, 784)
x_list = [np.repeat(0.5, 784)]
print("Entire Input Space [0,1]^784")
print("")

# OPTION II: random input balls
# eps_in = 0.1
# n_fuzz = 1
# x_ball = np.repeat(eps_in, 784)
# x_list = [np.random.rand(784) for _ in range(n_fuzz)]
# print("Number of Random Inputs:", n_fuzz)
# print("Size of Input Balls:", eps_in)
# print("")

# granularity of polynomial degrees
d_scan = np.concatenate([np.arange(1,10), np.arange(10,201,10)])

# compute output error for each degree
out_b = []
for degree in d_scan:
    d_list = [np.repeat(degree, len(b)) for b in b_list[:-1]]
    print("Computing with degree", degree)
    
    # OPTION III: choose mode = "approx" for approximate error bounds
    # OPTION IV: choose mode = "chebyshev" for exact error bounds
    result = ligar_cycle_2(w_list, b_list, x_list, x_ball, d_list,
                           mode="chebyshev", verbose=True)
    e_approx, pot_min, pot_max, out_bound = result
    out_b.append(out_bound)

# save results on file
np.savetxt("results.txt", np.column_stack([d_scan, out_b]), delimiter=",")

# plot results
plt.figure()
plt.semilogy(d_scan, out_b, "-k")
plt.grid()
plt.show()

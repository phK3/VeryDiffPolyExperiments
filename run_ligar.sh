#!/bin/bash

LIGAR_PATH="deps/ligar_reloaded_2/ligar_main.py"
EPS=0.5

# run LIGAR on MNIST with approximate formula for error
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_approx_global.csv --dataset mnist --eps_in $EPS 
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_approx_global.csv --dataset mnist --eps_in $EPS  
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_approx_global.csv --dataset mnist --eps_in $EPS 
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_approx_global.csv --dataset mnist --eps_in $EPS  
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_approx_global.csv --dataset mnist --eps_in $EPS  
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_approx_global.csv --dataset mnist --eps_in $EPS  

# run LIGAR on MNIST with concrete Chebyshev approximation for error computation
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev 
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev 
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev 
#taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev

# run LIGAR on HELOC with approximate formula for error
#taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc.onnx --output results/heloc/ligar_heloc_approx_global.csv --dataset heloc --eps_in $EPS
#taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc_2e5.onnx --output results/heloc/ligar_heloc_2e5_approx_global.csv --dataset heloc --eps_in $EPS 

# run LIGAR on HELOC with concrete Chebyshev approximation for error computation
#taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc.onnx --output results/heloc/ligar_heloc_cheby_global.csv --dataset heloc --eps_in $EPS --chebyshev
#taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc_2e5.onnx --output results/heloc/ligar_heloc_2e5_cheby_global.csv --dataset heloc --eps_in $EPS --chebyshev

# HAR is normalized to [-1, 1]
#EPS=1

# run LIGAR on HAR with approximate formula for error
#taskset -c 0 python "$LIGAR_PATH" --model networks/har/har.onnx --output results/har/ligar_har_approx_global.csv --dataset har --eps_in $EPS
#taskset -c 0 python "$LIGAR_PATH" --model networks/har/har_2e5.onnx --output results/har/ligar_har_2e5_approx_global.csv --dataset har --eps_in $EPS 
#taskset -c 0 python "$LIGAR_PATH" --model networks/har/har_1e4.onnx --output results/har/ligar_har_1e4_approx_global.csv --dataset har --eps_in $EPS 

# run LIGAR on HAR with concrete Chebyshev approximation for error computation
#taskset -c 0 python "$LIGAR_PATH" --model networks/har/har.onnx --output results/har/ligar_har_cheby_global.csv --dataset har --eps_in $EPS --chebyshev
#taskset -c 0 python "$LIGAR_PATH" --model networks/har/har_2e5.onnx --output results/har/ligar_har_2e5_cheby_global.csv --dataset har --eps_in $EPS --chebyshev
#taskset -c 0 python "$LIGAR_PATH" --model networks/har/har_1e4.onnx --output results/har/ligar_har_1e4_cheby_global.csv --dataset har --eps_in $EPS --chebyshev

# LIGAR on MNIST for small epsilon regions around training data for degree 200
EPS=0.01
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_approx_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_approx_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_approx_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_approx_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_approx_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_approx_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200 

taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_cheby_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_cheby_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_cheby_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_cheby_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_cheby_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_cheby_eps_001.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200


EPS=0.03
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_approx_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_approx_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_approx_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_approx_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_approx_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_approx_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200

taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_cheby_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_cheby_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_cheby_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_cheby_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_cheby_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_cheby_eps_003.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200


EPS=0.05
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_approx_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_approx_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_approx_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_approx_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_approx_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_approx_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --degrees 200

taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_cheby_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_cheby_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_cheby_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_cheby_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_cheby_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_cheby_eps_005.csv --dataset mnist --eps_in $EPS --n_inputs 10 --chebyshev --degrees 200




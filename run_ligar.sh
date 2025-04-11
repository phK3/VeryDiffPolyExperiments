#!/bin/bash

LIGAR_PATH="deps/ligar_reloaded/ligar_main.py"
EPS=1

# run LIGAR on MNIST with approximate formula for error
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_approx_global.csv --dataset mnist --eps_in $EPS 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_approx_global.csv --dataset mnist --eps_in $EPS  
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_approx_global.csv --dataset mnist --eps_in $EPS 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_approx_global.csv --dataset mnist --eps_in $EPS  
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_approx_global.csv --dataset mnist --eps_in $EPS  
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_approx_global.csv --dataset mnist --eps_in $EPS  

# run LIGAR on MNIST with concrete Chebyshev approximation for error computation
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_1e4.onnx --output results/mnist/ligar_mnist_256x4_1e4_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x4_2e5.onnx --output results/mnist/ligar_mnist_256x4_2e5_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_1e4.onnx --output results/mnist/ligar_mnist_256x6_1e4_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist_256x6_2e5.onnx --output results/mnist/ligar_mnist_256x6_2e5_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x4.onnx --output results/mnist/ligar_mnist-net_256x4_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev 
taskset -c 0 python "$LIGAR_PATH" --model networks/mnist/mnist-net_256x6.onnx --output results/mnist/ligar_mnist-net_256x6_cheby_global.csv --dataset mnist --eps_in $EPS --chebyshev

# run LIGAR on HELOC with approximate formula for error
taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc.onnx --output results/heloc/ligar_heloc_approx_global.csv --dataset heloc --eps_in $EPS
taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc_2e5.onnx --output results/heloc/ligar_heloc_2e5_approx_global.csv --dataset heloc --eps_in $EPS 

# run LIGAR on HELOC with concrete Chebyshev approximation for error computation
taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc.onnx --output results/heloc/ligar_heloc_cheby_global.csv --dataset heloc --eps_in $EPS --chebyshev
taskset -c 0 python "$LIGAR_PATH" --model networks/heloc/heloc_2e5.onnx --output results/heloc/ligar_heloc_2e5_cheby_global.csv --dataset heloc --eps_in $EPS --chebyshev

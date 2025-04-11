
import argparse
import csv
from typing import List, Optional

import numpy as np
import onnx

import time

from algorithms.ligar.ligar_2 import ligar_design, ligar_output_bound


MNIST_PATH = "../../datasets/mnist_train.csv"
HELOC_PATH = "../../datasets/heloc_dataset.csv"


def load_model(model_path: str):
    """Load the ONNX model from the specified path."""
    model = None
    try:
        model = onnx.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    W_list = []
    b_list = []
    params = model.graph.initializer

    # stupid special case for mnist-net_256x6: ONNX sorts alphabetically instead of numerically
    # and the activation layers are also numbered, so we have layer.12.weight, while the others only
    # have fc6.weight, so alphabetically sorting works in that case
    # We just divide the number by two, so we crudely fix it!
    if params[0].name.startswith("layers.0"):
        for param in params:
            parts = param.name.split(".")
            n = int(parts[1])
            param.name = f"{parts[0]}.{int(n/2)}.{parts[-1]}"

        # if we divide the layer number by two, we get below 10, so alphabetical sorting works again!
        params.sort(key=lambda x: x.name)

    for param in params:
        if "bias" in param.name:
            b_list.append(onnx.numpy_helper.to_array(param))
        elif "weight" in param.name:
            W_list.append(onnx.numpy_helper.to_array(param))
        else:
            raise RuntimeError(f"Unknown parameter name: {param.name}")
        
    return W_list, b_list


def n_inputs_to_indices(n_inputs: int) -> List[int]:
    """Convert n_inputs to a list of indices."""
    indices = list(range(n_inputs))
    if len(indices) == 0:
        return None
    return indices


def load_mnist(indices: Optional[List[int]] = None) -> List[np.ndarray]:
    """Load MNIST dataset from a CSV file.
        If indices are provided, only those rows will be loaded.
    """
    inputs = []
    try:
        with open(MNIST_PATH, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if indices is None or i in indices:
                    inputs.append(np.array([float(x) for x in row[1:]]))
    except Exception as e:
        raise RuntimeError(f"Failed to load inputs from {MNIST_PATH}: {e}")
    
    X = [x[1:] / 255 for x in inputs]  # Normalize the inputs to [0, 1]
    y = [x[0] for x in inputs]  # Extract labels
    
    return X, y


def load_heloc(indices: Optional[List[int]] = None) -> List[np.ndarray]:
    """Load HELOC dataset from a CSV file.
        If indices are provided, only those rows will be loaded.
    """
    X = []
    y = []
    try:
        with open(HELOC_PATH, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0:
                    # first row is the header, skip it
                    continue

                if indices is None or i in indices:
                    y.append(row[0]) # first column is the label
                    X.append(np.array([float(x) for x in row[1:]])) # rest are the features
    except Exception as e:
        raise RuntimeError(f"Failed to load inputs from {HELOC_PATH}: {e}")


    # Normalize the inputs to [0, 1]
    data_min = -9
    data_max = np.array([93, 789, 383, 383, 74, 17, 16, 100, 83, 9, 8, 87, 17, 100, 24, 66, 66, 232, 471, 32, 23, 18, 100])
    X = [(x - data_min) / (data_max - data_min) for x in X]

    return X, y

"""
Arguments:
    w_list: list of weight matrices
    b_list: list of bias vectors
    x_list: list of (concrete) inputs
    x_ball: perturbation distance for each network input
    degrees: list of polynomial degrees to repeat the experiments on
    mode: either "approx" or "chebyshev"
"""
def compute_epsilon_bounds(w_list, b_list, x_list, x_ball, degrees, mode="chebyshev"):
    """Compute the output error for each degree (and the runtime of the algorithm)."""
    out_bounds = []
    times = []
    for degree in degrees:
        d_list = [np.repeat(degree, len(b)) for b in b_list[:-1]]
        print(f"Computing with degree {degree}")
        
        start = time.time()
        
        # PHASE I: design the polynomial network
        
        # entire input space [0,1]^n
        n = len(x_ball)
        x_range = np.column_stack([np.zeros(n), np.ones(n)])
        
        # run a FastLin-style forward pass (parallel linear bounds)
        # we only care about the approximation error at each ReLU
        _, _, e_ball = ligar_design(w_list, b_list, x_range, d_list, mode=mode)
        
        # PHASE II: compute the output error bound
        
        # LiGAR-style Lipschitz estimation
        b = ligar_output_bound(w_list, b_list, x_list, x_ball, e_ball)
        out_bounds.append(b)
        
        # record the elapsed time
        t_elapsed = time.time() - start
        times.append(t_elapsed)
    
    return out_bounds, times

def main():
    parser = argparse.ArgumentParser(description="Lipschitz guided Synthesis of Polynomial Neural Networks")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Name of a predefined dataset to test (options: mnist, heloc)")
    
    # optional arguments
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="List of indices to select specific inputs from the dataset")
    parser.add_argument("--n_inputs", type=int, default=-1,
                        help="Load first n_inputs from the dataset (only if --indices is not provided)")
    parser.add_argument("--eps_in", type=float, default=1.0,
                        help="Size of the input ball")
    parser.add_argument("--chebyshev", action="store_true",
                        help="Use Chebyshev mode for exact error bounds")
    # list of polynomial degrees, default is [1, 2, ..., 10, 20, ..., 200]
    parser.add_argument("--degrees", nargs="+", type=int, default=None,
                        help="List of polynomial degrees to scan")
    
    args = parser.parse_args()

    if args.degrees is None:
        args.degrees = np.concatenate([np.arange(1, 10), np.arange(10, 201, 10)]).tolist()

    input_dim = -1
    center = None
    whole_input_space = False
    assert not (args.indices and (args.n_inputs > -1)), "Either --indices or --n_inputs should be provided, not both"
    if (not args.indices) and (args.n_inputs == -1):
        whole_input_space = True

    if args.n_inputs > -1:
        args.indices = n_inputs_to_indices(args.n_inputs)
    
    # handle predifined datasets
    if args.dataset == "mnist":
        input_dim = 784
        center = 0.5
        if whole_input_space:
            x_list = [np.repeat(center, input_dim)]
        else:
            x_list, _ = load_mnist(args.indices)
    elif args.dataset == "heloc":
        input_dim = 23
        center = 0.5
        if whole_input_space:
            x_list = [np.repeat(center, input_dim)]
        else:
            x_list, _ = load_heloc(args.indices) 

    # TODO: the input balls are x_list +/- eps_in
    x_ball = np.repeat(args.eps_in, input_dim)

    w_list, b_list = load_model(args.model)

    out_bounds, times = compute_epsilon_bounds(w_list, b_list, x_list, x_ball, args.degrees,
                                      mode="chebyshev" if args.chebyshev else "approx")
    # Save results to CSV
    np.savetxt(args.output, np.column_stack([args.degrees, out_bounds, times]), delimiter=",")

if __name__ == "__main__":
    main()


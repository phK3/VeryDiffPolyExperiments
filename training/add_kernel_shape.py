
import onnx 

import argparse


def main(input_path, output_path):
    model = onnx.load(input_path)
    for node in model.graph.node:
        if node.op_type == "Conv":
            has_kernel_shape = any(attr.name == "kernel_shape" for attr in node.attribute)
            # Add kernel_shape attribute if it doesn't exist
            if not has_kernel_shape:
                weight_name = node.input[1]
                for init in model.graph.initializer:
                    if init.name == weight_name:
                        shape = init.dims
                        kernel_shape = shape[2:]  # (height, width)

                        kernel_shape_attr = onnx.helper.make_attribute("kernel_shape", kernel_shape)
                        node.attribute.append(kernel_shape_attr)
    
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add kernel_shape attribute to Conv nodes in an ONNX model.")
    parser.add_argument("input_path", type=str, help="Path to the input ONNX model.")
    parser.add_argument("output_path", type=str, help="Path to save the modified ONNX model.")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path)
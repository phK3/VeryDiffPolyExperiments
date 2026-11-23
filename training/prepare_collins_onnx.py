"""Prepare the three legacy small Collins exports for Julia, without retraining.

Run from any directory with Python's onnx and numpy packages installed:
    python training/prepare_collins_onnx.py

Originals are never overwritten. Each output is checked against the original
with ONNX's reference evaluator on dataset samples and seeded random inputs.
This deliberately recognizes only the known small window-20 architectures.
"""

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ROOT / "networks" / "collins"
SOURCES = (
    "NN_rul_small_window_20.onnx",
    "NN_rul_small_window_20_gelu.onnx",
    "NN_rul_small_window_20_gelu_2e-5l1.onnx",
)
WEIGHT_SHAPES = (
    (5, 1, 5, 1), (10, 5, 5, 1), (5, 10, 5, 1),
    (5, 5, 3, 1), (100, 5, 6, 20), (1, 100, 1, 1),
)


def prepare_model(original):
    model = copy.deepcopy(original)
    initializers = {t.name: t for t in model.graph.initializer}
    inputs = [v for v in model.graph.input if v.name not in initializers]
    if len(inputs) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one data input and one output")
    ops = [n.op_type for n in model.graph.node]
    is_gelu = "Gelu" in ops
    expected = (
        ["Unsqueeze", "Conv", "Squeeze", "Gelu"] * 5
        + ["Unsqueeze", "Conv", "Squeeze", "Reshape"]
        if is_gelu else ["Conv", "Relu"] * 5 + ["Dropout", "Conv", "Flatten"]
    )
    if ops != expected:
        raise ValueError("Unexpected graph topology; refusing a generic rewrite")
    shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
    if shape != ([1, 20, 20] if is_gelu else [0, 1, 20, 20]):
        raise ValueError(f"Unexpected input shape: {shape}")

    old_input = inputs[0].name
    previous = old_input
    aliases = {old_input: "input"}
    nodes = []
    conv_shapes = []
    for node in model.graph.node:
        # The known graph is a single chain, with no branches or mask outputs.
        if node.input[0] != previous or len(node.output) != 1:
            raise ValueError("Expected a single-chain graph")
        previous = node.output[0]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type in ("Squeeze", "Unsqueeze"):
            axes = numpy_helper.to_array(initializers[node.input[1]])
            if axes.tolist() != [0]:
                raise ValueError("Only the singleton batch-axis wrappers are supported")
            aliases[node.output[0]] = aliases.get(node.input[0], node.input[0])
            continue
        if node.op_type == "Dropout":
            # This legacy opset-8 export's inference Dropout is identity, as in
            # the notebook's onnx2torch converter. Never strip training Dropout.
            version = next(v.version for v in model.opset_import if v.domain == "")
            if version != 8 or attrs.get("is_test", 1) != 1:
                raise ValueError("Expected legacy inference-mode Dropout")
            aliases[node.output[0]] = aliases.get(node.input[0], node.input[0])
            continue
        for i, name in enumerate(node.input):
            node.input[i] = aliases.get(name, name)
        if node.op_type == "Conv":
            dims = tuple(initializers[node.input[1]].dims)
            conv_shapes.append(dims)
            kernel = list(dims[2:])
            if "kernel_shape" not in attrs:
                node.attribute.append(helper.make_attribute("kernel_shape", kernel))
            elif list(attrs["kernel_shape"]) != kernel:
                raise ValueError("Convolution kernel metadata disagrees with weights")
        nodes.append(node)
    if tuple(conv_shapes) != WEIGHT_SHAPES:
        raise ValueError("Not the expected small window-20 network")

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Freeze initializer inputs from the MATLAB export; expose only data.
    del model.graph.input[:]
    model.graph.input.append(helper.make_tensor_value_info(
        "input", inputs[0].type.tensor_type.elem_type, [1, 1, 20, 20]
    ))
    model.graph.output[0].CopyFrom(helper.make_tensor_value_info(
        model.graph.output[0].name,
        model.graph.output[0].type.tensor_type.elem_type, [1, 1]
    ))
    del model.graph.value_info[:]
    model.ir_version = max(model.ir_version, 4)
    model.doc_string += "\nJulia-compatible inference copy; prepared by training/prepare_collins_onnx.py."
    # Every initializer, including all trained weights and biases, is unchanged.
    assert [t.SerializeToString() for t in model.graph.initializer] == [
        t.SerializeToString() for t in original.graph.initializer
    ]
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    return model


def validate_outputs(original, prepared, samples):
    source = ReferenceEvaluator(original)
    target = ReferenceEvaluator(prepared)
    weights = {t.name for t in original.graph.initializer}
    input_info = next(v for v in original.graph.input if v.name not in weights)
    rank = len(input_info.type.tensor_type.shape.dim)
    max_error = 0.0
    for x in samples:
        x = np.asarray(x, dtype=np.float32).reshape(1, 1, 20, 20)
        expected = source.run(None, {input_info.name: x if rank == 4 else x[0]})[0]
        actual = target.run(None, {"input": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return max_error


def main():
    rows = np.loadtxt(ROOT / "datasets" / "collins_X20_test.csv", delimiter=",")
    rng = np.random.default_rng(0)
    samples = [*rows[:8], *rng.uniform(-1, 1, size=(4, 400))]
    for filename in SOURCES:
        source = NETWORKS / filename
        output = source.with_name(source.stem + "_julia.onnx")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        original = onnx.load(source)
        prepared = prepare_model(original)
        error = validate_outputs(original, prepared, samples)
        onnx.save(prepared, output)
        print(f"{output.name}: {len(samples)} equivalence checks passed; max error={error}", flush=True)


if __name__ == "__main__":
    main()

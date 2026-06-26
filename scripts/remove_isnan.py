#!/usr/bin/env python3
"""
Remove IsNaN + Where (safe_softmax) pattern from CKIP BERT ONNX models.

PyTorch opset-17 export inserts IsNaN checks after Softmax in self-attention:
    softmax_out → IsNaN → Where(isnan, 0, softmax_out) → next_op

The RK3588S NPU (librknnrt) does not support the IsNaN op.
In normal inference softmax never produces NaN, so these nodes are safely removable.

Usage:
    python remove_isnan.py --input ckip_bert_ws.onnx --output ckip_bert_ws_clean.onnx
    python remove_isnan.py --input-dir /path/to/onnx/ --output-dir /path/to/clean/
"""
import argparse
import os
import onnx


def remove_isnan_pattern(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    Remove IsNaN + Where pairs, rewiring Where consumers to use Softmax output directly.

    Returns: (cleaned_model, num_pairs_removed)
    """
    nodes_to_remove = set()
    rewire = {}  # where_output -> softmax_output

    for i, node in enumerate(model.graph.node):
        if node.op_type != "IsNaN":
            continue

        isnan_output = node.output[0]
        softmax_output = node.input[0]  # IsNaN input = Softmax output

        # Find the Where node that consumes this IsNaN output
        for j, n2 in enumerate(model.graph.node):
            if n2.op_type == "Where" and isnan_output in n2.input:
                where_output = n2.output[0]
                rewire[where_output] = softmax_output
                nodes_to_remove.add(i)
                nodes_to_remove.add(j)
                break

    # Rewire: replace all references to Where outputs with Softmax outputs
    for node in model.graph.node:
        for k, inp in enumerate(node.input):
            if inp in rewire:
                node.input[k] = rewire[inp]

    # Remove nodes in reverse index order
    for idx in sorted(nodes_to_remove, reverse=True):
        del model.graph.node[idx]

    return model, len(nodes_to_remove) // 2


def process_file(input_path: str, output_path: str) -> None:
    """Process a single ONNX file."""
    print(f"  Loading: {input_path}")
    model = onnx.load(input_path)

    isnan_count = sum(1 for n in model.graph.node if n.op_type == "IsNaN")
    if isnan_count == 0:
        print(f"    No IsNaN nodes found, skipping.")
        return

    model, pairs_removed = remove_isnan_pattern(model)
    onnx.save(model, output_path)

    in_size = os.path.getsize(input_path) / 1024 / 1024
    out_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"    Removed {pairs_removed} IsNaN+Where pairs")
    print(f"    Saved: {output_path} ({out_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Remove IsNaN ops from CKIP BERT ONNX for RKNN compatibility")
    parser.add_argument("--input", help="Single ONNX file to process")
    parser.add_argument("--output", help="Output path for single file")
    parser.add_argument("--input-dir", help="Directory of ONNX files to process")
    parser.add_argument("--output-dir", help="Output directory for cleaned files")
    args = parser.parse_args()

    if args.input:
        output = args.output or args.input.replace(".onnx", "_clean.onnx")
        process_file(args.input, output)
    elif args.input_dir:
        output_dir = args.output_dir or os.path.join(args.input_dir, "clean")
        os.makedirs(output_dir, exist_ok=True)
        for f in sorted(os.listdir(args.input_dir)):
            if f.endswith(".onnx"):
                process_file(
                    os.path.join(args.input_dir, f),
                    os.path.join(output_dir, f),
                )
    else:
        parser.error("Provide either --input or --input-dir")


if __name__ == "__main__":
    main()

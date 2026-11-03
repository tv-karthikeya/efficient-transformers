#!/usr/bin/env python3
"""Write ONNX graph nodes to a TSV in stable topological order."""

import argparse
import csv
import heapq
import json
from pathlib import Path

import onnx


def topological_node_indices(nodes: list[onnx.NodeProto]) -> list[int]:
    producer_by_value = {}
    for node_index, node in enumerate(nodes):
        for output_name in node.output:
            if not output_name:
                continue
            if output_name in producer_by_value:
                raise ValueError(f"Multiple graph nodes produce tensor {output_name!r}")
            producer_by_value[output_name] = node_index

    dependencies = [set() for _ in nodes]
    consumers = [[] for _ in nodes]
    for node_index, node in enumerate(nodes):
        for input_name in node.input:
            producer_index = producer_by_value.get(input_name)
            if producer_index is not None:
                dependencies[node_index].add(producer_index)
        for producer_index in dependencies[node_index]:
            consumers[producer_index].append(node_index)

    ready = [node_index for node_index, deps in enumerate(dependencies) if not deps]
    heapq.heapify(ready)
    ordered_indices = []

    while ready:
        node_index = heapq.heappop(ready)
        ordered_indices.append(node_index)
        for consumer_index in consumers[node_index]:
            dependencies[consumer_index].remove(node_index)
            if not dependencies[consumer_index]:
                heapq.heappush(ready, consumer_index)

    if len(ordered_indices) != len(nodes):
        unresolved = [nodes[index].name or f"node_at_index_{index}" for index, deps in enumerate(dependencies) if deps]
        raise ValueError(f"ONNX graph contains a cycle; unresolved nodes: {unresolved[:10]}")

    return ordered_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "onnx_path",
        nargs="?",
        type=Path,
        default=Path(
            "/home/vtirumal/mainline/compiler_issue/Qwen3_5ForConditionalGeneration/"
            "Qwen3_5DecoderWrapper-3d54630abda335ef/Qwen3_5DecoderWrapper.onnx"
        ),
        help="Path to the ONNX model (defaults to the Qwen3.5 decoder ONNX path)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output TSV path (default: <onnx_path>.nodes.tsv)",
    )
    args = parser.parse_args()

    output_path = args.output or Path(f"{args.onnx_path}.nodes.tsv")
    model = onnx.load(args.onnx_path, load_external_data=False)
    nodes = list(model.graph.node)
    ordered_indices = topological_node_indices(nodes)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["topological_index", "onnx_index", "node_name", "op_type", "inputs", "outputs"])
        for topological_index, node_index in enumerate(ordered_indices):
            node = nodes[node_index]
            node_name = node.name or f"node_at_index_{node_index}"
            writer.writerow(
                [
                    topological_index,
                    node_index,
                    node_name,
                    node.op_type,
                    json.dumps(list(node.input), ensure_ascii=False),
                    json.dumps(list(node.output), ensure_ascii=False),
                ]
            )

    print(f"Wrote {len(nodes)} nodes to {output_path}")


if __name__ == "__main__":
    main()

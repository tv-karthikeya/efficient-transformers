#!/usr/bin/env python3
"""Report target-node inputs that depend on a selected ONNX node."""

import argparse
import sys
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import TextIO

import onnx

DEFAULT_ONNX_PATH = Path(
    "/home/vtirumal/mainline/compiler_issue/Qwen3_5ForConditionalGeneration/"
    "Qwen3_5DecoderWrapper-3d54630abda335ef/Qwen3_5DecoderWrapper.onnx"
)
DEFAULT_SOURCE_NODE = "/Gather_5"
DEFAULT_TARGET_NODE = "/language_model/layers.3/QEffQwen3_5FullAttentionDecoderLayer"


def find_unique_node(nodes: list[onnx.NodeProto], node_name: str) -> tuple[int, onnx.NodeProto]:
    matches = [(index, node) for index, node in enumerate(nodes) if node.name == node_name]
    if not matches:
        raise ValueError(f"No ONNX node named {node_name!r} was found")
    if len(matches) > 1:
        indices = [index for index, _ in matches]
        raise ValueError(f"Node name {node_name!r} is ambiguous; matching ONNX indices: {indices}")
    return matches[0]


def find_reachable_predecessors(
    source_index: int,
    nodes: list[onnx.NodeProto],
    producer_by_tensor: dict[str, int],
) -> dict[int, tuple[int, str] | None]:
    consumers_by_producer = [[] for _ in nodes]
    for consumer_index, node in enumerate(nodes):
        for input_name in node.input:
            producer_index = producer_by_tensor.get(input_name)
            if producer_index is not None:
                consumers_by_producer[producer_index].append((consumer_index, input_name))

    predecessors: dict[int, tuple[int, str] | None] = {source_index: None}
    nodes_to_visit = deque([source_index])
    while nodes_to_visit:
        producer_index = nodes_to_visit.popleft()
        for consumer_index, tensor_name in consumers_by_producer[producer_index]:
            if consumer_index not in predecessors:
                predecessors[consumer_index] = (producer_index, tensor_name)
                nodes_to_visit.append(consumer_index)
    return predecessors


def find_ancestry_path(
    tensor_name: str,
    source_index: int,
    producer_by_tensor: dict[str, int],
    predecessors: dict[int, tuple[int, str] | None],
) -> list[tuple[int, str]] | None:
    producer_index = producer_by_tensor.get(tensor_name)
    if producer_index is None or producer_index not in predecessors:
        return None

    reverse_path = []
    current_index = producer_index
    while current_index != source_index:
        predecessor = predecessors[current_index]
        if predecessor is None:
            return None
        previous_index, tensor_name = predecessor
        reverse_path.append((previous_index, tensor_name))
        current_index = previous_index
    reverse_path.reverse()
    reverse_path.append((producer_index, tensor_name))
    return reverse_path


def node_label(index: int, node: onnx.NodeProto) -> str:
    name = node.name or f"node_at_index_{index}"
    return f"{name} [{node.op_type}]"


def format_path(path: list[tuple[int, str]], target_input: str, nodes: list[onnx.NodeProto]) -> str:
    path_parts = [node_label(path[0][0], nodes[path[0][0]])]
    for (_, output_name), (next_index, _) in pairwise(path):
        path_parts.extend((f"--[{output_name}]-->", node_label(next_index, nodes[next_index])))
    path_parts.append(f"--[{path[-1][1]}]--> target input {target_input!r}")
    return " ".join(path_parts)


def write_report(
    output: TextIO,
    nodes: list[onnx.NodeProto],
    source_index: int,
    target_index: int,
    producer_by_tensor: dict[str, int],
    predecessors: dict[int, tuple[int, str] | None],
) -> None:
    source = nodes[source_index]
    target = nodes[target_index]
    print(f"ONNX source: {node_label(source_index, source)} (index {source_index})", file=output)
    print(f"ONNX target: {node_label(target_index, target)} (index {target_index})", file=output)

    dependent_inputs = 0
    for input_index, input_name in enumerate(target.input):
        path = find_ancestry_path(input_name, source_index, producer_by_tensor, predecessors)
        print(f"\nInput {input_index}: {input_name!r}", file=output)
        if path is None:
            print(f"  No dependency path from {source.name!r}", file=output)
            continue

        dependent_inputs += 1
        dependency_kind = "direct" if len(path) == 1 else "indirect"
        print(f"  {dependency_kind.capitalize()} dependency from {source.name!r}", file=output)
        print(f"  Path: {format_path(path, input_name, nodes)}", file=output)

    print(f"\nSummary: {dependent_inputs} of {len(target.input)} target inputs depend on the source node.", file=output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", nargs="?", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--source-node", default=DEFAULT_SOURCE_NODE, help="Exact source node name")
    parser.add_argument("--target-node", default=DEFAULT_TARGET_NODE, help="Exact target node name")
    parser.add_argument("-o", "--output", type=Path, help="Write the report to this file instead of stdout")
    args = parser.parse_args()

    model = onnx.load(args.onnx_path, load_external_data=False)
    nodes = list(model.graph.node)

    try:
        source_index, _ = find_unique_node(nodes, args.source_node)
        target_index, _ = find_unique_node(nodes, args.target_node)
        producer_by_tensor = {}
        for node_index, node in enumerate(nodes):
            for output_name in node.output:
                if not output_name:
                    continue
                if output_name in producer_by_tensor:
                    raise ValueError(f"Multiple graph nodes produce tensor {output_name!r}")
                producer_by_tensor[output_name] = node_index
        predecessors = find_reachable_predecessors(source_index, nodes, producer_by_tensor)
    except ValueError as error:
        parser.error(str(error))

    if args.output is None:
        write_report(sys.stdout, nodes, source_index, target_index, producer_by_tensor, predecessors)
    else:
        with args.output.open("w", encoding="utf-8") as output:
            write_report(output, nodes, source_index, target_index, producer_by_tensor, predecessors)
        print(f"Wrote ancestry report to {args.output}")


if __name__ == "__main__":
    main()

"""Helpers for converting model tensors into browser-friendly reduced graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .schema import (
    AttentionCollection,
    AttentionGraph,
    GraphDocument,
    GraphEdge,
    GraphNode,
    Position,
    TokenRecord,
)


def architecture_graph(num_layers: int, num_heads: int, intermediate_size: int) -> GraphDocument:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    stages: list[tuple[str, str, str, dict[str, int]]] = [
        ("tokens", "input", "Input tokens", {}),
        ("embedding", "embedding", "Token + position embeddings", {}),
    ]
    stages.extend(
        (
            f"layer-{layer}",
            "transformer_layer",
            f"Layer {layer}",
            {"attention_heads": num_heads, "mlp_width": intermediate_size},
        )
        for layer in range(num_layers)
    )
    stages.extend(
        [
            ("final-norm", "normalization", "Final layer norm", {}),
            ("logits", "output", "Vocabulary logits", {}),
        ]
    )

    for index, (node_id, node_type, label, metadata) in enumerate(stages):
        layer = index - 2 if node_type == "transformer_layer" else None
        nodes.append(
            GraphNode(
                id=node_id,
                type=node_type,
                label=label,
                layer=layer,
                position=Position(x=80 + index * 230, y=120),
                metadata=metadata,
            )
        )
        if index:
            source = stages[index - 1][0]
            edges.append(
                GraphEdge(
                    id=f"{source}->{node_id}",
                    source=source,
                    target=node_id,
                    metric="model_flow",
                    weight=1.0,
                )
            )

    return GraphDocument(metric="model_flow", nodes=nodes, edges=edges)


def attention_graphs(
    attentions: Sequence[torch.Tensor],
    tokens: list[TokenRecord],
    *,
    top_edges_per_token: int,
    minimum_weight: float,
) -> AttentionCollection:
    graphs: list[AttentionGraph] = []
    token_count = len(tokens)

    for layer_index, layer_attention in enumerate(attentions):
        tensor = layer_attention.detach().float().cpu()[0]
        for head_index in range(tensor.shape[0]):
            matrix = tensor[head_index]
            nodes = [
                GraphNode(
                    id=f"token-{token.index}",
                    type="token",
                    label=token.display,
                    token_index=token.index,
                    position=Position(x=80 + token.index * 170, y=160),
                    metadata={"token_id": token.token_id, "raw_text": token.text},
                )
                for token in tokens
            ]
            edges: list[GraphEdge] = []

            for query_index in range(token_count):
                row = matrix[query_index, : query_index + 1]
                k = min(top_edges_per_token, row.numel())
                values, indices = torch.topk(row, k=k)
                for value, key_index in zip(values.tolist(), indices.tolist(), strict=True):
                    if value < minimum_weight:
                        continue
                    edges.append(
                        GraphEdge(
                            id=f"l{layer_index}-h{head_index}-q{query_index}-k{key_index}",
                            source=f"token-{query_index}",
                            target=f"token-{key_index}",
                            metric="attention_probability",
                            weight=float(value),
                            label=f"{value:.3f}",
                            metadata={
                                "layer": layer_index,
                                "head": head_index,
                                "query_index": query_index,
                                "key_index": key_index,
                            },
                        )
                    )

            graphs.append(
                AttentionGraph(
                    layer=layer_index,
                    head=head_index,
                    graph=GraphDocument(
                        metric="attention_probability",
                        nodes=nodes,
                        edges=edges,
                    ),
                )
            )

    return AttentionCollection(graphs=graphs)


def neuron_graph(
    activations: Sequence[torch.Tensor],
    tokens: list[TokenRecord],
    *,
    top_neurons: int,
) -> GraphDocument:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    visible_tokens: set[int] = set()

    candidates: list[tuple[float, float, int, int, int]] = []
    for layer_index, activation in enumerate(activations):
        tensor = activation.detach().float().cpu()[0]
        flat = tensor.abs().flatten()
        k = min(top_neurons, flat.numel())
        _, flat_indices = torch.topk(flat, k=k)
        width = tensor.shape[-1]
        for flat_index in flat_indices.tolist():
            token_index = flat_index // width
            neuron_index = flat_index % width
            value = float(tensor[token_index, neuron_index])
            candidates.append((abs(value), value, layer_index, token_index, neuron_index))

    candidates.sort(reverse=True, key=lambda item: item[0])
    candidates = candidates[:top_neurons]

    for _, value, layer_index, token_index, neuron_index in candidates:
        if token_index not in visible_tokens:
            token = tokens[token_index]
            nodes.append(
                GraphNode(
                    id=f"token-{token_index}",
                    type="token",
                    label=token.display,
                    token_index=token_index,
                    position=Position(x=80, y=80 + token_index * 110),
                    metadata={"token_id": token.token_id, "raw_text": token.text},
                )
            )
            visible_tokens.add(token_index)

        node_id = f"mlp-l{layer_index}-t{token_index}-n{neuron_index}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="mlp_neuron",
                label=f"L{layer_index} · N{neuron_index}",
                layer=layer_index,
                token_index=token_index,
                position=Position(
                    x=350 + layer_index * 190,
                    y=80 + token_index * 110 + (neuron_index % 5) * 12,
                ),
                metadata={
                    "neuron_index": neuron_index,
                    "activation": value,
                    "absolute_activation": abs(value),
                },
            )
        )
        edges.append(
            GraphEdge(
                id=f"token-{token_index}->{node_id}",
                source=f"token-{token_index}",
                target=node_id,
                metric="mlp_activation",
                weight=abs(value),
                label=f"{value:+.3f}",
                metadata={"signed_value": value},
            )
        )

    return GraphDocument(metric="mlp_activation", nodes=nodes, edges=edges)

import torch

from circuit_lens.graph import attention_graphs, neuron_graph
from circuit_lens.schema import TokenRecord


def tokens():
    return [
        TokenRecord(index=0, token_id=1, text="A", display="A"),
        TokenRecord(index=1, token_id=2, text=" B", display="␠B"),
    ]


def test_attention_graph_respects_causal_rows():
    attention = torch.tensor([[[[1.0, 0.0], [0.4, 0.6]]]])
    result = attention_graphs([attention], tokens(), top_edges_per_token=2, minimum_weight=0.01)
    graph = result.graphs[0].graph
    assert len(graph.nodes) == 2
    assert all(edge.metadata["key_index"] <= edge.metadata["query_index"] for edge in graph.edges)


def test_neuron_graph_selects_global_top_k():
    activation = torch.tensor([[[0.1, 4.0], [3.0, -5.0]]])
    graph = neuron_graph([activation], tokens(), top_neurons=2)
    neuron_nodes = [node for node in graph.nodes if node.type == "mlp_neuron"]
    assert len(neuron_nodes) == 2
    values = sorted(node.metadata["absolute_activation"] for node in neuron_nodes)
    assert values == [4.0, 5.0]

"""Deterministic demo bundle generation without downloading a model."""

from __future__ import annotations

from datetime import UTC, datetime

from .graph import architecture_graph
from .schema import (
    AnalysisBundle,
    AttentionCollection,
    AttentionGraph,
    CapabilityDocument,
    GraphDocument,
    GraphEdge,
    GraphNode,
    LayerPrediction,
    LogitRecord,
    LogitsDocument,
    Manifest,
    ModelDocument,
    Position,
    PromptDocument,
    TokenRecord,
)


def create_demo_bundle() -> AnalysisBundle:
    raw_tokens = ["Baku", " is", " the", " capital", " of"]
    tokens = [
        TokenRecord(index=i, token_id=100 + i, text=text, display=text.replace(" ", "␠"))
        for i, text in enumerate(raw_tokens)
    ]

    attention_nodes = [
        GraphNode(
            id=f"token-{token.index}",
            type="token",
            label=token.display,
            token_index=token.index,
            position=Position(x=100 + token.index * 180, y=160),
            metadata={"raw_text": token.text, "token_id": token.token_id},
        )
        for token in tokens
    ]
    attention_edges = [
        GraphEdge(
            id="demo-a1",
            source="token-4",
            target="token-0",
            metric="attention_probability",
            weight=0.61,
            label="0.610",
        ),
        GraphEdge(
            id="demo-a2",
            source="token-4",
            target="token-3",
            metric="attention_probability",
            weight=0.27,
            label="0.270",
        ),
        GraphEdge(
            id="demo-a3",
            source="token-3",
            target="token-0",
            metric="attention_probability",
            weight=0.48,
            label="0.480",
        ),
    ]

    neuron_nodes = [
        attention_nodes[0].model_copy(update={"position": Position(x=80, y=100)}),
        attention_nodes[3].model_copy(update={"position": Position(x=80, y=250)}),
        GraphNode(
            id="mlp-l3-t0-n418",
            type="mlp_neuron",
            label="L3 · N418",
            layer=3,
            token_index=0,
            position=Position(x=390, y=100),
            metadata={"neuron_index": 418, "activation": 4.82, "absolute_activation": 4.82},
        ),
        GraphNode(
            id="mlp-l7-t3-n914",
            type="mlp_neuron",
            label="L7 · N914",
            layer=7,
            token_index=3,
            position=Position(x=650, y=250),
            metadata={"neuron_index": 914, "activation": 3.71, "absolute_activation": 3.71},
        ),
    ]
    neuron_edges = [
        GraphEdge(
            id="demo-n1",
            source="token-0",
            target="mlp-l3-t0-n418",
            metric="mlp_activation",
            weight=4.82,
            label="+4.820",
        ),
        GraphEdge(
            id="demo-n2",
            source="token-3",
            target="mlp-l7-t3-n914",
            metric="mlp_activation",
            weight=3.71,
            label="+3.710",
        ),
    ]

    predictions = [
        LogitRecord(token_id=200, token=" Azerbaijan", probability=0.72, logit=14.1),
        LogitRecord(token_id=201, token=" Georgia", probability=0.08, logit=11.9),
        LogitRecord(token_id=202, token=" Turkey", probability=0.05, logit=11.4),
    ]

    files = [
        "manifest.json",
        "model.json",
        "prompt.json",
        "tokens.json",
        "architecture.json",
        "logits.json",
        "graphs/attention.json",
        "graphs/neurons.json",
    ]
    return AnalysisBundle(
        manifest=Manifest(
            run_id="gpt2-demo",
            created_at=datetime.now(UTC),
            model_id="gpt2",
            capabilities=CapabilityDocument(),
            files=files,
        ),
        model=ModelDocument(
            id="gpt2",
            architecture="GPT2LMHeadModel",
            parameter_count=124_439_808,
            num_layers=12,
            num_attention_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            vocabulary_size=50_257,
            context_length=1024,
            dtype="float32",
            device="demo",
        ),
        prompt=PromptDocument(text="Baku is the capital of", target_token=" Azerbaijan"),
        tokens=tokens,
        architecture=architecture_graph(12, 12, 3072),
        attention=AttentionCollection(
            graphs=[
                AttentionGraph(
                    layer=7,
                    head=4,
                    graph=GraphDocument(
                        metric="attention_probability",
                        nodes=attention_nodes,
                        edges=attention_edges,
                    ),
                )
            ]
        ),
        neurons=GraphDocument(metric="mlp_activation", nodes=neuron_nodes, edges=neuron_edges),
        logits=LogitsDocument(
            top=predictions,
            by_layer=[
                LayerPrediction(
                    layer="embedding",
                    predictions=[
                        LogitRecord(token_id=203, token=" city", probability=0.12, logit=7.1),
                        LogitRecord(token_id=204, token=" country", probability=0.09, logit=6.8),
                    ],
                ),
                LayerPrediction(layer="layer-6", predictions=predictions[1:]),
                LayerPrediction(layer="final", predictions=predictions),
            ],
        ),
    )

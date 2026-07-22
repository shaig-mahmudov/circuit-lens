"""Versioned data contracts shared by the analyzer and web viewer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Position(StrictModel):
    x: float
    y: float


class GraphNode(StrictModel):
    id: str
    type: str
    label: str
    position: Position
    layer: int | None = None
    token_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(StrictModel):
    id: str
    source: str
    target: str
    metric: str
    weight: float
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphDocument(StrictModel):
    metric: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class AttentionGraph(StrictModel):
    layer: int
    head: int
    graph: GraphDocument


class AttentionCollection(StrictModel):
    graphs: list[AttentionGraph]


class TokenRecord(StrictModel):
    index: int
    token_id: int
    text: str
    display: str


class LogitRecord(StrictModel):
    token_id: int
    token: str
    probability: float
    logit: float


class LayerPrediction(StrictModel):
    layer: str
    predictions: list[LogitRecord]


class LogitsDocument(StrictModel):
    top: list[LogitRecord]
    by_layer: list[LayerPrediction]


class ModelDocument(StrictModel):
    id: str
    architecture: str
    parameter_count: int
    num_layers: int
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    vocabulary_size: int
    context_length: int
    dtype: str
    device: str


class PromptDocument(StrictModel):
    text: str
    target_token: str | None = None


class CapabilityDocument(StrictModel):
    architecture: bool = True
    attention: bool = True
    mlp_activations: bool = True
    logit_lens: bool = True
    ablation: bool = False
    activation_patching: bool = False


class Manifest(StrictModel):
    format: Literal["llmgraph"] = "llmgraph"
    format_version: str = "0.1.0"
    engine_version: str = "0.1.0"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    capabilities: CapabilityDocument = Field(default_factory=CapabilityDocument)
    files: list[str]


class AnalysisBundle(StrictModel):
    manifest: Manifest
    model: ModelDocument
    prompt: PromptDocument
    tokens: list[TokenRecord]
    architecture: GraphDocument
    attention: AttentionCollection
    neurons: GraphDocument
    logits: LogitsDocument

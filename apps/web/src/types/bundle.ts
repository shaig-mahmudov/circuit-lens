export type Position = { x: number; y: number };

export type GraphNode = {
  id: string;
  type: string;
  label: string;
  position: Position;
  layer?: number | null;
  token_index?: number | null;
  metadata: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  metric: string;
  weight: number;
  label?: string | null;
  metadata: Record<string, unknown>;
};

export type GraphDocument = {
  metric: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type AttentionGraph = {
  layer: number;
  head: number;
  graph: GraphDocument;
};

export type TokenRecord = {
  index: number;
  token_id: number;
  text: string;
  display: string;
};

export type LogitRecord = {
  token_id: number;
  token: string;
  probability: number;
  logit: number;
};

export type LayerPrediction = {
  layer: string;
  predictions: LogitRecord[];
};

export type CircuitLensBundle = {
  manifest: {
    format: "llmgraph";
    format_version: string;
    engine_version: string;
    run_id: string;
    created_at: string;
    model_id: string;
    capabilities: Record<string, boolean>;
    files: string[];
  };
  model: {
    id: string;
    architecture: string;
    parameter_count: number;
    num_layers: number;
    num_attention_heads: number;
    hidden_size: number;
    intermediate_size: number;
    vocabulary_size: number;
    context_length: number;
    dtype: string;
    device: string;
  };
  prompt: { text: string; target_token?: string | null };
  tokens: TokenRecord[];
  architecture: GraphDocument;
  attention: { graphs: AttentionGraph[] };
  neurons: GraphDocument;
  logits: { top: LogitRecord[]; by_layer: LayerPrediction[] };
};

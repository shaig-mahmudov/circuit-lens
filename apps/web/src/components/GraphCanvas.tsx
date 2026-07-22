"use client";

import { useEffect, useMemo } from "react";
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeMouseHandler,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type { GraphDocument, GraphNode } from "@/types/bundle";

type Props = {
  graph: GraphDocument;
  onNodeSelect: (node: GraphNode | null) => void;
};

const nodeColors: Record<string, string> = {
  input: "#22c55e",
  embedding: "#38bdf8",
  transformer_layer: "#8b5cf6",
  normalization: "#f59e0b",
  output: "#f43f5e",
  token: "#38bdf8",
  mlp_neuron: "#a78bfa",
};

export function GraphCanvas({ graph, onNodeSelect }: Props) {
  const converted = useMemo(() => {
    const nodes: Node[] = graph.nodes.map((node) => ({
      id: node.id,
      position: node.position,
      data: { label: node.label, source: node },
      style: {
        minWidth: node.type === "transformer_layer" ? 140 : 100,
        borderRadius: 14,
        border: `1px solid ${nodeColors[node.type] ?? "#64748b"}`,
        background: "rgba(10, 15, 28, 0.96)",
        color: "#f8fafc",
        boxShadow: "0 10px 30px rgba(0,0,0,.25)",
        fontSize: 12,
        fontWeight: 650,
        padding: 10,
      },
    }));

    const maximum = Math.max(...graph.edges.map((edge) => Math.abs(edge.weight)), 1);
    const edges: Edge[] = graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label ?? undefined,
      animated: edge.metric === "model_flow",
      markerEnd: { type: MarkerType.ArrowClosed },
      style: {
        strokeWidth: 1.5 + (Math.abs(edge.weight) / maximum) * 5,
        opacity: 0.35 + (Math.abs(edge.weight) / maximum) * 0.65,
      },
      labelStyle: { fill: "#cbd5e1", fontSize: 10 },
      data: edge,
    }));
    return { nodes, edges };
  }, [graph]);

  const [nodes, setNodes, onNodesChange] = useNodesState(converted.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(converted.edges);

  useEffect(() => {
    setNodes(converted.nodes);
    setEdges(converted.edges);
    onNodeSelect(null);
  }, [converted, onNodeSelect, setEdges, setNodes]);

  const onNodeClick: NodeMouseHandler = (_event, node) => {
    const source = node.data.source as GraphNode | undefined;
    onNodeSelect(source ?? null);
  };

  return (
    <div className="graph-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={() => onNodeSelect(null)}
        minZoom={0.08}
        maxZoom={2.5}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        proOptions={{ hideAttribution: false }}
      >
        <Background gap={22} size={1} />
        <MiniMap pannable zoomable />
        <Controls />
      </ReactFlow>
    </div>
  );
}

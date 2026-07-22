"use client";

import { useCallback, useMemo, useState } from "react";

import { parseBundle } from "@/lib/bundle-parser";
import type { CircuitLensBundle, GraphDocument, GraphNode } from "@/types/bundle";
import { FileDropzone } from "./FileDropzone";
import { GraphCanvas } from "./GraphCanvas";
import { LogitsPanel } from "./LogitsPanel";
import { NodeInspector } from "./NodeInspector";

type View = "architecture" | "attention" | "neurons";

export function Explorer() {
  const [bundle, setBundle] = useState<CircuitLensBundle | null>(null);
  const [view, setView] = useState<View>("architecture");
  const [attentionIndex, setAttentionIndex] = useState(0);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadBlob = useCallback(async (blob: Blob | ArrayBuffer) => {
    setBusy(true);
    setError(null);
    try {
      const parsed = await parseBundle(blob);
      setBundle(parsed);
      setView("architecture");
      setAttentionIndex(0);
      setSelectedNode(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not read bundle.");
    } finally {
      setBusy(false);
    }
  }, []);

  const graph = useMemo<GraphDocument | null>(() => {
    if (!bundle) return null;
    if (view === "architecture") return bundle.architecture;
    if (view === "neurons") return bundle.neurons;
    return bundle.attention.graphs[attentionIndex]?.graph ?? {
      metric: "attention_probability",
      nodes: [],
      edges: [],
    };
  }, [attentionIndex, bundle, view]);

  const selectedAttention = bundle?.attention.graphs[attentionIndex];

  return (
    <main>
      <header className="hero">
        <div>
          <div className="brand-mark">CL</div>
          <div>
            <p className="eyebrow">Open-source mechanistic interpretability</p>
            <h1>CircuitLens</h1>
            <p>Inspect reduced attention and MLP activation graphs exported from Kaggle.</p>
          </div>
        </div>
        <div className="status-pill">MVP · bundle format 0.1.0</div>
      </header>

      <FileDropzone
        busy={busy}
        onFile={(file) => loadBlob(file)}
        onDemo={async () => {
          const response = await fetch("/examples/gpt2-demo.llmgraph.zip");
          if (!response.ok) throw new Error("Demo bundle is unavailable.");
          await loadBlob(await response.arrayBuffer());
        }}
      />

      {error ? <div className="error-banner">{error}</div> : null}

      {!bundle ? (
        <section className="empty-state">
          <div className="empty-orbit"><span /><span /><span /></div>
          <h2>Load the included GPT‑2 demo or a Kaggle export.</h2>
          <p>The MVP supports architecture, attention, top MLP activations and final logits.</p>
        </section>
      ) : (
        <>
          <section className="summary-grid">
            <article><span>Model</span><strong>{bundle.model.id}</strong><small>{bundle.model.architecture}</small></article>
            <article><span>Parameters</span><strong>{Intl.NumberFormat("en", { notation: "compact" }).format(bundle.model.parameter_count)}</strong><small>{bundle.model.num_layers} layers</small></article>
            <article><span>Prompt tokens</span><strong>{bundle.tokens.length}</strong><small>{bundle.prompt.text}</small></article>
            <article><span>Runtime</span><strong>{bundle.model.device}</strong><small>{bundle.model.dtype}</small></article>
          </section>

          <section className="workspace">
            <div className="workspace-main">
              <div className="toolbar">
                <div className="tabs">
                  {(["architecture", "attention", "neurons"] as View[]).map((item) => (
                    <button
                      type="button"
                      className={view === item ? "tab-active" : ""}
                      key={item}
                      onClick={() => setView(item)}
                    >
                      {item}
                    </button>
                  ))}
                </div>
                {view === "attention" && bundle.attention.graphs.length ? (
                  <label>
                    Layer / head
                    <select
                      value={attentionIndex}
                      onChange={(event) => setAttentionIndex(Number(event.target.value))}
                    >
                      {bundle.attention.graphs.map((item, index) => (
                        <option value={index} key={`${item.layer}-${item.head}`}>
                          Layer {item.layer} · Head {item.head}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <div className="metric-label">
                  Metric: <strong>{graph?.metric}</strong>
                  {selectedAttention ? <span> · L{selectedAttention.layer} H{selectedAttention.head}</span> : null}
                </div>
              </div>
              {graph ? <GraphCanvas graph={graph} onNodeSelect={setSelectedNode} /> : null}
            </div>

            <aside>
              <NodeInspector node={selectedNode} />
              <section className="inspector-card">
                <div className="eyebrow">Next-token distribution</div>
                <h3>Top logits</h3>
                <LogitsPanel logits={bundle.logits.top} />
              </section>
              <section className="inspector-card disclaimer">
                <div className="eyebrow">Interpretation</div>
                <p>Attention and activation are descriptive signals. They are not proof of causal reasoning.</p>
              </section>
            </aside>
          </section>
        </>
      )}
    </main>
  );
}

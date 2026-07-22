import type { GraphNode } from "@/types/bundle";

export function NodeInspector({ node }: { node: GraphNode | null }) {
  return (
    <section className="inspector-card">
      <div className="eyebrow">Node inspector</div>
      {node ? (
        <>
          <h3>{node.label}</h3>
          <dl>
            <div><dt>Type</dt><dd>{node.type}</dd></div>
            {node.layer !== null && node.layer !== undefined ? (
              <div><dt>Layer</dt><dd>{node.layer}</dd></div>
            ) : null}
            {node.token_index !== null && node.token_index !== undefined ? (
              <div><dt>Token position</dt><dd>{node.token_index}</dd></div>
            ) : null}
          </dl>
          <pre>{JSON.stringify(node.metadata, null, 2)}</pre>
        </>
      ) : (
        <p>Select a graph node to inspect its metadata.</p>
      )}
    </section>
  );
}

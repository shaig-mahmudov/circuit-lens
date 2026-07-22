import type { LogitRecord } from "@/types/bundle";

export function LogitsPanel({ logits }: { logits: LogitRecord[] }) {
  const maximum = Math.max(...logits.map((item) => item.probability), 0.001);
  return (
    <div className="logits-list">
      {logits.map((item) => (
        <div className="logit-row" key={item.token_id}>
          <div className="logit-label">
            <code>{item.token.replaceAll(" ", "␠") || "∅"}</code>
            <span>{(item.probability * 100).toFixed(2)}%</span>
          </div>
          <div className="logit-track">
            <div style={{ width: `${(item.probability / maximum) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

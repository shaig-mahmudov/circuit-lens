"use client";

import { useRef, useState, type DragEvent } from "react";

type Props = {
  busy: boolean;
  onFile: (file: File) => Promise<void>;
  onDemo: () => Promise<void>;
};

export function FileDropzone({ busy, onFile, onDemo }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  async function accept(file?: File) {
    if (!file || busy) return;
    await onFile(file);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    void accept(event.dataTransfer.files[0]);
  }

  return (
    <div
      className={`dropzone ${dragging ? "dropzone-active" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        hidden
        type="file"
        accept=".zip,.llmgraph"
        onChange={(event) => void accept(event.target.files?.[0])}
      />
      <div>
        <strong>{busy ? "Reading bundle…" : "Drop a .llmgraph.zip bundle"}</strong>
        <span>All parsing stays in your browser.</span>
      </div>
      <div className="drop-actions">
        <button type="button" onClick={() => inputRef.current?.click()} disabled={busy}>
          Choose file
        </button>
        <button className="button-secondary" type="button" onClick={() => void onDemo()} disabled={busy}>
          Load demo
        </button>
      </div>
    </div>
  );
}

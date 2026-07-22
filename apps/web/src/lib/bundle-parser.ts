import JSZip from "jszip";
import { z } from "zod";

import type { CircuitLensBundle } from "@/types/bundle";

const manifestSchema = z.object({
  format: z.literal("llmgraph"),
  format_version: z.string(),
  engine_version: z.string(),
  run_id: z.string().min(1),
  created_at: z.string(),
  model_id: z.string().min(1),
  capabilities: z.record(z.string(), z.boolean()),
  files: z.array(z.string()),
});

async function readJson<T>(zip: JSZip, path: string): Promise<T> {
  const entry = zip.file(path);
  if (!entry) {
    throw new Error(`Bundle is missing ${path}`);
  }
  return JSON.parse(await entry.async("string")) as T;
}

export async function parseBundle(input: Blob | ArrayBuffer): Promise<CircuitLensBundle> {
  const zip = await JSZip.loadAsync(input);
  const manifest = manifestSchema.parse(await readJson(zip, "manifest.json"));
  if (manifest.format_version !== "0.1.0") {
    throw new Error(`Unsupported bundle version: ${manifest.format_version}`);
  }

  return {
    manifest,
    model: await readJson(zip, "model.json"),
    prompt: await readJson(zip, "prompt.json"),
    tokens: await readJson(zip, "tokens.json"),
    architecture: await readJson(zip, "architecture.json"),
    logits: await readJson(zip, "logits.json"),
    attention: await readJson(zip, "graphs/attention.json"),
    neurons: await readJson(zip, "graphs/neurons.json"),
  } as CircuitLensBundle;
}

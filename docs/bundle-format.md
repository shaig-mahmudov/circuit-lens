# `.llmgraph.zip` format — 0.1.0

Every bundle is a ZIP archive containing:

```text
manifest.json
model.json
prompt.json
tokens.json
architecture.json
logits.json
graphs/attention.json
graphs/neurons.json
```

`manifest.json` is the compatibility entry point. Consumers must reject unknown major/minor formats until a migration exists.

Large raw tensors are intentionally excluded from `0.1.0`. The analysis engine filters attention edges and neuron activations before export so bundles remain suitable for browser use.

# Architecture

CircuitLens is split into two independent parts:

1. **Analysis engine** — runs in Kaggle, captures model internals, reduces tensors, and writes a versioned bundle.
2. **Web explorer** — statically hosted browser application that parses the bundle and renders interactive graphs.

The browser never downloads model weights and the Kaggle notebook is not treated as a persistent API.

```text
Kaggle / Jupyter
  -> circuit_lens.analyze()
  -> reduced Pydantic models
  -> .llmgraph.zip
  -> browser upload
  -> JSZip + validation
  -> React Flow graph
```

## MVP boundaries

Version `0.1.0` supports GPT-2-style Hugging Face models exposing `transformer.h` blocks. Attention edges represent attention probabilities and neuron edges represent post-activation magnitude. They do not claim causality.

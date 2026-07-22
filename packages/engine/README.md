# CircuitLens Engine

Python package used by Kaggle notebooks to inspect supported language models and export portable `.llmgraph.zip` bundles.

```bash
pip install -e ./packages/engine
circuit-lens analyze --model-id gpt2 --prompt "Baku is the capital of" --output /kaggle/working/run.llmgraph.zip
```

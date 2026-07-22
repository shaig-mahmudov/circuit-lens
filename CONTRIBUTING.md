# Contributing

Keep model-specific behavior behind the Python engine and keep the web viewer dependent only on the versioned bundle contract.

Before submitting a change:

```bash
PYTHONPATH=packages/engine/src python -m pytest packages/engine/tests
npm run typecheck
npm run build
```

Do not commit model weights, Hugging Face caches, raw activation dumps, Kaggle credentials, or generated experiment bundles other than the small deterministic demo fixture.

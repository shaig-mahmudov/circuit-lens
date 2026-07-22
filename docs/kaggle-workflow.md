# Kaggle workflow

1. Create a Kaggle Notebook and enable Internet for the first model download.
2. Enable a GPU accelerator when available. GPT-2 Small can also run on CPU.
3. Clone the repository.
4. Install `packages/engine`.
5. Run `AnalysisConfig` and export the bundle to `/kaggle/working`.
6. Download the generated ZIP from the notebook output panel.
7. Open the static CircuitLens web viewer and upload the bundle.

The notebook is a compute job, not a hosted backend. This keeps the viewer deployable as static files and avoids depending on a temporary Kaggle session URL.

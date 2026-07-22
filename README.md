# CircuitLens

**An open-source interactive microscope for exploring attention heads, MLP activations, neuron contributions, and causal circuits inside small language models.**

CircuitLens allows researchers, students, and developers to inspect how information moves through a Transformer model.

Instead of treating a language model as a black box, CircuitLens extracts internal activations in a Kaggle Notebook and converts them into an interactive graph that can be explored in a web browser.

> Prompt → Tokens → Attention Heads → MLP Activations → Residual Stream → Output Logits

---

## Overview

Modern language models contain millions or billions of parameters, making their internal behavior difficult to understand.

CircuitLens focuses on small open-source language models and provides tools for visualizing:

* Transformer architecture
* Token-to-token attention patterns
* Attention-head behavior
* MLP neuron activations
* Direct logit contributions
* Layer-by-layer prediction changes
* Component ablation results
* Activation-patching experiments
* Simplified causal circuits

The model analysis is executed inside a Kaggle Notebook. The generated results are exported as a portable `.llmgraph.zip` bundle and opened inside the CircuitLens web application.

No permanent GPU server is required.

---

## How It Works

```text
┌──────────────────────────────────────┐
│           Kaggle Notebook            │
│                                      │
│  Load the language model             │
│  Tokenize the prompt                 │
│  Capture internal activations        │
│  Extract attention patterns          │
│  Calculate neuron contributions      │
│  Run ablation experiments            │
│  Build a reduced graph               │
└──────────────────┬───────────────────┘
                   │
                   │ Export
                   ▼
          experiment.llmgraph.zip
                   │
                   │ Upload
                   ▼
┌──────────────────────────────────────┐
│       CircuitLens Web Explorer       │
│                                      │
│  Architecture Explorer               │
│  Attention Explorer                  │
│  Neuron Explorer                     │
│  Logit Lens                          │
│  Causal Circuit Viewer               │
└──────────────────────────────────────┘
```

CircuitLens separates model computation from visualization.

The Kaggle Notebook handles expensive model operations, while the browser-based viewer renders the extracted information interactively.

---

## Main Features

### Architecture Explorer

Explore the model from a high-level Transformer view.

```text
Input tokens
     ↓
Token embeddings
     ↓
Transformer layer 0
     ↓
Transformer layer 1
     ↓
...
     ↓
Final normalization
     ↓
Output logits
```

Each Transformer layer can be expanded to display:

* Attention block
* Attention heads
* MLP block
* Residual connections
* Normalization layers
* Tensor dimensions
* Parameter statistics

---

### Attention Explorer

Visualize how individual tokens attend to previous tokens.

Users can select:

* Transformer layer
* Attention head
* Source token
* Target token
* Minimum attention threshold
* Maximum number of visible edges
* Attention metric

Supported metrics may include:

* Raw attention probability
* Attention-weighted value contribution
* Attention-output magnitude

Edge thickness represents the selected attention metric.

---

### Neuron Explorer

Inspect the most active MLP neurons for a particular prompt and token position.

Each neuron node may display:

* Layer index
* Neuron index
* Activation value
* Associated token position
* Direct logit contribution
* Ablation effect
* Top affected output tokens

CircuitLens does not display every neuron simultaneously. It uses top-k filtering and activation thresholds to create readable graphs.

---

### Logit Lens

Observe how the model's prediction changes across Transformer layers.

For each layer, CircuitLens can project the residual stream into vocabulary space and display the top predicted tokens.

Example:

```text
Layer 2:
  city          12%
  country        9%
  capital        7%

Layer 6:
  Azerbaijan    31%
  Georgia       11%
  Turkey         8%

Final layer:
  Azerbaijan    72%
  Georgia        6%
  Turkey         4%
```

---

### Component Ablation

Measure how individual model components affect the final prediction.

CircuitLens can temporarily disable:

* Attention heads
* MLP blocks
* Individual MLP neurons
* Residual-stream components

Example:

```text
Original probability:
Azerbaijan: 72%

After ablating layer 7, attention head 4:
Azerbaijan: 39%

Measured effect:
-33 percentage points
```

A high activation does not necessarily imply causal importance. Ablation experiments help distinguish correlation from causal contribution.

---

### Causal Circuit Viewer

Display a reduced graph containing components that meaningfully affect a selected output token.

Example:

```text
Token: "Baku"
      ↓
Layer 3 · Attention Head 6
      ↓
Layer 8 · MLP Neuron 241
      ↓
Layer 10 · Attention Head 2
      ↓
Output: "Azerbaijan"
```

Circuit edges can represent different metrics:

* Attention probability
* Direct logit attribution
* Activation correlation
* Ablation effect
* Activation-patching effect

Each edge must retain its metric type to avoid presenting correlation as causation.

---

## Supported Models

### Initial Support

* GPT-2 Small

GPT-2 Small is the first supported model because it has a relatively simple architecture and extensive interpretability tooling.

### Planned Support

* SmolLM2-135M
* Pythia-70M
* Pythia-160M
* Other Hugging Face causal language models

Each model is integrated through a model adapter.

```python
class ModelAdapter:
    def tokenize(self, text: str):
        ...

    def run_with_cache(self, tokens):
        ...

    def get_attention(self, layer: int):
        ...

    def get_mlp_activations(self, layer: int):
        ...

    def ablate_component(self, component_id: str):
        ...

    def compute_logit_attribution(self, token_id: int):
        ...
```

---

## Kaggle Workflow

CircuitLens uses Kaggle as the model-computation environment.

### Basic Workflow

1. Open the CircuitLens Kaggle Notebook.
2. Enable a GPU accelerator when available.
3. Set the model and prompt.
4. Run all notebook cells.
5. Download the generated `.llmgraph.zip` file.
6. Open the CircuitLens web application.
7. Drag and drop the bundle into the explorer.

Example notebook configuration:

```python
MODEL_ID = "gpt2"
PROMPT = "Baku is the capital of"
TARGET_TOKEN = " Azerbaijan"

TOP_NEURONS = 30
TOP_ATTENTION_EDGES = 5

CAPTURE_ATTENTION = True
CAPTURE_MLP = True
CAPTURE_RESIDUAL_STREAM = True

RUN_LOGIT_ATTRIBUTION = True
RUN_ABLATION = False
RUN_ACTIVATION_PATCHING = False
```

The notebook should contain minimal implementation logic. Most analysis code lives inside the reusable Python engine.

```python
from circuit_lens import analyze

result = analyze(
    model_id=MODEL_ID,
    prompt=PROMPT,
    target_token=TARGET_TOKEN,
)

result.export("/kaggle/working/experiment.llmgraph.zip")
```

---

## Bundle Format

CircuitLens uses a portable experiment format:

```text
experiment.llmgraph.zip
├── manifest.json
├── model.json
├── prompt.json
├── tokens.json
├── architecture.json
├── logits.json
├── graphs/
│   ├── attention.json
│   ├── neurons.json
│   └── circuit.json
├── tensors/
│   ├── attention.npz
│   ├── activations.npz
│   └── residuals.npz
└── experiments/
    ├── attribution.json
    ├── ablation.json
    └── activation_patching.json
```

Example manifest:

```json
{
  "format": "llmgraph",
  "format_version": "0.1.0",
  "run_id": "gpt2-baku-capital",
  "model": {
    "id": "gpt2",
    "revision": "main"
  },
  "prompt": "Baku is the capital of",
  "target_token": " Azerbaijan",
  "capabilities": {
    "attention": true,
    "mlp_activations": true,
    "logit_attribution": true,
    "ablation": false,
    "activation_patching": false
  }
}
```

The web application validates every uploaded bundle before displaying it.

---

## Repository Structure

```text
circuit-lens/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── ROADMAP.md
├── CITATION.cff
│
├── apps/
│   └── web/
│       ├── src/
│       │   ├── app/
│       │   ├── components/
│       │   ├── graph/
│       │   ├── parsers/
│       │   ├── schemas/
│       │   ├── stores/
│       │   ├── types/
│       │   └── utils/
│       └── package.json
│
├── packages/
│   ├── engine/
│   │   ├── src/
│   │   │   └── circuit_lens/
│   │   │       ├── models/
│   │   │       ├── capture/
│   │   │       ├── attribution/
│   │   │       ├── experiments/
│   │   │       ├── graph/
│   │   │       └── export/
│   │   └── tests/
│   │
│   └── schema/
│       ├── manifest.schema.json
│       ├── graph.schema.json
│       └── experiment.schema.json
│
├── notebooks/
│   ├── 00_kaggle_quickstart.ipynb
│   ├── 01_model_architecture.ipynb
│   ├── 02_attention_explorer.ipynb
│   ├── 03_neuron_activations.ipynb
│   ├── 04_logit_attribution.ipynb
│   ├── 05_component_ablation.ipynb
│   └── 06_circuit_export.ipynb
│
├── examples/
│   ├── example.llmgraph.zip
│   └── screenshots/
│
├── scripts/
│   ├── inspect_model.py
│   ├── validate_bundle.py
│   └── create_example_bundle.py
│
├── docs/
│   ├── architecture.md
│   ├── bundle-format.md
│   ├── kaggle-workflow.md
│   ├── model-adapters.md
│   ├── metrics.md
│   └── interpretability-limitations.md
│
└── .github/
    ├── workflows/
    ├── ISSUE_TEMPLATE/
    └── pull_request_template.md
```

---

## Technology Stack

### Analysis Engine

* Python
* PyTorch
* Hugging Face Transformers
* TransformerLens
* NumPy
* Safetensors
* Pydantic
* JSON Schema

### Web Application

* Next.js
* React
* TypeScript
* React Flow
* Sigma.js
* Canvas or D3.js
* Zustand
* JSZip

### Infrastructure

* Kaggle Notebooks for model computation
* Static hosting for the web explorer
* GitHub Actions for testing and validation
* Kaggle Datasets or GitHub Releases for large example bundles

---

# Development Roadmap

## Phase 0 — Project Foundation

The goal of this phase is to create a stable repository and shared data format.

### Tasks

* Create the monorepo
* Configure Python packaging
* Configure the Next.js application
* Define the `.llmgraph` bundle format
* Add JSON schemas
* Configure linting and formatting
* Add Python and frontend test workflows
* Add a basic example bundle
* Write project documentation

### Completion Criteria

* Python engine can be installed
* Web application can run
* Example bundle passes schema validation
* CI runs successfully

---

## Phase 1 — Model Architecture Explorer

The goal is to extract and display the structure of GPT-2 Small.

### Analysis Engine

* Load GPT-2 Small
* Extract model configuration
* Extract layer information
* Extract tensor dimensions
* Calculate parameter counts
* Export architecture metadata

### Web Explorer

* Upload `.llmgraph.zip`
* Validate the uploaded bundle
* Display Transformer layers
* Expand individual layers
* Display attention and MLP blocks
* Add zoom, pan, and minimap controls
* Add node-information panel

### Completion Criteria

A user can upload an experiment bundle and interactively inspect the GPT-2 architecture.

---

## Phase 2 — Attention Explorer

The goal is to visualize token-to-token attention patterns.

### Analysis Engine

* Capture attention patterns
* Extract per-layer attention data
* Extract per-head attention data
* Apply top-k edge filtering
* Apply attention thresholds
* Export attention graphs

### Web Explorer

* Add layer selector
* Add attention-head selector
* Add token selector
* Render token-to-token edges
* Scale edge width by attention score
* Add hover details
* Add attention heatmap
* Add incoming and outgoing attention views

### Completion Criteria

A user can select a layer and attention head and inspect which tokens attend to one another.

---

## Phase 3 — Neuron Explorer

The goal is to inspect important MLP activations.

### Analysis Engine

* Capture MLP pre-activations
* Capture MLP post-activations
* Select the most active neurons
* Associate neurons with token positions
* Calculate activation statistics
* Export reduced neuron graphs

### Web Explorer

* Display top activated neurons
* Filter by layer
* Filter by token
* Filter by activation magnitude
* Display positive and negative activations
* Add neuron-details panel

### Completion Criteria

A user can inspect the most active MLP neurons for each token and layer.

---

## Phase 4 — Logit Lens and Attribution

The goal is to connect internal components to output-token predictions.

### Analysis Engine

* Calculate intermediate vocabulary projections
* Extract top predictions by layer
* Calculate direct logit attribution
* Calculate component-level contributions
* Export output-token statistics

### Web Explorer

* Add layer-by-layer prediction view
* Add output-token selector
* Display top output probabilities
* Display positive and negative contributions
* Highlight components affecting the selected token

### Completion Criteria

A user can observe when a prediction begins to form and which components contribute to it.

---

## Phase 5 — Component Ablation

The goal is to measure causal importance.

### Analysis Engine

* Add attention-head ablation
* Add MLP-block ablation
* Add individual-neuron ablation
* Compare original and modified logits
* Calculate probability changes
* Rank components by causal effect

### Web Explorer

* Display original and ablated results
* Add causal-effect metric
* Rank components by impact
* Filter the graph using ablation scores
* Compare multiple experiments

### Completion Criteria

A user can identify components whose removal changes the selected model prediction.

---

## Phase 6 — Causal Circuit Tracing

The goal is to generate simplified causal paths through the model.

### Analysis Engine

* Add activation patching
* Add path-patching experiments
* Calculate circuit importance scores
* Remove low-impact nodes and edges
* Generate reduced causal graphs

### Web Explorer

* Add Circuit Mode
* Display causal paths
* Animate information flow
* Compare attention, attribution, and causal metrics
* Isolate a selected output token

### Completion Criteria

A user can explore a reduced circuit containing the components that causally influence a selected prediction.

---

## Phase 7 — Additional Model Support

The goal is to support multiple small open-source language models.

### Tasks

* Implement SmolLM2-135M adapter
* Implement Pythia-70M adapter
* Implement Pythia-160M adapter
* Normalize architecture metadata
* Add model capability registry
* Document unsupported model features

### Completion Criteria

At least three different model families can generate compatible `.llmgraph` bundles.

---

## Phase 8 — Sparse Autoencoder Features

The goal is to move from raw neurons to more interpretable latent features.

### Tasks

* Add sparse autoencoder support
* Capture feature activations
* Display top activating examples
* Connect features to output logits
* Add feature ablation
* Add feature steering experiments
* Create feature-labeling workflow

### Completion Criteria

Users can explore interpretable SAE features in addition to individual neurons.

---

## Scientific Limitations

CircuitLens is an interpretability tool, not a complete explanation of model reasoning.

Important limitations include:

* Attention does not automatically represent causal importance.
* High activation does not necessarily mean that a neuron caused an output.
* Individual neurons may represent multiple unrelated concepts.
* Causal conclusions require controlled interventions.
* Top-k filtering can hide smaller but distributed effects.
* Graph visualizations simplify high-dimensional computations.
* Direct logit attribution does not capture every nonlinear interaction.
* Sparse autoencoder features may still be incomplete or ambiguous.

CircuitLens should clearly distinguish between:

```text
Correlation
Contribution
Attribution
Intervention
Causation
```

---

## MVP Scope

The first usable version will include:

* GPT-2 Small support
* Kaggle Quickstart Notebook
* Prompt tokenization
* Model architecture extraction
* Attention extraction
* Top MLP activation extraction
* Top output logits
* `.llmgraph.zip` export
* Bundle upload
* Architecture graph
* Attention graph
* Neuron activation graph
* Node-details panel

The following features are intentionally excluded from the first MVP:

* Activation patching
* Sparse autoencoders
* Feature steering
* Automatic neuron labeling
* Large language models
* Real-time hosted model inference
* Permanent GPU backend

---

## Contributing

Contributions are welcome.

Possible contribution areas include:

* New model adapters
* Interpretability methods
* Graph layouts
* Performance optimization
* Kaggle notebooks
* Documentation
* Example experiments
* Bundle-format improvements

Before opening a pull request, please create or reference an issue describing the proposed change.

---

## License

CircuitLens is released under the Apache License 2.0.

Model weights and datasets may use different licenses. Users are responsible for reviewing the license of each model they analyze.

---

## Disclaimer

CircuitLens is intended for education and research.

The visualizations produced by the project are simplified representations of high-dimensional model computations and should not be interpreted as complete or definitive explanations of model behavior.

"""GPT-2 MVP analyzer used in Kaggle and other Jupyter environments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import torch

from .graph import architecture_graph, attention_graphs, neuron_graph
from .schema import (
    AnalysisBundle,
    CapabilityDocument,
    LayerPrediction,
    LogitRecord,
    LogitsDocument,
    Manifest,
    ModelDocument,
    PromptDocument,
    TokenRecord,
)


@dataclass(slots=True)
class AnalysisConfig:
    model_id: str = "gpt2"
    prompt: str = "Baku is the capital of"
    target_token: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    top_logits: int = 10
    top_neurons: int = 30
    top_attention_edges: int = 5
    minimum_attention: float = 0.02


def _safe_run_id(model_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-").lower()
    return f"{slug}-{timestamp}"


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    return torch.float16 if device.type == "cuda" else torch.float32


def _display_token(text: str) -> str:
    if not text:
        return "∅"
    return text.replace("\n", "↵").replace("\t", "⇥").replace(" ", "␠")


def _top_logits(logits: torch.Tensor, tokenizer: Any, count: int) -> list[LogitRecord]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    values, token_ids = torch.topk(probabilities, k=min(count, probabilities.numel()))
    records: list[LogitRecord] = []
    for probability, token_id in zip(values.tolist(), token_ids.tolist(), strict=True):
        records.append(
            LogitRecord(
                token_id=token_id,
                token=tokenizer.decode([token_id]),
                probability=float(probability),
                logit=float(logits[token_id]),
            )
        )
    return records


def _load_transformers() -> tuple[Any, Any, int]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Transformers is not installed. Run `pip install -e ./packages/engine` first."
        ) from exc
    try:
        major_version = int(__version__.split(".", maxsplit=1)[0])
    except ValueError:
        major_version = 5
    return AutoModelForCausalLM, AutoTokenizer, major_version


def analyze(config: AnalysisConfig) -> AnalysisBundle:
    """Analyze a GPT-2-style causal LM and return a reduced portable bundle."""

    AutoModelForCausalLM, AutoTokenizer, transformers_major = _load_transformers()
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    # Transformers v5 prefers `dtype`; v4 uses `torch_dtype`.
    dtype_argument = "dtype" if transformers_major >= 5 else "torch_dtype"
    load_kwargs: dict[str, Any] = {dtype_argument: dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError:  # Transformers versions before attn_implementation support
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)

    model.to(device)
    model.eval()

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError(
            "The MVP analyzer currently supports GPT-2-style models exposing transformer.h blocks."
        )

    mlp_activations: list[torch.Tensor | None] = [None] * len(model.transformer.h)
    hooks: list[Any] = []

    for layer_index, block in enumerate(model.transformer.h):
        def capture_mlp(_module: Any, _inputs: Any, output: torch.Tensor, *, index: int = layer_index) -> None:
            mlp_activations[index] = model.transformer.h[index].mlp.act(output).detach()

        hooks.append(block.mlp.c_fc.register_forward_hook(capture_mlp))

    encoded = tokenizer(config.prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    try:
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    if outputs.attentions is None:
        raise RuntimeError(
            "The model did not return attention tensors. Use the eager attention implementation."
        )
    if outputs.hidden_states is None:
        raise RuntimeError("The model did not return hidden states.")
    if any(item is None for item in mlp_activations):
        raise RuntimeError("One or more MLP activation hooks did not run.")

    input_ids = encoded["input_ids"][0].tolist()
    tokens = [
        TokenRecord(
            index=index,
            token_id=token_id,
            text=tokenizer.decode([token_id]),
            display=_display_token(tokenizer.decode([token_id])),
        )
        for index, token_id in enumerate(input_ids)
    ]

    final_logits = outputs.logits[0, -1].detach().cpu()
    by_layer: list[LayerPrediction] = []
    hidden_states = list(outputs.hidden_states)
    for index, hidden in enumerate(hidden_states):
        state = hidden[:, -1, :]
        # GPT-2's final hidden state is already normalized; earlier states are not.
        projected_state = state if index == len(hidden_states) - 1 else model.transformer.ln_f(state)
        layer_logits = model.lm_head(projected_state)[0].detach().cpu()
        label = "embedding" if index == 0 else (
            "final" if index == len(hidden_states) - 1 else f"layer-{index}"
        )
        by_layer.append(
            LayerPrediction(
                layer=label,
                predictions=_top_logits(layer_logits, tokenizer, config.top_logits),
            )
        )

    model_config = model.config
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    intermediate_size = int(
        getattr(model_config, "n_inner", None)
        or model.transformer.h[0].mlp.c_fc.weight.shape[-1]
    )

    file_names = [
        "manifest.json",
        "model.json",
        "prompt.json",
        "tokens.json",
        "architecture.json",
        "logits.json",
        "graphs/attention.json",
        "graphs/neurons.json",
    ]

    return AnalysisBundle(
        manifest=Manifest(
            run_id=_safe_run_id(config.model_id),
            model_id=config.model_id,
            capabilities=CapabilityDocument(),
            files=file_names,
        ),
        model=ModelDocument(
            id=config.model_id,
            architecture=model.__class__.__name__,
            parameter_count=parameter_count,
            num_layers=int(model_config.n_layer),
            num_attention_heads=int(model_config.n_head),
            hidden_size=int(model_config.n_embd),
            intermediate_size=intermediate_size,
            vocabulary_size=int(model_config.vocab_size),
            context_length=int(getattr(model_config, "n_positions", 0)),
            dtype=str(dtype).replace("torch.", ""),
            device=str(device),
        ),
        prompt=PromptDocument(text=config.prompt, target_token=config.target_token),
        tokens=tokens,
        architecture=architecture_graph(
            num_layers=int(model_config.n_layer),
            num_heads=int(model_config.n_head),
            intermediate_size=intermediate_size,
        ),
        attention=attention_graphs(
            outputs.attentions,
            tokens,
            top_edges_per_token=config.top_attention_edges,
            minimum_weight=config.minimum_attention,
        ),
        neurons=neuron_graph(
            [item for item in mlp_activations if item is not None],
            tokens,
            top_neurons=config.top_neurons,
        ),
        logits=LogitsDocument(
            top=_top_logits(final_logits, tokenizer, config.top_logits),
            by_layer=by_layer,
        ),
    )

"""Read and write the portable CircuitLens bundle format."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .schema import AnalysisBundle


def _json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump_json"):
        text = value.model_dump_json(indent=2)  # type: ignore[attr-defined]
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False)
    return text.encode("utf-8")


def export_bundle(bundle: AnalysisBundle, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[str, object] = {
        "manifest.json": bundle.manifest,
        "model.json": bundle.model,
        "prompt.json": bundle.prompt,
        "tokens.json": [token.model_dump(mode="json") for token in bundle.tokens],
        "architecture.json": bundle.architecture,
        "logits.json": bundle.logits,
        "graphs/attention.json": bundle.attention,
        "graphs/neurons.json": bundle.neurons,
    }

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in entries.items():
            archive.writestr(name, _json_bytes(value))

    return output


def read_bundle(path: str | Path) -> AnalysisBundle:
    with ZipFile(path, "r") as archive:
        def read_json(name: str) -> object:
            return json.loads(archive.read(name).decode("utf-8"))

        payload = {
            "manifest": read_json("manifest.json"),
            "model": read_json("model.json"),
            "prompt": read_json("prompt.json"),
            "tokens": read_json("tokens.json"),
            "architecture": read_json("architecture.json"),
            "attention": read_json("graphs/attention.json"),
            "neurons": read_json("graphs/neurons.json"),
            "logits": read_json("logits.json"),
        }
    return AnalysisBundle.model_validate(payload)

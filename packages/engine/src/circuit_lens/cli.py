"""Command-line interface for Kaggle and local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer import AnalysisConfig, analyze
from .bundle import export_bundle
from .demo import create_demo_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="circuit-lens")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a supported model")
    analyze_parser.add_argument("--model-id", default="gpt2")
    analyze_parser.add_argument("--prompt", required=True)
    analyze_parser.add_argument("--target-token")
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--device", default="auto")
    analyze_parser.add_argument("--dtype", default="auto")
    analyze_parser.add_argument("--top-logits", type=int, default=10)
    analyze_parser.add_argument("--top-neurons", type=int, default=30)
    analyze_parser.add_argument("--top-attention-edges", type=int, default=5)
    analyze_parser.add_argument("--minimum-attention", type=float, default=0.02)

    demo_parser = subparsers.add_parser("demo", help="Generate a deterministic demo bundle")
    demo_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        output = export_bundle(create_demo_bundle(), args.output)
    else:
        config = AnalysisConfig(
            model_id=args.model_id,
            prompt=args.prompt,
            target_token=args.target_token,
            device=args.device,
            dtype=args.dtype,
            top_logits=args.top_logits,
            top_neurons=args.top_neurons,
            top_attention_edges=args.top_attention_edges,
            minimum_attention=args.minimum_attention,
        )
        output = export_bundle(analyze(config), args.output)
    print(output)


if __name__ == "__main__":
    main()

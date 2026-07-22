from pathlib import Path

from circuit_lens.bundle import export_bundle
from circuit_lens.demo import create_demo_bundle


if __name__ == "__main__":
    destination = Path("apps/web/public/examples/gpt2-demo.llmgraph.zip")
    print(export_bundle(create_demo_bundle(), destination))

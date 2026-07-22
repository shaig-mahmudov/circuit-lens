from zipfile import ZipFile

from circuit_lens.bundle import export_bundle, read_bundle
from circuit_lens.demo import create_demo_bundle


def test_demo_bundle_round_trip(tmp_path):
    path = tmp_path / "demo.llmgraph.zip"
    export_bundle(create_demo_bundle(), path)

    with ZipFile(path) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "model.json",
            "prompt.json",
            "tokens.json",
            "architecture.json",
            "logits.json",
            "graphs/attention.json",
            "graphs/neurons.json",
        }

    restored = read_bundle(path)
    assert restored.manifest.format == "llmgraph"
    assert restored.model.id == "gpt2"
    assert restored.attention.graphs[0].layer == 7
    assert restored.logits.top[0].token.strip() == "Azerbaijan"

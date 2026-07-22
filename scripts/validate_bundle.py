import argparse

from circuit_lens.bundle import read_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    args = parser.parse_args()
    bundle = read_bundle(args.bundle)
    print(
        f"valid: format={bundle.manifest.format_version} "
        f"model={bundle.model.id} run={bundle.manifest.run_id}"
    )

"""Merge the revision analyses into one JSON digest that the revised text is checked against.

Reads output/revision/*.json (written by the run_revision_* scripts and the R1 digest build) and
writes output/revision/revision_digest.json. Every number quoted in the revised manuscript and the
response letter should be traceable to this file or to the text outputs of the controller reruns.
"""
from __future__ import annotations

import json
from pathlib import Path

REV = Path(__file__).resolve().parents[1] / "output" / "revision"
PARTS = {
    "digest_R1": "digest_R1.json",
    "fair_ladder": "fair_ladder.json",
    "envelope": "envelope.json",
    "reliability": "reliability_baselines.json",
    "selective": "selective.json",
    "decoded_fair_gpt2": "decoded_fair_gpt2.json",
    "decoded_fair_ngram5_kn": "decoded_fair_ngram5_kn.json",
    "prior_audit": "prior_audit.json",
}


def main() -> None:
    merged, missing = {}, []
    for key, name in PARTS.items():
        path = REV / name
        if path.exists():
            merged[key] = json.loads(path.read_text())
        else:
            missing.append(name)
    (REV / "revision_digest.json").write_text(json.dumps(merged, indent=1))
    print(f"wrote revision_digest.json with {len(merged)} parts; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()

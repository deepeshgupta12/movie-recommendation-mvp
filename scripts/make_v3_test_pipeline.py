# scripts/make_v3_test_pipeline.py

from __future__ import annotations

import importlib
import sys

PIPELINE = [
    "src.neural.export_user_embeddings_test_v3",
    "src.neural.ann_generate_candidates_test_v3",
    "src.sequence.generate_candidates_test_v3",
    "src.ranking.export_v2_candidates_test_v3",
    "src.ranking.blend_candidates_v3_test",
    "src.ranking.build_v3_test_pointwise",
    "src.ranking.train_ranker_v3_test",
    "scripts.eval_ranked_candidates_v3_test",
    # optionally swap last line with:
    # "scripts.eval_ranked_candidates_v3_test_strict",
]


def _run(module_name: str) -> None:
    print(f"\n========== RUN: {module_name} ==========")
    mod = importlib.import_module(module_name)

    if hasattr(mod, "main"):
        mod.main()
    else:
        raise RuntimeError(f"Module {module_name} has no main()")


def main():
    failed = []

    for m in PIPELINE:
        try:
            _run(m)
        except Exception as e:
            print(f"\n[ERROR] {m} failed: {e}")
            failed.append((m, str(e)))
            break

    if failed:
        print("\n[PIPELINE] Stopped due to error.")
        for m, err in failed:
            print(f" - {m}: {err}")
        sys.exit(1)

    print("\n[PIPELINE] V3 TEST pipeline complete.")


if __name__ == "__main__":
    main()
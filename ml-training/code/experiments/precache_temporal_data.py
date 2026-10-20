"""
Pre-caches raw window sequences to .npz for fast sequential loading in Phase 13B.
Operates strictly on Train (n=163) and Dev (n=56) splits. Official Test split is strictly quarantined.
"""

import sys
from pathlib import Path
import time
import pandas as pd

# Add ml-training/code to path
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core.config import DEEP_FEATURES_DIR, LABELS_DIR
from data.split_loader import load_official_splits
from data.temporal_loader import get_participant_sequence, TEMPORAL_CACHE_DIR


def main():
    print("=" * 70)
    print("Pre-caching Unpooled Window Sequences for Phase 13B")
    print(f"Cache Directory: {TEMPORAL_CACHE_DIR}")
    print("=" * 70)

    official = load_official_splits(labels_dir=LABELS_DIR)
    p_inv = pd.read_csv(DEEP_FEATURES_DIR / "metadata" / "participant_inventory.csv")

    pids_to_cache = [("Train", official.train_pids), ("Dev", official.dev_pids)]
    representations = ["vgg16", "densenet201"]

    t0 = time.time()
    total_processed = 0

    for split_name, pids in pids_to_cache:
        print(f"\nProcessing {split_name} split ({len(pids)} participants)...")
        for rep in representations:
            print(f"  Caching {rep} sequences...")
            for i, pid in enumerate(pids, 1):
                cache_file = TEMPORAL_CACHE_DIR / f"{pid}_{rep}.npz"
                if not cache_file.exists():
                    X, ts = get_participant_sequence(pid, rep, p_inv)
                    total_processed += 1
                if i % 40 == 0 or i == len(pids):
                    print(f"    [{rep}] {i}/{len(pids)} cached")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Pre-caching completed in {elapsed:.1f}s ({total_processed} new .npz files created).")
    print(f"Total cached files in {TEMPORAL_CACHE_DIR}: {len(list(TEMPORAL_CACHE_DIR.glob('*.npz')))}")
    print("=" * 70)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


# 5-way attention direction labels (used by the JS collector + MobileNetV2 softmax training)
LABELS = ["screen", "away_left", "away_right", "away_up", "away_down"]


@dataclass(frozen=True)
class SampleGroup:
    participant: str
    session: str
    placement: str
    label: str
    files: list[Path]


def iter_face_groups(root: Path) -> list[SampleGroup]:
    """
    Expects structure produced by the JS collector zip:
      face/<participant>/<session>/<placement>/<label>/*.jpg
    """
    groups: list[SampleGroup] = []
    face_root = root / "face"
    if not face_root.exists():
        return []

    for participant_dir in sorted([p for p in face_root.iterdir() if p.is_dir()]):
        for session_dir in sorted([p for p in participant_dir.iterdir() if p.is_dir()]):
            for placement_dir in sorted([p for p in session_dir.iterdir() if p.is_dir()]):
                for label in LABELS:
                    lab_dir = placement_dir / label
                    if not lab_dir.exists():
                        continue
                    files = sorted([p for p in lab_dir.glob("*.jpg") if p.is_file()])
                    if files:
                        groups.append(
                            SampleGroup(
                                participant=participant_dir.name,
                                session=session_dir.name,
                                placement=placement_dir.name,
                                label=label,
                                files=files,
                            )
                        )
    return groups


def main():
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits from collected face crops.")
    parser.add_argument("--runs-dir", default="data", help="Directory containing run_* folders")
    parser.add_argument("--out-dir", default="data/splits", help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--split-by",
        choices=["participant", "session"],
        default="participant",
        help="How to split to avoid leakage (recommended: participant if you have >=3 participants)",
    )
    parser.add_argument(
        "--holdout-participant",
        default=None,
        help="If set (requires split-by=participant), force this participant into the test split (LOPO).",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-per-group", type=int, default=800, help="Cap samples per (person/session/label) group")
    parser.add_argument("--copy", action="store_true", help="Copy files (default: symlink if possible, else copy)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in runs_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run_* folders found in {runs_dir}")

    groups: list[SampleGroup] = []
    for r in run_dirs:
        groups.extend(iter_face_groups(r))

    if not groups:
        raise SystemExit("No face crops found. Expected face/<participant>/<session>/<placement>/<label>/*.jpg")

    # Split keys
    if args.split_by == "participant":
        keys = sorted({g.participant for g in groups})
    else:
        keys = sorted({f"{g.participant}/{g.session}" for g in groups})

    rng = random.Random(args.seed)
    rng.shuffle(keys)

    holdout = (args.holdout_participant or "").strip() or None
    if holdout:
        if args.split_by != "participant":
            raise SystemExit("--holdout-participant requires --split-by participant")
        if holdout not in set(keys):
            raise SystemExit(f"Holdout participant '{holdout}' not found. Available: {sorted(set(keys))}")
        # Remove holdout from the pool used to build train/val; it will be assigned to test.
        keys = [k for k in keys if k != holdout]

    n = len(keys)
    n_test = max(1, int(round(n * args.test_frac))) if n >= 3 else 1
    n_val = max(1, int(round(n * args.val_frac))) if n >= 3 else 1
    n_train = max(1, n - n_val - n_test)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set([holdout]) if holdout else set(keys[n_train + n_val :])

    def key_for(g: SampleGroup) -> str:
        return g.participant if args.split_by == "participant" else f"{g.participant}/{g.session}"

    def split_for(g: SampleGroup) -> str:
        k = key_for(g)
        if k in test_keys:
            return "test"
        if k in val_keys:
            return "val"
        return "train"

    # Clear existing output
    for split in ["train", "val", "test"]:
        for lab in LABELS:
            d = out_dir / split / lab
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    def put(dst: Path, src: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not args.copy:
            try:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src)
                return
            except Exception:
                pass
        shutil.copy2(src, dst)

    counts = {(split, lab): 0 for split in ["train", "val", "test"] for lab in LABELS}

    for g in groups:
        split = split_for(g)
        files = g.files[:]
        rng.shuffle(files)
        if args.max_per_group > 0:
            files = files[: args.max_per_group]

        for src in files:
            # Preserve provenance in filename
            stem = src.stem
            dst_name = f"{g.participant}__{g.session}__{g.placement}__{stem}.jpg"
            dst = out_dir / split / g.label / dst_name
            put(dst, src)
            counts[(split, g.label)] += 1

    print("Wrote dataset splits to:", out_dir)
    print("Split keys:", args.split_by)
    if holdout:
        print("Holdout participant (forced test):", holdout)
    print("Counts:")
    for split in ["train", "val", "test"]:
        parts = "  ".join([f"{lab}={counts[(split, lab)]:6d}" for lab in LABELS])
        print(f"  {split:5s}  {parts}")
    print("\nTrain your model with:")
    print(f"  python train/train_tf.py --train-dir \"{out_dir / 'train'}\" --val-dir \"{out_dir / 'val'}\" --test-dir \"{out_dir / 'test'}\"")


if __name__ == "__main__":
    main()


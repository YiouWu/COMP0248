"""
Make split JSON for RGB-D dataset 

A) data_root/Subject/G01_call/clip01/...
B) data_root/Wrapper/Subject/G01_call/clip01/...
C) data_root/G01_call/clip01/...

- This script scans the dataset directory, discovers "subjects", then writes a JSON split file.
- Supports three folder layouts (A/B/C), skips invalid folders robustly.
- Default: subject-level train/val split.
- With --test_only: only writes test_subjects (can be just one subject or '.' when root itself is the subject).

"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List


GESTURES = ["G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",]


def is_subject_dir(p: Path) -> bool:
    """Subject dir contains gesture folders like G01_call ... (at least one)."""
    if not p.is_dir():
        return False
    name = p.name.lower()
    if name.startswith("__") or name == "__macosx":
        return False
    return any((p / g).is_dir() for g in GESTURES)


def find_subjects(data_root: Path, verbose: bool = False) -> List[str]:

    subjects: List[str] = []

    # Case C: data_root itself is a subject 
    if is_subject_dir(data_root):
        return ["."]

    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        pname = p.name.lower()
        if pname.startswith("__") or pname == "__macosx":
            continue

        # Case A: p is subject
        if is_subject_dir(p):
            subjects.append(p.name.replace("\\", "/"))
            continue

        # Case B: p is wrapper, check children
        found_child = 0
        for q in sorted(p.iterdir()):
            if is_subject_dir(q):
                subjects.append(f"{p.name}/{q.name}".replace("\\", "/"))
                found_child += 1

        if verbose and found_child == 0:
            print(f"[skip] not subject / wrapper: {p}")

    subjects = sorted(list(set(subjects)))
    return subjects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True) # dataset root folder (required)
    ap.add_argument("--out", type=str, default="splits.json")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42) 
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--test_only", action="store_true",
                    help="Only write test_subjects (no train/val). Useful for independent test folder.") # For test
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    subjects = find_subjects(data_root, verbose=args.verbose)

    if len(subjects) == 0:
        raise RuntimeError(
            f"No subjects found under {data_root}.\n"
            f"Expected either data_root/Subject/G01_call/... OR data_root/G01_call/..."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # test-only mode
    if args.test_only:
        payload = {
            "split_mode": "subject",
            "data_root": str(data_root),
            "seed": args.seed,
            "val_frac": args.val_frac,
            "train_subjects": [],
            "val_subjects": [],
            "test_subjects": subjects,
            "num_subjects": len(subjects),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] Wrote TEST split to: {out_path}")
        print(f"     subjects found: {len(subjects)}")
        print(f"     test subjects:  {len(subjects)}")
        return

    # normal train/val subject split 
    if len(subjects) < 2:
        raise RuntimeError(
            f"Not enough subjects for train/val split in {data_root} (found {len(subjects)}).\n"
            f"Tip: if this is an independent test folder, use --test_only."
        )

    random.seed(args.seed)
    random.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * args.val_frac)))
    val_subjects = sorted(subjects[:n_val])
    train_subjects = sorted(subjects[n_val:])

    payload = {
        "split_mode": "subject",
        "data_root": str(data_root),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "num_subjects": len(subjects),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote split to: {out_path}")
    print(f"     subjects found: {len(subjects)}")
    print(f"     train subjects: {len(train_subjects)}")
    print(f"     val subjects:   {len(val_subjects)}")




if __name__ == "__main__":
    main()
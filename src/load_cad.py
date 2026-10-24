"""
Load the Contextual Abuse Dataset (CAD) and map its labels to our three classes.

Data: Vidgen et al. (2021), "Introducing CAD: the Contextual Abuse Dataset".
Download from Zenodo (https://zenodo.org/records/4881008) and put these files in
data/raw/cad/:

    cad_v1_1_train.tsv
    cad_v1_1_dev.tsv
    cad_v1_1_test.tsv

Each split file has three tab-separated columns: id, text, labels. `labels` is a
comma-separated list because an entry can have more than one primary label.

Our three classes:
    group   - IdentityDirectedAbuse or AffiliationDirectedAbuse
    person  - PersonDirectedAbuse (and no group-directed label)
    neutral - everything else (Neutral, CounterSpeech, non-hateful slurs)

If an entry has both a group-directed and a person-directed label, it counts as
group, since group-directed abuse is our outcome of interest.

Usage:
    python src/load_cad.py                 # prints class counts per split
    python src/load_cad.py --data-dir path/to/cad
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "cad"
SPLITS = ("train", "dev", "test")

GROUP_LABELS = {"IdentityDirectedAbuse", "AffiliationDirectedAbuse"}
PERSON_LABELS = {"PersonDirectedAbuse"}
NEUTRAL_LABELS = {"Neutral", "CounterSpeech", "Slur"}
KNOWN_LABELS = GROUP_LABELS | PERSON_LABELS | NEUTRAL_LABELS

CLASSES = ["neutral", "person", "group"]


def map_labels(label_string: str) -> str:
    """Map CAD's comma-separated primary labels to one of our three classes."""
    labels = {l.strip() for l in label_string.split(",") if l.strip()}
    unknown = labels - KNOWN_LABELS
    if unknown:
        raise ValueError(f"Unexpected CAD label(s): {sorted(unknown)}")
    if labels & GROUP_LABELS:
        return "group"
    if labels & PERSON_LABELS:
        return "person"
    return "neutral"


def clean_text(text: str) -> str:
    """Undo CAD's [linebreak] encoding. Model preprocessing happens later."""
    return text.replace("[linebreak]", "\n").replace("\n ", "\n").strip()


def load_split(split: str, data_dir: Path = DEFAULT_DIR, version: str = "v1_1") -> pd.DataFrame:
    """Load one CAD split as a DataFrame with columns: id, text, cad_labels, label."""
    path = Path(data_dir) / f"cad_{version}_{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download CAD from https://zenodo.org/records/4881008 "
            f"and place the split files in {data_dir}."
        )
    # QUOTE_NONE and keep_default_na=False match how the CAD authors read these files.
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False, dtype=str)
    df = df.rename(columns={"labels": "cad_labels"})
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["cad_labels"].map(map_labels)
    df["split"] = split
    return df[["id", "text", "cad_labels", "label", "split"]]


def load_cad(data_dir: Path = DEFAULT_DIR, version: str = "v1_1") -> dict:
    """Load all three splits. Returns {"train": df, "dev": df, "test": df}."""
    return {s: load_split(s, data_dir, version) for s in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    splits = load_cad(args.data_dir)
    for name, df in splits.items():
        counts = df["label"].value_counts().reindex(CLASSES, fill_value=0)
        shares = (counts / len(df)).round(3)
        print(f"{name:5s}  n={len(df):6d}  " + "  ".join(f"{c}={counts[c]} ({shares[c]:.1%})" for c in CLASSES))


if __name__ == "__main__":
    main()
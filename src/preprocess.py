"""
Text cleaning for model input, plus a CAD loader that adds the cleaned text.

clean_text() is used on both CAD (training) and the Reddit sample (inference),
so it has to handle raw Reddit text as well as CAD's version of it. The two
differ a little: Arctic Shift returns HTML-escaped bodies ("&gt;" for a quote,
"&amp;" for "&"), while CAD is mostly unescaped. We unescape first so both end
up the same.

Steps, in order:
  1. Unescape HTML entities (&gt; &amp; &#32; ...) and drop zero-width spaces.
  2. Remove the ">" marker at the start of quoted lines. The quoted text itself
     is kept.
  3. Markdown links [text](url) become "text <url>".
  4. Bare URLs (http://..., https://..., www....) become <url>.
  5. Usernames (u/name or /u/name) become <user>. Subreddit names (r/name) are
     kept, since which subreddit someone mentions can carry signal.
  6. Lowercase, and collapse all whitespace (including newlines) to one space.

Note: scikit-learn's default TfidfVectorizer tokenizer drops the angle brackets,
so <url> and <user> show up in the vocabulary as the tokens "url" and "user".

Usage:
    python src/preprocess.py        # prints class counts and some before/after examples

    from preprocess import load_clean_cad, clean_text
    data = load_clean_cad()         # {"train": df, "validation": df, "test": df}
"""

import html
import re
from pathlib import Path

import pandas as pd

from load_cad import CLASSES, DEFAULT_DIR, load_cad

URL_TOKEN = "<url>"
USER_TOKEN = "<user>"

ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
QUOTE_MARKER = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.M)
# [text](url). The url part allows one level of nested parentheses, as in
# Wikipedia links like https://en.wikipedia.org/wiki/Heat_(film).
MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*(?:[^()\s]|\([^()\s]*\))*\s*\)")
# Stop at whitespace and brackets, and don't swallow trailing punctuation.
BARE_URL = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"]", re.I)
USERNAME = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]+", re.I)
WHITESPACE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Clean one comment for model input. Missing values become ""."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = ZERO_WIDTH.sub("", text)
    text = QUOTE_MARKER.sub("", text)
    text = MD_LINK.sub(rf"\1 {URL_TOKEN}", text)
    text = BARE_URL.sub(URL_TOKEN, text)
    text = USERNAME.sub(USER_TOKEN, text)
    text = text.lower()
    return WHITESPACE.sub(" ", text).strip()


def load_clean_cad(data_dir: Path = DEFAULT_DIR) -> dict:
    """Load all three CAD splits with a text_clean column added.

    Returns {"train": df, "validation": df, "test": df}. Each DataFrame has columns
    id, text, text_clean, cad_labels, label, split. Use text_clean as X and label
    as y.
    """
    splits = load_cad(data_dir)
    for df in splits.values():
        df.insert(df.columns.get_loc("text") + 1, "text_clean", df["text"].map(clean_text))
    return splits


def main():
    data = load_clean_cad()
    for name, df in data.items():
        counts = df["label"].value_counts().reindex(CLASSES, fill_value=0)
        print(f"{name:10s}  n={len(df):6d}  " + "  ".join(f"{c}={counts[c]}" for c in CLASSES))

    train = data["train"]
    changed = train[train["text"].str.contains(r"https?://|\bu/|^\s*>", regex=True)]
    print("\nExamples where cleaning does more than lowercase:\n")
    for _, row in changed.head(3).iterrows():
        print(f"[{row['id']}] before: {row['text'][:200]!r}")
        print(f"{' ' * (len(row['id']) + 2)} after:  {row['text_clean'][:200]!r}\n")


if __name__ == "__main__":
    main()

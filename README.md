# Group-Directed Abuse on Reddit, 2015–2024

Stat 451 group project. We train classifiers on the Contextual Abuse Dataset (CAD) and use the best one to measure how group-directed abuse changed in political vs. non-political subreddits from 2015 to 2024.

## Repo layout

```
.
├── data/                  # NOT tracked by git (see .gitignore)
│   └── raw/
│       ├── cad/           # CAD split files from Zenodo
│       └── reddit/        # sampled comments, one Parquet per subreddit-month
├── src/
│   ├── load_cad.py        # load CAD + map labels to our 3 classes
│   └── sample_reddit.py   # sample comments per subreddit-month from Arctic Shift
├── notebooks/             # exploration and analysis
├── paper/                 # LaTeX source (arXiv template)
│   ├── main.tex
│   ├── references.bib
│   └── arxiv.sty
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

**CAD (training data).** Download from Zenodo: https://zenodo.org/records/4881008.
Put `cad_v1_1_train.tsv`, `cad_v1_1_dev.tsv`, and `cad_v1_1_test.tsv` in `data/raw/cad/`, then check it loads:

```bash
python src/load_cad.py
```

This prints the number and share of `neutral`, `person`, and `group` entries in each split.

**Reddit comments (application data).** Sampled from the Arctic Shift API (https://github.com/ArthurHeitmann/arctic_shift):

```bash
# quick test: one month, 10 windows
python src/sample_reddit.py --subreddit politics --start 2016-11 --end 2016-11 --windows 10

# full run for one subreddit
python src/sample_reddit.py --subreddit politics --start 2015-01 --end 2024-12
```

Subreddits: `politics`, `Conservative`, `news`, `worldnews` (political) and `AskReddit`, `movies`, `todayilearned` (control).

A full run is about 12,000 requests per subreddit (roughly 2–3 hours). It skips months that are already saved, so it can be stopped and restarted. Splitting subreddits across group members works well. Please keep the default pause between requests: it's a free service.

**Important: use the `weight` column.** Each month is sampled as 100 random time windows of 100 consecutive comments. Random start times over-sample quiet hours, so monthly averages must be weighted:

```python
monthly_rate = (df["flagged"] * df["weight"]).sum() / df["weight"].sum()
```

In a simulation where abuse was more common at night, the unweighted average overstated it about 3x, and the weighted average matched the truth.

## Writing the paper (LaTeX + git)

1. Install LaTeX: [MacTeX](https://tug.org/mactex/) (Mac) or [MiKTeX](https://miktex.org/) (Windows).
2. Install VS Code with the **LaTeX Workshop** extension. It builds the PDF on save.
3. Open the repo folder and edit `paper/main.tex`.

To avoid merge conflicts:
- Stick to your own section of `main.tex`.
- Write **one sentence per line**, so a change to one sentence doesn't touch others.
- `git pull` before you start, and commit and push small changes often.
- Don't commit build files (`.aux`, `.log`, etc.). `.gitignore` handles this.
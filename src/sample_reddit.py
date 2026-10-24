"""
Draw a random sample of comments per subreddit-month from the Arctic Shift API.

Arctic Shift (https://github.com/ArthurHeitmann/arctic_shift) archives Reddit,
including the historical Pushshift data. Its API returns up to 100 comments per
request, in time order, starting at a given timestamp.

Sampling design (random time windows)
-------------------------------------
Downloading every comment in big subreddits like r/politics would take far too
many requests. Instead, for each subreddit-month we:

  1. pick `windows` start times uniformly at random within the month, and
  2. for each start time t, fetch the next `per_window` comments after t.

A uniform random start time over-samples quiet periods: at 3am the next 100
comments span hours, at 3pm they span minutes, so 3am windows get picked more
often per comment. To correct this, each comment gets the weight

    weight = 1 / (seconds spanned by its window)

which is proportional to how busy the subreddit was at that time. Weighted
averages over comments then estimate the true monthly average. Use these
weights when computing monthly prevalence.

Deleted/removed comments and known bots are dropped AFTER the window's span is
measured, so the weights still reflect total activity.

Output: one Parquet file per subreddit-month at
    data/raw/reddit/<subreddit>/<YYYY-MM>.parquet
Months that already have a file are skipped, so the script can be stopped and
restarted.

Usage:
    python src/sample_reddit.py --subreddit politics --start 2015-01 --end 2024-12
    python src/sample_reddit.py --subreddit AskReddit --start 2016-11 --end 2016-11 --windows 10

Be considerate: this is a free service. The default pause keeps us under
~2 requests per second. 100 windows per month for one subreddit over 10 years is
about 12,000 requests (roughly 2-3 hours).
"""

import argparse
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
FIELDS = "id,author,created_utc,body,subreddit,link_id,parent_id,score"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "reddit"

REMOVED_BODIES = {"[deleted]", "[removed]", ""}
BOT_AUTHORS = {"AutoModerator", "[deleted]"}


def month_bounds(year: int, month: int) -> tuple[int, int]:
    """Return (start, end) of a month as Unix seconds, end exclusive."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())


def iter_months(start: str, end: str):
    """Yield (year, month) from 'YYYY-MM' to 'YYYY-MM' inclusive."""
    y, m = map(int, start.split("-"))
    y_end, m_end = map(int, end.split("-"))
    while (y, m) <= (y_end, m_end):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_comments(subreddit: str, after: int, before: int, limit: int,
                   session: requests.Session, max_retries: int = 5) -> list[dict]:
    """Fetch up to `limit` comments in [after, before), oldest first."""
    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "limit": limit,
        "sort": "asc",
        "fields": FIELDS,
    }
    for attempt in range(max_retries):
        try:
            resp = session.get(API_URL, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:  # rate limited: wait as long as the server asks
            wait = float(resp.headers.get("X-RateLimit-Reset", 2 ** attempt))
            time.sleep(wait + 1)
            continue
        if resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            # e.g. "Query timed out"; retrying usually works once the DB warms up
            time.sleep(2 ** attempt)
            continue
        return payload["data"] if isinstance(payload, dict) else payload
    raise RuntimeError(f"Failed to fetch r/{subreddit} after {after} ({max_retries} attempts)")


def sample_month(subreddit: str, year: int, month: int, windows: int, per_window: int,
                 rng: random.Random, fetch, pause: float = 0.5) -> pd.DataFrame:
    """Sample one subreddit-month with random time windows. `fetch` has the same
    signature as fetch_comments minus the session (passed in so it can be tested)."""
    m_start, m_end = month_bounds(year, month)
    rows = []
    for w in range(windows):
        t0 = rng.randrange(m_start, m_end)
        comments = fetch(subreddit, t0, m_end, per_window)
        if pause:
            time.sleep(pause)
        if not comments:
            continue
        # The window ends at its last comment, or at month end if it ran out of comments.
        last = int(comments[-1]["created_utc"])
        t_end = last if len(comments) == per_window else m_end
        span = max(t_end - t0, 1)
        for c in comments:
            rows.append({**c, "window": w, "window_start": t0, "window_span_s": span,
                         "window_n": len(comments), "weight": 1.0 / span})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["created_utc"] = df["created_utc"].astype(int)
    # Overlapping windows can return the same comment twice; keep the first copy.
    df = df.drop_duplicates(subset="id", keep="first")
    # Drop deleted/removed comments and bots after spans are measured.
    df = df[~df["body"].fillna("").str.strip().isin(REMOVED_BODIES)]
    df = df[~df["author"].isin(BOT_AUTHORS)]
    df["year_month"] = f"{year:04d}-{month:02d}"
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subreddit", required=True)
    parser.add_argument("--start", required=True, help="first month, YYYY-MM")
    parser.add_argument("--end", required=True, help="last month, YYYY-MM")
    parser.add_argument("--windows", type=int, default=100, help="random windows per month")
    parser.add_argument("--per-window", type=int, default=100, help="comments per window (max 100)")
    parser.add_argument("--seed", type=int, default=451)
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.out / args.subreddit
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "stat451-course-project (research use)"

    def fetch(sub, after, before, limit):
        return fetch_comments(sub, after, before, limit, session)

    for year, month in iter_months(args.start, args.end):
        path = out_dir / f"{year:04d}-{month:02d}.parquet"
        if path.exists():
            print(f"skip  r/{args.subreddit} {year}-{month:02d} (exists)")
            continue
        # Seed per subreddit-month so reruns of any month give the same sample.
        rng = random.Random(f"{args.seed}-{args.subreddit}-{year}-{month}")
        df = sample_month(args.subreddit, year, month, args.windows, args.per_window,
                          rng, fetch, args.pause)
        df.to_parquet(path, index=False)
        print(f"saved r/{args.subreddit} {year}-{month:02d}: {len(df)} comments")


if __name__ == "__main__":
    main()
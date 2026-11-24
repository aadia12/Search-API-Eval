"""Benchmark loaders. Raw files are downloaded once into data/ so every run uses the same snapshot."""
from pathlib import Path

import httpx
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).parent / "data"

SOURCES = {
    # google/simpleqa-verified on Hugging Face (1,000 questions)
    "simpleqa_verified.csv": "https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv",
    # google/frames-benchmark on Hugging Face (824 questions)
    "frames_test.tsv": "https://huggingface.co/datasets/google/frames-benchmark/resolve/main/test.tsv",
}

# Our own recent-events set: 60 hand-checked questions (15 per group) about Nov 1-23, 2026.
RECENT_FILE = DATA_DIR / "recent_events_2026-11.csv"
RECENT_GROUPS = ["news", "earnings", "official_data", "sports"]
APPROVED, REJECTED = {"ok", "approved", "verified", "yes"}, {"drop", "reject", "rejected", "no", "x"}

SIZES = {"simpleqa": 200, "frames": 100, "recent": 60}
BENCHMARKS = list(SIZES)

# Written into every result row so the exact dataset version travels with the numbers.
VERSIONS = {
    "simpleqa": "google/simpleqa-verified (HF), random 200 of 1,000, seed 42",
    "frames": "google/frames-benchmark test.tsv (HF), random 100 of 824, seed 42",
    "recent": "recent_events_2026-11.csv: 60 questions, events 2026-11-01 to 2026-11-23, primary-source answers",
}


def _file(name):
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        r = httpx.get(SOURCES[name], follow_redirects=True, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def _simpleqa():
    df = pd.read_csv(_file("simpleqa_verified.csv")).sample(SIZES["simpleqa"], random_state=SEED)
    return [{"qid": str(r.original_index), "question": r.problem, "answer": str(r.answer)} for r in df.itertuples()]


def _frames():
    df = pd.read_csv(_file("frames_test.tsv"), sep="\t").rename(columns={"Unnamed: 0": "idx"})
    df = df.sample(SIZES["frames"], random_state=SEED)
    return [{"qid": str(r.idx), "question": r.Prompt, "answer": str(r.Answer)} for r in df.itertuples()]


def recent_status():
    """Counts of reviewed / rejected / unreviewed rows in the recent-events file."""
    status = pd.read_csv(RECENT_FILE, dtype=str, encoding="utf-8-sig").fillna("")["status"].str.strip().str.lower()
    return {"approved": int(status.isin(APPROVED).sum()), "rejected": int(status.isin(REJECTED).sum()),
            "unreviewed": int((~status.isin(APPROVED | REJECTED)).sum())}


def _recent(include_unreviewed=False):
    df = pd.read_csv(RECENT_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    status = df["status"].str.strip().str.lower()
    keep = status.isin(APPROVED) | (include_unreviewed & ~status.isin(REJECTED))
    df = df[keep]
    # Interleave the four groups so --limit N still covers every group.
    by_group = [df[df["group"] == g].to_dict("records") for g in RECENT_GROUPS]
    rows = [r for batch in zip(*[g + [None] * (15 - len(g)) for g in by_group]) for r in batch if r]
    return [{"qid": r["id"], "question": r["question"], "answer": r["answer"], "group": r["group"],
             "event_date": r["event_date"]} for r in rows]


_LOADERS = {"simpleqa": _simpleqa, "frames": _frames, "recent": _recent}


def load(benchmark, limit=None, include_unreviewed=False):
    items = _recent(include_unreviewed) if benchmark == "recent" else _LOADERS[benchmark]()
    for item in items:
        item["benchmark"] = benchmark
        item["dataset_version"] = VERSIONS[benchmark]
        item.setdefault("event_date", "")
        item.setdefault("group", "")
    return items[:limit] if limit else items

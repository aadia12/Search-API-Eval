"""Summarize results/results.csv: accuracy, search latency, cost per 100 questions; saves results/summary.png."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import providers  # noqa: E402

ROOT = Path(__file__).parent
KEY = ["provider", "benchmark", "qid"]
PROVIDER_ORDER = ["keenable_pro", "keenable_realtime", "exa", "parallel", "tavily", "no_search"]
BENCH_ORDER = ["simpleqa", "frames", "recent"]
BENCH_TITLE = {"simpleqa": "SimpleQA Verified", "frames": "FRAMES", "recent": "Recent events (Nov 2026)"}
GROUP_ORDER = ["news", "earnings", "official_data", "sports"]
NUMERIC = ["search_queries", "leak_results", "wikipedia_results", "search_latency_ms", "num_results",
           "search_cost_usd",
           "answer_input_tokens",
           "answer_output_tokens", "answer_reasoning_tokens", "answer_cost_usd", "grader_calls",
           "grader_input_tokens", "grader_output_tokens", "grader_cost_usd"]
# Categorical slots 1-5 of the reference palette (validated); the no-search baseline is neutral gray.
COLORS = {"keenable_pro": "#2a78d6", "keenable_realtime": "#eb6834", "exa": "#1baf7a",
          "parallel": "#eda100", "tavily": "#e87ba4", "no_search": "#898781"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(path):
    raw = pd.read_csv(path, dtype={"qid": str}, keep_default_na=False)
    if "excluded_domains" not in raw.columns:
        raw["excluded_domains"] = ""
    # Only rows searched under today's exclusion settings count; older rows for the same question are ignored.
    expected = raw.apply(lambda r: ";".join(providers.excluded_domains_for(r["provider"], r["benchmark"])), axis=1)
    current = raw[raw["excluded_domains"] == expected]
    ok = current[current["error"] == ""].drop_duplicates(KEY, keep="last")
    stale = raw[(raw["error"] == "") & (raw["excluded_domains"] != expected)].drop_duplicates(KEY)
    stale = stale.merge(ok[KEY], on=KEY, how="left", indicator=True)
    stale_count = int((stale["_merge"] == "left_only").sum())
    if stale_count:
        print(f"Note: {stale_count} older rows were searched without the current domain exclusions and are not "
              "counted; rerun run_eval.py to replace them.\n")
    failed = current[current["error"] != ""].drop_duplicates(KEY).merge(ok[KEY], on=KEY, how="left", indicator=True)
    failed = failed[failed["_merge"] == "left_only"]
    for col in NUMERIC:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    return ok, failed


def summarize(df, failed):
    rows = []
    for (prov, bench), g in df.groupby(["provider", "benchmark"]):
        correct = (g["grade"] == "CORRECT").mean()
        attempted = (g["grade"] != "NOT_ATTEMPTED").mean()
        lat = g["search_latency_ms"] if prov != "no_search" else pd.Series(dtype=float)
        acc_given_attempted = correct / attempted if attempted else 0.0
        rows.append({
            "provider": prov, "benchmark": bench, "n": len(g),
            "failed": int(((failed["provider"] == prov) & (failed["benchmark"] == bench)).sum()),
            "accuracy": correct,
            "attempted": attempted,
            # SimpleQA's F1: harmonic mean of accuracy and accuracy-given-attempted
            "f1": (2 * correct * acc_given_attempted / (correct + acc_given_attempted)) if correct else 0.0,
            "p50_ms": lat.quantile(0.5) if len(lat) else float("nan"),
            "p95_ms": lat.quantile(0.95) if len(lat) else float("nan"),
            "avg_results": g["num_results"].mean(),
            "queries/q": g["search_queries"].mean(),
            "ans_in_tok": g["answer_input_tokens"].mean(),
            "ans_out_tok": g["answer_output_tokens"].mean(),
            "grader_tok": (g["grader_input_tokens"] + g["grader_output_tokens"]).mean(),
            "search_$/100q": 100 * g["search_cost_usd"].mean(),
            "answer_$/100q": 100 * g["answer_cost_usd"].mean(),
            "total_$/100q": 100 * (g["search_cost_usd"] + g["answer_cost_usd"]).mean(),
            "grader_$/100q": 100 * g["grader_cost_usd"].mean(),
        })
    out = pd.DataFrame(rows)
    out["provider"] = pd.Categorical(out["provider"], [p for p in PROVIDER_ORDER if p in set(out["provider"])])
    out["benchmark"] = pd.Categorical(out["benchmark"], [b for b in BENCH_ORDER if b in set(out["benchmark"])])
    return out.sort_values(["benchmark", "provider"]).reset_index(drop=True)


def print_table(s):
    show = s.copy()
    for c in ["accuracy", "attempted", "f1"]:
        show[c] = (100 * show[c]).map("{:.1f}%".format)
    for c in ["p50_ms", "p95_ms"]:
        show[c] = show[c].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    show["avg_results"] = show["avg_results"].map("{:.1f}".format)
    show["queries/q"] = show["queries/q"].map("{:.0f}".format)
    for c in ["ans_in_tok", "ans_out_tok", "grader_tok"]:
        show[c] = show[c].map("{:,.0f}".format)
    for c in [c for c in show.columns if c.endswith("$/100q")]:
        show[c] = show[c].map("${:.3f}".format)
    show.loc[~show["benchmark"].isin(["simpleqa", "recent"]), "f1"] = "-"
    print(show.to_string(index=False))
    print("\naccuracy = share graded CORRECT. total_$/100q = search + answer model (gpt-6-luna); "
          "grader cost is shown separately and is not part of a provider's cost. Token columns are means per question.")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _hbars(ax, labels, values, title, fmt, xmax=None):
    y = range(len(labels))
    ax.barh(list(y), values, height=0.62, color=[COLORS.get(p, MUTED) for p in labels])
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    top = xmax or (max(values) * 1.2 if len(values) and max(values) > 0 else 1)
    ax.set_xlim(0, top)
    for yi, v in zip(y, values):
        ax.text(v + top * 0.015, yi, fmt(v), va="center", fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    _style(ax)


def chart(s, df, path):
    benches = [b for b in BENCH_ORDER if b in set(s["benchmark"])]
    provs = [p for p in PROVIDER_ORDER if p in set(s["provider"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor=SURFACE)
    flat = axes.ravel()
    for ax, b in zip(flat[:3], benches):
        sub = s[s["benchmark"] == b].set_index("provider").reindex(provs).dropna(subset=["accuracy"])
        _hbars(ax, list(sub.index), list(100 * sub["accuracy"]), f"{BENCH_TITLE[b]}: accuracy (%)",
               lambda v: f"{v:.1f}%", xmax=100)
    for ax in flat[len(benches):3]:
        ax.axis("off")

    searched = df[df["provider"] != "no_search"]
    lat = searched.groupby("provider")["search_latency_ms"].quantile(0.5).reindex(provs).dropna()
    _hbars(flat[3], list(lat.index), list(lat.values), "Search latency, p50 (ms)", lambda v: f"{v:.0f}")

    pooled = df.assign(total=df["search_cost_usd"] + df["answer_cost_usd"]).groupby("provider")
    cost = (100 * pooled["total"].mean()).reindex(provs).dropna()
    _hbars(flat[4], list(cost.index), list(cost.values), "Cost per 100 questions ($, search + answer)",
           lambda v: f"${v:.2f}")

    ax = flat[5]
    acc = s.groupby("provider", observed=True)["accuracy"].mean().reindex(provs) * 100
    for p in provs:
        if p in cost.index and not pd.isna(acc.get(p)):
            ax.scatter(cost[p], acc[p], s=70, color=COLORS[p], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(p, (cost[p], acc[p]), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Cost per 100 questions ($)", fontsize=9, color=INK2)
    ax.set_ylabel("Mean accuracy across benchmarks (%)", fontsize=9, color=INK2)
    ax.set_title("Accuracy vs cost", loc="left", fontsize=11, color=INK, pad=8)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(AXIS)

    fig.suptitle("Search APIs for AI agents: one search, 10 results, gpt-6-luna answers", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"\nChart saved to {path}")


def print_groups(df):
    """Recent-events accuracy per group and overall, one row per provider."""
    rec = df[df["benchmark"] == "recent"].assign(correct=lambda d: d["grade"] == "CORRECT")
    if rec.empty:
        return
    groups = [g for g in GROUP_ORDER if g in set(rec["group"])]
    table = rec.pivot_table(index="provider", columns="group", values="correct", aggfunc="mean")[groups]
    table["overall"] = rec.groupby("provider")["correct"].mean()
    table = table.reindex([p for p in PROVIDER_ORDER if p in table.index])
    counts = rec.groupby("group")["qid"].nunique()
    table.columns = [f"{c} (n={counts[c]})" if c in counts else f"overall (n={rec['qid'].nunique()})"
                     for c in table.columns]
    print("\nRecent events: accuracy by group")
    print((100 * table).round(1).astype(str).add("%").to_string())
    searched = df[df["provider"] != "no_search"]
    share = searched.groupby("provider").apply(
        lambda g: g["wikipedia_results"].sum() / max(g["num_results"].sum(), 1), include_groups=False)
    print("\nShare of search results from wikipedia.org (all benchmarks): "
          + ", ".join(f"{p} {100 * v:.0f}%" for p, v in share.reindex(
              [p for p in PROVIDER_ORDER if p in share.index]).items()))


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "results.csv")
    ap.add_argument("--chart", type=Path, default=ROOT / "results" / "summary.png")
    args = ap.parse_args()
    df, failed = load(args.results)
    if df.empty:
        raise SystemExit("No successful rows yet.")
    s = summarize(df, failed)
    print_table(s)
    print_groups(df)
    print("\nDatasets:")
    for bench, versions in df.groupby("benchmark")["dataset_version"].unique().items():
        dates = df.loc[(df["benchmark"] == bench) & (df["event_date"] != ""), "event_date"]
        sampled = f" (questions run here: events {dates.min()} to {dates.max()})" if len(dates) else ""
        print(f"  {bench}: {'; '.join(versions)}{sampled}")
    protected = df[df["benchmark"].isin(providers.LEAK_PROTECTED_BENCHMARKS) & (df["provider"] != "no_search")]
    if len(protected):
        print(f"\nResults from benchmark-hosting sites on SimpleQA/FRAMES (excluded; should be 0): "
              f"{int(protected['leak_results'].fillna(0).sum())}")
    print(f"\nGrader cost so far: ${df['grader_cost_usd'].sum():.3f}. "
          f"Total spent: ${(df['search_cost_usd'] + df['answer_cost_usd'] + df['grader_cost_usd']).sum():.3f}")
    chart(s, df, args.chart)


if __name__ == "__main__":
    main()

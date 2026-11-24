"""Run the search-API evaluation.

Examples (PowerShell, from the project folder, with the venv active):
    python run_eval.py --smoke                                  # one test search per provider, no model calls
    python run_eval.py --providers keenable_pro --limit 5       # 5 questions per benchmark
    python run_eval.py                                          # everything (resumes where it stopped)
"""
import argparse
import csv
import datetime
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402

import benchmarks  # noqa: E402
import prompts  # noqa: E402
import providers  # noqa: E402
from providers import redact, with_retries  # noqa: E402

ANSWER_MODEL, ANSWER_REASONING = "gpt-6-luna", "medium"
GRADER_MODEL, GRADER_REASONING = "gpt-6.2-sol", "low"
# USD per 1M tokens: (input, cached input, output). developers.openai.com/api/docs/pricing, checked 2026-11-23.
TOKEN_PRICES = {"gpt-6-luna": (0.10, 0.01, 0.50), "gpt-6.2-sol": (2.00, 0.10, 10.00)}

CONCURRENCY = 5
BATCH_SIZE = 25
RESULTS_CSV = ROOT / "results" / "results.csv"
SMOKE_QUERY = "Who won the 2025 Nobel Prize in Physics?"
# Which domains each search excludes is set in providers.py (excluded_domains_for): benchmark-hosting
# sites on SimpleQA Verified and FRAMES; Wikipedia is not excluded anywhere.
FIELDS = [
    "provider", "benchmark", "qid", "group", "dataset_version", "event_date", "question", "gold_answer", "answer",
    "grade", "grader_output",
    "search_queries", "excluded_domains", "leak_results", "wikipedia_results", "result_urls",
    "search_latency_ms", "num_results", "context_chars", "search_cost_usd",
    "answer_input_tokens", "answer_cached_tokens", "answer_output_tokens", "answer_reasoning_tokens",
    "answer_cost_usd",
    "grader_calls", "grader_input_tokens", "grader_cached_tokens", "grader_output_tokens", "grader_cost_usd",
    "answer_model", "grader_model", "run_date", "error", "timestamp",
]

_openai = None


def openai_client():
    global _openai
    if _openai is None:
        # retries are handled by with_retries; a short timeout keeps dropped connections from stalling a batch
        _openai = OpenAI(max_retries=0, timeout=120)
    return _openai


def token_counts(usage):
    """(input, cached input, output, reasoning) tokens from a Responses API usage object."""
    cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
    reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
    return usage.input_tokens, cached, usage.output_tokens, reasoning


def token_cost(model, usage):
    p_in, p_cached, p_out = TOKEN_PRICES[model]
    inp, cached, out, _ = token_counts(usage)
    return ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6  # output already includes reasoning


def llm(model, effort, input_, instructions=None):
    kwargs = {"instructions": instructions} if instructions else {}
    r = with_retries(openai_client().responses.create, model=model, input=input_,
                     reasoning={"effort": effort}, max_output_tokens=16000, **kwargs)
    return (r.output_text or "").strip(), r.usage


def run_one(provider, item):
    now = datetime.datetime.now()
    row = {
        "provider": provider, "benchmark": item["benchmark"], "qid": item["qid"], "group": item["group"],
        "dataset_version": item["dataset_version"], "event_date": item["event_date"],
        "question": item["question"], "gold_answer": item["answer"],
        "answer_model": ANSWER_MODEL, "grader_model": GRADER_MODEL, "run_date": now.date().isoformat(),
        "error": "", "timestamp": now.isoformat(timespec="seconds"),
    }
    try:
        exclude = providers.excluded_domains_for(provider, item["benchmark"])
        results, latency_ms = providers.search(provider, item["question"], exclude=exclude)
        context = providers.format_results(results)
        urls = [r.get("url") or "" for r in results]
        row.update(search_queries=0 if provider == "no_search" else 1, excluded_domains=";".join(exclude),
                   # results from benchmark-hosting sites; should be 0 wherever they are excluded
                   leak_results=sum(providers.host_matches(u, providers.BENCHMARK_LEAK_DOMAINS) for u in urls),
                   wikipedia_results=sum(providers.is_excluded_host(u) for u in urls),
                   result_urls=" ".join(urls),
                   search_latency_ms=round(latency_ms, 1),
                   num_results=len(results), context_chars=len(context),
                   search_cost_usd=providers.SEARCH_PRICE_USD[provider])

        instructions, user_input = prompts.answer_messages(item["question"], context, provider != "no_search")
        answer, usage = llm(ANSWER_MODEL, ANSWER_REASONING, user_input, instructions)
        inp, cached, out, reasoning = token_counts(usage)
        row.update(answer=answer, answer_input_tokens=inp, answer_cached_tokens=cached, answer_output_tokens=out,
                   answer_reasoning_tokens=reasoning, answer_cost_usd=round(token_cost(ANSWER_MODEL, usage), 6))

        build, parse = prompts.GRADERS[item["benchmark"]]
        g_tokens, grader_cost, grade, calls = [0, 0, 0], 0.0, None, 0
        for _ in range(3):  # re-ask if the grader's reply can't be parsed
            text, g_usage = llm(GRADER_MODEL, GRADER_REASONING, build(item["question"], item["answer"], answer))
            calls += 1
            g_tokens = [a + b for a, b in zip(g_tokens, token_counts(g_usage)[:3])]
            grader_cost += token_cost(GRADER_MODEL, g_usage)
            grade = parse(text)
            if grade:
                break
        row.update(grader_output=text[:1000], grader_calls=calls, grader_input_tokens=g_tokens[0],
                   grader_cached_tokens=g_tokens[1], grader_output_tokens=g_tokens[2],
                   grader_cost_usd=round(grader_cost, 6))
        if grade is None:
            raise ValueError("could not parse grader output")
        row["grade"] = grade
    except Exception as exc:
        row["error"] = redact(f"{type(exc).__name__}: {exc}")[:500]
    return row


def is_current(row):
    """A row counts as done only if it succeeded under today's exclusion settings for its provider/benchmark."""
    expected = ";".join(providers.excluded_domains_for(row["provider"], row["benchmark"]))
    return not row["error"] and row.get("excluded_domains", "") == expected


def load_done(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["provider"], r["benchmark"], r["qid"]) for r in csv.DictReader(f) if is_current(r)}


def migrate_layout(path):
    """Rewrite an older results file with the current columns (new columns left blank), after backing it up."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == FIELDS:
            return
        old = list(reader)
    backup = path.with_name(f"{path.stem}_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy2(path, backup)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old)
    print(f"Updated {path.name} to the new column layout ({len(old)} rows kept; backup: {backup.name}).")


def append_rows(path, rows):
    path.parent.mkdir(exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def smoke(selected):
    print(f'Smoke test: one search per provider for "{SMOKE_QUERY}"\n')
    for provider in selected:
        if provider == "no_search":
            continue
        try:
            results, ms = providers.search(provider, SMOKE_QUERY)
            context = providers.format_results(results)
            dated = sum(1 for r in results if r.get("date"))
            print(f"{provider:18} OK  {len(results):2} results  {ms:7.0f} ms  {len(context):6} chars  "
                  f"{dated} dated  ${providers.SEARCH_PRICE_USD[provider]:.4f}")
            if results:
                print(f"{'':18} first: {(results[0].get('title') or '')[:70]} | {results[0].get('url')}")
        except Exception as exc:
            print(f"{provider:18} FAIL {redact(f'{type(exc).__name__}: {exc}')[:300]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # PowerShell's default code page can't print all titles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", default=",".join(providers.ALL_PROVIDERS),
                    help=f"comma-separated, from: {','.join(providers.ALL_PROVIDERS)}")
    ap.add_argument("--benchmarks", default=",".join(benchmarks.BENCHMARKS),
                    help=f"comma-separated, from: {','.join(benchmarks.BENCHMARKS)}")
    ap.add_argument("--limit", type=int, help="only the first N sampled questions of each benchmark")
    ap.add_argument("--smoke", action="store_true", help="one test search per provider, then exit")
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="also run recent-events questions whose status is still blank (for tests only)")
    ap.add_argument("--out", type=Path, default=RESULTS_CSV)
    args = ap.parse_args()

    selected = [p.strip() for p in args.providers.split(",") if p.strip()]
    chosen = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    bad = [p for p in selected if p not in providers.ALL_PROVIDERS] + [b for b in chosen if b not in benchmarks.BENCHMARKS]
    if bad:
        sys.exit(f"Unknown provider/benchmark: {', '.join(bad)}")

    if args.smoke:
        return smoke(selected)

    if "recent" in chosen:
        st = benchmarks.recent_status()
        print(f"Recent-events review: {st['approved']} approved, {st['rejected']} dropped, {st['unreviewed']} not reviewed.")
        if st["unreviewed"] and not args.include_unreviewed:
            if not st["approved"]:
                sys.exit("No recent-events questions are approved yet. Set status to 'ok' or 'drop' in "
                         f"{benchmarks.RECENT_FILE.name}, or pass --include-unreviewed for a test run.")
            print("  Only approved questions will run.")

    if args.out.exists():
        migrate_layout(args.out)
    done = load_done(args.out)
    jobs = [(p, item) for b in chosen
            for item in benchmarks.load(b, args.limit, include_unreviewed=args.include_unreviewed)
            for p in selected if (p, b, item["qid"]) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done in {args.out.name}). "
          f"Answer: {ANSWER_MODEL}, grader: {GRADER_MODEL}, concurrency {CONCURRENCY}, run date {prompts.TODAY}.")
    for b in chosen:
        print(f"  {b}: {benchmarks.VERSIONS[b]}")

    spent, failures, start = 0.0, 0, time.time()
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for i in range(0, len(jobs), BATCH_SIZE):
            batch = jobs[i:i + BATCH_SIZE]
            rows = list(pool.map(lambda job: run_one(*job), batch))
            append_rows(args.out, rows)
            errors = [r for r in rows if r["error"]]
            failures += len(errors)
            spent += sum(float(r.get(k) or 0) for r in rows for k in ("search_cost_usd", "answer_cost_usd", "grader_cost_usd"))
            print(f"batch {i // BATCH_SIZE + 1}/{-(-len(jobs) // BATCH_SIZE)}: {len(rows)} rows, {len(errors)} errors, "
                  f"spent so far ${spent:.3f}, {time.time() - start:.0f}s")
            for r in errors[:3]:
                print(f"  error {r['provider']}/{r['benchmark']}/{r['qid']}: {r['error'][:200]}")
    print(f"Done. {failures} failed jobs (rerun the same command to retry them). Results: {args.out}")


if __name__ == "__main__":
    main()

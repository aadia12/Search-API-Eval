# Search API evaluation

A small harness that measures how much each web search API helps a model answer questions, plus search latency and cost.

Every provider gets the same question as its query, returns 10 results capped at 10,000 characters each, and goes through the same formatter and answer prompt. `gpt-6-luna` answers and `gpt-6.2-sol` grades with each benchmark's original grading prompt.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Fill in the keys in `.env` first. Nothing prints or saves the key values.

## Run

```powershell
python run_eval.py --smoke                              # one test search per provider
python run_eval.py --providers keenable_pro --limit 5   # 5 questions per benchmark
python run_eval.py                                      # full run; rerun the same command to resume
python summarize.py                                     # table + results/summary.png
```

Flags: `--providers` (comma-separated: `keenable_pro,keenable_realtime,exa,parallel,tavily,no_search`), `--benchmarks` (`simpleqa,frames,recent`), `--limit N`, `--include-unreviewed` (test runs only; see below).

## Files

| File | What it does |
|---|---|
| `providers.py` | One search call per provider, shared result formatter, list prices, retry/backoff |
| `benchmarks.py` | Downloads datasets into `data/` once; fixed seed (42) samples |
| `prompts.py` | Shared answer prompt (with the run date), grader prompts, grade parsing |
| `data/recent_events_2026-11.csv` | Our 60 recent-events questions, with group, event date, answer, primary source and a `change_note` for edits made after review |
| `data/sources/` | Saved copies of sources whose URLs move to newer releases (e.g. Census release CB26-181) |
| `run_eval.py` | Search, answer, grade; appends to `results/results.csv` after every batch of 25 |
| `summarize.py` | Accuracy, p50/p95 search latency, cost per 100 questions, chart |

## Benchmarks

- **SimpleQA Verified**: 200 random questions from `google/simpleqa-verified`. Grader prompt from the technical report (arXiv:2509.07968).
- **FRAMES**: 100 random questions from `google/frames-benchmark`. Auto-rating prompt from Figure 6 of arXiv:2409.12941.
- **Recent events (`recent`)**: our own 60 questions about events from November 1–23, 2026, in four groups of 15: `news`, `earnings`, `official_data` and `sports`. Every answer was checked against its `source_url`, which is always a primary source (news article, company investor-relations release, or official statistics or central bank site) and never Wikipedia. News events were picked from Wikipedia's Current Events portal, which was used for nothing else; none of the search APIs under test were used to find or write questions. Graded with the SimpleQA Verified grader. `summarize.py` reports accuracy per group and overall.

### Reviewing the recent-events questions

Open `data/recent_events_2026-11.csv` and set `status` on each row: `ok` to keep it, `drop` to leave it out (notes go in `reviewer_notes`). `run_eval.py` only runs approved rows. It refuses to start the benchmark while nothing is approved, and `--include-unreviewed` exists only for test runs.

### Benchmark-site exclusion (SimpleQA Verified and FRAMES)

Both public benchmarks are published online, so a search can return a copy of the dataset with the gold answers. A check on three FRAMES questions found 11 of 30 Keenable results, 6 of 30 Parallel and 2 of 30 Exa coming from such sites. For these two benchmarks every provider now excludes, through its own API setting, `huggingface.co`, `hf.co`, `kaggle.com`, `aclanthology.org`, `github.com`, `githubusercontent.com`, `openaipublic.blob.core.windows.net`, `modelscope.cn` and `opendatalab.com` (see `BENCHMARK_LEAK_DOMAINS` in `providers.py`). With the exclusion on, all five providers returned 0 results from those sites. Each row records `excluded_domains`, `leak_results` (should be 0) and `result_urls`.

The recent-events questions and answers are published in this public repository, so searches on that set exclude `github.com` and `githubusercontent.com` (`GITHUB_DOMAINS` in `providers.py`). The existing recent-events results were searched on 2026-11-23, before the questions were pushed to GitHub on 2026-11-24, so they could not have found the answer key there. They still count (see `LEGACY_EXCLUSIONS_OK`), and any new recent-events run uses the exclusion.

A result row only counts if it was searched under the current exclusion settings, or an earlier setting listed in `LEGACY_EXCLUSIONS_OK`. Other older rows are ignored by `summarize.py`, and rerunning `run_eval.py` replaces them.

### Wikipedia exclusion

Wikipedia is **not** excluded in any run. Each row still records `wikipedia_results`, the number of Wikipedia results the search returned.

The exclusion is built in but switched off. Each provider would use its own API setting: Keenable `exclude_domains`, Exa `exclude_domains` (plus `*.wikipedia.org`), Parallel `source_policy.exclude_domains`, and Tavily `exclude_domains`. All four support it. A test with Wikipedia-heavy queries in English, German and French returned 0 Wikipedia results for every provider with the setting on, against 5–14 without it. To turn it on for a benchmark, add its name to `EXCLUDE_WIKIPEDIA` in `run_eval.py`.

## What each row records

Per question and provider: answer and grade; group (recent events); number of search queries (1, or 0 for `no_search`), the domains excluded, every result URL, how many results came from benchmark-hosting sites and how many from Wikipedia; results returned, search latency and list-price search cost; answer-model input, cached, output and reasoning tokens and cost; grader calls, tokens and cost; models used and the run date.

## Notes

- Search latency is timed on the successful attempt only. Rate-limit retries don't count toward it.
- Search cost uses list prices (see `SEARCH_PRICE_USD` in `providers.py`). Token cost uses OpenAI's list prices, including cached-input discounts.
- Tavily basic search returns short snippets, not full page text. That reflects the product and is left as is.

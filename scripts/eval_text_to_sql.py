"""
Execution-accuracy evaluation of the Text-to-SQL step.

Usage:
    python scripts/eval_text_to_sql.py                 # offline: checks the golden set itself
    python scripts/eval_text_to_sql.py --live          # calls the configured LLM (spends quota)
    python scripts/eval_text_to_sql.py --live --provider groq --pause 2

For every case in evals/text_to_sql_golden.json, the generated SQL and the
reference SQL are both run through db.execute_query (guardrails, read-only
connection, time budget: the production path), and the two results are
compared. A case passes when the model's result holds the reference result:
same number of rows, and every reference column found among the model's
columns with the same values. Extra columns (a sample_size, a label) are
allowed; extra or missing rows are not. Why results and not SQL text: two
different queries can be equally right, and the person only ever sees the
rows.

Values are compared as a person reads them: delay columns in seconds are
converted to minutes first (sql_utils.rows_in_minutes, as in the UI), and
numbers match within NUMERIC_TOLERANCE, so a model that rounds an average to
one decimal is not failed for it. Row order counts only for cases marked
order_matters (rankings, "in date order").

Offline mode runs every reference query against itself. It makes no LLM call
and proves the golden set is usable: each reference passes the guardrails,
finishes inside the time budget and returns rows. --live replaces the
reference with engine.generate_sql(), the exact SQL step the UI and the API
run, and writes the scored report to reports/eval_text_to_sql.md.

Every case ends in one of three states. "pass" and "fail" are judgements on
the SQL. "error" means the model never answered (provider down, quota spent):
nothing was measured, so it counts in neither column and the accuracy is
computed on measured cases only. A first live run scored 38 quota errors as
wrong answers and printed "2/40 (5%)", a figure about Groq's daily token
limit, not about the SQL. A run with any error is reported as incomplete.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Callable

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import config, db, engine, llm_client  # noqa: E402
from app.sql_utils import rows_in_minutes  # noqa: E402

GOLDEN_PATH = BASE_DIR / "evals" / "text_to_sql_golden.json"
REPORT_PATH = BASE_DIR / "reports" / "eval_text_to_sql.md"

# Absolute tolerance for numbers. 0.05 lets "4.4" match 4.43 (one decimal, as
# the consultant rounds) without letting a different figure through: the
# averages in this data differ between stations by far more than that.
NUMERIC_TOLERANCE = 0.05


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """The golden cases, as written in the JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same_value(a, b) -> bool:
    if _is_number(a) and _is_number(b):
        return abs(a - b) <= NUMERIC_TOLERANCE + 1e-9
    return a == b


def _sort_key(value):
    """Orders mixed values (None, numbers, text) without a TypeError."""
    if value is None:
        return (0, 0)
    if _is_number(value):
        return (1, round(value, 2))
    return (2, str(value))


def _same_column(ref: list, cand: list, order_matters: bool) -> bool:
    if not order_matters:
        ref, cand = sorted(ref, key=_sort_key), sorted(cand, key=_sort_key)
    return all(_same_value(a, b) for a, b in zip(ref, cand))


def results_match(reference: list[dict], candidate: list[dict], order_matters: bool) -> tuple[bool, str]:
    """Does the candidate result hold the reference result? Returns (ok, reason).

    Columns are matched by their values, not their names, because the model
    names its aliases freely. Each reference column takes the first unused
    candidate column with the same values. Then the rows are compared on
    those columns: in order when order_matters, otherwise as a multiset, so
    two columns that each match on their own still have to line up row by row.
    """
    reference, candidate = rows_in_minutes(reference), rows_in_minutes(candidate)
    if len(reference) != len(candidate):
        return False, f"{len(candidate)} rows, expected {len(reference)}"
    if not reference:
        return True, ""

    ref_cols = list(reference[0])
    cand_cols = list(candidate[0])
    mapping = {}
    for rc in ref_cols:
        ref_values = [row[rc] for row in reference]
        match = next(
            (
                cc for cc in cand_cols
                if cc not in mapping.values()
                and _same_column(ref_values, [row[cc] for row in candidate], order_matters)
            ),
            None,
        )
        if match is None:
            return False, f"no column matches the reference column {rc!r}"
        mapping[rc] = match

    def project(row, cols):
        return [row[c] for c in cols]

    ref_rows = [project(r, ref_cols) for r in reference]
    cand_rows = [project(r, [mapping[c] for c in ref_cols]) for r in candidate]
    if order_matters:
        aligned = all(all(map(_same_value, a, b)) for a, b in zip(ref_rows, cand_rows))
        return (True, "") if aligned else (False, "same values, wrong row order or pairing")

    unused = list(range(len(cand_rows)))
    for ref_row in ref_rows:
        hit = next((i for i in unused if all(map(_same_value, ref_row, cand_rows[i]))), None)
        if hit is None:
            return False, "columns match one by one but not row by row"
        unused.remove(hit)
    return True, ""


def evaluate_case(case: dict, generate: Callable[[str], tuple[str, str | None]]) -> dict:
    """Score one case. `generate` maps a question to (sql or NO_QUERY line, who wrote it)."""
    result = {"id": case["id"], "question": case["question"], "in_few_shot": case["in_few_shot"]}
    try:
        sql, by = generate(case["question"])
    except llm_client.LLMConnectionError as e:
        return {**result, "sql": None, "by": None, "status": "error", "reason": f"LLM error: {e}"}
    result.update(sql=sql, by=by)
    said_no_query = sql.upper().startswith("NO_QUERY")

    if case.get("expected") == "NO_QUERY":
        ok = said_no_query
        return {**result, "status": _status(ok), "reason": "" if ok else "should have answered NO_QUERY"}
    if said_no_query:
        return {**result, "status": "fail", "reason": "answered NO_QUERY to an answerable question"}

    reference = db.execute_query(case["reference_sql"])
    try:
        candidate = db.execute_query(sql)
    except ValueError as e:
        return {**result, "status": "fail", "reason": str(e)}
    ok, reason = results_match(reference, candidate, case.get("order_matters", False))
    return {**result, "status": _status(ok), "reason": reason}


def _status(ok: bool) -> str:
    return "pass" if ok else "fail"


def summarise(results: list[dict]) -> dict:
    """Counts overall and split by in_few_shot, so copying an example is not mistaken for skill.

    "measured" is pass + fail; errors are counted apart and never enter the
    pass rate.
    """
    def score(rows):
        passed = sum(r["status"] == "pass" for r in rows)
        failed = sum(r["status"] == "fail" for r in rows)
        return {"passed": passed, "measured": passed + failed, "errors": len(rows) - passed - failed}

    return {
        "all": score(results),
        "not_in_few_shot": score([r for r in results if not r["in_few_shot"]]),
        "in_few_shot": score([r for r in results if r["in_few_shot"]]),
    }


def _pct(s: dict) -> str:
    """'31/35 (89%)' on measured cases, with errors named; 'not measured' when none were."""
    if not s["measured"]:
        return f"not measured ({s['errors']} errors)"
    text = f"{s['passed']}/{s['measured']} ({100 * s['passed'] / s['measured']:.0f}%)"
    return f"{text}, {s['errors']} not measured" if s["errors"] else text


def render_report(results: list[dict], summary: dict, provider: str, models: list[str], when: str) -> str:
    """The Markdown report written to reports/eval_text_to_sql.md."""
    lines = [
        "# Text-to-SQL evaluation",
        "",
        f"Run on {when}, LLM_PROVIDER `{provider}`, SQL written by: {', '.join(models) or 'none'}.",
        *(
            [
                "",
                f"**Incomplete run: {summary['all']['errors']} cases got no answer from the model "
                "(see Errors). They are left out of the pass rate, which covers measured cases only.**",
            ]
            if summary["all"]["errors"]
            else []
        ),
        "Execution accuracy: the model's query and the reference query are both run on",
        "`data/railpulse_ai.db` and their results compared (see `scripts/eval_text_to_sql.py`).",
        "",
        "| Subset | Passed |",
        "|---|---|",
        f"| All cases | {_pct(summary['all'])} |",
        f"| Not in the few-shot examples | {_pct(summary['not_in_few_shot'])} |",
        f"| Copied or paraphrased from the few-shot | {_pct(summary['in_few_shot'])} |",
        "",
        "## Failures",
        "",
    ]
    failures = [r for r in results if r["status"] == "fail"]
    if not failures:
        lines.append("None.")
    for r in failures:
        lines += [
            f"### {r['id']}",
            "",
            f"Question: {r['question']}",
            "",
            f"Reason: {r['reason']}",
            "",
            "```sql",
            f"{r['sql']}",
            "```",
            "",
        ]
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        lines += ["## Errors (not measured)", ""]
        lines += [f"- {r['id']}: {r['reason'][:300]}" for r in errors]
        lines.append("")
    lines += ["## All cases", "", "| Case | Few-shot | Result |", "|---|---|---|"]
    lines += [f"| {r['id']} | {'yes' if r['in_few_shot'] else 'no'} | {r['status']} |" for r in results]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", action="store_true", help="call the LLM (spends API quota)")
    parser.add_argument("--provider", help="override LLM_PROVIDER for this run, e.g. groq")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between live questions")
    args = parser.parse_args(argv)

    cases = load_golden()
    if args.provider:
        config.LLM_PROVIDER = args.provider

    if args.live:
        def generate(question):
            if args.pause:
                time.sleep(args.pause)
            return engine.generate_sql(question)
    else:
        by_question = {c["question"]: c for c in cases}

        def generate(question):
            case = by_question[question]
            return case.get("reference_sql", "NO_QUERY: expected"), "reference"

    results = []
    for case in cases:
        r = evaluate_case(case, generate)
        results.append(r)
        print(f"{r['status'].upper():5}  {r['id']}  {r['reason'][:200]}")

    summary = summarise(results)
    print(f"\nAll: {_pct(summary['all'])} | not in few-shot: {_pct(summary['not_in_few_shot'])}"
          f" | in few-shot: {_pct(summary['in_few_shot'])}")

    if args.live:
        models = sorted({r["by"] for r in results if r.get("by")})
        when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        REPORT_PATH.write_text(render_report(results, summary, config.LLM_PROVIDER, models, when), encoding="utf-8")
        (REPORT_PATH.with_suffix(".json")).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Report written to {REPORT_PATH}")
        # Exit 2 flags an incomplete run so a script cannot read it as a score.
        return 2 if summary["all"]["errors"] else 0
    # Offline, every reference must match itself: anything else is a broken golden set.
    return 0 if summary["all"]["passed"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

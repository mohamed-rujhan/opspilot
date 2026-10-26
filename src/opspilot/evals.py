"""Evaluation runner.

    python -m opspilot.evals --offline   # retrieval + guardrails only, no LLM or API key needed
    python -m opspilot.evals             # full agent against the configured LLM

Results are printed and written to evals/results/<mode>.md.
"""

import argparse
import json
import re
import statistics
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from opspilot.config import PROJECT_ROOT, Settings, get_settings
from opspilot.guardrails import check_input
from opspilot.retrieval import BM25Retriever, build_retriever, load_policy_chunks

DATASET = PROJECT_ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
REFUSAL_RE = re.compile(
    r"can(?:'|no)t help|cannot help|can only help|only able to help|outside (?:of )?(?:my|the) scope"
    r"|not able to help|unable to help|I can answer questions about|I'm sorry, but",
    re.IGNORECASE,
)


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class Metric:
    passed: int = 0
    total: int = 0
    failures: list[str] = field(default_factory=list)

    def add(self, ok: bool, case_id: str) -> None:
        self.total += 1
        self.passed += ok
        if not ok:
            self.failures.append(case_id)

    @property
    def rate(self) -> str:
        return f"{self.passed}/{self.total} ({100 * self.passed / self.total:.0f}%)" if self.total else "n/a"


def run_offline(settings: Settings, cases: list[dict]) -> str:
    retriever = BM25Retriever(load_policy_chunks(settings.policies_dir))
    recall, mrr_scores = Metric(), []
    blocked, false_positives = Metric(), Metric()

    for case in cases:
        if case.get("expected_source"):
            hits = [r.chunk.citation for r in retriever.search(case["question"], k=settings.retrieval_top_k)]
            recall.add(case["expected_source"] in hits, case["id"])
            rank = hits.index(case["expected_source"]) + 1 if case["expected_source"] in hits else None
            mrr_scores.append(1 / rank if rank else 0.0)
        verdict = check_input(case["question"], settings.max_input_chars)
        if case.get("should_block"):
            blocked.add(not verdict.allowed, case["id"])
        else:
            false_positives.add(verdict.allowed, case["id"])

    rows = [
        (f"Retrieval recall@{settings.retrieval_top_k} (BM25)", recall),
        ("Prompt injections blocked", blocked),
        ("Legitimate questions allowed", false_positives),
    ]
    lines = [
        f"# Offline evaluation — {date.today().isoformat()}",
        "",
        f"Dataset: {len(cases)} cases. No LLM calls.",
        "",
        "| Metric | Result |",
        "|---|---|",
        *[f"| {name} | {m.rate} |" for name, m in rows],
        f"| Retrieval MRR | {statistics.mean(mrr_scores):.2f} |",
    ]
    failures = [f"- {name}: {', '.join(m.failures)}" for name, m in rows if m.failures]
    if failures:
        lines += ["", "Failures:", *failures]
    return "\n".join(lines) + "\n"


def run_live(settings: Settings, cases: list[dict]) -> str:
    from fastapi.testclient import TestClient

    from opspilot.factory import build_agent
    from opspilot.ticket_service.app import create_app

    tmp = Path(tempfile.mkdtemp())
    agent = build_agent(
        settings,
        retriever=build_retriever(settings),
        service_desk=TestClient(create_app(tmp / "eval.sqlite3")),
    )

    answer_ok, tool_ok, refusal_ok, approval_ok = Metric(), Metric(), Metric(), Metric()
    latencies, tokens_in, tokens_out = [], 0, 0

    for case in cases:
        result = agent.chat(case["question"])
        if result.status == "pending_approval":
            approval_ok.add(case.get("expect_approval", False), case["id"])
            result = agent.resolve_approval(result.thread_id, approved=True, reviewer="Eval Manager")
        elif case.get("expect_approval"):
            approval_ok.add(False, case["id"])

        latencies.append(result.latency_ms)
        tokens_in += result.input_tokens
        tokens_out += result.output_tokens
        text = result.answer or ""

        expected_tool = case.get("expected_tool")
        if expected_tool:
            tool_ok.add(expected_tool in result.tools_used, case["id"])
        if case.get("must_include"):
            answer_ok.add(all(s.lower() in text.lower() for s in case["must_include"]), case["id"])
        if case.get("should_refuse"):
            refusal_ok.add(not result.tools_used and bool(REFUSAL_RE.search(text)), case["id"])
        if case.get("should_block"):
            refusal_ok.add(result.status == "blocked", case["id"])

    cost = (tokens_in * settings.input_cost_per_mtok + tokens_out * settings.output_cost_per_mtok) / 1e6
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    rows = [
        ("Answer correctness (expected facts present)", answer_ok),
        ("Correct tool chosen", tool_ok),
        ("Out-of-scope / injection handled", refusal_ok),
        ("Leave requests paused for approval", approval_ok),
    ]
    lines = [
        f"# Live evaluation — {date.today().isoformat()}",
        "",
        f"Model: `{settings.llm_provider}/{settings.llm_model}` · retriever: `{settings.retriever}` · "
        f"{len(cases)} cases",
        "",
        "| Metric | Result |",
        "|---|---|",
        *[f"| {name} | {m.rate} |" for name, m in rows],
        f"| Latency (mean / p95) | {statistics.mean(latencies):.0f} ms / {p95} ms |",
        f"| Tokens (input / output) | {tokens_in:,} / {tokens_out:,} |",
    ]
    if cost:
        lines.append(f"| Estimated cost | ${cost:.4f} |")
    failures = [f"- {name}: {', '.join(m.failures)}" for name, m in rows if m.failures]
    if failures:
        lines += ["", "Failures:", *failures]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="retrieval and guardrails only")
    args = parser.parse_args()

    settings, cases = get_settings(), load_cases()
    report = run_offline(settings, cases) if args.offline else run_live(settings, cases)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("offline.md" if args.offline else "live.md")).write_text(report)
    print(report)


if __name__ == "__main__":
    main()

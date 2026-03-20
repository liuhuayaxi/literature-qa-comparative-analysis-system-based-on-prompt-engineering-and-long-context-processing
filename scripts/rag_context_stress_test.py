#!/usr/bin/env python3
"""Run realistic RAG context stress tests with comprehensive QA evaluation.

Features:
- Build meaningful long history from real evidence chunks (no gibberish filler).
- Run practical domain questions about principles + parameters.
- Score model on:
  1) data finding ability,
  2) context recall ability,
  3) logical reasoning ability.
- Persist all conversations to SQLite and export JSON report.

Examples:
  .venv/bin/python scripts/rag_context_stress_test.py --preset 128k
  .venv/bin/python scripts/rag_context_stress_test.py --preset all --course-id 论文
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_utils import estimate_token_count
from src.config import AppConfig
from src.memory_store import SessionMemoryStore, SQLiteMemoryStore
from src.models import ChatTurn
from src.retrieval_engine import RAGChatService, VectorStoreService


@dataclass(slots=True)
class EvidenceChunk:
    file_name: str
    locator: str
    text: str


@dataclass(slots=True)
class StressCase:
    name: str
    model: str
    target_history_tokens: int
    model_context_window: int
    max_history_turns: int
    answer_token_reserve: int = 1000
    context_overflow_retries: int = 0


@dataclass(slots=True)
class EvalQuestion:
    qid: str
    ability: str
    prompt: str
    expected_terms: list[str]


@dataclass(slots=True)
class EvalResult:
    qid: str
    ability: str
    prompt: str
    answer_chars: int
    citations_count: int
    saw_error: bool
    error_text: str
    match_score: float
    matched_terms: list[str]
    missing_terms: list[str]


def _is_rate_limit_error(message: str) -> bool:
    text = str(message or "")
    low = text.lower()
    return ("频率限制" in text) or ("rate limit" in low) or ("too many requests" in low) or ("429" in low)


def _normalize_text(value: str, *, max_chars: int) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _extract_numeric_highlights(text: str) -> list[str]:
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:μmol|umol|mmol|mol)\s*/?\s*(?:g|h|L|mL)?(?:-1|⁻¹)?",
        r"\b\d+(?:\.\d+)?\s*°C",
        r"\b\d+(?:\.\d+)?\s*(?:W|kW)\b",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b(?:AM\s*1\.5|1\s*sun|full-spectrum)\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        for item in re.findall(pattern, text, flags=re.IGNORECASE):
            token = " ".join(str(item).split())
            if token and token not in found:
                found.append(token)
            if len(found) >= 6:
                return found
    return found


def _infer_focus_tags(text: str) -> str:
    low = text.lower()
    tag_rules = [
        ("等离子体预处理", ("plasma", "等离子体")),
        ("光催化过程", ("photocatal", "光催化")),
        ("ROS路径", ("ros", "活性氧", "自由基")),
        ("TiO2界面作用", ("tio2", "二氧化钛")),
        ("烯烃选择性", ("olefin", "烯烃", "selectivity", "选择性")),
        ("产率与能耗", ("yield", "产率", "energy", "能耗")),
        ("聚烯烃转化", ("polyolefin", "聚烯烃", "polyethylene", "聚乙烯")),
        ("中间体演化", ("intermediate", "中间体")),
    ]
    tags: list[str] = []
    for tag, keywords in tag_rules:
        if any(keyword in low for keyword in keywords):
            tags.append(tag)
    if not tags:
        return "反应机制、参数口径和可比性"
    return "、".join(tags[:3])


def _build_user_turn(index: int, evidence: EvidenceChunk, variant: int) -> str:
    focus = _infer_focus_tags(evidence.text)
    numeric = _extract_numeric_highlights(evidence.text)[:2]
    number_hint = (
        f"如果能检索到，顺便核对 { '、'.join(numeric) } 这些数字；查不到就直接说未检索到。"
        if numeric
        else "如果没有稳定数值，请直接说未检索到。"
    )
    prompts = [
        f"我准备把这组论文做成读书会汇报，你先别给大而全总结，先围绕 {focus} 帮我查最关键的证据链。{number_hint}",
        f"我对这块机制还是有点混乱。你按“起始条件-关键中间体-结果指标”三段来讲，重点看 {focus}，并告诉我哪些点现在还缺证据。{number_hint}",
        f"我们继续上一轮，你再帮我把 {focus} 相关内容查一遍。这次我只关心可对比的结论，尤其是实验条件变化后指标怎么变。{number_hint}",
        f"我想做跨文献对照，请你把 {focus} 这个方向先整理成可复核提纲，再标出哪些是明确结论、哪些只是推测。{number_hint}",
    ]
    return prompts[variant % len(prompts)]


def _build_assistant_turn(index: int, evidence: EvidenceChunk, variant: int) -> str:
    focus = _infer_focus_tags(evidence.text)
    nums = "、".join(_extract_numeric_highlights(evidence.text)[:4]) or "暂未检索到稳定数值"
    leads = [
        "我先给你一个工作版结论：",
        "这轮我先把脉络收拢一下：",
        "按你这个问题，我先给一版可复核摘要：",
        "先给你阶段性整理结果：",
    ]
    lead = leads[variant % len(leads)]
    return (
        f"{lead}当前最相关的是 {focus}。"
        f"为了后续能跨文献比较，我会按“条件、机制、指标、单位”四列去对齐。"
        f"目前先记下的量化线索是 {nums}。"
        "下一轮你可以指定要追问哪个参数或机制争议点，我再继续检索补齐。"
    )


async def _collect_evidence_chunks(
    vector_store: VectorStoreService,
    *,
    course_id: str,
    doc_ids: list[str],
    max_chunks: int,
    per_chunk_chars: int = 1200,
) -> list[EvidenceChunk]:
    if doc_ids:
        candidate_ids = doc_ids
    else:
        records = await vector_store.list_documents(course_id)
        candidate_ids = [item.doc_id for item in records]

    collected: list[EvidenceChunk] = []
    seen_hashes: set[str] = set()

    for doc_id in candidate_ids:
        chunks = await vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
        for chunk in chunks:
            text = _normalize_text(chunk.page_content, max_chars=per_chunk_chars)
            if len(text) < 120:
                continue
            key = sha1(text.encode("utf-8")).hexdigest()
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            meta = chunk.metadata
            locator = str(meta.get("page_label") or meta.get("section_label") or meta.get("section") or "")
            collected.append(
                EvidenceChunk(
                    file_name=str(meta.get("file_name") or doc_id),
                    locator=locator,
                    text=text,
                )
            )
            if len(collected) >= max_chunks:
                return collected
    return collected


def _build_meaningful_history(
    evidences: list[EvidenceChunk],
    *,
    target_tokens: int,
    model_name: str,
    max_turns: int,
) -> tuple[list[ChatTurn], int]:
    if not evidences:
        return [], 0
    turns: list[ChatTurn] = []
    running_tokens = 0
    now = datetime.now(timezone.utc)

    idx = 0
    round_no = 1
    while running_tokens < target_tokens and len(turns) < max_turns:
        evidence = evidences[idx % len(evidences)]
        variant = round_no % 4

        user_text = _build_user_turn(round_no, evidence, variant)
        assistant_text = _build_assistant_turn(round_no, evidence, variant)
        u_tok = estimate_token_count(user_text, model_name)
        a_tok = estimate_token_count(assistant_text, model_name)

        turns.append(ChatTurn(role="user", content=user_text, created_at=now))
        if len(turns) >= max_turns:
            running_tokens += u_tok
            break
        turns.append(ChatTurn(role="assistant", content=assistant_text, created_at=now))
        running_tokens += u_tok + a_tok

        idx += 1
        round_no += 1
    return turns, running_tokens


async def _seed_history(store: SQLiteMemoryStore, session_id: str, turns: list[ChatTurn]) -> None:
    for turn in turns:
        await store.append_turn(session_id=session_id, turn=turn)


def _build_eval_questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(
            qid="Q1",
            ability="data_finding",
            prompt=(
                "在《An Integrated Plasma–Photocatalytic System for Upcycling of Polyolefin Plastics》中，"
                "等离子体处理前后PE在TiO2表面的相互作用能分别是多少？请给出数值和单位。"
            ),
            expected_terms=["29.85", "45.04", "kcal"],
        ),
        EvalQuestion(
            qid="Q3",
            ability="data_finding",
            prompt=(
                "在 Yue 2024 这篇文章中，柴油烯烃的选择性和最高产率分别是多少？"
            ),
            expected_terms=["85", "76.1"],
        ),
        EvalQuestion(
            qid="Q4",
            ability="logical_reasoning",
            prompt=(
                "比较这两篇工作的核心机制差异：一个是等离子体预处理+光催化，"
                "一个是单一ROS调控。请给出差异点并指出证据缺失项。"
            ),
            expected_terms=["等离子体", "ROS", "差异", "未"],
        ),
        EvalQuestion(
            qid="Q5",
            ability="context_recall",
            prompt=(
                "回到你前面给出的两个定量结果（108.95 和 76.1），分别对应哪篇文献？"
                "这两个指标能否直接比较？请说明原因（单位/归一化维度）。"
            ),
            expected_terms=["108.95", "76.1", "单位", "比较"],
        ),
    ]


def _score_answer(answer: str, expected_terms: list[str]) -> tuple[float, list[str], list[str]]:
    lower = answer.lower()
    matched: list[str] = []
    missing: list[str] = []
    for term in expected_terms:
        if term.lower() in lower:
            matched.append(term)
        else:
            missing.append(term)
    score = (len(matched) / len(expected_terms)) if expected_terms else 0.0
    return score, matched, missing


async def _run_eval_question(
    rag: RAGChatService,
    *,
    session_id: str,
    course_id: str,
    doc_ids: list[str],
    question: EvalQuestion,
    streaming_mode: str,
) -> EvalResult:
    saw_error = False
    error_text = ""
    answer_text = ""
    citations_count = 0

    attempt = 0
    while True:
        attempt += 1
        saw_error = False
        error_text = ""
        answer_text = ""
        citations_count = 0

        async for event in rag.stream_answer(
            session_id=session_id,
            course_id=course_id,
            question=question.prompt,
            memory_mode="persistent",
            language="zh",
            doc_ids=doc_ids,
            enable_query_rewrite=False,
            streaming_mode=streaming_mode,
            retrieval_top_k=6,
            retrieval_fetch_k=20,
            citation_limit=4,
        ):
            event_type = str(event.get("type", ""))
            content = event.get("content")
            if event_type == "error":
                saw_error = True
                error_text = str(content or "")
            if event_type == "done":
                payload = content or {}
                if isinstance(payload, dict):
                    answer_text = str(payload.get("answer", ""))
                    citations = payload.get("citations")
                    if isinstance(citations, list):
                        citations_count = len(citations)

        if not saw_error:
            break
        if _is_rate_limit_error(error_text):
            await asyncio.sleep(min(10, 1 + attempt))
            continue
        break

    if saw_error:
        score, matched, missing = 0.0, [], list(question.expected_terms)
    else:
        score, matched, missing = _score_answer(answer_text, question.expected_terms)

    return EvalResult(
        qid=question.qid,
        ability=question.ability,
        prompt=question.prompt,
        answer_chars=len(answer_text),
        citations_count=citations_count,
        saw_error=saw_error,
        error_text=error_text,
        match_score=round(score, 4),
        matched_terms=matched,
        missing_terms=missing,
    )


def _aggregate_scores(results: list[EvalResult]) -> dict[str, Any]:
    if not results:
        return {
            "data_finding": 0.0,
            "context_recall": 0.0,
            "logical_reasoning": 0.0,
            "citation_coverage": 0.0,
            "overall": 0.0,
        }

    def avg(items: list[EvalResult]) -> float:
        if not items:
            return 0.0
        return sum(item.match_score for item in items) / len(items)

    data_items = [item for item in results if item.ability == "data_finding"]
    recall_items = [item for item in results if item.ability == "context_recall"]
    reasoning_items = [item for item in results if item.ability == "logical_reasoning"]

    data_score = round(avg(data_items), 4)
    recall_score = round(avg(recall_items), 4)
    reasoning_score = round(avg(reasoning_items), 4)
    citation_cov = round(sum(1 for item in results if item.citations_count > 0) / len(results), 4)

    overall = round(0.4 * data_score + 0.3 * recall_score + 0.3 * reasoning_score, 4)
    return {
        "data_finding": data_score,
        "context_recall": recall_score,
        "logical_reasoning": reasoning_score,
        "citation_coverage": citation_cov,
        "overall": overall,
    }


async def _run_case(
    base_config_path: str,
    case: StressCase,
    *,
    course_id: str,
    doc_ids: list[str],
    streaming_mode: str,
    evidences: list[EvidenceChunk],
) -> dict[str, Any]:
    cfg = AppConfig.from_file(base_config_path)
    cfg.chat_model = case.model
    cfg.model_context_window = case.model_context_window
    cfg.answer_token_reserve = case.answer_token_reserve
    cfg.context_overflow_retries = case.context_overflow_retries
    cfg.rate_limit_retry_forever = True
    cfg.rate_limit_retry_attempts = max(20, int(cfg.rate_limit_retry_attempts))
    cfg.api_retry_attempts = max(2, int(cfg.api_retry_attempts))
    cfg.prompt_compression_turn_token_limit = 320
    cfg.history_summary_token_limit = 2000

    seeded_turns, seeded_tokens = _build_meaningful_history(
        evidences,
        target_tokens=case.target_history_tokens,
        model_name=cfg.chat_model,
        max_turns=case.max_history_turns,
    )
    avg_turn_tokens = max(160, int(seeded_tokens / max(1, len(seeded_turns))))
    effective_recent_budget = int(case.target_history_tokens)
    if case.target_history_tokens <= 20000:
        effective_recent_budget = min(effective_recent_budget, 19000)
    cfg.recent_history_turns = min(
        len(seeded_turns),
        max(6, int(effective_recent_budget / avg_turn_tokens)),
    )

    session_id = f"auto_realstress_{case.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sqlite_store = SQLiteMemoryStore(cfg.db_path)
    await _seed_history(sqlite_store, session_id=session_id, turns=seeded_turns)

    rag = RAGChatService(
        config=cfg,
        vector_store=VectorStoreService(cfg),
        session_memory_store=SessionMemoryStore(),
        persistent_memory_store=sqlite_store,
    )

    print(f"\n[CASE] {case.name}")
    print(f"[MODEL] {case.model}")
    print(f"[SESSION] {session_id}")
    print(f"[SEEDED] turns={len(seeded_turns)}, approx_tokens={seeded_tokens}")

    question_results: list[EvalResult] = []
    for index, q in enumerate(_build_eval_questions(), start=1):
        if index > 1:
            await asyncio.sleep(2)
        print(f"[QUESTION] {q.qid} {q.ability}: {q.prompt[:80]}...")
        result = await _run_eval_question(
            rag,
            session_id=session_id,
            course_id=course_id,
            doc_ids=doc_ids,
            question=q,
            streaming_mode=streaming_mode,
        )
        question_results.append(result)
        if result.saw_error:
            print(f"[QUESTION_RESULT] {q.qid} error={result.error_text}")
            # if context limit already exceeded, stop subsequent questions
            if "上下文" in result.error_text or "context" in result.error_text.lower():
                break
        else:
            print(
                f"[QUESTION_RESULT] {q.qid} score={result.match_score} "
                f"citations={result.citations_count} matched={result.matched_terms}"
            )

    profile = await sqlite_store.get_session_profile(session_id=session_id)
    turns = await sqlite_store.get_recent_turns(session_id=session_id, limit=len(seeded_turns) + 40)

    results_payload = [
        {
            "qid": item.qid,
            "ability": item.ability,
            "prompt": item.prompt,
            "answer_chars": item.answer_chars,
            "citations_count": item.citations_count,
            "saw_error": item.saw_error,
            "error_text": item.error_text,
            "match_score": item.match_score,
            "matched_terms": item.matched_terms,
            "missing_terms": item.missing_terms,
        }
        for item in question_results
    ]

    scores = _aggregate_scores(question_results)
    summary = {
        "case": case.name,
        "model": case.model,
        "session_id": session_id,
        "seeded_history_turns": len(seeded_turns),
        "seeded_history_tokens_est": seeded_tokens,
        "db_turns_total": len(turns),
        "profile_prompt_tokens_est": profile.get("last_prompt_token_estimate"),
        "profile_compressed": profile.get("last_context_compressed"),
        "profile_strategies": profile.get("last_context_strategies"),
        "scores": scores,
        "questions": results_payload,
    }
    print(f"[SUMMARY] {json.dumps(summary, ensure_ascii=False)[:600]}...")
    return summary


def _build_cases(model_32k: str, model_large: str) -> dict[str, StressCase]:
    return {
        "32k": StressCase(
            name="ctx32k_real",
            model=model_32k,
            target_history_tokens=42000,
            model_context_window=32000,
            max_history_turns=220,
        ),
        "64k": StressCase(
            name="ctx64k_real",
            model=model_large,
            target_history_tokens=78000,
            model_context_window=32000,
            max_history_turns=320,
        ),
        "128k": StressCase(
            name="ctx128k_real",
            model=model_large,
            target_history_tokens=128000,
            model_context_window=32000,
            max_history_turns=460,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realistic RAG stress tests with capability evaluation.")
    parser.add_argument("--preset", choices=["32k", "64k", "128k", "all"], default="all")
    parser.add_argument("--course-id", default="论文")
    parser.add_argument(
        "--doc-ids",
        default="doc_09958df4131e,doc_4da824237577",
        help="Comma-separated document IDs for retrieval scope.",
    )
    parser.add_argument("--config", default="config/app_config.json")
    parser.add_argument("--streaming-mode", choices=["stream", "non_stream"], default="non_stream")
    parser.add_argument("--model-32k", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--model-large", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--max-evidence-chunks", type=int, default=1200)
    return parser.parse_args()


async def _amain() -> int:
    args = _parse_args()
    doc_ids = [item.strip() for item in args.doc_ids.split(",") if item.strip()]
    cases = _build_cases(args.model_32k, args.model_large)

    if args.preset == "all":
        selected_cases = [cases["32k"], cases["64k"], cases["128k"]]
    else:
        selected_cases = [cases[args.preset]]

    base_cfg = AppConfig.from_file(args.config)
    evidence_store = VectorStoreService(base_cfg)
    evidences = await _collect_evidence_chunks(
        evidence_store,
        course_id=args.course_id,
        doc_ids=doc_ids,
        max_chunks=max(200, int(args.max_evidence_chunks)),
    )
    if not evidences:
        raise RuntimeError("未找到可用证据切片，无法构造真实对话压测。")
    print(f"[EVIDENCE] loaded={len(evidences)}")

    reports: list[dict[str, Any]] = []
    for case in selected_cases:
        reports.append(
            await _run_case(
                args.config,
                case,
                course_id=args.course_id,
                doc_ids=doc_ids,
                streaming_mode=args.streaming_mode,
                evidences=evidences,
            )
        )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"rag_real_stress_eval_{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[REPORT] {report_path}")
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())

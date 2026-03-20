#!/usr/bin/env python3
"""Sweep multiple context budgets with realistic multi-turn QA and merged evaluation.

Runs context budgets (default: 64k/32k/20k/15k/10k/5k/1k) and evaluates:
- data_finding
- logical_reasoning
- context_recall

Outputs:
- reports/rag_context_sweep_*.json
- reports/rag_context_sweep_*.md
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
from statistics import mean
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
class EvalQuestion:
    qid: str
    ability: str
    prompt: str
    expected_terms: list[str]


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_TERM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "单位": ("单位", "归一化", "normalized", "normalization", "口径"),
    "比较": ("比较", "可比", "横向比较", "directly compare", "直接比较"),
    "不同": ("不同", "并非同一", "两个指标", "不是同一个指标"),
    "选择性": ("选择性", "selectivity"),
    "产率": ("产率", "yield"),
    "路径": ("路径", "route", "reaction path"),
    "差异": ("差异", "difference", "区别"),
    "可比": ("可比", "可比较", "可直接比较"),
    "不可比": ("不可比", "不可直接比较", "不能直接比较"),
    "kcal": ("kcal", "kcal/mol", "千卡"),
}


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


async def collect_evidence_chunks(
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

    chunks: list[EvidenceChunk] = []
    seen: set[str] = set()
    for doc_id in candidate_ids:
        docs = await vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
        for doc in docs:
            text = _normalize_text(doc.page_content, max_chars=per_chunk_chars)
            if len(text) < 120:
                continue
            h = sha1(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            m = doc.metadata
            locator = str(m.get("page_label") or m.get("section_label") or m.get("section") or "")
            chunks.append(EvidenceChunk(file_name=str(m.get("file_name") or doc_id), locator=locator, text=text))
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def build_meaningful_history(
    evidences: list[EvidenceChunk],
    *,
    target_tokens: int,
    model_name: str,
    max_turns: int,
) -> tuple[list[ChatTurn], int]:
    turns: list[ChatTurn] = []
    running = 0
    now = datetime.now(timezone.utc)
    anchor_rounds = [
        (
            "我先确认读书会里最容易记错的两个数字：PE 在 TiO2 上处理前后相互作用能各是多少？",
            "先记录成备忘：处理前后大约是 29.85 和 45.04 kcal/mol，后面如果检索到更细节再补充。",
        ),
        (
            "再确认另一篇里柴油区间这组数据，选择性和最高产率分别记成什么？",
            "这组我先记为 85%（选择性）和 76.1（最高产率），单位与归一化口径后面单独核对。",
        ),
        (
            "如果后面我要横向比较这两篇，你先提醒我最容易出错的点。",
            "最容易错在把不同统计口径直接对比。通常要先对齐单位、归一化维度和测试条件。",
        ),
    ]
    for user_text, assistant_text in anchor_rounds:
        if len(turns) + 2 > max_turns:
            break
        ut = estimate_token_count(user_text, model_name)
        at = estimate_token_count(assistant_text, model_name)
        turns.append(ChatTurn(role="user", content=user_text, created_at=now))
        turns.append(ChatTurn(role="assistant", content=assistant_text, created_at=now))
        running += ut + at
        if running >= target_tokens:
            return turns, running

    idx = 0
    round_no = 1
    while running < target_tokens and len(turns) < max_turns and evidences:
        ev = evidences[idx % len(evidences)]
        variant = round_no % 4
        u = _build_user_turn(round_no, ev, variant)
        a = _build_assistant_turn(round_no, ev, variant)
        ut = estimate_token_count(u, model_name)
        at = estimate_token_count(a, model_name)
        turns.append(ChatTurn(role="user", content=u, created_at=now))
        if len(turns) >= max_turns:
            running += ut
            break
        turns.append(ChatTurn(role="assistant", content=a, created_at=now))
        running += ut + at
        idx += 1
        round_no += 1
    return turns, running


async def seed_history(store: SQLiteMemoryStore, session_id: str, turns: list[ChatTurn]) -> None:
    for turn in turns:
        await store.append_turn(session_id=session_id, turn=turn)


def build_question_banks() -> dict[str, list[EvalQuestion]]:
    return {
        "baseline": [
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
                qid="Q2",
                ability="data_finding",
                prompt="在 Yue 2024 这篇文章中，柴油烯烃的选择性和最高产率分别是多少？",
                expected_terms=["85", "76.1"],
            ),
            EvalQuestion(
                qid="Q3",
                ability="logical_reasoning",
                prompt=(
                    "比较这两篇工作的核心机制差异：一个是等离子体预处理+光催化，一个是单一ROS调控。"
                    "请给出差异点并指出证据缺失项。"
                ),
                expected_terms=["等离子体", "ROS", "差异", "未"],
            ),
            EvalQuestion(
                qid="Q4",
                ability="context_recall",
                prompt=(
                    "回到你前面给出的两个定量结果（108.95 和 76.1），分别对应哪篇文献？"
                    "这两个指标能否直接比较？请说明原因（单位/归一化维度）。"
                ),
                expected_terms=["108.95", "76.1", "单位", "比较"],
            ),
        ],
        "strict_numeric": [
            EvalQuestion(
                qid="Q1",
                ability="data_finding",
                prompt=(
                    "如果只看 Plasma-Photocatalytic 这篇，PE 在 TiO2 表面的相互作用能在处理前后各是多少？"
                    "另外请补一句变化幅度。"
                ),
                expected_terms=["29.85", "45.04", "15.19"],
            ),
            EvalQuestion(
                qid="Q2",
                ability="data_finding",
                prompt=(
                    "Yue 2024 里柴油区间烯烃这部分，选择性和最高产率分别是多少？"
                    "请明确这是两个不同指标。"
                ),
                expected_terms=["85", "76.1", "不同"],
            ),
            EvalQuestion(
                qid="Q3",
                ability="logical_reasoning",
                prompt=(
                    "如果把这两篇工作放到同一张对比图里，至少要先统一哪两类统计口径，"
                    "不然结论会误导？请按检索证据回答。"
                ),
                expected_terms=["单位", "归一化", "比较"],
            ),
            EvalQuestion(
                qid="Q4",
                ability="context_recall",
                prompt="回顾你刚才给出的关键数字，哪几组属于界面作用能，哪几组属于产物分布指标？",
                expected_terms=["29.85", "45.04", "85", "76.1"],
            ),
        ],
        "cross_retrieval": [
            EvalQuestion(
                qid="Q1",
                ability="data_finding",
                prompt="在这两篇文献里，108.95 这个数值对应什么指标，来自哪篇研究？",
                expected_terms=["108.95", "Yue", "产率"],
            ),
            EvalQuestion(
                qid="Q2",
                ability="data_finding",
                prompt="和 85% 同一研究里，另一个最关键的产率数字是多少？两者分别表示什么？",
                expected_terms=["85", "76.1", "选择性"],
            ),
            EvalQuestion(
                qid="Q3",
                ability="logical_reasoning",
                prompt=(
                    "按检索结果，等离子体预处理主要改变了哪一段反应路径？"
                    "它和单一 ROS 调控路线最本质的差别是什么？"
                ),
                expected_terms=["等离子体", "ROS", "路径", "差异"],
            ),
            EvalQuestion(
                qid="Q4",
                ability="context_recall",
                prompt="结合你前面的回答，哪个数字最不适合跨论文直接横向比较？为什么？",
                expected_terms=["108.95", "单位", "归一化", "比较"],
            ),
        ],
        "mechanism_audit": [
            EvalQuestion(
                qid="Q1",
                ability="data_finding",
                prompt=(
                    "先只看 plasma-photocatalytic 这篇，文中给出的 PE 与 TiO2 相互作用能前后两个值是多少？"
                    "我只要原始数字和单位。"
                ),
                expected_terms=["29.85", "45.04", "kcal"],
            ),
            EvalQuestion(
                qid="Q2",
                ability="data_finding",
                prompt=(
                    "再看 Yue 2024，柴油区间烯烃这部分哪个是选择性，哪个是最高产率？"
                    "请把两个数值分别对应上。"
                ),
                expected_terms=["85", "76.1", "选择性", "产率"],
            ),
            EvalQuestion(
                qid="Q3",
                ability="logical_reasoning",
                prompt=(
                    "如果要论证“等离子体预处理不只是提高活性，而是改变反应路径”，"
                    "现有检索证据能支撑到哪一步？哪一步仍然缺证据？"
                ),
                expected_terms=["等离子体", "路径", "证据", "缺"],
            ),
            EvalQuestion(
                qid="Q4",
                ability="context_recall",
                prompt=(
                    "你刚才提到的关键数字里，哪个最不适合直接跨论文比较？"
                    "请给出原因（至少包含单位或归一化口径）。"
                ),
                expected_terms=["108.95", "单位", "归一化", "比较"],
            ),
        ],
        "attribution_trace": [
            EvalQuestion(
                qid="Q1",
                ability="data_finding",
                prompt="我怕把数字记串：29.85 和 45.04 分别对应什么指标，来自哪条机制链？",
                expected_terms=["29.85", "45.04", "TiO2"],
            ),
            EvalQuestion(
                qid="Q2",
                ability="data_finding",
                prompt="108.95 和 76.1 分别来自哪篇研究、各代表什么指标？",
                expected_terms=["108.95", "76.1", "Yue", "产率"],
            ),
            EvalQuestion(
                qid="Q3",
                ability="logical_reasoning",
                prompt=(
                    "如果只允许使用已检索证据，你能把两篇工作的“可比结论”和“不可比结论”分开吗？"
                    "核心依据写清楚。"
                ),
                expected_terms=["可比", "不可比", "单位", "归一化"],
            ),
            EvalQuestion(
                qid="Q4",
                ability="context_recall",
                prompt=(
                    "回到最初对比：等离子体路线和单一 ROS 路线，你总结的本质差异是什么？"
                ),
                expected_terms=["等离子体", "ROS", "差异", "路径"],
            ),
        ],
    }


def _extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _numeric_match(term: str, answer_numbers: list[float]) -> bool:
    if not _NUMBER_RE.search(term):
        return False
    try:
        expected_value = float(_NUMBER_RE.search(term).group(0))  # type: ignore[union-attr]
    except (ValueError, AttributeError):
        return False
    tolerance = max(0.2, abs(expected_value) * 0.02)
    for value in answer_numbers:
        if abs(value - expected_value) <= tolerance:
            return True
    return False


def _term_match(term: str, answer: str, low: str, answer_numbers: list[float]) -> bool:
    normalized_term = str(term or "").strip()
    if not normalized_term:
        return False
    lower_term = normalized_term.lower()
    if lower_term in low:
        return True
    if _numeric_match(normalized_term, answer_numbers):
        return True
    for alias in _TERM_SYNONYMS.get(normalized_term, ()):
        if alias.lower() in low:
            return True
    if normalized_term == "Yue" and ("yue" in low or "2024" in low):
        return True
    if normalized_term == "TiO2" and ("tio2" in low or "二氧化钛" in answer):
        return True
    return False


def score_answer(answer: str, expected_terms: list[str]) -> tuple[float, list[str], list[str]]:
    normalized_answer = str(answer or "")
    low = normalized_answer.lower()
    answer_numbers = _extract_numbers(normalized_answer)
    matched, missing = [], []
    for term in expected_terms:
        if _term_match(term, normalized_answer, low, answer_numbers):
            matched.append(term)
        else:
            missing.append(term)
    score = (len(matched) / len(expected_terms)) if expected_terms else 0.0
    return score, matched, missing


async def run_question(
    rag: RAGChatService,
    *,
    session_id: str,
    course_id: str,
    doc_ids: list[str],
    q: EvalQuestion,
    streaming_mode: str,
    question_timeout_sec: int,
) -> dict[str, Any]:
    async def _consume_one_attempt() -> tuple[str, int, bool, str, list[int]]:
        answer = ""
        citations_count = 0
        saw_error = False
        error_text = ""
        estimate_tokens_seen = []

        async for event in rag.stream_answer(
            session_id=session_id,
            course_id=course_id,
            question=q.prompt,
            memory_mode="persistent",
            language="zh",
            doc_ids=doc_ids,
            enable_query_rewrite=False,
            streaming_mode=streaming_mode,
            retrieval_top_k=6,
            retrieval_fetch_k=20,
            citation_limit=4,
        ):
            et = str(event.get("type", ""))
            content = str(event.get("content", ""))
            if et == "status":
                m = re.search(r"预计输入\s*(\d+)\s*tokens", content)
                if m:
                    estimate_tokens_seen.append(int(m.group(1)))
            if et == "error":
                saw_error = True
                error_text = content
            if et == "done":
                payload = event.get("content") or {}
                if isinstance(payload, dict):
                    answer = str(payload.get("answer", ""))
                    c = payload.get("citations")
                    if isinstance(c, list):
                        citations_count = len(c)
        return answer, citations_count, saw_error, error_text, estimate_tokens_seen

    answer = ""
    citations_count = 0
    saw_error = False
    error_text = ""
    estimate_tokens_seen: list[int] = []

    attempt = 0
    while True:
        attempt += 1
        try:
            answer, citations_count, saw_error, error_text, estimate_tokens_seen = await asyncio.wait_for(
                _consume_one_attempt(),
                timeout=max(30, int(question_timeout_sec)),
            )
        except asyncio.TimeoutError:
            saw_error = True
            error_text = f"timeout>{int(question_timeout_sec)}s"
            answer = ""
            citations_count = 0
            estimate_tokens_seen = []
        if not saw_error:
            break
        if _is_rate_limit_error(error_text):
            await asyncio.sleep(min(10, 1 + attempt))
            continue
        if "timeout" in str(error_text).lower() and attempt < 6:
            await asyncio.sleep(min(10, 1 + attempt))
            continue
        break

    if saw_error:
        score, matched, missing = 0.0, [], list(q.expected_terms)
    else:
        score, matched, missing = score_answer(answer, q.expected_terms)

    return {
        "qid": q.qid,
        "ability": q.ability,
        "prompt": q.prompt,
        "answer_chars": len(answer),
        "citations_count": citations_count,
        "saw_error": saw_error,
        "error_text": error_text,
        "match_score": round(score, 4),
        "matched_terms": matched,
        "missing_terms": missing,
        "max_prompt_tokens_estimate": max(estimate_tokens_seen) if estimate_tokens_seen else None,
    }


def aggregate_scores(question_results: list[dict[str, Any]]) -> dict[str, float]:
    def avg(values: list[float]) -> float:
        return round(mean(values), 4) if values else 0.0

    data = [float(x["match_score"]) for x in question_results if x["ability"] == "data_finding"]
    logic = [float(x["match_score"]) for x in question_results if x["ability"] == "logical_reasoning"]
    recall = [float(x["match_score"]) for x in question_results if x["ability"] == "context_recall"]
    citation_cov = round(
        sum(1 for x in question_results if int(x.get("citations_count", 0)) > 0) / len(question_results),
        4,
    ) if question_results else 0.0

    data_s = avg(data)
    logic_s = avg(logic)
    recall_s = avg(recall)
    overall = round(0.4 * data_s + 0.3 * logic_s + 0.3 * recall_s, 4)

    return {
        "data_finding": data_s,
        "logical_reasoning": logic_s,
        "context_recall": recall_s,
        "citation_coverage": citation_cov,
        "overall": overall,
    }


async def run_case(
    base_config: str,
    *,
    context_budget: int,
    question_bank: str,
    questions: list[EvalQuestion],
    repeat_index: int,
    history_retention: str,
    model: str,
    course_id: str,
    doc_ids: list[str],
    evidences: list[EvidenceChunk],
    streaming_mode: str,
    question_timeout_sec: int,
) -> dict[str, Any]:
    cfg = AppConfig.from_file(base_config)
    cfg.chat_model = model
    cfg.answer_token_reserve = 1000
    cfg.model_context_window = 32000
    cfg.context_overflow_retries = 0
    cfg.rate_limit_retry_forever = True
    cfg.rate_limit_retry_attempts = max(20, int(cfg.rate_limit_retry_attempts))
    cfg.api_retry_attempts = max(2, int(cfg.api_retry_attempts))
    cfg.prompt_compression_turn_token_limit = 320
    cfg.history_summary_token_limit = 2000

    target_tokens = max(1000, int(context_budget))
    max_turns = min(520, max(12, int(target_tokens / 220)))
    seeded_turns, seeded_tokens = build_meaningful_history(
        evidences,
        target_tokens=target_tokens,
        model_name=cfg.chat_model,
        max_turns=max_turns,
    )
    if history_retention == "case_full":
        # Keep full case history for every budget bucket.
        # This removes an evaluation bias where small buckets dropped early seeded
        # turns (including anchor facts) before retrieval+answering.
        anticipated_eval_turns = len(questions) * 2 + 4
        cfg.recent_history_turns = max(12, len(seeded_turns) + anticipated_eval_turns)
    else:
        avg_turn_tokens = max(160, int(seeded_tokens / max(1, len(seeded_turns))))
        effective_recent_budget = int(context_budget)
        if context_budget <= 20000:
            effective_recent_budget = min(effective_recent_budget, 19000)
        cfg.recent_history_turns = min(
            len(seeded_turns),
            max(6, int(effective_recent_budget / avg_turn_tokens)),
        )

    session_id = (
        f"auto_sweep_{question_bank}_{context_budget}_r{repeat_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    sqlite_store = SQLiteMemoryStore(cfg.db_path)
    await seed_history(sqlite_store, session_id, seeded_turns)

    rag = RAGChatService(
        config=cfg,
        vector_store=VectorStoreService(cfg),
        session_memory_store=SessionMemoryStore(),
        persistent_memory_store=sqlite_store,
    )

    print(f"\n[CASE] bank={question_bank} budget={context_budget} repeat={repeat_index} model={model}")
    print(f"[SESSION] {session_id}")
    print(f"[SEEDED] turns={len(seeded_turns)} tokens~{seeded_tokens}")

    q_results: list[dict[str, Any]] = []
    for idx, q in enumerate(questions, start=1):
        if idx > 1:
            await asyncio.sleep(1.5)
        print(f"[QUESTION] {q.qid} {q.ability}")
        qr = await run_question(
            rag,
            session_id=session_id,
            course_id=course_id,
            doc_ids=doc_ids,
            q=q,
            streaming_mode=streaming_mode,
            question_timeout_sec=question_timeout_sec,
        )
        q_results.append(qr)
        if qr["saw_error"]:
            print(f"[QUESTION_RESULT] {q.qid} error={qr['error_text'][:120]}")
        else:
            print(
                f"[QUESTION_RESULT] {q.qid} score={qr['match_score']} "
                f"citations={qr['citations_count']} matched={qr['matched_terms']}"
            )

    profile = await sqlite_store.get_session_profile(session_id)
    turns = await sqlite_store.get_recent_turns(session_id=session_id, limit=len(seeded_turns) + 50)

    case_scores = aggregate_scores(q_results)
    prompt_estimates = [x["max_prompt_tokens_estimate"] for x in q_results if x.get("max_prompt_tokens_estimate")]

    result = {
        "question_bank": question_bank,
        "repeat_index": repeat_index,
        "history_retention": history_retention,
        "context_budget": context_budget,
        "session_id": session_id,
        "model": model,
        "seeded_history_turns": len(seeded_turns),
        "seeded_history_tokens_est": seeded_tokens,
        "configured_recent_history_turns": int(cfg.recent_history_turns),
        "db_turns_total": len(turns),
        "profile_prompt_tokens_est": profile.get("last_prompt_token_estimate"),
        "profile_compressed": profile.get("last_context_compressed"),
        "profile_strategies": profile.get("last_context_strategies"),
        "max_prompt_tokens_estimate_seen": max(prompt_estimates) if prompt_estimates else None,
        "scores": case_scores,
        "questions": q_results,
    }
    print(f"[CASE_SUMMARY] {json.dumps(result, ensure_ascii=False)[:400]}...")
    return result


def _score_level(score: float) -> str:
    if score >= 0.85:
        return "优秀"
    if score >= 0.70:
        return "良好"
    if score >= 0.50:
        return "中等"
    if score >= 0.30:
        return "偏弱"
    return "较弱"


def _collect_missing_terms(case_result: dict[str, Any], ability: str) -> list[str]:
    missing: list[str] = []
    for item in case_result.get("questions", []):
        if str(item.get("ability")) != ability:
            continue
        for term in item.get("missing_terms", []):
            t = str(term).strip()
            if t and t not in missing:
                missing.append(t)
    return missing


def _ability_comment(ability: str, score: float, missing_terms: list[str]) -> str:
    if ability == "data_finding":
        if score >= 0.85:
            return "关键参数提取稳定，定量命中率高。"
        if score >= 0.70:
            return "主要参数可定位，但仍有个别数值遗漏。"
        if score >= 0.50:
            return "可提取部分参数，数值完整性一般。"
        if score >= 0.30:
            return "参数抽取不稳定，关键数值经常缺失。"
        return "参数定位能力较弱，关键定量结果大量缺失。"
    if ability == "logical_reasoning":
        if score >= 0.85:
            return "机制对比完整，差异与证据边界表达清晰。"
        if score >= 0.70:
            return "核心推理可成立，但细节要点存在遗漏。"
        if score >= 0.50:
            return "推理主线可读，但论证完整度不足。"
        if score >= 0.30:
            return "推理链条不稳定，机制差异说明偏弱。"
        return "推理能力不足，难以形成可靠机制对比。"
    if ability == "context_recall":
        if score >= 0.85:
            return "上下文追问可稳定回指，跨轮对应关系清楚。"
        if score >= 0.70:
            return "大部分跨轮回指正确，仍有小幅错漏。"
        if score >= 0.50:
            return "可回忆部分历史结论，关联一致性一般。"
        if score >= 0.30:
            return "跨轮找回不稳定，存在明显对应错误。"
        return "上下文找回能力较弱，历史信息难以可靠复现。"
    if ability == "citation_coverage":
        if score >= 0.85:
            return "回答基本都带引用，证据可追溯性强。"
        if score >= 0.70:
            return "引用覆盖较好，少量回答缺引用。"
        if score >= 0.50:
            return "引用覆盖中等，证据链条不够稳定。"
        if score >= 0.30:
            return "引用覆盖偏低，较多回答缺少出处。"
        return "引用覆盖较弱，证据可追溯性不足。"
    return ""


def _compression_comment(case_result: dict[str, Any], peak_tokens: int) -> str:
    compressed = bool(case_result.get("profile_compressed"))
    strategies = [str(item).strip() for item in case_result.get("profile_strategies", []) if str(item).strip()]
    strategies_text = "；".join(strategies) if strategies else "无"
    if compressed:
        return f"已触发压缩。策略={strategies_text}。峰值prompt估算={peak_tokens}。"
    return f"未触发压缩。门控/策略记录={strategies_text}。峰值prompt估算={peak_tokens}。"


def _aggregate_case_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    first = group[0]
    score_keys = ["data_finding", "logical_reasoning", "context_recall", "citation_coverage", "overall"]
    aggregated_scores = {
        key: round(mean(float(item["scores"][key]) for item in group), 4)
        for key in score_keys
    }
    peaks = [
        int(item.get("max_prompt_tokens_estimate_seen") or item.get("profile_prompt_tokens_est") or 0)
        for item in group
    ]
    question_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in group:
        for q in case.get("questions", []):
            qid = str(q.get("qid", "")).strip()
            ability = str(q.get("ability", "")).strip()
            if not qid:
                continue
            question_map.setdefault((qid, ability), []).append(q)

    aggregated_questions: list[dict[str, Any]] = []
    for (qid, ability), items in sorted(question_map.items(), key=lambda x: x[0][0]):
        matched_counter: dict[str, int] = {}
        missing_counter: dict[str, int] = {}
        for item in items:
            for term in item.get("matched_terms", []):
                matched_counter[str(term)] = matched_counter.get(str(term), 0) + 1
            for term in item.get("missing_terms", []):
                missing_counter[str(term)] = missing_counter.get(str(term), 0) + 1
        top_matched = [
            key for key, _ in sorted(matched_counter.items(), key=lambda x: (-x[1], x[0]))[:4]
        ]
        top_missing = [
            key for key, _ in sorted(missing_counter.items(), key=lambda x: (-x[1], x[0]))[:4]
        ]
        aggregated_questions.append(
            {
                "qid": qid,
                "ability": ability,
                "prompt": str(items[0].get("prompt", "")),
                "match_score": round(mean(float(item.get("match_score", 0.0)) for item in items), 4),
                "citations_count": round(mean(float(item.get("citations_count", 0)) for item in items), 2),
                "error_rate": round(
                    sum(1 for item in items if bool(item.get("saw_error"))) / len(items),
                    4,
                ),
                "matched_terms": top_matched,
                "missing_terms": top_missing,
            }
        )

    return {
        "question_bank": str(first.get("question_bank") or "baseline"),
        "context_budget": int(first["context_budget"]),
        "repeats": len(group),
        "scores": aggregated_scores,
        "avg_prompt_tokens_peak": round(mean(peaks), 2) if peaks else 0.0,
        "compression_rate": round(
            sum(1 for item in group if bool(item.get("profile_compressed"))) / len(group),
            4,
        ),
        "seeded_history_tokens_est_avg": int(
            round(mean(float(item.get("seeded_history_tokens_est", 0)) for item in group))
        ),
        "seeded_turns_avg": int(round(mean(float(item.get("seeded_history_turns", 0)) for item in group))),
        "db_turns_total_avg": int(round(mean(float(item.get("db_turns_total", 0)) for item in group))),
        "session_ids": [str(item.get("session_id", "")) for item in group if str(item.get("session_id", ""))],
        "questions": aggregated_questions,
    }


def _aggregate_by_bank(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for item in results:
        bank = str(item.get("question_bank") or "baseline")
        budget = int(item.get("context_budget") or 0)
        grouped.setdefault(bank, {}).setdefault(budget, []).append(item)
    out: dict[str, list[dict[str, Any]]] = {}
    for bank, budget_map in grouped.items():
        out[bank] = [
            _aggregate_case_group(group)
            for _, group in sorted(budget_map.items(), key=lambda x: x[0], reverse=True)
        ]
    return out


def _count_expected_decline_violations(ordered: list[dict[str, Any]], tolerance: float = 0.03) -> int:
    """Count violations against expectation: higher context should not score better."""

    if len(ordered) < 2:
        return 0
    violations = 0
    for idx in range(len(ordered) - 1):
        high = float(ordered[idx]["scores"]["overall"])
        lower = float(ordered[idx + 1]["scores"]["overall"])
        if high > lower + tolerance:
            violations += 1
    return violations


def _append_bank_report(lines: list[str], bank_name: str, bank_results: list[dict[str, Any]]) -> None:
    ordered = sorted(bank_results, key=lambda x: int(x["context_budget"]), reverse=True)
    lines.extend(
        [
            f"## 题库：{bank_name}",
            "",
            "### 总览",
            "",
            "| 预算 | 重复数 | 平均prompt峰值 | 数据定位 | 逻辑推理 | 上下文找回 | 引用覆盖 | 总分 | 压缩触发率 | 总体等级 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in ordered:
        s = r["scores"]
        peak = float(r.get("avg_prompt_tokens_peak") or 0.0)
        lines.append(
            f"| {r['context_budget']} | {r['repeats']} | {peak:.2f} | {s['data_finding']:.4f} | {s['logical_reasoning']:.4f} | {s['context_recall']:.4f} | {s['citation_coverage']:.4f} | {s['overall']:.4f} | {float(r.get('compression_rate', 0.0)):.4f} | {_score_level(float(s['overall']))} |"
        )

    lines.extend(["", "### 关键观察", ""])
    if ordered:
        best_data = max(ordered, key=lambda x: float(x["scores"]["data_finding"]))
        best_logic = max(ordered, key=lambda x: float(x["scores"]["logical_reasoning"]))
        best_recall = max(ordered, key=lambda x: float(x["scores"]["context_recall"]))
        lines.append(
            f"- 数据定位最佳：{best_data['context_budget']}（{best_data['scores']['data_finding']:.4f}，{_score_level(float(best_data['scores']['data_finding']))}）"
        )
        lines.append(
            f"- 逻辑推理最佳：{best_logic['context_budget']}（{best_logic['scores']['logical_reasoning']:.4f}，{_score_level(float(best_logic['scores']['logical_reasoning']))}）"
        )
        lines.append(
            f"- 上下文找回最佳：{best_recall['context_budget']}（{best_recall['scores']['context_recall']:.4f}，{_score_level(float(best_recall['scores']['context_recall']))}）"
        )
        violation_count = _count_expected_decline_violations(ordered)
        lines.append(
            f"- 趋势检查（期望“上下文越长，总分越低”）：违反次数={violation_count}（容差=0.03）。"
        )

    lines.extend(["", "### 分档文字评估", ""])
    for r in ordered:
        s = r["scores"]
        budget = int(r["context_budget"])
        peak = float(r.get("avg_prompt_tokens_peak") or 0.0)
        data_score = float(s["data_finding"])
        logic_score = float(s["logical_reasoning"])
        recall_score = float(s["context_recall"])
        cite_score = float(s["citation_coverage"])
        overall_score = float(s["overall"])
        data_missing = _collect_missing_terms(r, "data_finding")
        logic_missing = _collect_missing_terms(r, "logical_reasoning")
        recall_missing = _collect_missing_terms(r, "context_recall")
        lines.append(f"#### {budget} tokens")
        lines.append(f"- 综合表现：{overall_score:.4f}（{_score_level(overall_score)}）。")
        lines.append(
            f"- 数据定位：{data_score:.4f}（{_score_level(data_score)}）。{_ability_comment('data_finding', data_score, data_missing)} 缺失词={data_missing or ['无']}。"
        )
        lines.append(
            f"- 逻辑推理：{logic_score:.4f}（{_score_level(logic_score)}）。{_ability_comment('logical_reasoning', logic_score, logic_missing)} 缺失词={logic_missing or ['无']}。"
        )
        lines.append(
            f"- 上下文找回：{recall_score:.4f}（{_score_level(recall_score)}）。{_ability_comment('context_recall', recall_score, recall_missing)} 缺失词={recall_missing or ['无']}。"
        )
        lines.append(
            f"- 引用覆盖：{cite_score:.4f}（{_score_level(cite_score)}）。{_ability_comment('citation_coverage', cite_score, [])}"
        )
        lines.append(f"- 压缩触发率：{float(r.get('compression_rate', 0.0)):.4f}，平均prompt峰值={peak:.2f}。")
        lines.append(
            f"- 运行信息：重复={r['repeats']}，seeded_history_tokens均值≈{r['seeded_history_tokens_est_avg']}，seeded_turns均值={r['seeded_turns_avg']}，DB_turns均值={r['db_turns_total_avg']}。"
        )
        preview_sessions = ", ".join(r.get("session_ids", [])[:3])
        if preview_sessions:
            lines.append(f"- 会话样例：{preview_sessions}")
        lines.append("")

    lines.extend(["### 题目级明细", ""])
    for r in ordered:
        lines.append(f"#### {r['context_budget']} tokens")
        lines.append("| 题号 | 能力 | 均分 | 高频命中词 | 高频缺失词 | 平均引用 | 错误率 |")
        lines.append("|---|---|---:|---|---|---:|---|")
        for q in r.get("questions", []):
            matched = ",".join(str(item) for item in q.get("matched_terms", [])) or "-"
            missing = ",".join(str(item) for item in q.get("missing_terms", [])) or "-"
            lines.append(
                f"| {q['qid']} | {q['ability']} | {float(q['match_score']):.4f} | {matched} | {missing} | {float(q['citations_count']):.2f} | {float(q['error_rate']):.4f} |"
            )
        lines.append("")

    lines.extend(["### 本题库提问清单", ""])
    question_items: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    for r in ordered:
        for q in r.get("questions", []):
            qid = str(q.get("qid", "")).strip()
            if not qid or qid in seen_ids:
                continue
            seen_ids.add(qid)
            question_items.append(
                (
                    qid,
                    str(q.get("ability", "")).strip(),
                    str(q.get("prompt", "")).strip(),
                )
            )
    for qid, ability, prompt in sorted(question_items, key=lambda item: item[0]):
        lines.append(f"- {qid}（{ability}）：{prompt}")
    lines.append("")


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    grouped = _aggregate_by_bank(results)
    retention_modes = sorted(
        {
            str(item.get("history_retention") or "").strip()
            for item in results
            if str(item.get("history_retention") or "").strip()
        }
    )

    lines = [
        "# 上下文梯度真实问答评测报告（多题库）",
        "",
        "## 评测范围",
        "",
        "- 上下文预算：64k, 32k, 20k, 15k, 10k, 5k, 1k",
        f"- 题库集合：{', '.join(sorted(grouped))}",
        f"- 历史保留策略：{', '.join(retention_modes) if retention_modes else '未记录'}",
        "- 问题类型：参数抽取、跨文献机制比较、上下文追问",
        "- 能力维度：数据定位 / 逻辑推理 / 上下文找回 / 引用覆盖",
        "",
        "## 评判标准",
        "",
        "| 维度 | 计算方式 | 评分含义（统一阈值） |",
        "|---|---|---|",
        "| 数据定位 | data_finding 题目的术语命中率平均值 | 0.85-1.00=优秀，0.70-0.85=良好，0.50-0.70=中等，0.30-0.50=偏弱，<0.30=较弱 |",
        "| 逻辑推理 | logical_reasoning 题目的术语命中率平均值 | 0.85-1.00=优秀，0.70-0.85=良好，0.50-0.70=中等，0.30-0.50=偏弱，<0.30=较弱 |",
        "| 上下文找回 | context_recall 题目的术语命中率平均值 | 0.85-1.00=优秀，0.70-0.85=良好，0.50-0.70=中等，0.30-0.50=偏弱，<0.30=较弱 |",
        "| 引用覆盖 | 有引用回答数 / 总回答数 | 0.85-1.00=优秀，0.70-0.85=良好，0.50-0.70=中等，0.30-0.50=偏弱，<0.30=较弱 |",
        "| 总分 | 0.4*数据定位 + 0.3*逻辑推理 + 0.3*上下文找回 | 使用同一阈值分档，强调参数抽取权重更高 |",
        "",
        "## 跨题库总览",
        "",
        "| 题库 | 预算 | 重复数 | 平均prompt峰值 | 数据定位 | 逻辑推理 | 上下文找回 | 总分 | 压缩触发率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bank in sorted(grouped):
        ordered = sorted(grouped[bank], key=lambda x: int(x["context_budget"]), reverse=True)
        for r in ordered:
            s = r["scores"]
            peak = float(r.get("avg_prompt_tokens_peak") or 0.0)
            lines.append(
                f"| {bank} | {r['context_budget']} | {r['repeats']} | {peak:.2f} | {s['data_finding']:.4f} | {s['logical_reasoning']:.4f} | {s['context_recall']:.4f} | {s['overall']:.4f} | {float(r.get('compression_rate', 0.0)):.4f} |"
            )

    for bank in sorted(grouped):
        lines.append("")
        _append_bank_report(lines, bank, grouped[bank])

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run context sweep eval.")
    p.add_argument("--contexts", default="64000,32000,20000,15000,10000,5000,1000")
    p.add_argument(
        "--question-banks",
        default="baseline,strict_numeric,cross_retrieval,mechanism_audit,attribution_trace",
    )
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--config", default="config/app_config.json")
    p.add_argument("--course-id", default="论文")
    p.add_argument("--doc-ids", default="doc_09958df4131e,doc_4da824237577")
    p.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--streaming-mode", choices=["stream", "non_stream"], default="non_stream")
    p.add_argument(
        "--history-retention",
        choices=["case_full", "budget_adaptive"],
        default="case_full",
        help="case_full keeps all seeded+eval turns for each case; budget_adaptive scales by budget.",
    )
    p.add_argument("--question-timeout-sec", type=int, default=240)
    p.add_argument("--max-evidence-chunks", type=int, default=1200)
    return p.parse_args()


async def amain() -> int:
    args = parse_args()
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    question_banks = [x.strip() for x in args.question_banks.split(",") if x.strip()]
    repeats = max(1, int(args.repeats))
    doc_ids = [x.strip() for x in args.doc_ids.split(",") if x.strip()]
    all_banks = build_question_banks()
    unknown_banks = [name for name in question_banks if name not in all_banks]
    if unknown_banks:
        raise RuntimeError(
            f"未知题库: {unknown_banks}。可选题库: {sorted(all_banks)}"
        )

    base_cfg = AppConfig.from_file(args.config)
    evidence_store = VectorStoreService(base_cfg)
    evidences = await collect_evidence_chunks(
        evidence_store,
        course_id=args.course_id,
        doc_ids=doc_ids,
        max_chunks=max(200, int(args.max_evidence_chunks)),
    )
    if not evidences:
        raise RuntimeError("未找到证据切片，无法执行评测。")
    print(f"[EVIDENCE] loaded={len(evidences)}")

    results: list[dict[str, Any]] = []
    total_cases = repeats * len(question_banks) * len(contexts)
    case_index = 0
    for repeat_index in range(1, repeats + 1):
        print(f"\n[REPEAT] {repeat_index}/{repeats}")
        for bank in question_banks:
            questions = all_banks[bank]
            print(f"\n[BANK] {bank} questions={len(questions)}")
            for budget in contexts:
                case_index += 1
                if case_index > 1:
                    await asyncio.sleep(2)
                print(f"[PROGRESS] case {case_index}/{total_cases}")
                result = await run_case(
                    args.config,
                    context_budget=budget,
                    question_bank=bank,
                    questions=questions,
                    repeat_index=repeat_index,
                    history_retention=args.history_retention,
                    model=args.model,
                    course_id=args.course_id,
                    doc_ids=doc_ids,
                    evidences=evidences,
                    streaming_mode=args.streaming_mode,
                    question_timeout_sec=max(30, int(args.question_timeout_sec)),
                )
                results.append(result)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path("reports") / f"rag_context_sweep_{ts}.json"
    md_path = Path("reports") / f"rag_context_sweep_{ts}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(results), encoding="utf-8")

    print(f"\n[REPORT_JSON] {json_path}")
    print(f"[REPORT_MD] {md_path}")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())

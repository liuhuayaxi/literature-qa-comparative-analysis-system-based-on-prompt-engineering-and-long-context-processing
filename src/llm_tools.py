"""Prompt templates, model builders, lifecycle helpers, and health checks."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import suppress
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Iterable, Literal
from uuid import uuid4

from .app_utils import (
    check_task_cancelled,
    estimate_token_count,
    sleep_with_task_control,
    sleep_with_task_control_blocking,
)
from .errors import OperationCancelledError, ProviderRequestError, classify_provider_exception
from .models import ChatTurn, ChunkCitation, ExtractionFieldSpec, SingleDocAnalysis, SourceDocument

if TYPE_CHECKING:
    from .config import AppConfig


StatusCallback = Callable[[str], Awaitable[None] | None]
_CHAT_QUEUE_LOCK = threading.Lock()
_CHAT_QUEUE_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_EMBED_QUEUE_LOCK = threading.Lock()
_EMBED_QUEUE_SEMAPHORES: dict[str, threading.Semaphore] = {}


DEFAULT_QUERY_REWRITE_INSTRUCTION_ZH = (
    "你是检索改写器，不负责直接回答问题。请把当前用户问题改写成适合知识库检索的独立查询。\n"
    "要求：\n"
    "1. 结合摘要历史和最近对话，补全代词、省略和上下文指代。\n"
    "2. 保留课程名、文件名、专业术语、缩写、时间、版本号等检索关键词。\n"
    "3. 如果原问题已经独立清晰，只做最小改写。\n"
    "4. 不要扩写成答案，不要解释，不要添加对话中没有出现的新事实。\n"
    "5. 输出语言跟随当前问题。\n"
    "只返回一行改写后的检索问题。"
)
DEFAULT_QUERY_REWRITE_INSTRUCTION_EN = (
    "You are a retrieval-query rewriter, not an answer generator. Rewrite the user's latest question into a standalone query for knowledge-base retrieval.\n"
    "Requirements:\n"
    "1. Use the conversation summary and recent turns to resolve pronouns, ellipsis, and missing references.\n"
    "2. Preserve course names, file names, technical terms, abbreviations, dates, and version numbers.\n"
    "3. If the question is already standalone, apply only minimal edits.\n"
    "4. Do not answer the question, do not explain your reasoning, and do not add new facts.\n"
    "5. Keep the output in the same language as the user's current question.\n"
    "Return only the rewritten retrieval query on a single line."
)
DEFAULT_QA_SYSTEM_PROMPT_ZH = (
    "你是课程资料问答助手，你的首要任务是基于检索到的原始片段回答问题。\n"
    "规则：\n"
    "1. 只有检索片段中的内容可以作为可引用依据。\n"
    "2. 若证据不足、无法定位、存在冲突，必须明确说明未在当前知识库中找到可引用依据。\n"
    "3. 优先给出直接结论，再补充必要解释；不要编造事实、页码、章节或引用。\n"
    "4. 若用户要求常识性补充，可单独标注“非知识库补充”，并与有依据的结论分开。\n"
    "5. 回答语言跟随用户问题或当前语言设置。"
)
DEFAULT_QA_SYSTEM_PROMPT_EN = (
    "You are a course-material QA assistant. Your primary job is to answer from the retrieved source passages.\n"
    "Rules:\n"
    "1. Treat only the retrieved passages as citable evidence.\n"
    "2. If the evidence is insufficient, missing, ambiguous, or conflicting, explicitly say that no citable evidence was found in the current knowledge base.\n"
    "3. Give the direct answer first, then add only the necessary explanation. Do not invent facts, page numbers, sections, or citations.\n"
    "4. If the user asks for general background knowledge, separate it clearly as a non-knowledge-base supplement.\n"
    "5. Match the answer language to the user's question or the selected language setting."
)
DEFAULT_QA_ANSWER_INSTRUCTION_ZH = (
    "请只输出答案正文。先直接回答，再按需要补充简短要点。"
    "不要输出“来源”或“参考资料”标题，也不要自行拼接引用编号；引用将由系统追加。"
)
DEFAULT_QA_ANSWER_INSTRUCTION_EN = (
    "Return only the answer body. Lead with the direct answer, then add brief supporting points only if needed. "
    'Do not include a "Sources" heading and do not format citation numbers yourself; the system appends citations.'
)
DEFAULT_SINGLE_ANALYSIS_PROMPT_ZH = (
    "你是课程资料分析助手。请阅读输入文本并输出严格的 JSON，对应字段必须为 "
    "summary, sentiment, keywords, main_topics, risk_points。\n"
    "要求：\n"
    "1. summary 只保留核心观点、方法、结论和限定条件。\n"
    "2. sentiment 只能是 positive / neutral / negative / mixed 之一；学术与课程材料通常偏 neutral。\n"
    "3. keywords 保留原文中的专业术语、模型名、方法名、课程概念。\n"
    "4. main_topics 提炼 3 到 6 个核心主题。\n"
    "5. risk_points 提取风险、局限、假设、注意事项；没有明确风险时给出谨慎表述。\n"
    "6. 不要输出 markdown，不要补充 JSON 之外的解释。"
)
DEFAULT_SINGLE_ANALYSIS_PROMPT_EN = (
    "You are a course-material analysis assistant. Read the input text and return strict JSON with the fields "
    "summary, sentiment, keywords, main_topics, risk_points.\n"
    "Requirements:\n"
    "1. Keep summary focused on key ideas, methods, conclusions, and caveats.\n"
    "2. sentiment must be exactly one of positive / neutral / negative / mixed; academic and course materials are often neutral.\n"
    "3. Preserve domain terms, model names, method names, and course concepts in keywords.\n"
    "4. Extract 3 to 6 main topics.\n"
    "5. risk_points should capture risks, limitations, assumptions, or cautions; if no explicit risk exists, use a cautious formulation.\n"
    "6. Do not output markdown and do not add explanations outside the JSON object."
)
DEFAULT_COMPARE_REPORT_PROMPT_ZH = (
    "你是课程文献对比报告助手。请基于多篇文档的单文档分析结果，生成结构化 Markdown 报告。\n"
    "要求：\n"
    "1. 报告结构尽量覆盖：数据概览、单文档摘要、主题共性、关键差异、方法/观点对比、风险点与局限、对课程学习的启发、关键结论引用。\n"
    "2. 先写共性，再写差异，再写风险和启发。\n"
    "3. 对关键结论尽量引用候选引用中的文件名和定位信息。\n"
    "4. 不要编造文档中不存在的结论。\n"
    "5. 输出必须是 Markdown 正文。"
)
DEFAULT_COMPARE_REPORT_PROMPT_EN = (
    "You are a course-literature comparison assistant. Generate a structured Markdown report from multiple single-document analyses.\n"
    "Requirements:\n"
    "1. Cover, as much as possible: data overview, single-document summaries, shared themes, key differences, methods/viewpoints comparison, risks and limitations, learning implications, and key citations.\n"
    "2. Present shared themes first, then differences, then risks and implications.\n"
    "3. Use the candidate citations for major claims whenever possible, including file names and locators.\n"
    "4. Do not invent conclusions that are not supported by the analyses.\n"
    "5. Output Markdown only."
)
DEFAULT_DATA_EXTRACTION_PROMPT_ZH = (
    "你是课程论文数据抽取助手。请只基于提供的切片，为每个目标字段抽取结构化结果。\n"
    "输出必须是严格 JSON，格式为：\n"
    '{"fields":[{"field_name":"","value":"","normalized_value":"","unit":"","status":"found|not_found|conflict|uncertain","evidence_chunk_id":"","notes":""}]}\n'
    "要求：\n"
    "1. 只能依据提供的切片，不能编造未出现的数据。\n"
    "2. 如果同一字段出现多个互相冲突的数据，status 设为 conflict，并在 notes 里简要说明。\n"
    "3. 如果没有找到，status 设为 not_found。\n"
    "4. 若只能找到模糊表述，status 设为 uncertain。\n"
    "5. evidence_chunk_id 必须填写提供切片中的 chunk_id；未找到时可为空。\n"
    "6. normalized_value 应尽量做轻量标准化，unit 只写单位本身。"
)
DEFAULT_DATA_EXTRACTION_PROMPT_EN = (
    "You are a paper-data extraction assistant. Extract one structured result for each requested field using only the provided chunks.\n"
    "Return strict JSON in this format:\n"
    '{"fields":[{"field_name":"","value":"","normalized_value":"","unit":"","status":"found|not_found|conflict|uncertain","evidence_chunk_id":"","notes":""}]}\n'
    "Requirements:\n"
    "1. Use only the supplied chunks and do not invent missing values.\n"
    "2. If conflicting values are present for the same field, set status to conflict and explain briefly in notes.\n"
    "3. If no value is found, set status to not_found.\n"
    "4. If the evidence is vague or incomplete, set status to uncertain.\n"
    "5. evidence_chunk_id must match one of the provided chunk IDs; leave it empty only when nothing is found.\n"
    "6. normalized_value should be a lightly standardized form and unit should contain only the unit text."
)
DEFAULT_TABLE_SUMMARY_PROMPT_ZH = (
    "你是论文数据对比总结助手。请基于结构化字段抽取结果，输出 Markdown 小节。\n"
    "要求：\n"
    "1. 先总结字段覆盖情况，再总结关键差异和异常值。\n"
    "2. 明确指出缺失值、冲突值和需要人工复查的条目。\n"
    "3. 不要重复完整表格，重点写结论。"
)
DEFAULT_TABLE_SUMMARY_PROMPT_EN = (
    "You are a data-comparison summary assistant. Write a Markdown section from structured field-extraction results.\n"
    "Requirements:\n"
    "1. Summarize field coverage first, then highlight key differences and anomalies.\n"
    "2. Explicitly call out missing values, conflicting values, and items that need manual review.\n"
    "3. Do not repeat the full table; focus on conclusions."
)


def build_query_rewrite_prompt(
    question: str,
    history: Iterable[ChatTurn],
    language: str,
    instruction_override: str | None = None,
    history_summary: str | None = None,
    focus_sources: list[str] | None = None,
    selected_document_titles: list[str] | None = None,
) -> str:
    history_block = "\n".join(f"{item.role}: {item.content}" for item in history)
    instruction = instruction_override.strip() if instruction_override else (
        DEFAULT_QUERY_REWRITE_INSTRUCTION_ZH if language == "zh" else DEFAULT_QUERY_REWRITE_INSTRUCTION_EN
    )
    summary_heading = "摘要历史" if language == "zh" else "Conversation summary"
    history_heading = "最近对话" if language == "zh" else "Recent turns"
    focus_heading = "上一轮重点来源" if language == "zh" else "Previous focus sources"
    scope_heading = "当前已选文档" if language == "zh" else "Selected documents"
    question_heading = "当前问题" if language == "zh" else "Question"
    sections = [instruction]
    if history_summary and history_summary.strip():
        sections.append(f"{summary_heading}:\n{history_summary.strip()}")
    if history_block:
        sections.append(f"{history_heading}:\n{history_block}")
    if focus_sources:
        sections.append(f"{focus_heading}:\n" + "\n".join(f"- {item}" for item in focus_sources if str(item).strip()))
    if selected_document_titles:
        sections.append(
            f"{scope_heading}:\n" + "\n".join(f"- {item}" for item in selected_document_titles if str(item).strip())
        )
    sections.append(f"{question_heading}:\n{question}")
    return "\n\n".join(sections)


def build_rag_prompt(
    question: str,
    context_docs: list[SourceDocument],
    language: str,
    system_prompt_override: str | None = None,
    answer_instruction_override: str | None = None,
    recent_history: Iterable[ChatTurn] | None = None,
    history_summary: str | None = None,
    focus_sources: list[str] | None = None,
    selected_document_titles: list[str] | None = None,
) -> str:
    instructions = system_prompt_override.strip() if system_prompt_override else (
        DEFAULT_QA_SYSTEM_PROMPT_ZH if language == "zh" else DEFAULT_QA_SYSTEM_PROMPT_EN
    )
    context_blocks = []
    for index, doc in enumerate(context_docs, start=1):
        metadata = doc.metadata
        locator = metadata.get("page_label") or metadata.get("section_label") or metadata.get("section") or ""
        context_blocks.append(
            f"[{index}] file={metadata.get('file_name')} locator={locator}\n{doc.page_content}"
        )
    joined_context = "\n\n".join(context_blocks)
    answer_instruction = answer_instruction_override.strip() if answer_instruction_override else (
        DEFAULT_QA_ANSWER_INSTRUCTION_ZH if language == "zh" else DEFAULT_QA_ANSWER_INSTRUCTION_EN
    )
    summary_heading = "摘要历史" if language == "zh" else "Conversation summary"
    history_heading = "最近对话" if language == "zh" else "Recent turns"
    focus_heading = "上一轮重点来源" if language == "zh" else "Previous focus sources"
    scope_heading = "当前已选文档" if language == "zh" else "Selected documents"
    question_heading = "当前问题" if language == "zh" else "Question"
    context_heading = "参考资料" if language == "zh" else "Context"
    history_block = "\n".join(f"{item.role}: {item.content}" for item in (recent_history or []))
    sections = [
        instructions,
        answer_instruction,
    ]
    if history_summary and history_summary.strip():
        sections.append(f"{summary_heading}:\n{history_summary.strip()}")
    if history_block:
        sections.append(f"{history_heading}:\n{history_block}")
    if focus_sources:
        sections.append(f"{focus_heading}:\n" + "\n".join(f"- {item}" for item in focus_sources if str(item).strip()))
    if selected_document_titles:
        sections.append(
            f"{scope_heading}:\n" + "\n".join(f"- {item}" for item in selected_document_titles if str(item).strip())
        )
    sections.append(f"{question_heading}:\n{question}")
    sections.append(f"{context_heading}:\n{joined_context}")
    return "\n\n".join(sections)


def build_single_analysis_prompt(
    text: str,
    title: str,
    language: str,
    instruction_override: str | None = None,
) -> str:
    instruction = instruction_override.strip() if instruction_override else (
        DEFAULT_SINGLE_ANALYSIS_PROMPT_ZH if language == "zh" else DEFAULT_SINGLE_ANALYSIS_PROMPT_EN
    )
    return f"{instruction}\n\nTitle: {title}\n\nText:\n{text}"


def build_window_summary_prompt(
    *,
    title: str,
    text: str,
    language: str,
    window_index: int,
    total_windows: int,
) -> str:
    instruction = (
        "你正在处理长文档的一个滑动窗口片段。请提炼该片段中的关键事实、核心术语、方法、风险点和结论。"
        "保留专业名词，不要编造。输出简洁要点，不要输出 JSON。"
        if language == "zh"
        else "You are processing one sliding-window segment of a long document. Extract the key facts, domain terms, methods, risks, and conclusions. "
        "Preserve technical terms, do not invent details, and return concise bullet points instead of JSON."
    )
    window_label = f"{window_index}/{total_windows}"
    return f"{instruction}\n\nTitle: {title}\nWindow: {window_label}\n\nText:\n{text}"


def build_recursive_summary_prompt(
    *,
    title: str,
    summaries: list[str],
    language: str,
    level: int,
) -> str:
    instruction = (
        "下面是长文档多个窗口的中间摘要。请将它们压缩成更短的综合摘要，保留贯穿全文的术语、关键结论、方法和风险点。"
        "删除重复内容，但不要丢掉重要限定条件。输出简洁要点，不要输出 JSON。"
        if language == "zh"
        else "Below are intermediate summaries from multiple long-document windows. Compress them into a shorter integrated summary while preserving recurring terms, key conclusions, methods, and risks. "
        "Remove repetition without losing important caveats. Return concise bullet points instead of JSON."
    )
    summary_block = "\n\n".join(f"[Summary {index}]\n{item}" for index, item in enumerate(summaries, start=1))
    return f"{instruction}\n\nTitle: {title}\nSummary level: {level}\n\nSummaries:\n{summary_block}"


def build_compare_prompt(
    analyses: list[SingleDocAnalysis],
    language: str,
    citations: list[ChunkCitation],
    instruction_override: str | None = None,
) -> str:
    instruction = instruction_override.strip() if instruction_override else (
        DEFAULT_COMPARE_REPORT_PROMPT_ZH if language == "zh" else DEFAULT_COMPARE_REPORT_PROMPT_EN
    )
    analyses_block = "\n\n".join(
        json.dumps(
            {
                "title": item.title,
                "language": item.language,
                "summary": item.summary,
                "sentiment": item.sentiment,
                "keywords": item.keywords,
                "main_topics": item.main_topics,
                "risk_points": item.risk_points,
            },
            ensure_ascii=False,
            indent=2,
        )
        for item in analyses
    )
    citations_block = "\n".join(
        f"- {item.file_name} {item.page_label or item.section_label or ''}: {item.quote}"
        for item in citations
    )
    return f"{instruction}\n\nAnalyses:\n{analyses_block}\n\nCandidate citations:\n{citations_block}"


def build_data_extraction_prompt(
    *,
    title: str,
    fields: list[ExtractionFieldSpec],
    chunks: list[SourceDocument],
    language: str,
    instruction_override: str | None = None,
) -> str:
    """Build the prompt used to extract user-defined fields from selected chunks."""

    instruction = instruction_override.strip() if instruction_override else (
        DEFAULT_DATA_EXTRACTION_PROMPT_ZH if language == "zh" else DEFAULT_DATA_EXTRACTION_PROMPT_EN
    )
    field_lines = []
    for index, field in enumerate(fields, start=1):
        suffix = []
        if field.instruction.strip():
            suffix.append(f"instruction={field.instruction.strip()}")
        if field.expected_unit.strip():
            suffix.append(f"expected_unit={field.expected_unit.strip()}")
        aliases = _field_alias_terms(field)
        if aliases:
            suffix.append(f"aliases={', '.join(aliases[:6])}")
        extra = f" ({'; '.join(suffix)})" if suffix else ""
        field_lines.append(f"{index}. {field.name}{extra}")
    chunk_blocks = []
    for chunk in chunks:
        metadata = chunk.metadata
        locator = metadata.get("page_label") or metadata.get("section_label") or metadata.get("section") or ""
        chunk_blocks.append(
            f"[chunk_id={metadata.get('chunk_id')}] file={metadata.get('file_name')} locator={locator}\n{chunk.page_content}"
        )
    fields_heading = "目标字段" if language == "zh" else "Target fields"
    chunks_heading = "候选证据切片" if language == "zh" else "Candidate evidence chunks"
    title_heading = "标题" if language == "zh" else "Title"
    return (
        f"{instruction}\n\n"
        + (
            "如果目标字段是中文而证据是英文，请优先参考 aliases 进行匹配；返回的 value 必须能在证据原文中找到，不要重复拼接单位。\n\n"
            if language == "zh"
            else "If a target field is written in Chinese while the evidence is in English, use aliases first. The returned value must be grounded in the original evidence text and must not duplicate the unit.\n\n"
        )
        + f"{title_heading}: {title}\n\n"
        f"{fields_heading}:\n" + "\n".join(field_lines) + "\n\n"
        f"{chunks_heading}:\n" + "\n\n".join(chunk_blocks)
    )


def _field_alias_terms(field: ExtractionFieldSpec) -> list[str]:
    """Expose common English aliases for bilingual field extraction prompts."""

    text = " ".join(
        part.strip().lower()
        for part in [field.name, field.instruction, field.expected_unit]
        if part and part.strip()
    )
    aliases: list[str] = []
    mapping = {
        "reactant": ["reactant", "reactants", "substrate", "feedstock", "plastic feedstock"],
        "product": ["product", "products", "generated product", "value-added chemicals", "liquid product"],
        "hydrogen_rate": [
            "hydrogen production rate",
            "hydrogen evolution rate",
            "h2 production rate",
            "h2 evolution rate",
            "hydrogen generation rate",
        ],
        "temperature": ["reaction temperature", "temperature", "operating temperature"],
        "light_intensity": ["light intensity", "irradiance", "AM 1.5", "1 sun", "solar simulator", "visible light"],
        "onset_potential": ["onset potential", "onset voltage", "starting potential", "applied bias", "potential"],
    }
    if any(token in text for token in {"反应物", "reactant", "substrate", "feedstock"}):
        aliases.extend(mapping["reactant"])
    if any(token in text for token in {"产物", "product", "products"}):
        aliases.extend(mapping["product"])
    if any(token in text for token in {"氢气产生速率", "产氢速率", "hydrogen", "h2", "evolution rate", "production rate"}):
        aliases.extend(mapping["hydrogen_rate"])
    if any(token in text for token in {"反应温度", "温度", "temperature"}):
        aliases.extend(mapping["temperature"])
    if any(token in text for token in {"反应光强", "光强", "光照", "light", "irradiance", "am"}):
        aliases.extend(mapping["light_intensity"])
    if any(token in text for token in {"起始电位", "onset potential", "potential", "voltage", "bias"}):
        aliases.extend(mapping["onset_potential"])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def build_table_summary_prompt(
    *,
    title: str,
    extraction_json: str,
    language: str,
    instruction_override: str | None = None,
) -> str:
    """Build the prompt for summarizing extracted field tables into short insights."""

    instruction = instruction_override.strip() if instruction_override else (
        DEFAULT_TABLE_SUMMARY_PROMPT_ZH if language == "zh" else DEFAULT_TABLE_SUMMARY_PROMPT_EN
    )
    payload_heading = "抽取结果" if language == "zh" else "Extraction results"
    title_heading = "标题" if language == "zh" else "Title"
    return f"{instruction}\n\n{title_heading}: {title}\n\n{payload_heading}:\n{extraction_json}"


def build_chat_llm(config: "AppConfig", streaming: bool = False):
    """Build the chat model client used by both QA and analysis flows."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError(
            "langchain-openai is required. Install requirements.txt before using the chat model."
        ) from exc

    return ChatOpenAI(
        api_key=config.resolved_chat_api_key,
        base_url=config.resolved_chat_base_url,
        model=config.chat_model,
        streaming=streaming,
        timeout=(
            config.chat_stream_request_timeout_seconds
            if streaming
            else config.chat_timeout_seconds
        ),
        max_retries=0,
    )


def build_embedding_model(config: "AppConfig"):
    """Build the embedding model used by the vector store backend."""

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError(
            "langchain-openai is required. Install requirements.txt before using embeddings."
        ) from exc

    backend = OpenAIEmbeddings(
        api_key=config.resolved_embedding_api_key,
        base_url=config.resolved_embedding_base_url,
        model=config.embedding_model,
        chunk_size=max(1, int(getattr(config, "vector_upsert_batch_size", 8))),
        max_retries=0,
    )
    return LoggedEmbeddingModel(config=config, backend=backend)


async def _acquire_chat_slot(
    config: "AppConfig",
    status_callback: StatusCallback | None,
    *,
    transport: str,
    language: Literal["zh", "en"],
) -> asyncio.Semaphore:
    semaphore = _get_chat_semaphore(config)
    warned = False
    wait_started = time.perf_counter()
    last_notice_seconds = 0.0
    while True:
        check_task_cancelled()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.25)
            if warned:
                waited_seconds = max(0.0, time.perf_counter() - wait_started)
                _append_api_log(
                    config,
                    {
                        "event": "queue_resume",
                        "transport": transport,
                        "model": config.chat_model,
                        "base_url": config.resolved_chat_base_url,
                        "wait_seconds": round(waited_seconds, 3),
                    },
                )
                await _emit_status(
                    status_callback,
                    (
                        f"已轮到当前请求，正在继续处理。累计排队约 {int(round(waited_seconds))} 秒。"
                        if language == "zh"
                        else f"The request left the queue and is now running after about {int(round(waited_seconds))} seconds."
                    ),
                )
            return semaphore
        except TimeoutError:
            waited_seconds = max(0.0, time.perf_counter() - wait_started)
            should_notify = (not warned) or (waited_seconds - last_notice_seconds >= 5.0)
            if should_notify:
                warned = True
                last_notice_seconds = waited_seconds
                await _emit_status(
                    status_callback,
                    (
                        f"当前模型请求较多，正在等待服务端队列，已等待约 {int(round(waited_seconds))} 秒。"
                        if language == "zh"
                        else f"The model is busy, and this request has waited in the provider queue for about {int(round(waited_seconds))} seconds."
                    ),
                )
            if not warned or should_notify:
                _append_api_log(
                    config,
                    {
                        "event": "queue_wait",
                        "transport": transport,
                        "model": config.chat_model,
                        "base_url": config.resolved_chat_base_url,
                        "wait_seconds": round(waited_seconds, 3),
                    },
                )


def _acquire_embedding_slot(config: "AppConfig", *, transport: str) -> threading.Semaphore:
    semaphore = _get_embedding_semaphore(config)
    warned = False
    while True:
        check_task_cancelled()
        if semaphore.acquire(timeout=0.25):
            if warned:
                _append_api_log(
                    config,
                    {
                        "event": "queue_resume",
                        "transport": transport,
                        "model": config.embedding_model,
                        "base_url": config.resolved_embedding_base_url,
                    },
                )
            return semaphore
        if not warned:
            warned = True
            _append_api_log(
                config,
                {
                    "event": "queue_wait",
                    "transport": transport,
                    "model": config.embedding_model,
                    "base_url": config.resolved_embedding_base_url,
                },
            )


def _get_chat_semaphore(config: "AppConfig") -> asyncio.Semaphore:
    key = f"{config.resolved_chat_base_url}|{config.chat_model}"
    with _CHAT_QUEUE_LOCK:
        semaphore = _CHAT_QUEUE_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(1, int(config.chat_provider_concurrency)))
            _CHAT_QUEUE_SEMAPHORES[key] = semaphore
        return semaphore


def _get_embedding_semaphore(config: "AppConfig") -> threading.Semaphore:
    key = f"{config.resolved_embedding_base_url}|{config.embedding_model}"
    with _EMBED_QUEUE_LOCK:
        semaphore = _EMBED_QUEUE_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.Semaphore(max(1, int(config.embedding_provider_concurrency)))
            _EMBED_QUEUE_SEMAPHORES[key] = semaphore
        return semaphore


async def _await_with_task_control(
    awaitable: Awaitable[Any],
    *,
    poll_interval_seconds: float = 0.2,
) -> Any:
    """Await one async operation while polling stop signals for faster cancellation."""

    inner_task = asyncio.ensure_future(awaitable)
    interval = max(0.05, float(poll_interval_seconds))
    try:
        while True:
            check_task_cancelled()
            done, _ = await asyncio.wait({inner_task}, timeout=interval)
            if done:
                return inner_task.result()
    except (OperationCancelledError, asyncio.CancelledError):
        inner_task.cancel()
        with suppress(BaseException):
            await asyncio.wait_for(inner_task, timeout=1.0)
        raise


async def invoke_chat_text(
    config: "AppConfig",
    prompt: str,
    language: Literal["zh", "en"],
    status_callback: StatusCallback | None = None,
) -> str:
    """Run one non-streaming chat request with retries and normalized errors."""

    request_id = uuid4().hex
    attempts = max(1, int(config.api_retry_attempts))
    last_error: ProviderRequestError | None = None
    general_retries = 0
    rate_limit_retries = 0
    retry_forever = bool(config.rate_limit_retry_forever)
    _append_api_log(
        config,
        {
            "event": "request",
            "request_id": request_id,
            "transport": "chat_non_stream",
            "model": config.chat_model,
            "base_url": config.resolved_chat_base_url,
            "prompt": prompt,
            "prompt_tokens_estimate": estimate_token_count(prompt, config.chat_model),
            "language": language,
        },
    )
    while True:
        check_task_cancelled()
        queue_slot = await _acquire_chat_slot(
            config,
            status_callback,
            transport="chat_non_stream",
            language=language,
        )
        slot_released = False
        try:
            llm = build_chat_llm(config, streaming=False)
            started = time.perf_counter()
            async with asyncio.timeout(config.chat_timeout_seconds):
                response = await _await_with_task_control(llm.ainvoke(prompt))
            content = _message_content(response)
            usage = _extract_token_usage(response)
            _append_api_log(
                config,
                {
                    "event": "response",
                    "request_id": request_id,
                    "transport": "chat_non_stream",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "content": content,
                    "prompt_tokens_estimate": estimate_token_count(prompt, config.chat_model),
                    "completion_tokens_estimate": estimate_token_count(content, config.chat_model),
                    "provider_prompt_tokens": usage.get("prompt_tokens"),
                    "provider_completion_tokens": usage.get("completion_tokens"),
                    "provider_total_tokens": usage.get("total_tokens"),
                },
            )
            return content
        except OperationCancelledError:
            _append_api_log(
                config,
                {
                    "event": "cancelled",
                    "request_id": request_id,
                    "transport": "chat_non_stream",
                },
            )
            raise
        except Exception as exc:  # pragma: no cover - provider behavior.
            mapped = classify_provider_exception(exc, language)
            last_error = mapped
            _append_api_log(
                config,
                {
                    "event": "error",
                    "request_id": request_id,
                    "transport": "chat_non_stream",
                    "category": mapped.code,
                    "detail": mapped.detail,
                },
            )
            if mapped.code == "rate_limit":
                if not retry_forever and rate_limit_retries >= max(0, int(config.rate_limit_retry_attempts)):
                    raise _finalize_retry_error(mapped, language, config)
                rate_limit_retries += 1
                wait_seconds = max(1, int(config.rate_limit_retry_delay_seconds))
                queue_slot.release()
                slot_released = True
                await _emit_status(
                    status_callback,
                    _rate_limit_notice(
                        language,
                        wait_seconds,
                        rate_limit_retries,
                        None if retry_forever else max(0, int(config.rate_limit_retry_attempts)),
                    ),
                )
                _append_api_log(
                    config,
                    {
                        "event": "retry_wait",
                        "request_id": request_id,
                        "transport": "chat_non_stream",
                        "category": mapped.code,
                        "wait_seconds": wait_seconds,
                        "retry_index": rate_limit_retries,
                    },
                )
                await sleep_with_task_control(wait_seconds)
                continue
            if not mapped.retryable or general_retries >= attempts - 1:
                raise mapped
            general_retries += 1
            wait_seconds = _retry_delay_seconds(config, general_retries)
            queue_slot.release()
            slot_released = True
            await _emit_status(
                status_callback,
                _generic_retry_notice(language, mapped, general_retries, attempts - 1, wait_seconds),
            )
            _append_api_log(
                config,
                {
                    "event": "retry_wait",
                    "request_id": request_id,
                    "transport": "chat_non_stream",
                    "category": mapped.code,
                    "wait_seconds": wait_seconds,
                    "retry_index": general_retries,
                },
            )
            await sleep_with_task_control(wait_seconds)
        finally:
            if not slot_released:
                queue_slot.release()
    if last_error is not None:
        raise last_error
    raise ProviderRequestError(
        "模型服务暂时不可用，请稍后重试。" if language == "zh" else "The model service is temporarily unavailable. Please try again later.",
        code="provider",
        retryable=False,
    )


async def stream_chat_text(
    config: "AppConfig",
    prompt: str,
    language: Literal["zh", "en"],
    request_id: str | None = None,
    status_callback: StatusCallback | None = None,
) -> AsyncIterator[str]:
    """Run one streaming chat request and log the full prompt and collected response."""

    stream_request_id = request_id or uuid4().hex
    collected_chunks: list[str] = []
    _append_api_log(
        config,
        {
            "event": "request",
            "request_id": stream_request_id,
            "transport": "chat_stream",
            "model": config.chat_model,
            "base_url": config.resolved_chat_base_url,
            "prompt": prompt,
            "prompt_tokens_estimate": estimate_token_count(prompt, config.chat_model),
            "language": language,
        },
    )
    started = time.perf_counter()
    queue_slot = await _acquire_chat_slot(
        config,
        status_callback,
        transport="chat_stream",
        language=language,
    )
    usage: dict[str, int] = {}
    try:
        llm = build_chat_llm(config, streaming=True)
        async for chunk in _iterate_stream_with_timeouts(
            llm.astream(prompt),
            first_token_timeout_seconds=config.chat_stream_first_token_timeout_seconds,
            idle_timeout_seconds=config.chat_stream_idle_timeout_seconds,
        ):
            text = _message_content(chunk)
            chunk_usage = _extract_token_usage(chunk)
            if chunk_usage:
                usage = chunk_usage
            if text:
                collected_chunks.append(text)
                yield text
            check_task_cancelled()
        _append_api_log(
            config,
            {
                "event": "response",
                "request_id": stream_request_id,
                "transport": "chat_stream",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "content": "".join(collected_chunks),
                "prompt_tokens_estimate": estimate_token_count(prompt, config.chat_model),
                "completion_tokens_estimate": estimate_token_count("".join(collected_chunks), config.chat_model),
                "provider_prompt_tokens": usage.get("prompt_tokens"),
                "provider_completion_tokens": usage.get("completion_tokens"),
                "provider_total_tokens": usage.get("total_tokens"),
            },
        )
    except OperationCancelledError:
        _append_api_log(
            config,
            {
                "event": "cancelled",
                "request_id": stream_request_id,
                "transport": "chat_stream",
                "partial_content": "".join(collected_chunks),
            },
        )
        raise
    except Exception as exc:  # pragma: no cover - provider behavior.
        mapped = classify_provider_exception(exc, language)
        _append_api_log(
            config,
            {
                "event": "error",
                "request_id": stream_request_id,
                "transport": "chat_stream",
                "category": mapped.code,
                "detail": mapped.detail,
                "partial_content": "".join(collected_chunks),
            },
        )
        raise mapped
    finally:
        queue_slot.release()


async def run_chat_healthcheck(config: "AppConfig", timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Check whether both non-streaming and streaming chat calls work."""

    if not config.has_chat_model_credentials:
        return {
            "ok": False,
            "message": "Missing OPENAI_CHAT_API_KEY / OPENAI_CHAT_BASE_URL / OPENAI_CHAT_MODEL.",
            "base_url": config.resolved_chat_base_url,
            "chat_model": config.chat_model,
        }

    non_stream = await _test_non_streaming(config, timeout_seconds=timeout_seconds)
    stream = await _test_streaming(config, timeout_seconds=timeout_seconds)
    return {
        "ok": bool(non_stream["ok"] and stream["ok"]),
        "base_url": config.resolved_chat_base_url,
        "chat_model": config.chat_model,
        "non_stream": non_stream,
        "stream": stream,
    }


async def _test_non_streaming(config: "AppConfig", timeout_seconds: float) -> dict[str, Any]:
    prompt = "Reply with exactly OK."
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(invoke_chat_text(config, prompt, "en"), timeout=timeout_seconds)
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "elapsed_seconds": round(elapsed, 2),
            "preview": response[:120],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _test_streaming(config: "AppConfig", timeout_seconds: float) -> dict[str, Any]:
    prompt = "Reply with exactly OK."
    started = time.perf_counter()
    try:
        text = await asyncio.wait_for(_collect_stream(config, prompt), timeout=timeout_seconds)
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "elapsed_seconds": round(elapsed, 2),
            "preview": text[:120],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _collect_stream(config: "AppConfig", prompt: str) -> str:
    chunks: list[str] = []
    async for text in stream_chat_text(config, prompt, "en"):
        if text:
            chunks.append(text)
    return "".join(chunks)


def _extract_content(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


async def _iterate_stream_with_timeouts(
    stream: AsyncIterator[Any],
    *,
    first_token_timeout_seconds: int,
    idle_timeout_seconds: int,
) -> AsyncIterator[Any]:
    """Stream tokens with separate first-token and idle timeout controls."""

    iterator = stream.__aiter__()
    is_first_chunk = True
    while True:
        timeout_seconds = (
            max(1, int(first_token_timeout_seconds))
            if is_first_chunk
            else max(1, int(idle_timeout_seconds))
        )
        try:
            chunk = await _await_with_task_control(
                asyncio.wait_for(iterator.__anext__(), timeout=timeout_seconds)
            )
        except StopAsyncIteration:
            return
        is_first_chunk = False
        yield chunk


def _retry_delay_seconds(config: "AppConfig", attempt: int) -> float:
    minimum = max(0, int(config.api_retry_backoff_min_seconds))
    maximum = max(minimum, int(config.api_retry_backoff_max_seconds))
    return float(min(maximum, minimum * (2 ** max(0, attempt - 1))))


def _message_content(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _extract_token_usage(value: Any) -> dict[str, int]:
    """Extract normalized token usage fields from provider response objects."""

    candidates: list[dict[str, Any]] = []
    usage_metadata = getattr(value, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        candidates.append(usage_metadata)
    response_metadata = getattr(value, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            payload = response_metadata.get(key)
            if isinstance(payload, dict):
                candidates.append(payload)
    additional_kwargs = getattr(value, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            payload = additional_kwargs.get(key)
            if isinstance(payload, dict):
                candidates.append(payload)

    prompt_tokens = _extract_token_value(candidates, "prompt_tokens", "input_tokens", "prompt_token_count")
    completion_tokens = _extract_token_value(candidates, "completion_tokens", "output_tokens", "completion_token_count")
    total_tokens = _extract_token_value(candidates, "total_tokens", "total_token_count")

    usage: dict[str, int] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return usage


def _extract_token_value(candidates: list[dict[str, Any]], *keys: str) -> int | None:
    for payload in candidates:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
    return None


class LoggedEmbeddingModel:
    """Wrap the embedding client with local request logging and explicit retry rules."""

    def __init__(self, config: "AppConfig", backend: Any) -> None:
        self.config = config
        self.backend = backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run_embedding_call("embedding_documents", texts=texts)

    def embed_query(self, text: str) -> list[float]:
        result = self._run_embedding_call("embedding_query", texts=[text])
        return result[0] if result else []

    def _run_embedding_call(self, transport: str, *, texts: list[str]) -> list[list[float]]:
        request_id = uuid4().hex
        attempts = max(1, int(self.config.api_retry_attempts))
        general_retries = 0
        rate_limit_retries = 0
        retry_forever = bool(self.config.rate_limit_retry_forever)
        _append_api_log(
            self.config,
            {
                "event": "request",
                "request_id": request_id,
                "transport": transport,
                "model": self.config.embedding_model,
                "base_url": self.config.resolved_embedding_base_url,
                "texts": texts,
            },
        )
        while True:
            check_task_cancelled()
            queue_slot = _acquire_embedding_slot(self.config, transport=transport)
            try:
                started = time.perf_counter()
                if transport == "embedding_query":
                    response = [self.backend.embed_query(texts[0])]
                else:
                    response = self.backend.embed_documents(texts)
                _append_api_log(
                    self.config,
                    {
                        "event": "response",
                        "request_id": request_id,
                        "transport": transport,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "vectors": response,
                    },
                )
                return response
            except OperationCancelledError:
                _append_api_log(
                    self.config,
                    {
                        "event": "cancelled",
                        "request_id": request_id,
                        "transport": transport,
                    },
                )
                raise
            except Exception as exc:  # pragma: no cover - provider behavior.
                mapped = classify_provider_exception(exc, "zh")
                _append_api_log(
                    self.config,
                    {
                        "event": "error",
                        "request_id": request_id,
                        "transport": transport,
                        "category": mapped.code,
                        "detail": mapped.detail,
                    },
                )
                if mapped.code == "rate_limit":
                    if not retry_forever and rate_limit_retries >= max(0, int(self.config.rate_limit_retry_attempts)):
                        raise _finalize_retry_error(mapped, "zh", self.config)
                    rate_limit_retries += 1
                    wait_seconds = max(1, int(self.config.rate_limit_retry_delay_seconds))
                    _append_api_log(
                        self.config,
                        {
                            "event": "retry_wait",
                            "request_id": request_id,
                            "transport": transport,
                            "category": mapped.code,
                            "wait_seconds": wait_seconds,
                            "retry_index": rate_limit_retries,
                        },
                    )
                    sleep_with_task_control_blocking(wait_seconds)
                    continue
                if not mapped.retryable or general_retries >= attempts - 1:
                    raise mapped
                general_retries += 1
                wait_seconds = _retry_delay_seconds(self.config, general_retries)
                _append_api_log(
                    self.config,
                    {
                        "event": "retry_wait",
                        "request_id": request_id,
                        "transport": transport,
                        "category": mapped.code,
                        "wait_seconds": wait_seconds,
                        "retry_index": general_retries,
                    },
                )
                sleep_with_task_control_blocking(wait_seconds)
            finally:
                queue_slot.release()


async def _emit_status(callback: StatusCallback | None, message: str) -> None:
    """Forward retry or lifecycle messages to an optional UI callback."""

    if callback is None:
        return
    result = callback(message)
    if asyncio.iscoroutine(result):
        await result


def _rate_limit_notice(
    language: Literal["zh", "en"],
    wait_seconds: int,
    retry_index: int,
    retry_total: int | None,
) -> str:
    if language == "zh":
        if retry_total is None:
            return f"触发频率限制，{wait_seconds} 秒后继续自动重试（第 {retry_index} 次）..."
        return f"触发频率限制，{wait_seconds} 秒后自动重试（第 {retry_index}/{retry_total} 次）..."
    if retry_total is None:
        return f"Rate limited. Retrying again in {wait_seconds} seconds (attempt {retry_index})..."
    return f"Rate limited. Retrying in {wait_seconds} seconds ({retry_index}/{retry_total})..."


def _generic_retry_notice(
    language: Literal["zh", "en"],
    error: ProviderRequestError,
    retry_index: int,
    retry_total: int,
    wait_seconds: float,
) -> str:
    if language == "zh":
        return f"{error.user_message} 系统将在 {int(wait_seconds)} 秒后自动重试（第 {retry_index}/{retry_total} 次）..."
    return f"{error.user_message} Retrying in {int(wait_seconds)} seconds ({retry_index}/{retry_total})..."


def _finalize_retry_error(
    error: ProviderRequestError,
    language: Literal["zh", "en"],
    config: "AppConfig",
) -> ProviderRequestError:
    if error.code != "rate_limit":
        return error
    retry_total = max(0, int(config.rate_limit_retry_attempts))
    wait_seconds = max(1, int(config.rate_limit_retry_delay_seconds))
    message = (
        f"已触发频率限制，系统每 {wait_seconds} 秒自动重试一次，连续重试 {retry_total} 次仍失败，请稍后再试。"
        if language == "zh"
        else f"The request remained rate-limited after {retry_total} retries spaced {wait_seconds} seconds apart. Please try again later."
    )
    return ProviderRequestError(
        message,
        code=error.code,
        retryable=False,
        detail=error.detail,
    )


def _append_api_log(config: "AppConfig", payload: dict[str, Any]) -> None:
    """Append one JSONL API trace entry to the local log file."""

    path = Path(config.active_api_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_make_json_safe(safe_payload), ensure_ascii=False) + "\n")


def _make_json_safe(value: Any) -> Any:
    """Convert nested provider payloads to JSON-serializable Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _make_json_safe(value.model_dump())
    return str(value)

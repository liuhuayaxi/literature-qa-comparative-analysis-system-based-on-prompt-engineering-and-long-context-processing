"""Notebook UI for knowledge-base management, chat, and analysis workflows."""

from __future__ import annotations

import asyncio
import re
import shutil
import traceback
from datetime import datetime
from math import ceil
from pathlib import Path

from .analysis_engine import (
    BatchComparisonService,
    SingleDocumentAnalysisService,
    parse_extraction_field_specs,
)
from .config import AppConfig
from .errors import AppServiceError, OperationCancelledError, OperationPausedError
from .knowledge_ingestion import DocumentIndexer
from .knowledge_base_manager import KnowledgeBaseManager
from .memory_store import SessionMemoryStore, SQLiteMemoryStore
from .retrieval_engine import RAGChatService, VectorStoreService
from .migration_bundle import (
    export_migration_bundle,
    import_migration_bundle,
    inspect_migration_bundle,
)
from .models import ChatTurn, DocumentRecord
from .app_utils import (
    TaskControl,
    load_json_mapping,
    new_session_id,
    sanitize_storage_key,
    save_json_mapping,
    task_control_context,
    update_json_config_file,
)
from .llm_tools import run_chat_healthcheck


def build_app(config: AppConfig):
    """Build the ipywidgets application used in the main notebook."""
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, Javascript, display
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("ipywidgets is required to render the notebook UI.") from exc

    if not config.has_model_credentials:
        return _build_recovery_app(widgets, config)

    vector_store = VectorStoreService(config)
    session_store = SessionMemoryStore()
    sqlite_store = SQLiteMemoryStore(config.db_path)
    indexer = DocumentIndexer(config, vector_store)
    manager = KnowledgeBaseManager(config, vector_store, indexer)
    rag_service = RAGChatService(config, vector_store, session_store, sqlite_store)
    single_doc_service = SingleDocumentAnalysisService(config, vector_store)
    batch_service = BatchComparisonService(config, vector_store, single_doc_service)
    field_templates = load_json_mapping(config.field_template_path)
    manage_task_state = {"task": None, "control": None, "label": ""}
    chat_task_state = {"task": None, "control": None, "label": ""}
    analysis_task_state = {"task": None, "control": None, "label": ""}

    manage_kb_dropdown = widgets.Dropdown(
        options=[("请选择知识库", "")],
        value="",
        description="知识库",
        layout=widgets.Layout(width="320px"),
    )
    refresh_kb_button = widgets.Button(description="刷新列表")
    create_kb_input = widgets.Text(
        value="",
        description="新建知识库",
        placeholder="输入名称后点击新建知识库",
        layout=widgets.Layout(width="320px"),
    )
    create_kb_button = widgets.Button(description="新建知识库", button_style="primary")
    target_kb_dropdown = widgets.Dropdown(
        options=[("请选择上传目标", "")],
        value="",
        description="上传目标",
        layout=widgets.Layout(width="360px"),
    )
    rename_kb_input = widgets.Text(
        value="",
        description="重命名知识库",
        placeholder="输入当前知识库的新名称",
        layout=widgets.Layout(width="320px"),
    )
    rename_kb_button = widgets.Button(description="重命名")

    source_type_dropdown = widgets.Dropdown(
        options=[("讲义", "lecture"), ("作业", "assignment"), ("论文", "paper")],
        value="lecture",
        description="文件类型",
    )
    rebuild_checkbox = widgets.Checkbox(value=False, description="重建索引")
    chunk_size_input = widgets.BoundedIntText(
        value=config.chunk_size,
        min=200,
        max=4000,
        step=50,
        description="切片大小",
    )
    chunk_overlap_input = widgets.BoundedIntText(
        value=config.chunk_overlap,
        min=0,
        max=1000,
        step=20,
        description="重叠大小",
    )
    merge_small_chunks_checkbox = widgets.Checkbox(value=config.merge_small_chunks, description="智能合并小块")
    min_chunk_size_input = widgets.BoundedIntText(
        value=config.min_chunk_size,
        min=80,
        max=4000,
        step=20,
        description="最小块长",
    )
    upload_widget = widgets.FileUpload(accept=".pdf,.md,.txt,.docx", multiple=True, description="选择文件")
    upload_status = widgets.HTML("<i>还没有选中文件。提示：在系统文件选择框中按住 Cmd/Ctrl 或 Shift 可以多选；如果按钮上显示已选文件，但这里仍显示未选择，说明 JupyterHub 没把文件同步到 kernel，请改用下方“从工作区导入”。</i>")
    ingest_button = widgets.Button(description="导入并建索引", button_style="primary", disabled=True)
    save_index_settings_button = widgets.Button(description="保存索引参数到配置文件", button_style="info")
    index_settings_status = widgets.HTML(value="")
    import_dir_input = widgets.Text(
        value="imports",
        description="工作区目录",
        placeholder="扫描这个目录中的现有文件",
        layout=widgets.Layout(width="420px"),
    )
    scan_workspace_button = widgets.Button(description="扫描目录")
    workspace_file_selector = widgets.SelectMultiple(
        options=[],
        description="工作区文件",
        layout=widgets.Layout(width="100%", height="180px"),
    )
    import_workspace_button = widgets.Button(description="从工作区导入并建索引", button_style="info")
    migration_export_name_input = widgets.Text(
        value="",
        description="导出文件名",
        placeholder="留空则自动生成 zip 文件名",
        layout=widgets.Layout(width="420px"),
    )
    export_migration_button = widgets.Button(description="导出迁移包", button_style="success")
    migration_bundle_upload = widgets.FileUpload(accept=".zip", multiple=False, description="上传迁移包")
    migration_bundle_path_input = widgets.Text(
        value="",
        description="工作区包路径",
        placeholder="例如 release/migration_bundles/migration_bundle_xxx.zip",
        layout=widgets.Layout(width="520px"),
    )
    import_migration_button = widgets.Button(description="导入迁移包", button_style="warning")
    migration_confirm_checkbox = widgets.Checkbox(
        value=False,
        description="我已确认导入会覆盖当前配置和数据文件（日志不会改动）",
    )
    migration_status_html = widgets.HTML(value="")

    refresh_manage_files_button = widgets.Button(description="刷新文件列表")
    delete_files_button = widgets.Button(description="删除所选文件", button_style="danger")
    vectorize_button = widgets.Button(description="向量化处理", button_style="success")
    manage_stop_button = widgets.Button(description="停止当前操作", button_style="warning", disabled=True)
    file_action_confirm_checkbox = widgets.Checkbox(value=False, description="我已确认执行删除 / 移动 / 重建这类敏感操作")
    manage_select_all_button = widgets.Button(description="全选文件")
    manage_clear_button = widgets.Button(description="清空选择")
    manage_file_page_size_dropdown = widgets.Dropdown(
        options=[("20 / 页", 20), ("40 / 页", 40), ("80 / 页", 80)],
        value=40,
        description="文件分页",
        layout=widgets.Layout(width="180px"),
    )
    manage_file_prev_button = widgets.Button(description="上一页")
    manage_file_next_button = widgets.Button(description="下一页")
    manage_file_page_status_html = widgets.HTML(value="<i>第 1 / 1 页，共 0 个文件（已选 0）</i>")
    manage_doc_selector = widgets.SelectMultiple(
        options=[],
        description="文件列表",
        layout=widgets.Layout(width="100%", height="220px"),
    )
    rename_file_input = widgets.Text(
        value="",
        description="新文件名",
        placeholder="选中一个文件后重命名",
        layout=widgets.Layout(width="420px"),
    )
    rename_file_button = widgets.Button(description="修改文件名")
    view_chunks_button = widgets.Button(description="查看切片内容")
    move_target_kb_dropdown = widgets.Dropdown(
        options=[("请选择目标知识库", "")],
        value="",
        description="移动到",
        layout=widgets.Layout(width="320px"),
    )
    move_target_type_dropdown = widgets.Dropdown(
        options=[("讲义", "lecture"), ("作业", "assignment"), ("论文", "paper")],
        value="lecture",
        description="目标类型",
        layout=widgets.Layout(width="220px"),
    )
    move_files_button = widgets.Button(description="移动所选文件", button_style="info")
    manage_output = widgets.Output(
        layout={
            "width": "100%",
            "height": "260px",
            "max_height": "260px",
            "overflow": "auto",
        }
    )
    manage_output_panel = widgets.Box(
        [manage_output],
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="8px",
            height="260px",
            max_height="260px",
            overflow="auto",
        ),
    )
    manage_output_panel.add_class("assistant-log-panel")
    chunk_search_input = widgets.Text(
        value="",
        description="内容搜索",
        placeholder="按文件名或切片内容搜索",
        continuous_update=False,
        layout=widgets.Layout(width="420px"),
    )
    search_selected_only_checkbox = widgets.Checkbox(value=True, description="仅在已选文件中搜索")
    detail_page_size_dropdown = widgets.Dropdown(
        options=[("10 / 页", 10), ("20 / 页", 20), ("50 / 页", 50)],
        value=10,
        description="分页",
        layout=widgets.Layout(width="170px"),
    )
    detail_prev_button = widgets.Button(description="上一页")
    detail_next_button = widgets.Button(description="下一页")
    detail_page_status_html = widgets.HTML(value="<i>第 1 / 1 页</i>")
    chunk_search_button = widgets.Button(description="搜索")
    chunk_search_clear_button = widgets.Button(description="清空搜索")
    chunk_search_status = widgets.HTML(value="<i>请选择知识库后查看详情。</i>")
    manage_detail_hint_html = widgets.HTML(
        value=(
            "<div style='padding:8px 10px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;color:#5b6472;'>"
            "建议先在上面的文件列表中勾选需要查看的文件。未选择文件时，详情区默认先展示知识库概览，不立即加载所有切片正文。"
            "如果开启“仅在已选文件中搜索”，内容搜索也只会在上面选中的文件里执行。"
            "</div>"
        )
    )
    chunk_output = widgets.HTML(
        value=_manage_details_placeholder_html("请选择一个知识库后查看文件概览和切片详情。"),
        layout=widgets.Layout(border="1px solid #ddd", padding="8px", max_height="420px", overflow="auto"),
    )
    knowledge_base_status_html = widgets.HTML(value="")
    knowledge_base_change_html = widgets.HTML(value="")
    delete_kb_confirm_checkbox = widgets.Checkbox(value=False, description="我确认删除当前知识库")
    delete_kb_button = widgets.Button(description="删除当前知识库", button_style="danger", disabled=True)
    knowledge_help_html = widgets.HTML(
        value=(
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;'>"
            "<b>说明</b><br>"
            "颜色标识：<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:12px;font-weight:600;'>讲义</span> "
            "<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#fef3c7;color:#b45309;font-size:12px;font-weight:600;'>作业</span> "
            "<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#dcfce7;color:#15803d;font-size:12px;font-weight:600;'>论文</span><br>"
            "<b>重建索引</b>：会先清空当前知识库已有的向量索引，再按本次文件重新切片和向量化。<br>"
            "<b>删除当前知识库</b>：只删除当前选中的知识库目录及其向量索引，不影响其他知识库和会话数据库。"
            "</div>"
        )
    )
    file_type_legend_html = widgets.HTML(
        value=(
            "<div style='margin:4px 0 8px 0;color:#5b6472;'>"
            "文件颜色标识："
            "<span style='margin-left:8px;'>🟦 讲义</span>"
            "<span style='margin-left:12px;'>🟨 作业</span>"
            "<span style='margin-left:12px;'>🟩 论文</span>"
            "</div>"
        )
    )

    chat_kb_dropdown = widgets.Dropdown(
        options=[("请选择知识库", "")],
        value="",
        description="知识库",
        layout=widgets.Layout(width="320px"),
    )
    chat_refresh_files_button = widgets.Button(description="刷新文件")
    chat_doc_selector = widgets.SelectMultiple(
        options=[],
        description="问答文件",
        layout=widgets.Layout(width="100%", height="180px"),
    )
    chat_select_all_button = widgets.Button(description="全选文件")
    chat_clear_button = widgets.Button(description="清空选择")
    chat_scope_hint = widgets.HTML("<i>未选择文件时，默认检索整个知识库。</i>")
    session_selector = widgets.Select(
        options=[],
        description="会话",
        layout=widgets.Layout(width="100%", height="360px"),
    )
    session_search_input = widgets.Text(
        value="",
        description="搜索会话",
        placeholder="按标题筛选",
        layout=widgets.Layout(width="100%"),
    )
    refresh_sessions_button = widgets.Button(description="刷新会话")
    delete_session_button = widgets.Button(description="删除选中会话", button_style="danger")
    stop_chat_button = widgets.Button(description="停止当前回答", button_style="warning", disabled=True)
    delete_session_confirm_checkbox = widgets.Checkbox(value=False, description="我已确认删除这段会话")
    rename_session_input = widgets.Text(
        value="",
        description="会话标题",
        placeholder="修改当前会话标题",
        layout=widgets.Layout(width="100%"),
    )
    rename_session_button = widgets.Button(description="重命名会话")
    session_summary_html = widgets.HTML(value="<i>当前没有会话。</i>")
    session_id_input = widgets.Text(
        value="",
        description="内部会话ID",
        placeholder="内部使用",
        layout=widgets.Layout(display="none"),
    )
    new_session_button = widgets.Button(description="新建会话")
    memory_mode_toggle = widgets.ToggleButtons(
        options=[("会话记忆", "session"), ("持久记忆", "persistent")],
        value=config.default_memory_mode,
        description="记忆模式",
    )
    language_toggle = widgets.ToggleButtons(
        options=[("自动", "auto"), ("中文", "zh"), ("英文", "en")],
        value=config.default_language,
        description="语言",
    )
    streaming_mode_toggle = widgets.ToggleButtons(
        options=[("流式输出", "stream"), ("仅普通回答", "non_stream")],
        value=config.default_streaming_mode,
        description="回答方式",
    )
    question_area = widgets.Textarea(
        placeholder="请输入问题 / Ask a question",
        layout=widgets.Layout(width="100%", height="120px"),
    )
    retrieval_top_k_input = widgets.BoundedIntText(
        value=config.retrieval_top_k,
        min=1,
        max=20,
        step=1,
        description="返回数量",
        layout=widgets.Layout(width="220px"),
    )
    retrieval_fetch_k_input = widgets.BoundedIntText(
        value=config.retrieval_fetch_k,
        min=1,
        max=60,
        step=1,
        description="候选数量",
        layout=widgets.Layout(width="220px"),
    )
    model_context_window_input = widgets.BoundedIntText(
        value=config.model_context_window,
        min=2048,
        max=200000,
        step=1024,
        description="上下文上限",
        layout=widgets.Layout(width="220px"),
    )
    answer_token_reserve_input = widgets.BoundedIntText(
        value=config.answer_token_reserve,
        min=256,
        max=64000,
        step=256,
        description="回答预留",
        layout=widgets.Layout(width="220px"),
    )
    long_context_window_tokens_input = widgets.BoundedIntText(
        value=config.long_context_window_tokens,
        min=128,
        max=20000,
        step=64,
        description="滑窗大小",
        layout=widgets.Layout(width="220px"),
    )
    long_context_window_overlap_tokens_input = widgets.BoundedIntText(
        value=config.long_context_window_overlap_tokens,
        min=0,
        max=4000,
        step=32,
        description="滑窗重叠",
        layout=widgets.Layout(width="220px"),
    )
    recursive_summary_target_tokens_input = widgets.BoundedIntText(
        value=config.recursive_summary_target_tokens,
        min=128,
        max=16000,
        step=64,
        description="摘要目标",
        layout=widgets.Layout(width="220px"),
    )
    recursive_summary_batch_size_input = widgets.BoundedIntText(
        value=config.recursive_summary_batch_size,
        min=2,
        max=12,
        step=1,
        description="摘要批大小",
        layout=widgets.Layout(width="220px"),
    )
    prompt_compression_turn_token_limit_input = widgets.BoundedIntText(
        value=config.prompt_compression_turn_token_limit,
        min=24,
        max=2000,
        step=24,
        description="历史压缩上限",
        layout=widgets.Layout(width="220px"),
    )
    recent_history_turns_input = widgets.BoundedIntText(
        value=config.recent_history_turns,
        min=0,
        max=30,
        step=1,
        description="原文历史条数",
        layout=widgets.Layout(width="220px"),
    )
    citation_limit_input = widgets.BoundedIntText(
        value=config.citation_limit,
        min=1,
        max=8,
        step=1,
        description="引用数",
        layout=widgets.Layout(width="220px"),
    )
    enable_rerank_checkbox = widgets.Checkbox(value=config.enable_rerank, description="启用切片评分过滤")
    rerank_min_score_input = widgets.BoundedFloatText(
        value=config.rerank_min_score,
        min=0.0,
        max=1.0,
        step=0.01,
        description="最低分数",
        layout=widgets.Layout(width="220px"),
    )
    rerank_min_keep_input = widgets.BoundedIntText(
        value=config.rerank_min_keep,
        min=0,
        max=20,
        step=1,
        description="最低保留",
        layout=widgets.Layout(width="220px"),
    )
    rerank_weight_vector_input = widgets.BoundedFloatText(
        value=config.rerank_weight_vector,
        min=0.0,
        max=2.0,
        step=0.05,
        description="向量权重",
        layout=widgets.Layout(width="220px"),
    )
    rerank_weight_keyword_input = widgets.BoundedFloatText(
        value=config.rerank_weight_keyword,
        min=0.0,
        max=2.0,
        step=0.05,
        description="关键词权重",
        layout=widgets.Layout(width="220px"),
    )
    rerank_weight_phrase_input = widgets.BoundedFloatText(
        value=config.rerank_weight_phrase,
        min=0.0,
        max=2.0,
        step=0.05,
        description="短语权重",
        layout=widgets.Layout(width="220px"),
    )
    rerank_weight_metadata_input = widgets.BoundedFloatText(
        value=config.rerank_weight_metadata,
        min=0.0,
        max=2.0,
        step=0.05,
        description="元数据权重",
        layout=widgets.Layout(width="220px"),
    )
    enable_rewrite_checkbox = widgets.Checkbox(value=config.enable_query_rewrite, description="启用问题改写")
    enable_migration_ui_checkbox = widgets.Checkbox(
        value=config.enable_migration_ui,
        description="显示迁移与备份区域",
    )
    qa_system_prompt_zh_input = widgets.Textarea(
        value=config.qa_system_prompt_zh,
        description="中文系统提示词",
        layout=widgets.Layout(width="100%", height="120px"),
    )
    rewrite_prompt_zh_input = widgets.Textarea(
        value=config.query_rewrite_instruction_zh,
        description="中文改写提示词",
        layout=widgets.Layout(width="100%", height="140px"),
    )
    answer_instruction_zh_input = widgets.Textarea(
        value=config.qa_answer_instruction_zh,
        description="中文回答格式",
        layout=widgets.Layout(width="100%", height="100px"),
    )
    qa_system_prompt_en_input = widgets.Textarea(
        value=config.qa_system_prompt_en,
        description="English system prompt",
        layout=widgets.Layout(width="100%", height="140px"),
    )
    rewrite_prompt_en_input = widgets.Textarea(
        value=config.query_rewrite_instruction_en,
        description="English rewrite prompt",
        layout=widgets.Layout(width="100%", height="170px"),
    )
    answer_instruction_en_input = widgets.Textarea(
        value=config.qa_answer_instruction_en,
        description="English answer format",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    single_analysis_prompt_zh_input = widgets.Textarea(
        value=config.single_analysis_prompt_zh,
        description="中文单文档分析",
        layout=widgets.Layout(width="100%", height="180px"),
    )
    compare_report_prompt_zh_input = widgets.Textarea(
        value=config.compare_report_prompt_zh,
        description="中文对比报告",
        layout=widgets.Layout(width="100%", height="160px"),
    )
    single_analysis_prompt_en_input = widgets.Textarea(
        value=config.single_analysis_prompt_en,
        description="English single-doc analysis",
        layout=widgets.Layout(width="100%", height="210px"),
    )
    compare_report_prompt_en_input = widgets.Textarea(
        value=config.compare_report_prompt_en,
        description="English comparison report",
        layout=widgets.Layout(width="100%", height="180px"),
    )
    data_extraction_prompt_zh_input = widgets.Textarea(
        value=config.data_extraction_prompt_zh,
        description="中文字段抽取",
        layout=widgets.Layout(width="100%", height="220px"),
    )
    table_summary_prompt_zh_input = widgets.Textarea(
        value=config.table_summary_prompt_zh,
        description="中文表格总结",
        layout=widgets.Layout(width="100%", height="160px"),
    )
    data_extraction_prompt_en_input = widgets.Textarea(
        value=config.data_extraction_prompt_en,
        description="English extraction prompt",
        layout=widgets.Layout(width="100%", height="220px"),
    )
    table_summary_prompt_en_input = widgets.Textarea(
        value=config.table_summary_prompt_en,
        description="English table summary",
        layout=widgets.Layout(width="100%", height="160px"),
    )
    save_settings_button = widgets.Button(description="保存参数到配置文件", button_style="info")
    test_model_button = widgets.Button(description="测试模型连接", button_style="warning")
    model_test_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "8px"})
    session_history_button = widgets.Button(description="载入这段会话")
    clear_chat_view_button = widgets.Button(description="清空当前窗口")
    history_page_size_dropdown = widgets.Dropdown(
        options=[("20/页", 20), ("50/页", 50), ("100/页", 100)],
        value=20,
        description="每页",
        layout=widgets.Layout(width="170px"),
    )
    history_first_button = widgets.Button(description="首页")
    history_prev_button = widgets.Button(description="上一页")
    history_next_button = widgets.Button(description="下一页")
    history_last_button = widgets.Button(description="末页")
    history_page_input = widgets.BoundedIntText(
        value=1,
        min=1,
        max=1,
        description="页码",
        layout=widgets.Layout(width="170px"),
    )
    history_go_button = widgets.Button(description="跳转")
    history_page_info = widgets.HTML(value="<i>第 1 / 1 页，共 0 条</i>")
    send_button = widgets.Button(description="发送", button_style="success")
    chat_history_output = widgets.HTML(value=_render_chat_history([]))
    citation_output = widgets.HTML(value="<div>引用将在这里显示。</div>")
    settings_status = widgets.HTML(value="")
    chat_status = widgets.HTML(value="")
    chat_progress_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "8px"})

    analysis_kb_dropdown = widgets.Dropdown(
        options=[("请选择知识库", "")],
        value="",
        description="知识库",
        layout=widgets.Layout(width="320px"),
    )
    refresh_docs_button = widgets.Button(description="刷新文档")
    doc_selector = widgets.SelectMultiple(
        options=[],
        description="分析文件",
        layout=widgets.Layout(display="none"),
    )
    analysis_doc_search_input = widgets.Text(
        value="",
        description="查找文件",
        placeholder="按文件名筛选，勾选后可直接批量分析",
        continuous_update=False,
        layout=widgets.Layout(width="420px"),
    )
    analysis_select_all_button = widgets.Button(description="全选当前列表")
    analysis_clear_button = widgets.Button(description="清空选择")
    analysis_doc_page_size_dropdown = widgets.Dropdown(
        options=[("20 / 页", 20), ("40 / 页", 40), ("80 / 页", 80)],
        value=20,
        description="分页",
        layout=widgets.Layout(width="170px"),
    )
    analysis_doc_prev_button = widgets.Button(description="上一页")
    analysis_doc_next_button = widgets.Button(description="下一页")
    analysis_doc_page_status_html = widgets.HTML(value="<i>第 1 / 1 页，共 0 个文件（已选 0）</i>")
    analysis_doc_status_html = widgets.HTML(
        value="<i>请选择知识库后，在这里勾选需要分析的文件；点击文件名可以查看切片详情。</i>"
    )
    analysis_doc_list_box = widgets.VBox(
        [],
        layout=widgets.Layout(
            border="1px solid #d0d7de",
            padding="10px",
            min_height="500px",
            max_height="560px",
            overflow="auto",
        ),
    )
    analysis_doc_detail_html = widgets.HTML(
        value=_manage_details_placeholder_html("点击上面的文件名后，这里会显示该文件的切片详情。"),
        layout=widgets.Layout(border="1px solid #ddd", padding="8px", max_height="420px", overflow="auto"),
    )
    add_field_button = widgets.Button(description="添加字段", button_style="primary")
    add_echem_template_button = widgets.Button(description="光催化模板")
    add_common_template_button = widgets.Button(description="常用模板")
    clear_fields_button = widgets.Button(description="清空字段")
    template_name_input = widgets.Text(
        value="",
        description="模板名称",
        placeholder="把当前字段保存成模板",
        layout=widgets.Layout(width="340px"),
    )
    field_template_dropdown = widgets.Dropdown(
        options=[("请选择模板", "")],
        value="",
        description="字段模板",
        layout=widgets.Layout(width="340px"),
    )
    save_template_button = widgets.Button(description="保存模板")
    load_template_button = widgets.Button(description="载入模板")
    delete_template_button = widgets.Button(description="删除模板", button_style="danger")
    target_fields_summary_html = widgets.HTML(value="<i>当前未配置目标字段。</i>")
    target_fields_box = widgets.VBox([])
    export_csv_checkbox = widgets.Checkbox(value=True, description="同时导出 CSV 表格")
    single_button = widgets.Button(description="单文档分析")
    compare_button = widgets.Button(description="批量对比")
    resume_compare_button = widgets.Button(description="继续上次批量对比")
    clear_checkpoint_button = widgets.Button(description="清除这次进度", button_style="danger")
    pause_analysis_button = widgets.Button(description="暂停并保存进度", button_style="info", disabled=True)
    stop_analysis_button = widgets.Button(description="停止当前分析", button_style="warning", disabled=True)
    analysis_checkpoint_status_html = widgets.HTML(value="<i>当前没有可继续的批量对比进度。</i>")
    clear_selected_cache_button = widgets.Button(description="清除选中文件缓存", button_style="warning")
    report_output = widgets.Textarea(
        value=_render_report_log_text([]),
        disabled=True,
        layout=widgets.Layout(
            width="100%",
            height="304px",
        ),
    )
    report_output_panel = widgets.Box(
        [report_output],
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="0",
            height="320px",
            max_height="320px",
            overflow="hidden",
        ),
    )
    report_output_panel.add_class("assistant-log-panel")
    report_markdown_output = widgets.HTML(
        value=_render_markdown_html(""),
        layout=widgets.Layout(
            width="100%",
        ),
    )
    report_markdown_output_panel = widgets.Box(
        [report_markdown_output],
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="12px",
            width="100%",
            overflow="visible",
        ),
    )
    report_progress_output = widgets.HTML(
        value="<i>当前还没有运行中的分析任务。</i>",
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="8px",
            min_height="180px",
            max_height="360px",
            overflow="auto",
        ),
    )
    report_table_output = widgets.HTML(
        value="<i>这里会显示结构化表格结果，方便你快速核对与筛选。</i>",
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="8px",
            min_height="420px",
            max_height="560px",
            overflow="auto",
        ),
    )
    target_field_rows: list[dict[str, object]] = []
    manage_detail_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
    report_log_lines: list[str] = []
    analysis_doc_records: list[DocumentRecord] = []
    analysis_doc_records_by_id: dict[str, DocumentRecord] = {}
    analysis_doc_cache_status: dict[str, bool] = {}
    analysis_doc_preview_state = {"doc_id": ""}
    refresh_tokens = {
        "manage_files": 0,
        "manage_details": 0,
        "kb_status": 0,
        "analysis_docs": 0,
    }
    manage_detail_page = {"current": 1}
    manage_file_all_options: list[tuple[str, str]] = []
    manage_file_selected_paths: set[str] = set()
    manage_file_sync_state = {"updating": False}
    manage_file_pagination = {
        "current": 1,
        "page_size": max(1, int(manage_file_page_size_dropdown.value)),
    }
    analysis_doc_pagination = {
        "current": 1,
        "page_size": max(1, int(analysis_doc_page_size_dropdown.value)),
    }

    def _task_is_running(task_state: dict[str, object]) -> bool:
        task = task_state.get("task")
        return bool(task and not task.done())

    def _update_stop_buttons() -> None:
        manage_stop_button.disabled = not _task_is_running(manage_task_state)
        stop_chat_button.disabled = not _task_is_running(chat_task_state)
        stop_analysis_button.disabled = not _task_is_running(analysis_task_state)
        pause_analysis_button.disabled = not _task_is_running(analysis_task_state)

    def _active_log_path() -> Path:
        return config.active_api_log_path

    def _refresh_field_template_options(preferred: str | None = None) -> None:
        template_names = sorted(name for name, value in field_templates.items() if isinstance(value, list))
        field_template_dropdown.options = [("请选择模板", "")] + [(name, name) for name in template_names]
        valid_values = {value for _, value in field_template_dropdown.options}
        target = preferred if preferred is not None else field_template_dropdown.value
        field_template_dropdown.value = target if target in valid_values else ""

    def _save_field_templates() -> None:
        save_json_mapping(config.field_template_path, field_templates)

    def _render_report_table(headers: list[str], rows: list[dict[str, str]], title: str = "结构化表格") -> str:
        if not headers or not rows:
            return "<i>当前还没有可展示的结构化表格。</i>"
        head_chunks: list[str] = []
        for index, cell in enumerate(headers):
            extra_style = "left:0;z-index:3;" if index == 0 else ""
            head_chunks.append(
                f"<th style='padding:10px 12px;border-bottom:1px solid #d0d7de;background:#f6f8fa;text-align:left;"
                f"position:sticky;top:0;{extra_style}min-width:180px;word-break:break-word;'>{_escape_html(cell)}</th>"
            )
        head_cells = "".join(head_chunks)
        body_rows: list[str] = []
        for row in rows:
            cell_chunks: list[str] = []
            for index, header in enumerate(headers):
                extra_style = "position:sticky;left:0;background:#fff;z-index:1;" if index == 0 else ""
                cell_chunks.append(
                    f"<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;vertical-align:top;"
                    f"min-width:180px;max-width:320px;word-break:break-word;line-height:1.7;{extra_style}'>"
                    f"{_escape_html(str(row.get(header, '-')))}</td>"
                )
            cells = "".join(cell_chunks)
            body_rows.append(f"<tr>{cells}</tr>")
        return (
            "<div style='margin-top:8px;'>"
            f"<div style='font-weight:600;margin-bottom:8px;'>{_escape_html(title)}</div>"
            "<div style='overflow:auto;max-height:520px;border:1px solid #d0d7de;border-radius:8px;'>"
            "<table style='width:max-content;min-width:100%;border-collapse:collapse;font-size:13px;background:#fff;'>"
            f"<thead><tr>{head_cells}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
            "</div>"
        )

    def _single_extraction_table_payload(extraction) -> tuple[list[str], list[dict[str, str]]]:
        headers = ["字段", "结果", "状态", "来源", "定位", "是否换算"]
        rows: list[dict[str, str]] = []
        for field in extraction.fields:
            rows.append(
                {
                    "字段": field.field_name,
                    "结果": field.normalized_value or field.value or "未提及",
                    "状态": field.status,
                    "来源": field.source_file or extraction.title,
                    "定位": field.page_label or field.section_label or "-",
                    "是否换算": "是" if field.converted else "否",
                }
            )
        return headers, rows

    def clear_manage_detail_cache() -> None:
        manage_detail_cache.clear()

    def reset_manage_detail_page() -> None:
        manage_detail_page["current"] = 1

    async def append_manage_message(message: str) -> None:
        """Append one knowledge-base progress line to the management output box."""

        manage_output.append_stdout(f"{_shorten_manage_message(message)}\n")

    async def append_report_message(message: str) -> None:
        """Append one analysis progress line to the report output box."""

        compact = _shorten_report_message(message)
        report_log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {compact}")
        report_output.value = _render_report_log_text(report_log_lines)

    def clear_report_markdown_view() -> None:
        report_markdown_output.value = _render_markdown_html("")

    def render_report_markdown(markdown_text: str) -> None:
        report_markdown_output.value = _render_markdown_html(markdown_text)

    def clear_report_log() -> None:
        report_log_lines.clear()
        report_output.value = _render_report_log_text(report_log_lines)

    def write_report_log(message: str) -> None:
        report_log_lines.append(str(message))
        report_output.value = _render_report_log_text(report_log_lines)

    def reset_report_progress_view(message: str = "<i>当前还没有运行中的分析任务。</i>") -> None:
        report_progress_output.value = message

    def reset_report_table_view(message: str = "<i>这里会显示结构化表格结果，方便你快速核对与筛选。</i>") -> None:
        report_table_output.value = message

    def _render_analysis_progress_panel(*, title: str, summary: str, detail: str = "") -> str:
        body = [f"<b>{_escape_html(title)}</b>", f"<div style='margin-top:6px;'>{_escape_html(summary)}</div>"]
        if detail:
            body.append(
                "<div style='margin-top:8px;color:#5b6472;font-size:12px;line-height:1.5;'>"
                f"{_escape_html(detail)}"
                "</div>"
            )
        return (
            "<div style='padding:12px;border:1px solid #d0d7de;border-radius:10px;background:#f6f8fa;"
            "line-height:1.6;'>"
            + "".join(body)
            + "</div>"
        )

    async def append_chat_progress(message: str) -> None:
        """Append one chat-stage progress line to the dedicated progress box."""

        chat_progress_output.append_stdout(f"{message}\n")

    def clear_chat_progress() -> None:
        chat_progress_output.clear_output()

    def update_manage_pagination(total_items: int, total_pages: int, current_page: int) -> None:
        detail_page_status_html.value = f"<i>第 {current_page} / {max(1, total_pages)} 页，共 {total_items} 个文件卡片</i>"
        detail_prev_button.disabled = current_page <= 1 or total_items == 0
        detail_next_button.disabled = current_page >= total_pages or total_items == 0

    def reset_manage_file_page() -> None:
        manage_file_pagination["current"] = 1

    def _selected_manage_file_paths() -> tuple[str, ...]:
        valid_values = {value for _, value in manage_file_all_options}
        manage_file_selected_paths.intersection_update(valid_values)
        return tuple(value for _, value in manage_file_all_options if value in manage_file_selected_paths)

    def _render_manage_file_selector_page() -> None:
        total_items = len(manage_file_all_options)
        page_size = max(1, int(manage_file_pagination["page_size"]))
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        current_page = max(1, min(int(manage_file_pagination["current"]), total_pages))
        manage_file_pagination["current"] = current_page

        start = (current_page - 1) * page_size
        end = start + page_size
        page_options = manage_file_all_options[start:end]
        visible_values = {value for _, value in page_options}
        visible_selected = tuple(value for value in _selected_manage_file_paths() if value in visible_values)

        manage_file_sync_state["updating"] = True
        try:
            manage_doc_selector.options = page_options
            manage_doc_selector.value = visible_selected
        finally:
            manage_file_sync_state["updating"] = False

        manage_file_page_status_html.value = (
            f"<i>第 {current_page} / {total_pages} 页，共 {total_items} 个文件（已选 {len(manage_file_selected_paths)}）</i>"
        )
        manage_file_prev_button.disabled = current_page <= 1 or total_items == 0
        manage_file_next_button.disabled = current_page >= total_pages or total_items == 0

    def _set_manage_file_options(options: list[tuple[str, str]]) -> None:
        valid_values = {value for _, value in options}
        manage_file_selected_paths.intersection_update(valid_values)
        manage_file_all_options[:] = options
        _render_manage_file_selector_page()

    def refresh_target_fields_summary() -> None:
        specs = collect_target_field_specs()
        if not specs:
            target_fields_summary_html.value = "<i>当前未配置目标字段。你可以点击“添加字段”逐项填写，或直接插入模板。</i>"
            _schedule(refresh_analysis_checkpoint_status())
            return
        chips = []
        for spec in specs:
            detail = f" | {_escape_html(spec.expected_unit)}" if spec.expected_unit else ""
            chips.append(
                "<span style='display:inline-block;margin:4px 6px 0 0;padding:4px 10px;border-radius:999px;"
                "background:#eef2ff;color:#3730a3;font-size:12px;font-weight:600;'>"
                f"{_escape_html(spec.name)}{detail}</span>"
            )
        target_fields_summary_html.value = "<b>当前字段：</b><br>" + "".join(chips)
        _schedule(refresh_analysis_checkpoint_status())

    async def refresh_analysis_checkpoint_status() -> None:
        course_id = analysis_kb_dropdown.value.strip()
        selected_doc_ids = list(doc_selector.value)
        if not course_id or not selected_doc_ids:
            analysis_checkpoint_status_html.value = "<i>选择知识库和文档后，这里会显示是否存在可继续的批量对比进度。</i>"
            return
        info = await batch_service.inspect_compare_checkpoint(
            course_id=course_id,
            doc_ids=selected_doc_ids,
            output_language=language_toggle.value,
            target_fields=collect_target_field_specs() or None,
            export_csv=export_csv_checkbox.value,
        )
        if not info.get("exists"):
            analysis_checkpoint_status_html.value = "<i>当前选择还没有可继续的批量对比进度。</i>"
            return
        active_docs = info.get("active_docs", []) or []
        active_doc_lines = []
        for item in active_docs[:3]:
            locator = ""
            if int(item.get("current_window", 0) or 0) and int(item.get("total_windows", 0) or 0):
                locator = f" | 滑窗 {int(item.get('current_window', 0))}/{int(item.get('total_windows', 0))}"
            active_doc_lines.append(
                f"- {_escape_html(str(item.get('title', item.get('doc_id', '-'))))} | {_escape_html(str(item.get('stage', '-')))}"
                f" | {_escape_html(str(item.get('status', '-')))}{_escape_html(locator)}<br>"
                f"&nbsp;&nbsp;{_escape_html(str(item.get('detail', '-')))}"
            )
        active_doc_html = "".join(active_doc_lines) if active_doc_lines else "暂无正在处理的文档"
        analysis_checkpoint_status_html.value = (
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:10px;background:#f6f8fa;'>"
            f"<b>发现可继续的进度</b><br>"
            f"状态: {_escape_html(str(info.get('status', '-')))}<br>"
            f"已完成单文档分析: {_escape_html(str(info.get('analysis_done', 0)))} / {int(info.get('total_docs', len(selected_doc_ids)))}<br>"
            f"还剩待分析: {_escape_html(str(info.get('analysis_remaining', 0)))} 篇<br>"
            f"已完成字段抽取: {_escape_html(str(info.get('extraction_done', 0)))} / {int(info.get('total_docs', len(selected_doc_ids)))}<br>"
            f"还剩待抽取: {_escape_html(str(info.get('extraction_remaining', 0)))} 篇<br>"
            f"最近阶段: {_escape_html(str(info.get('last_stage', '-')))}<br>"
            f"最近信息: {_escape_html(str(info.get('last_message', '-')))}<br>"
            f"当前文档:<br>{active_doc_html}<br>"
            f"最后更新: {_escape_html(str(info.get('updated_at', '-')))}<br>"
            f"进度快照: {_escape_html(str(info.get('progress_output_path', '-')))}"
            "</div>"
        )

    async def clear_compare_checkpoint(_=None) -> None:
        clear_report_log()
        if not analysis_kb_dropdown.value:
            write_report_log("请先选择知识库。")
            return
        if not doc_selector.value:
            write_report_log("请先选择至少一篇文档。")
            return
        field_specs = collect_target_field_specs()
        result = await batch_service.clear_compare_checkpoint(
            course_id=analysis_kb_dropdown.value.strip(),
            doc_ids=list(doc_selector.value),
            output_language=language_toggle.value,
            target_fields=field_specs or None,
            export_csv=export_csv_checkbox.value,
        )
        if not result.get("removed"):
            write_report_log("当前选择没有可清除的断点进度。")
            return
        write_report_log("已清除这次批量对比的断点进度。")
        if result.get("progress_output_path"):
            write_report_log(f"已删除进度快照: {result.get('progress_output_path')}")
        await refresh_analysis_checkpoint_status()

    def collect_target_field_specs():
        specs = []
        for row in target_field_rows:
            name = str(row["name"].value).strip()
            instruction = str(row["instruction"].value).strip()
            expected_unit = str(row["unit"].value).strip()
            if not name:
                continue
            specs.append(parse_extraction_field_specs(f"{name} | {instruction} | {expected_unit}")[0])
        return specs

    def render_target_field_rows() -> None:
        if not target_field_rows:
            target_fields_box.children = (
                widgets.HTML(
                    "<div style='padding:12px;border:1px dashed #d0d7de;border-radius:10px;color:#6b7280;'>"
                    "还没有配置字段。点击“添加字段”开始，或使用模板快速生成。"
                    "</div>"
                ),
            )
            refresh_target_fields_summary()
            return
        target_fields_box.children = tuple(row["container"] for row in target_field_rows)
        refresh_target_fields_summary()

    def add_target_field_row(name: str = "", instruction: str = "", expected_unit: str = "") -> None:
        name_input = widgets.Text(
            value=name,
            placeholder="字段名，例如：氢气产生速率",
            layout=widgets.Layout(width="28%"),
        )
        instruction_input = widgets.Text(
            value=instruction,
            placeholder="补充说明，例如：优先提取主结果中的数值",
            layout=widgets.Layout(width="52%"),
        )
        unit_input = widgets.Text(
            value=expected_unit,
            placeholder="期望单位",
            layout=widgets.Layout(width="14%"),
        )
        remove_button = widgets.Button(description="删除", button_style="danger", layout=widgets.Layout(width="78px"))
        header = widgets.HTML(
            "<div style='font-weight:600;margin-bottom:6px;color:#374151;'>字段配置</div>"
        )
        container = widgets.VBox(
            [
                header,
                widgets.HBox([name_input, instruction_input, unit_input, remove_button]),
            ],
            layout=widgets.Layout(
                border="1px solid #d0d7de",
                padding="10px 12px",
                margin="0 0 8px 0",
            ),
        )
        row = {
            "name": name_input,
            "instruction": instruction_input,
            "unit": unit_input,
            "container": container,
        }

        def _remove(_):
            if row in target_field_rows:
                target_field_rows.remove(row)
            render_target_field_rows()

        def _refresh(_):
            refresh_target_fields_summary()

        remove_button.on_click(_remove)
        name_input.observe(_refresh, names="value")
        instruction_input.observe(_refresh, names="value")
        unit_input.observe(_refresh, names="value")
        target_field_rows.append(row)
        render_target_field_rows()

    def add_field_template(items: list[tuple[str, str, str]]) -> None:
        existing = {(str(row["name"].value).strip(), str(row["unit"].value).strip()) for row in target_field_rows}
        for name, instruction, unit in items:
            if (name.strip(), unit.strip()) in existing:
                continue
            add_target_field_row(name=name, instruction=instruction, expected_unit=unit)

    def clear_target_fields() -> None:
        target_field_rows.clear()
        render_target_field_rows()

    def save_current_template(_=None) -> None:
        name = template_name_input.value.strip()
        specs = collect_target_field_specs()
        if not name:
            clear_report_log()
            write_report_log("请先填写模板名称。")
            return
        if not specs:
            clear_report_log()
            write_report_log("当前没有可保存的字段。")
            return
        field_templates[name] = [
            {
                "name": spec.name,
                "instruction": spec.instruction,
                "expected_unit": spec.expected_unit,
            }
            for spec in specs
        ]
        _save_field_templates()
        _refresh_field_template_options(preferred=name)

    def load_selected_template(_=None) -> None:
        template_name = field_template_dropdown.value
        items = field_templates.get(template_name)
        if not template_name or not isinstance(items, list):
            return
        clear_target_fields()
        for item in items:
            if not isinstance(item, dict):
                continue
            add_target_field_row(
                name=str(item.get("name", "")).strip(),
                instruction=str(item.get("instruction", "")).strip(),
                expected_unit=str(item.get("expected_unit", "")).strip(),
            )
        template_name_input.value = template_name

    def delete_selected_template(_=None) -> None:
        template_name = field_template_dropdown.value
        if not template_name or template_name not in field_templates:
            return
        field_templates.pop(template_name, None)
        _save_field_templates()
        _refresh_field_template_options()

    async def get_cached_chunk_details(course_id: str, record: DocumentRecord) -> list[dict[str, object]]:
        cache_key = (course_id, str(Path(record.file_path).resolve(strict=False)))
        if cache_key in manage_detail_cache:
            return manage_detail_cache[cache_key]
        if not record.doc_id or not record.is_vectorized:
            manage_detail_cache[cache_key] = []
            return []
        chunks = await vector_store.get_document_chunks(course_id=course_id, doc_id=record.doc_id)
        details: list[dict[str, object]] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata
            details.append(
                {
                    "chunk_index": int(metadata.get("chunk_index", index)),
                    "chunk_id": str(metadata.get("chunk_id", "")),
                    "page_label": metadata.get("page_label"),
                    "section_label": metadata.get("section_label"),
                    "merged_from_count": int(metadata.get("merged_from_count", 1)),
                    "length": len(chunk.page_content),
                    "content": chunk.page_content,
                }
            )
        manage_detail_cache[cache_key] = details
        return details

    async def refresh_knowledge_base_options(preferred_course_id: str | None = None):
        course_ids = await manager.list_knowledge_bases()
        options = [("请选择知识库", "")] + [(course_id, course_id) for course_id in course_ids]
        for dropdown in (manage_kb_dropdown, chat_kb_dropdown, analysis_kb_dropdown, target_kb_dropdown, move_target_kb_dropdown):
            current_value = preferred_course_id if preferred_course_id is not None else dropdown.value
            dropdown.options = options
            valid_values = {value for _, value in options}
            dropdown.value = current_value if current_value in valid_values else ""

    async def refresh_document_options(course_id: str, selector, empty_hint: str | None = None):
        if selector is doc_selector:
            await refresh_analysis_doc_options(course_id, empty_hint=empty_hint)
            return
        if not course_id:
            selector.options = []
            selector.value = ()
            if selector is chat_doc_selector and empty_hint:
                chat_scope_hint.value = f"<i>{empty_hint}</i>"
            return
        records = await manager.list_documents(course_id)
        options = [(_format_doc_option(record), record.doc_id) for record in records]
        selected_ids = tuple(doc_id for doc_id in selector.value if doc_id in {value for _, value in options})
        selector.options = options
        selector.value = selected_ids
        if selector is chat_doc_selector:
            chat_scope_hint.value = "<i>未选择文件时，默认检索整个知识库。</i>" if options else f"<i>{empty_hint or '该知识库下暂无文件。'}</i>"

    async def refresh_manageable_file_options(course_id: str):
        refresh_tokens["manage_files"] += 1
        token = refresh_tokens["manage_files"]
        if not course_id:
            if manage_kb_dropdown.value:
                return
            manage_file_selected_paths.clear()
            _set_manage_file_options([])
            await refresh_manage_details_preview(course_id="", all_records=[])
            return
        records = await manager.list_manageable_files(course_id)
        if token != refresh_tokens["manage_files"] or course_id != manage_kb_dropdown.value:
            return
        options = [(_format_manage_doc_option(record), record.file_path) for record in records]
        _set_manage_file_options(options)
        await refresh_manage_details_preview(course_id=course_id, all_records=records)

    async def refresh_manage_details_preview(course_id: str, all_records: list[DocumentRecord] | None = None):
        try:
            refresh_tokens["manage_details"] += 1
            token = refresh_tokens["manage_details"]
            if not course_id:
                if manage_kb_dropdown.value:
                    return
                chunk_search_status.value = "<i>请选择知识库后查看详情。</i>"
                chunk_output.value = _manage_details_placeholder_html("请选择一个知识库后查看文件概览和切片详情。")
                update_manage_pagination(total_items=0, total_pages=1, current_page=1)
                return
            if course_id == manage_kb_dropdown.value:
                chunk_search_status.value = f"<i>正在加载知识库“{_escape_html(course_id)}”的详情...</i>"
                chunk_output.value = _manage_details_placeholder_html("正在读取文件列表与概览信息，请稍候。")
            if all_records is None:
                all_records = await manager.list_manageable_files(course_id)
            if token != refresh_tokens["manage_details"] or course_id != manage_kb_dropdown.value:
                return
            selected_paths = {str(Path(path).resolve(strict=False)) for path in _selected_manage_file_paths()}
            search_query = chunk_search_input.value.strip()
            selected_records = [
                record
                for record in all_records
                if str(Path(record.file_path).resolve(strict=False)) in selected_paths
            ]
            base_records = selected_records if selected_paths else all_records
            if search_query and search_selected_only_checkbox.value and not selected_paths:
                overview_html = _render_manage_overview_card(
                    course_id,
                    all_records,
                    sum(int(record.chunk_count or 0) for record in all_records),
                    open_by_default=True,
                )
                chunk_search_status.value = (
                    "<i>当前已启用“仅在已选文件中搜索”。请先在上面的文件列表中选择需要加载的文件，再执行内容搜索。</i>"
                )
                chunk_output.value = (
                    '<div style="margin-bottom:10px;padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#f6f8fa;">'
                    f"<b>知识库详情: {_escape_html(course_id)}</b>"
                    "</div>"
                    + overview_html
                    + _manage_details_placeholder_html("请先在上面的文件列表中选择需要搜索的文件，然后再次点击搜索。")
                )
                update_manage_pagination(total_items=0, total_pages=1, current_page=1)
                return
            records = base_records
            should_load_details = bool(selected_paths or search_query)
            if should_load_details:
                detail_tasks = [get_cached_chunk_details(course_id, record) for record in records]
                detail_lists = await asyncio.gather(*detail_tasks)
            else:
                detail_lists = [[] for _ in records]
            if token != refresh_tokens["manage_details"] or course_id != manage_kb_dropdown.value:
                return
            rendered_html, status_html, total_items, total_pages, current_page = _render_manage_details_html(
                course_id=course_id,
                records=records,
                detail_lists=detail_lists,
                search_query=search_query,
                selected_file_count=len(selected_paths),
                total_file_count=len(all_records),
                details_loaded=should_load_details,
                total_chunk_count=sum(int(record.chunk_count or 0) for record in all_records),
                current_page=manage_detail_page["current"],
                page_size=int(detail_page_size_dropdown.value),
            )
            manage_detail_page["current"] = current_page
            update_manage_pagination(total_items=total_items, total_pages=total_pages, current_page=current_page)
            chunk_search_status.value = status_html
            chunk_output.value = rendered_html
        except Exception:
            chunk_search_status.value = "<b>刷新详情预览失败</b>"
            chunk_output.value = _manage_details_placeholder_html(_escape_html(traceback.format_exc()))
            update_manage_pagination(total_items=0, total_pages=1, current_page=1)

    async def refresh_knowledge_base_status(course_id: str):
        refresh_tokens["kb_status"] += 1
        token = refresh_tokens["kb_status"]
        if not course_id:
            if manage_kb_dropdown.value:
                return
            knowledge_base_status_html.value = (
                "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;'>"
                "<b>当前未选择知识库</b>"
                "<div style='margin-top:6px;color:#5b6472;'>请选择一个知识库后查看文件数、向量化状态和切片统计。</div>"
                "</div>"
            )
            knowledge_base_change_html.value = ""
            return
        records = await manager.list_manageable_files(course_id)
        change_info = await manager.detect_knowledge_base_changes(course_id)
        if token != refresh_tokens["kb_status"] or course_id != manage_kb_dropdown.value:
            return
        total_files = len(records)
        vectorized_files = sum(1 for record in records if record.is_vectorized)
        unvectorized_files = total_files - vectorized_files
        total_chunks = sum(int(record.chunk_count or 0) for record in records)
        changed = bool(change_info.get("changed", False))
        change_text = (
            "检测到知识库内容有变化，建议重新向量化相关文件。"
            if changed
            else "当前知识库与最近一次记录状态一致。"
        )
        knowledge_base_status_html.value = (
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;'>"
            f"<b>当前知识库: {_escape_html(course_id)}</b>"
            f"<div style='margin-top:6px;color:#5b6472;'>文件总数: {total_files}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>已向量化文件: {vectorized_files}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>未向量化文件: {unvectorized_files}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>当前切片总数: {total_chunks}</div>"
            "<div style='margin-top:4px;color:#5b6472;'>删除当前知识库只会删除这一项的原始文件和向量索引，不影响其他知识库或会话数据。</div>"
            "</div>"
        )
        knowledge_base_change_html.value = (
            "<div style='margin-top:8px;padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;"
            f"background:{'#fff7ed' if changed else '#f0fdf4'};color:{'#9a3412' if changed else '#166534'};'>"
            f"<b>变更检查</b><div style='margin-top:4px;'>{_escape_html(change_text)}</div></div>"
        )

    def _filtered_analysis_records() -> list[DocumentRecord]:
        query = analysis_doc_search_input.value.strip().lower()
        if not query:
            return list(analysis_doc_records)
        return [record for record in analysis_doc_records if query in record.file_name.lower()]

    def _update_analysis_doc_status(records: list[DocumentRecord]) -> None:
        total = len(analysis_doc_records)
        visible = len(records)
        selected = len(doc_selector.value)
        if not analysis_kb_dropdown.value:
            analysis_doc_status_html.value = "<i>请选择知识库后，在这里勾选需要分析的文件；点击文件名可以查看切片详情。</i>"
            return
        scope_text = (
            f"当前共 {total} 个文件，筛选后显示 {visible} 个，已勾选 {selected} 个。"
            if analysis_doc_search_input.value.strip()
            else f"当前共 {total} 个文件，已勾选 {selected} 个。"
        )
        analysis_doc_status_html.value = (
            "<div style='padding:8px 10px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;color:#5b6472;'>"
            f"{_escape_html(scope_text)} 点击文件名前往下方查看该文件的切片详情。"
            "</div>"
        )

    def reset_analysis_doc_page() -> None:
        analysis_doc_pagination["current"] = 1

    def _update_analysis_doc_pagination(*, total_items: int, total_pages: int, current_page: int) -> None:
        analysis_doc_page_status_html.value = (
            f"<i>第 {current_page} / {max(1, total_pages)} 页，共 {total_items} 个文件（已选 {len(doc_selector.value)}）</i>"
        )
        analysis_doc_prev_button.disabled = current_page <= 1 or total_items == 0
        analysis_doc_next_button.disabled = current_page >= total_pages or total_items == 0

    async def show_analysis_doc_details(doc_id: str) -> None:
        course_id = analysis_kb_dropdown.value.strip()
        record = analysis_doc_records_by_id.get(doc_id)
        if not course_id or record is None:
            analysis_doc_preview_state["doc_id"] = ""
            analysis_doc_detail_html.value = _manage_details_placeholder_html("点击上面的文件名后，这里会显示该文件的切片详情。")
            return
        analysis_doc_preview_state["doc_id"] = doc_id
        analysis_doc_detail_html.value = _manage_details_placeholder_html(f"正在加载 {_escape_html(record.file_name)} 的切片详情...")
        chunk_details = await get_cached_chunk_details(course_id, record)
        rendered = _render_manage_detail_card(record, chunk_details, "", details_loaded=True)
        detail_html = str(rendered.get("html", ""))
        if detail_html.startswith("<details "):
            detail_html = detail_html.replace("<details ", "<details open ", 1)
        analysis_doc_detail_html.value = (
            "<div style='margin-bottom:10px;padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#f6f8fa;'>"
            "<b>文件切片详情</b><br>"
            "<span style='color:#5b6472;'>点击其他文件名可切换预览。</span>"
            "</div>"
            + detail_html
        )

    def render_analysis_doc_selector() -> None:
        filtered_records = _filtered_analysis_records()
        selected_ids = set(doc_selector.value)
        preview_doc_id = str(analysis_doc_preview_state.get("doc_id", "") or "")
        page_size = max(1, int(analysis_doc_pagination["page_size"]))
        total_items = len(filtered_records)
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        current_page = max(1, min(int(analysis_doc_pagination["current"]), total_pages))
        analysis_doc_pagination["current"] = current_page
        if not filtered_records:
            analysis_doc_list_box.children = (
                widgets.HTML(
                    "<div style='padding:12px;color:#6b7280;'>当前没有可显示的文件。可以换一个知识库，或清空筛选条件后再试。</div>"
                ),
            )
            _update_analysis_doc_status(filtered_records)
            _update_analysis_doc_pagination(total_items=0, total_pages=1, current_page=1)
            return
        start = (current_page - 1) * page_size
        end = start + page_size
        paged_records = filtered_records[start:end]
        rows = []
        for record in paged_records:
            checkbox = widgets.Checkbox(
                value=record.doc_id in selected_ids,
                indent=False,
                layout=widgets.Layout(width="28px", min_width="28px"),
            )
            display_name = _truncate_middle(record.file_name, 92)
            file_button = widgets.Button(
                description=display_name,
                tooltip=record.file_name,
                layout=widgets.Layout(flex="1 1 auto", width="auto", height="38px"),
                button_style="info" if record.doc_id == preview_doc_id else "",
            )
            file_button.style.button_color = "#f8fafc" if record.doc_id != preview_doc_id else None
            has_cache = analysis_doc_cache_status.get(record.doc_id, False)
            cache_badge = (
                "<span style='margin-left:8px;color:#16a34a;font-size:11px;font-weight:600;'>🟢 有缓存</span>"
                if has_cache
                else "<span style='margin-left:8px;color:#9ca3af;font-size:11px;'>⚪ 无缓存</span>"
            )
            meta_html = widgets.HTML(
                value=(
                    "<div style='text-align:right;white-space:nowrap;'>"
                    f"{_source_type_badge_html(record.source_type)}"
                    f"<span style='margin-left:8px;color:#5b6472;font-size:12px;'>{int(record.chunk_count or 0)} 个切片</span>"
                    f"{cache_badge}"
                    "</div>"
                ),
                layout=widgets.Layout(width="260px", min_width="260px"),
            )

            def _on_analysis_checked(change, *, doc_id=record.doc_id):
                if change.get("name") != "value":
                    return
                selected = list(doc_selector.value)
                if change["new"]:
                    if doc_id not in selected:
                        selected.append(doc_id)
                else:
                    selected = [item for item in selected if item != doc_id]
                doc_selector.value = tuple(selected)

            def _on_analysis_preview(_button, *, doc_id=record.doc_id):
                analysis_doc_preview_state["doc_id"] = doc_id
                render_analysis_doc_selector()
                _schedule(show_analysis_doc_details(doc_id))

            checkbox.observe(_on_analysis_checked, names="value")
            file_button.on_click(_on_analysis_preview)
            rows.append(
                widgets.HBox(
                    [checkbox, file_button, meta_html],
                    layout=widgets.Layout(
                        width="100%",
                        align_items="center",
                        padding="6px 4px",
                        border_bottom="1px solid #eef2f7",
                    ),
                )
            )
        analysis_doc_list_box.children = tuple(rows)
        _update_analysis_doc_status(filtered_records)
        _update_analysis_doc_pagination(total_items=total_items, total_pages=total_pages, current_page=current_page)

    async def refresh_analysis_doc_options(course_id: str, empty_hint: str | None = None) -> None:
        refresh_tokens["analysis_docs"] += 1
        token = refresh_tokens["analysis_docs"]
        analysis_doc_preview_state["doc_id"] = ""
        if not course_id:
            doc_selector.options = []
            doc_selector.value = ()
            analysis_doc_records.clear()
            analysis_doc_records_by_id.clear()
            analysis_doc_list_box.children = (
                widgets.HTML("<div style='padding:12px;color:#6b7280;'>请选择知识库后查看分析文件列表。</div>"),
            )
            analysis_doc_detail_html.value = _manage_details_placeholder_html("点击上面的文件名后，这里会显示该文件的切片详情。")
            _update_analysis_doc_status([])
            _update_analysis_doc_pagination(total_items=0, total_pages=1, current_page=1)
            return
        analysis_doc_list_box.children = (
            widgets.HTML("<div style='padding:12px;color:#6b7280;'>正在加载文件列表，请稍候。</div>"),
        )
        records = await manager.list_documents(course_id)
        if token != refresh_tokens["analysis_docs"] or course_id != analysis_kb_dropdown.value:
            return
        analysis_doc_records[:] = records
        analysis_doc_records_by_id.clear()
        analysis_doc_records_by_id.update({record.doc_id: record for record in records})
        options = [(_format_doc_option(record), record.doc_id) for record in records]
        valid_ids = {value for _, value in options}
        selected_ids = tuple(doc_id for doc_id in doc_selector.value if doc_id in valid_ids)
        doc_selector.options = options
        doc_selector.value = selected_ids
        reset_analysis_doc_page()
        # Batch check analysis cache status for all documents.
        analysis_doc_cache_status.clear()
        if records and config.enable_result_cache:
            async def _check_cache(rec):
                try:
                    info = await single_doc_service.inspect_analysis_cache(
                        course_id=course_id, doc_id=rec.doc_id,
                    )
                    return rec.doc_id, bool(info.get("cached"))
                except Exception:
                    return rec.doc_id, False
            cache_results = await asyncio.gather(*(_check_cache(rec) for rec in records))
            analysis_doc_cache_status.update(dict(cache_results))
        render_analysis_doc_selector()
        if records:
            analysis_doc_detail_html.value = _manage_details_placeholder_html("点击上面的文件名后，这里会显示该文件的切片详情。")
        else:
            analysis_doc_detail_html.value = _manage_details_placeholder_html(empty_hint or "该知识库下暂无文件。")
        await refresh_analysis_checkpoint_status()

    def _select_visible_analysis_docs(_=None) -> None:
        doc_selector.value = tuple(record.doc_id for record in _filtered_analysis_records())

    def _clear_analysis_docs(_=None) -> None:
        doc_selector.value = ()

    session_summaries_by_id: dict[str, dict[str, object]] = {}
    history_pagination_state = {
        "current_page": 1,
        "total_pages": 1,
        "total_turns": 0,
        "page_size": int(history_page_size_dropdown.value),
    }

    async def refresh_session_options(preferred_session_id: str | None = None):
        nonlocal session_summaries_by_id
        summaries = await _active_memory_store().list_sessions(limit=100)
        query = session_search_input.value.strip().lower()
        if query:
            summaries = [
                item
                for item in summaries
                if query in str(item.title).lower()
            ]
        session_summaries_by_id = {item.session_id: item.model_dump() for item in summaries}
        options = [(_format_session_option(item), item.session_id) for item in summaries]
        if not options:
            options = [("暂无会话", "")]
        session_selector.options = options
        valid_values = {value for _, value in options}
        target_session_id = preferred_session_id if preferred_session_id is not None else session_id_input.value.strip()
        if target_session_id in valid_values:
            session_selector.value = target_session_id
        elif options and options[0][1]:
            session_selector.value = options[0][1]
        else:
            session_selector.value = ""
        _refresh_session_summary()

    def _refresh_session_summary() -> None:
        session_id = session_selector.value or session_id_input.value.strip()
        current_kb = chat_kb_dropdown.value.strip() or "未选择"
        selected_count = len(chat_doc_selector.value)
        file_scope = "整个知识库" if selected_count == 0 else f"已选 {selected_count} 个文件"
        memory_mode_text = _translate_memory_mode(memory_mode_toggle.value)
        answer_mode_text = "流式输出" if streaming_mode_toggle.value == "stream" else "仅普通回答"
        memory_help = (
            "仅保留在当前 Notebook / Kernel 运行期间"
            if memory_mode_toggle.value == "session"
            else "会保存到本地数据库，重启后仍可恢复"
        )
        if not session_id:
            session_summary_html.value = (
                "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;'>"
                "<b>当前没有会话</b>"
                f"<div style='margin-top:6px;color:#5b6472;'>当前知识库: {_escape_html(current_kb)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>检索范围: {_escape_html(file_scope)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>记忆模式: {_escape_html(memory_mode_text)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>回答方式: {_escape_html(answer_mode_text)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>说明: {_escape_html(memory_help)}</div>"
                "</div>"
            )
            return
        summary = session_summaries_by_id.get(session_id)
        if not summary:
            session_summary_html.value = (
                "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;'>"
                f"<div><b>当前会话</b></div>"
                f"<div style='margin-top:4px;color:#5b6472;'>当前知识库: {_escape_html(current_kb)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>检索范围: {_escape_html(file_scope)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>记忆模式: {_escape_html(memory_mode_text)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>回答方式: {_escape_html(answer_mode_text)}</div>"
                f"<div style='margin-top:4px;color:#5b6472;'>说明: {_escape_html(memory_help)}</div>"
                "</div>"
            )
            return
        title = _escape_html(str(summary.get("title", session_id)))
        turns = int(summary.get("turn_count", 0))
        memory_mode = _escape_html(_translate_memory_mode(str(summary.get("memory_mode", memory_mode_toggle.value))))
        last_updated = summary.get("last_updated")
        last_text = _escape_html(str(last_updated).replace("T", " ")[:19]) if last_updated else "-"
        prompt_tokens = int(summary.get("last_prompt_token_estimate", 0) or 0)
        context_doc_count = int(summary.get("last_context_doc_count", 0) or 0)
        context_compressed = bool(summary.get("last_context_compressed", False))
        compression_text = "已触发" if context_compressed else "未触发"
        candidate_doc_count = int(summary.get("last_candidate_doc_count", 0) or 0)
        rerank_kept_count = int(summary.get("last_rerank_kept_count", 0) or 0)
        rerank_filtered_count = int(summary.get("last_rerank_filtered_count", 0) or 0)
        low_score_filtered = bool(summary.get("last_low_score_filtered", False))
        low_score_text = "已触发" if low_score_filtered else "未触发"
        strategy_names = [str(item).strip() for item in summary.get("last_context_strategies", []) if str(item).strip()]
        strategy_text = "、".join(strategy_names) if strategy_names else "未触发滑窗 / 递归摘要 / 历史压缩"
        session_summary_html.value = (
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fafbfc;'>"
            f"<div><b>{title}</b></div>"
            f"<div style='margin-top:4px;color:#5b6472;'>消息数: {turns} | 模式: {memory_mode}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>当前知识库: {_escape_html(current_kb)}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>检索范围: {_escape_html(file_scope)}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>回答方式: {_escape_html(answer_mode_text)}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>最近一次上下文估算: {prompt_tokens} tokens</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>候选切片数: {candidate_doc_count}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>评分后保留: {rerank_kept_count}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>低分丢弃数: {rerank_filtered_count}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>低分过滤: {low_score_text}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>最近一次参考片段数: {context_doc_count}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>上下文压缩: {compression_text}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>本次策略: {_escape_html(strategy_text)}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>最近更新: {last_text}</div>"
            f"<div style='margin-top:4px;color:#5b6472;'>说明: {_escape_html(memory_help)}</div>"
            "</div>"
        )

    async def refresh_all_views(course_id: str | None = None):
        await refresh_knowledge_base_options(preferred_course_id=course_id)
        current_manage = course_id if course_id is not None else manage_kb_dropdown.value
        current_chat = course_id if course_id is not None else chat_kb_dropdown.value
        current_analysis = course_id if course_id is not None else analysis_kb_dropdown.value
        await refresh_manageable_file_options(current_manage)
        await refresh_document_options(current_chat, chat_doc_selector, empty_hint="该知识库下暂无文件。")
        await refresh_document_options(current_analysis, doc_selector, empty_hint="该知识库下暂无文件。")
        await refresh_session_options()
        await refresh_knowledge_base_status(current_manage)

    def _resolve_import_bundle_path() -> Path:
        uploads = _normalize_upload_value(migration_bundle_upload.value)
        if uploads:
            item = uploads[0]
            bundle_name = Path(str(item.get("name", "migration_bundle.zip"))).name
            destination_dir = config.project_root / "release" / "migration_uploads"
            destination_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = destination_dir / f"{timestamp}_{bundle_name}"
            content = item.get("content", b"")
            if isinstance(content, memoryview):
                content = content.tobytes()
            destination.write_bytes(content if isinstance(content, bytes) else bytes(content))
            return destination
        raw_path = migration_bundle_path_input.value.strip()
        if not raw_path:
            raise ValueError("请先上传迁移包，或填写工作区中的 zip 路径。")
        return _resolve_workspace_bundle_path(config.project_root, raw_path)

    def _render_migration_status(title: str, result: dict[str, object], *, note: str = "") -> str:
        roots = "、".join(str(item) for item in result.get("roots", []) if str(item).strip()) or "-"
        bundle_path = str(result.get("bundle_path", ""))
        backup_path = str(result.get("backup_path", ""))
        file_count = int(result.get("file_count", 0) or 0)
        total_size = _human_readable_bytes(int(result.get("total_size_bytes", 0) or 0))
        lines = [
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#f6f8fa;'>",
            f"<b>{_escape_html(title)}</b><br>",
            f"包路径: {_escape_html(bundle_path)}<br>",
            f"包含目录: {_escape_html(roots)}<br>",
            f"文件数: {file_count} | 总大小: {_escape_html(total_size)}",
        ]
        if backup_path:
            lines.append(f"<br>导入前备份: {_escape_html(backup_path)}")
        if note:
            lines.append(
                "<br><span style='color:#5b6472;'>"
                f"{_escape_html(note)}"
                "</span>"
            )
        lines.append("</div>")
        return "".join(lines)

    def _render_migration_progress(title: str, detail: str = "") -> str:
        detail_html = (
            "<div style='margin-top:6px;color:#5b6472;'>"
            f"{_escape_html(detail)}"
            "</div>"
            if detail
            else ""
        )
        return (
            "<style>"
            "@keyframes codex-progress-slide {"
            "0% { transform: translateX(-70%); }"
            "50% { transform: translateX(80%); }"
            "100% { transform: translateX(220%); }"
            "}"
            "</style>"
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#f6f8fa;'>"
            f"<b>{_escape_html(title)}</b>"
            f"{detail_html}"
            "<div style='margin-top:10px;height:8px;border-radius:999px;background:#e5e7eb;overflow:hidden;'>"
            "<div style='width:45%;height:100%;border-radius:999px;background:#2563eb;"
            "animation:codex-progress-slide 1.2s ease-in-out infinite;'></div>"
            "</div>"
            "</div>"
        )

    def _set_migration_section_visibility(enabled: bool) -> None:
        migration_section.layout.display = "flex" if enabled else "none"

    async def export_project_bundle(_):
        export_migration_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                migration_status_html.value = _render_migration_progress(
                    "正在导出迁移包",
                    "正在收集配置、运行数据和知识库文件，请稍候。",
                )
                print("正在打包迁移数据...")
                await asyncio.sleep(0.05)
                result = await asyncio.to_thread(
                    export_migration_bundle,
                    config,
                    bundle_name=migration_export_name_input.value.strip(),
                )
                print(f"迁移包已生成: {result['bundle_path']}")
                print(f"包含目录: {', '.join(str(item) for item in result.get('roots', []))}")
                print(f"文件数: {result.get('file_count', 0)}")
                print(f"总大小: {_human_readable_bytes(int(result.get('total_size_bytes', 0) or 0))}")
                print("说明: 已排除 logs 目录及运行日志文件。")
                migration_status_html.value = _render_migration_status("迁移包导出完成", result)
            except Exception:
                print("导出迁移包失败:")
                print(traceback.format_exc())
                migration_status_html.value = (
                    "<b>导出失败:</b><pre>"
                    f"{_escape_html(traceback.format_exc())}"
                    "</pre>"
                )
            finally:
                export_migration_button.disabled = False

    async def import_project_bundle(_):
        import_migration_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                if not migration_confirm_checkbox.value:
                    print("导入会覆盖当前配置和运行数据。请先勾选确认框。")
                    return
                bundle_path = _resolve_import_bundle_path()
                migration_status_html.value = _render_migration_progress(
                    "正在校验迁移包",
                    "正在检查 zip 文件和迁移清单。",
                )
                print(f"准备读取迁移包: {bundle_path}")
                await asyncio.sleep(0.05)
                manifest = await asyncio.to_thread(inspect_migration_bundle, bundle_path)
                print(f"准备导入迁移包: {bundle_path}")
                print(f"打包时间: {manifest.get('created_at', '-')}")
                print(f"包含目录: {', '.join(str(item) for item in manifest.get('roots', []))}")
                print(f"文件数: {manifest.get('file_count', len(manifest.get('files', [])))}")
                migration_status_html.value = _render_migration_progress(
                    "正在导入迁移包",
                    "正在自动备份当前数据，并恢复迁移包内容。大文件较多时会花一些时间。",
                )
                print("正在自动备份当前数据并导入迁移包...")
                await asyncio.sleep(0.05)
                result = await asyncio.to_thread(import_migration_bundle, config, bundle_path)
                print("导入完成。")
                if result.get("backup_path"):
                    print(f"已自动备份当前数据: {result['backup_path']}")
                print("注意: 当前运行中的服务仍可能持有旧连接。建议立即重启 Notebook / Kernel 后再继续使用。")
                migration_status_html.value = _render_migration_status(
                    "迁移包导入完成",
                    result,
                    note="为了确保配置、向量索引和会话数据库全部切换到新数据，建议现在重启当前 Notebook / Kernel。",
                )
                migration_confirm_checkbox.value = False
                migration_bundle_path_input.value = ""
                migration_bundle_upload.value = ()
                migration_bundle_upload.error = ""
            except Exception:
                print("导入迁移包失败:")
                print(traceback.format_exc())
                migration_status_html.value = (
                    "<b>导入失败:</b><pre>"
                    f"{_escape_html(traceback.format_exc())}"
                    "</pre>"
                )
            finally:
                import_migration_button.disabled = False

    async def initialize_ui():
        _refresh_field_template_options()
        _update_stop_buttons()
        _set_migration_section_visibility(enable_migration_ui_checkbox.value)
        _update_history_pagination_ui(
            total_turns=0,
            total_pages=1,
            current_page=1,
            page_size=max(1, int(history_page_size_dropdown.value)),
        )
        await refresh_all_views()

    async def ingest_files(_):
        ingest_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                if not upload_widget.value:
                    print("请先上传文件。")
                    return
                chunk_settings_error = _validate_chunk_settings(
                    chunk_size=chunk_size_input.value,
                    chunk_overlap=chunk_overlap_input.value,
                    merge_small_chunks=merge_small_chunks_checkbox.value,
                    min_chunk_size=min_chunk_size_input.value,
                )
                if chunk_settings_error:
                    print(chunk_settings_error)
                    return
                course_id = target_kb_dropdown.value.strip()
                if not course_id:
                    print("请先选择上传目标知识库。")
                    return
                if rebuild_checkbox.value and not file_action_confirm_checkbox.value:
                    print("当前勾选了“重建索引”。请先勾选敏感操作确认，再继续。")
                    return
                selected_uploads = _normalize_upload_value(upload_widget.value)
                print(f"准备导入 {len(selected_uploads)} 个文件到知识库“{course_id}”。")
                print(f"运行日志: {_active_log_path()}")
                print(f"向量日志: {config.vector_operation_log_path}")
                file_paths = []
                source_type = source_type_dropdown.value
                destination_dir = config.data_root / course_id / f"{source_type}s"
                destination_dir.mkdir(parents=True, exist_ok=True)
                for item in selected_uploads:
                    destination = destination_dir / item["name"]
                    content = item["content"]
                    if isinstance(content, memoryview):
                        content = content.tobytes()
                    destination.write_bytes(content)
                    file_paths.append(destination)
                    print(f"已保存: {destination}")
                result = await manager.add_files(
                    course_id=course_id,
                    file_paths=file_paths,
                    source_type=source_type,
                    rebuild_index=rebuild_checkbox.value,
                    chunk_size=chunk_size_input.value,
                    chunk_overlap=chunk_overlap_input.value,
                    merge_small_chunks=merge_small_chunks_checkbox.value,
                    min_chunk_size=min_chunk_size_input.value,
                    progress_callback=append_manage_message,
                )
                print(f"导入完成: {result}")
                print(f"当前检索后端: {vector_store.backend_mode}")
                print(f"运行日志: {_active_log_path()}")
                print(f"切片参数: chunk_size={chunk_size_input.value}, chunk_overlap={chunk_overlap_input.value}")
                print(
                    f"智能合并: {'开启' if merge_small_chunks_checkbox.value else '关闭'}, "
                    f"min_chunk_size={min_chunk_size_input.value}"
                )
                print(f"本次实际接收文件数: {len(selected_uploads)}")
                clear_manage_detail_cache()
                await refresh_all_views(course_id=course_id)
                upload_widget.value = ()
                upload_widget.error = ""
                file_action_confirm_checkbox.value = False
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("导入失败:")
                print(traceback.format_exc())
                print(f"运行日志: {_active_log_path()}")
                print(f"向量日志: {config.vector_operation_log_path}")
            finally:
                ingest_button.disabled = not bool(upload_widget.value)

    async def scan_workspace_files(_):
        scan_workspace_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                directory = _resolve_workspace_dir(import_dir_input.value)
                if not directory.exists():
                    print(f"目录不存在: {directory}")
                    workspace_file_selector.options = []
                    workspace_file_selector.value = ()
                    return
                files = _scan_supported_files(directory)
                workspace_file_selector.options = [(str(path.relative_to(Path.cwd())), str(path)) for path in files]
                workspace_file_selector.value = ()
                print(f"扫描完成，找到 {len(files)} 个可导入文件。")
            except Exception:
                print("扫描目录失败:")
                print(traceback.format_exc())
            finally:
                scan_workspace_button.disabled = False

    async def import_workspace_files(_):
        import_workspace_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                if not target_kb_dropdown.value:
                    print("请先选择上传目标知识库。")
                    return
                if not workspace_file_selector.value:
                    print("请先选择至少一个工作区文件。")
                    return
                chunk_settings_error = _validate_chunk_settings(
                    chunk_size=chunk_size_input.value,
                    chunk_overlap=chunk_overlap_input.value,
                    merge_small_chunks=merge_small_chunks_checkbox.value,
                    min_chunk_size=min_chunk_size_input.value,
                )
                if chunk_settings_error:
                    print(chunk_settings_error)
                    return
                course_id = target_kb_dropdown.value.strip()
                source_type = source_type_dropdown.value
                destination_dir = config.data_root / course_id / f"{source_type}s"
                destination_dir.mkdir(parents=True, exist_ok=True)
                copied_paths: list[Path] = []
                for raw_path in workspace_file_selector.value:
                    source_path = Path(raw_path)
                    if not source_path.exists():
                        print(f"跳过不存在文件: {source_path}")
                        continue
                    destination = _unique_destination(destination_dir / source_path.name)
                    shutil.copy2(source_path, destination)
                    copied_paths.append(destination)
                    print(f"已导入: {source_path} -> {destination}")
                if not copied_paths:
                    print("没有成功导入任何文件。")
                    return
                result = await manager.add_files(
                    course_id=course_id,
                    file_paths=copied_paths,
                    source_type=source_type,
                    rebuild_index=rebuild_checkbox.value,
                    chunk_size=chunk_size_input.value,
                    chunk_overlap=chunk_overlap_input.value,
                    merge_small_chunks=merge_small_chunks_checkbox.value,
                    min_chunk_size=min_chunk_size_input.value,
                    progress_callback=append_manage_message,
                )
                print(f"导入完成: {result}")
                print(f"当前检索后端: {vector_store.backend_mode}")
                print(f"运行日志: {_active_log_path()}")
                print(
                    f"智能合并: {'开启' if merge_small_chunks_checkbox.value else '关闭'}, "
                    f"min_chunk_size={min_chunk_size_input.value}"
                )
                clear_manage_detail_cache()
                await refresh_all_views(course_id=course_id)
                file_action_confirm_checkbox.value = False
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("从工作区导入失败:")
                print(traceback.format_exc())
            finally:
                import_workspace_button.disabled = False

    async def create_knowledge_base(_):
        create_kb_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                course_id = await manager.create_knowledge_base(create_kb_input.value)
                print(f"知识库已创建: {course_id}")
                create_kb_input.value = ""
                await refresh_all_views(course_id=course_id)
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("新建知识库失败:")
                print(traceback.format_exc())
            finally:
                create_kb_button.disabled = False

    async def delete_selected_files(_):
        delete_files_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                selected_file_paths = list(_selected_manage_file_paths())
                if not manage_kb_dropdown.value:
                    print("请先选择知识库。")
                    return
                if not selected_file_paths:
                    print("请先选择至少一个文件。")
                    return
                if not file_action_confirm_checkbox.value:
                    print("删除文件前，请先勾选敏感操作确认。删除原始文件后无法恢复。")
                    return
                result = await manager.delete_file_paths(
                    course_id=manage_kb_dropdown.value,
                    file_paths=selected_file_paths,
                )
                print(f"删除完成: {result}")
                manage_file_selected_paths.clear()
                _render_manage_file_selector_page()
                clear_manage_detail_cache()
                await refresh_all_views(course_id=manage_kb_dropdown.value)
                file_action_confirm_checkbox.value = False
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("删除文件失败:")
                print(traceback.format_exc())
            finally:
                delete_files_button.disabled = False

    async def rename_selected_file(_):
        rename_file_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                selected_file_paths = list(_selected_manage_file_paths())
                if not manage_kb_dropdown.value:
                    print("请先选择知识库。")
                    return
                if len(selected_file_paths) != 1:
                    print("请只选择一个文件进行重命名。")
                    return
                record = await manager.rename_file_by_path(
                    course_id=manage_kb_dropdown.value,
                    file_path=selected_file_paths[0],
                    new_file_name=rename_file_input.value,
                )
                print(f"文件已重命名为: {record.file_name}")
                rename_file_input.value = record.file_name
                clear_manage_detail_cache()
                await refresh_all_views(course_id=manage_kb_dropdown.value)
                manage_file_selected_paths.clear()
                manage_file_selected_paths.add(record.file_path)
                _render_manage_file_selector_page()
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("修改文件名失败:")
                print(traceback.format_exc())
            finally:
                rename_file_button.disabled = False

    async def rename_selected_kb(_):
        rename_kb_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                old_course_id = manage_kb_dropdown.value.strip()
                new_course_id = rename_kb_input.value.strip()
                if not old_course_id:
                    print("请先选择知识库。")
                    return
                result = await manager.rename_knowledge_base(old_course_id=old_course_id, new_course_id=new_course_id)
                print(f"知识库已重命名: {old_course_id} -> {new_course_id}")
                print(f"更新 chunk 数: {result['updated_chunks']}")
                clear_manage_detail_cache()
                await refresh_all_views(course_id=new_course_id)
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("修改知识库名失败:")
                print(traceback.format_exc())
            finally:
                rename_kb_button.disabled = False

    async def view_selected_chunks(_):
        view_chunks_button.disabled = True
        try:
            await refresh_manage_details_preview(manage_kb_dropdown.value)
        finally:
            view_chunks_button.disabled = False

    async def vectorize_selected_files(_):
        vectorize_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                selected_file_paths = list(_selected_manage_file_paths())
                if not manage_kb_dropdown.value:
                    print("请先选择知识库。")
                    return
                if not selected_file_paths:
                    print("请先选择至少一个文件。")
                    return
                if rebuild_checkbox.value and not file_action_confirm_checkbox.value:
                    print("当前勾选了“重建索引”。请先勾选敏感操作确认，再继续。")
                    return
                chunk_settings_error = _validate_chunk_settings(
                    chunk_size=chunk_size_input.value,
                    chunk_overlap=chunk_overlap_input.value,
                    merge_small_chunks=merge_small_chunks_checkbox.value,
                    min_chunk_size=min_chunk_size_input.value,
                )
                if chunk_settings_error:
                    print(chunk_settings_error)
                    return
                result = await manager.vectorize_files(
                    course_id=manage_kb_dropdown.value,
                    file_paths=selected_file_paths,
                    chunk_size=chunk_size_input.value,
                    chunk_overlap=chunk_overlap_input.value,
                    merge_small_chunks=merge_small_chunks_checkbox.value,
                    min_chunk_size=min_chunk_size_input.value,
                    progress_callback=append_manage_message,
                )
                print(f"向量化完成: {result}")
                print(f"运行日志: {_active_log_path()}")
                print(
                    f"智能合并: {'开启' if merge_small_chunks_checkbox.value else '关闭'}, "
                    f"min_chunk_size={min_chunk_size_input.value}"
                )
                clear_manage_detail_cache()
                await refresh_all_views(course_id=manage_kb_dropdown.value)
                file_action_confirm_checkbox.value = False
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("向量化处理失败:")
                print(traceback.format_exc())
            finally:
                vectorize_button.disabled = False

    async def move_selected_files(_):
        move_files_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                selected_file_paths = list(_selected_manage_file_paths())
                if not manage_kb_dropdown.value:
                    print("请先选择当前知识库。")
                    return
                if not selected_file_paths:
                    print("请先选择至少一个文件。")
                    return
                if not move_target_kb_dropdown.value:
                    print("请先选择目标知识库。")
                    return
                if not file_action_confirm_checkbox.value:
                    print("移动文件前，请先勾选敏感操作确认。若中途失败，系统会自动回滚已完成的移动。")
                    return
                result = await manager.move_file_paths(
                    source_course_id=manage_kb_dropdown.value,
                    file_paths=selected_file_paths,
                    target_course_id=move_target_kb_dropdown.value,
                    target_source_type=move_target_type_dropdown.value,
                )
                print(f"文件移动完成: {result}")
                print("若移动过程中失败，系统会自动回滚已完成的文件移动和向量元数据更新。")
                manage_file_selected_paths.clear()
                _render_manage_file_selector_page()
                clear_manage_detail_cache()
                await refresh_all_views(course_id=manage_kb_dropdown.value)
                file_action_confirm_checkbox.value = False
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("移动文件失败:")
                print(traceback.format_exc())
            finally:
                move_files_button.disabled = False

    def _active_memory_store():
        return sqlite_store if memory_mode_toggle.value == "persistent" else session_store

    def _update_history_pagination_ui(*, total_turns: int, total_pages: int, current_page: int, page_size: int) -> None:
        history_pagination_state["total_turns"] = int(total_turns)
        history_pagination_state["total_pages"] = int(total_pages)
        history_pagination_state["current_page"] = int(current_page)
        history_pagination_state["page_size"] = int(page_size)
        history_page_size_dropdown.value = int(page_size)
        history_page_input.max = max(1, int(total_pages))
        history_page_input.value = max(1, int(current_page))
        history_first_button.disabled = current_page <= 1
        history_prev_button.disabled = current_page <= 1
        history_next_button.disabled = current_page >= total_pages
        history_last_button.disabled = current_page >= total_pages
        history_go_button.disabled = total_turns <= 0
        history_page_info.value = (
            f"<div style='color:#5b6472;'>第 <b>{current_page}</b> / <b>{total_pages}</b> 页，"
            f"共 <b>{total_turns}</b> 条消息，每页 {page_size} 条。</div>"
        )

    async def _read_chat_history_page(
        session_id: str,
        *,
        target_page: int | None = None,
        anchor: str = "latest",
    ) -> tuple[list[ChatTurn], int, int, int]:
        store = _active_memory_store()
        page_size = max(1, int(history_page_size_dropdown.value))
        total_turns = await store.get_turn_count(session_id)
        total_pages = max(1, ceil(total_turns / page_size))
        if target_page is not None:
            page = max(1, min(int(target_page), total_pages))
        elif anchor == "oldest":
            page = 1
        elif anchor == "current":
            page = max(1, min(int(history_pagination_state["current_page"]), total_pages))
        else:
            page = total_pages
        offset = (page - 1) * page_size
        turns = await store.get_turns_page(session_id=session_id, limit=page_size, offset=offset)
        return turns, total_turns, total_pages, page

    async def load_chat_history(
        _=None,
        *,
        target_page: int | None = None,
        anchor: str = "latest",
        announce: bool = True,
    ):
        session_id = session_id_input.value.strip()
        if not session_id:
            chat_status.value = "<b>失败:</b> 请先在左侧选择会话或新建会话。"
            return
        if announce:
            await append_chat_progress("正在加载历史消息...")
        turns, total_turns, total_pages, current_page = await _read_chat_history_page(
            session_id,
            target_page=target_page,
            anchor=anchor,
        )
        _update_history_pagination_ui(
            total_turns=total_turns,
            total_pages=total_pages,
            current_page=current_page,
            page_size=max(1, int(history_page_size_dropdown.value)),
        )
        chat_history_output.value = _render_chat_history(turns)
        if total_turns <= 0:
            chat_status.value = "<b>当前会话暂无历史消息</b>"
            if announce:
                await append_chat_progress("当前会话暂无历史消息。")
        else:
            chat_status.value = (
                f"<b>已加载第 {current_page}/{total_pages} 页，显示 {len(turns)} 条，共 {total_turns} 条历史消息</b>"
            )
            if announce:
                await append_chat_progress(
                    f"历史消息加载完成：第 {current_page}/{total_pages} 页，当前页 {len(turns)} 条，总计 {total_turns} 条。"
                )
        await refresh_session_options(preferred_session_id=session_id)

    async def create_new_session(_=None):
        new_id = new_session_id()
        session_id_input.value = new_id
        await _active_memory_store().set_session_profile(
            new_id,
            {
                "session_title": "新会话",
                "memory_mode": memory_mode_toggle.value,
            },
        )
        clear_chat_display()
        clear_chat_progress()
        await refresh_session_options(preferred_session_id=new_id)
        rename_session_input.value = "新会话"
        chat_status.value = "<b>已新建会话</b>"

    async def delete_selected_session(_=None):
        session_id = session_selector.value or session_id_input.value.strip()
        if not session_id:
            chat_status.value = "<b>失败:</b> 请先选择会话。"
            return
        if not delete_session_confirm_checkbox.value:
            chat_status.value = "<b>失败:</b> 删除会话前，请先勾选确认。"
            return
        await _active_memory_store().delete_session(session_id)
        clear_chat_display()
        clear_chat_progress()
        await refresh_session_options()
        if session_selector.value:
            session_id_input.value = str(session_selector.value)
            await load_chat_history()
        else:
            session_id_input.value = ""
            session_summary_html.value = "<i>当前没有会话。</i>"
        delete_session_confirm_checkbox.value = False
        chat_status.value = "<b>会话已删除</b>"

    async def rename_selected_session(_=None):
        session_id = session_selector.value or session_id_input.value.strip()
        title = rename_session_input.value.strip()
        if not session_id:
            chat_status.value = "<b>失败:</b> 请先选择会话。"
            return
        if not title:
            chat_status.value = "<b>失败:</b> 请输入新的会话标题。"
            return
        await manager.rename_session_title(_active_memory_store(), session_id, title)
        await refresh_session_options(preferred_session_id=session_id)
        chat_status.value = "<b>会话标题已更新</b>"

    def clear_chat_display(_=None):
        chat_history_output.value = _render_chat_history([])
        _update_history_pagination_ui(total_turns=0, total_pages=1, current_page=1, page_size=max(1, int(history_page_size_dropdown.value)))
        citation_output.value = "<div>引用将在这里显示。</div>"
        chat_status.value = ""
        clear_chat_progress()

    async def save_runtime_settings(_):
        save_settings_button.disabled = True
        try:
            chunk_settings_error = _validate_chunk_settings(
                chunk_size=chunk_size_input.value,
                chunk_overlap=chunk_overlap_input.value,
                merge_small_chunks=merge_small_chunks_checkbox.value,
                min_chunk_size=min_chunk_size_input.value,
            )
            if chunk_settings_error:
                settings_status.value = f"<b>失败:</b> {_escape_html(chunk_settings_error)}"
                return
            if retrieval_fetch_k_input.value < retrieval_top_k_input.value:
                settings_status.value = "<b>失败:</b> 候选数量必须大于或等于返回数量。"
                return
            generation_settings_error = _validate_generation_settings(
                model_context_window=model_context_window_input.value,
                answer_token_reserve=answer_token_reserve_input.value,
            )
            if generation_settings_error:
                settings_status.value = f"<b>失败:</b> {_escape_html(generation_settings_error)}"
                return
            context_strategy_error = _validate_context_strategy_settings(
                window_tokens=long_context_window_tokens_input.value,
                overlap_tokens=long_context_window_overlap_tokens_input.value,
                summary_target_tokens=recursive_summary_target_tokens_input.value,
                summary_batch_size=recursive_summary_batch_size_input.value,
                prompt_compression_turn_token_limit=prompt_compression_turn_token_limit_input.value,
            )
            if context_strategy_error:
                settings_status.value = f"<b>失败:</b> {_escape_html(context_strategy_error)}"
                return
            rerank_error = _validate_rerank_settings(
                min_score=rerank_min_score_input.value,
                min_keep=rerank_min_keep_input.value,
                weight_vector=rerank_weight_vector_input.value,
                weight_keyword=rerank_weight_keyword_input.value,
                weight_phrase=rerank_weight_phrase_input.value,
                weight_metadata=rerank_weight_metadata_input.value,
            )
            if rerank_error:
                settings_status.value = f"<b>失败:</b> {_escape_html(rerank_error)}"
                return
            config_path = update_json_config_file(
                config.config_path,
                {
                    "APP_DEFAULT_LANGUAGE": language_toggle.value,
                    "APP_DEFAULT_MEMORY_MODE": memory_mode_toggle.value,
                    "APP_DEFAULT_STREAMING_MODE": streaming_mode_toggle.value,
                    "APP_MODEL_CONTEXT_WINDOW": model_context_window_input.value,
                    "APP_ANSWER_TOKEN_RESERVE": answer_token_reserve_input.value,
                    "APP_LONG_CONTEXT_WINDOW_TOKENS": long_context_window_tokens_input.value,
                    "APP_LONG_CONTEXT_WINDOW_OVERLAP_TOKENS": long_context_window_overlap_tokens_input.value,
                    "APP_RECURSIVE_SUMMARY_TARGET_TOKENS": recursive_summary_target_tokens_input.value,
                    "APP_RECURSIVE_SUMMARY_BATCH_SIZE": recursive_summary_batch_size_input.value,
                    "APP_PROMPT_COMPRESSION_TURN_TOKEN_LIMIT": prompt_compression_turn_token_limit_input.value,
                    "APP_RECENT_HISTORY_TURNS": recent_history_turns_input.value,
                    "APP_RETRIEVAL_TOP_K": retrieval_top_k_input.value,
                    "APP_RETRIEVAL_FETCH_K": retrieval_fetch_k_input.value,
                    "APP_CITATION_LIMIT": citation_limit_input.value,
                    "APP_ENABLE_RERANK": enable_rerank_checkbox.value,
                    "APP_RERANK_MIN_SCORE": rerank_min_score_input.value,
                    "APP_RERANK_MIN_KEEP": rerank_min_keep_input.value,
                    "APP_RERANK_WEIGHT_VECTOR": rerank_weight_vector_input.value,
                    "APP_RERANK_WEIGHT_KEYWORD": rerank_weight_keyword_input.value,
                    "APP_RERANK_WEIGHT_PHRASE": rerank_weight_phrase_input.value,
                    "APP_RERANK_WEIGHT_METADATA": rerank_weight_metadata_input.value,
                    "APP_ENABLE_QUERY_REWRITE": enable_rewrite_checkbox.value,
                    "APP_ENABLE_MIGRATION_UI": enable_migration_ui_checkbox.value,
                    "APP_QA_SYSTEM_PROMPT_ZH": qa_system_prompt_zh_input.value,
                    "APP_QUERY_REWRITE_INSTRUCTION_ZH": rewrite_prompt_zh_input.value,
                    "APP_QA_ANSWER_INSTRUCTION_ZH": answer_instruction_zh_input.value,
                    "APP_QA_SYSTEM_PROMPT_EN": qa_system_prompt_en_input.value,
                    "APP_QUERY_REWRITE_INSTRUCTION_EN": rewrite_prompt_en_input.value,
                    "APP_QA_ANSWER_INSTRUCTION_EN": answer_instruction_en_input.value,
                    "APP_SINGLE_ANALYSIS_PROMPT_ZH": single_analysis_prompt_zh_input.value,
                    "APP_COMPARE_REPORT_PROMPT_ZH": compare_report_prompt_zh_input.value,
                    "APP_DATA_EXTRACTION_PROMPT_ZH": data_extraction_prompt_zh_input.value,
                    "APP_TABLE_SUMMARY_PROMPT_ZH": table_summary_prompt_zh_input.value,
                    "APP_SINGLE_ANALYSIS_PROMPT_EN": single_analysis_prompt_en_input.value,
                    "APP_COMPARE_REPORT_PROMPT_EN": compare_report_prompt_en_input.value,
                    "APP_DATA_EXTRACTION_PROMPT_EN": data_extraction_prompt_en_input.value,
                    "APP_TABLE_SUMMARY_PROMPT_EN": table_summary_prompt_en_input.value,
                    "APP_CHUNK_SIZE": chunk_size_input.value,
                    "APP_CHUNK_OVERLAP": chunk_overlap_input.value,
                    "APP_MERGE_SMALL_CHUNKS": merge_small_chunks_checkbox.value,
                    "APP_MIN_CHUNK_SIZE": min_chunk_size_input.value,
                },
            )
            config.default_language = language_toggle.value
            config.default_memory_mode = memory_mode_toggle.value
            config.default_streaming_mode = streaming_mode_toggle.value
            config.model_context_window = model_context_window_input.value
            config.answer_token_reserve = answer_token_reserve_input.value
            config.long_context_window_tokens = long_context_window_tokens_input.value
            config.long_context_window_overlap_tokens = long_context_window_overlap_tokens_input.value
            config.recursive_summary_target_tokens = recursive_summary_target_tokens_input.value
            config.recursive_summary_batch_size = recursive_summary_batch_size_input.value
            config.prompt_compression_turn_token_limit = prompt_compression_turn_token_limit_input.value
            config.recent_history_turns = recent_history_turns_input.value
            config.retrieval_top_k = retrieval_top_k_input.value
            config.retrieval_fetch_k = retrieval_fetch_k_input.value
            config.citation_limit = citation_limit_input.value
            config.enable_rerank = enable_rerank_checkbox.value
            config.rerank_min_score = rerank_min_score_input.value
            config.rerank_min_keep = rerank_min_keep_input.value
            config.rerank_weight_vector = rerank_weight_vector_input.value
            config.rerank_weight_keyword = rerank_weight_keyword_input.value
            config.rerank_weight_phrase = rerank_weight_phrase_input.value
            config.rerank_weight_metadata = rerank_weight_metadata_input.value
            config.enable_query_rewrite = enable_rewrite_checkbox.value
            config.enable_migration_ui = enable_migration_ui_checkbox.value
            config.qa_system_prompt_zh = qa_system_prompt_zh_input.value
            config.query_rewrite_instruction_zh = rewrite_prompt_zh_input.value
            config.qa_answer_instruction_zh = answer_instruction_zh_input.value
            config.qa_system_prompt_en = qa_system_prompt_en_input.value
            config.query_rewrite_instruction_en = rewrite_prompt_en_input.value
            config.qa_answer_instruction_en = answer_instruction_en_input.value
            config.single_analysis_prompt_zh = single_analysis_prompt_zh_input.value
            config.compare_report_prompt_zh = compare_report_prompt_zh_input.value
            config.data_extraction_prompt_zh = data_extraction_prompt_zh_input.value
            config.table_summary_prompt_zh = table_summary_prompt_zh_input.value
            config.single_analysis_prompt_en = single_analysis_prompt_en_input.value
            config.compare_report_prompt_en = compare_report_prompt_en_input.value
            config.data_extraction_prompt_en = data_extraction_prompt_en_input.value
            config.table_summary_prompt_en = table_summary_prompt_en_input.value
            config.chunk_size = chunk_size_input.value
            config.chunk_overlap = chunk_overlap_input.value
            config.merge_small_chunks = merge_small_chunks_checkbox.value
            config.min_chunk_size = min_chunk_size_input.value
            _set_migration_section_visibility(enable_migration_ui_checkbox.value)
            settings_status.value = f"<b>已保存到配置文件:</b> {_escape_html(str(config_path))}"
        except Exception:
            settings_status.value = f"<b>失败:</b><pre>{_escape_html(traceback.format_exc())}</pre>"
        finally:
            save_settings_button.disabled = False

    async def save_index_settings(_):
        save_index_settings_button.disabled = True
        try:
            chunk_settings_error = _validate_chunk_settings(
                chunk_size=chunk_size_input.value,
                chunk_overlap=chunk_overlap_input.value,
                merge_small_chunks=merge_small_chunks_checkbox.value,
                min_chunk_size=min_chunk_size_input.value,
            )
            if chunk_settings_error:
                index_settings_status.value = f"<b>失败:</b> {_escape_html(chunk_settings_error)}"
                return
            config_path = update_json_config_file(
                config.config_path,
                {
                    "APP_CHUNK_SIZE": chunk_size_input.value,
                    "APP_CHUNK_OVERLAP": chunk_overlap_input.value,
                    "APP_MERGE_SMALL_CHUNKS": merge_small_chunks_checkbox.value,
                    "APP_MIN_CHUNK_SIZE": min_chunk_size_input.value,
                },
            )
            config.chunk_size = chunk_size_input.value
            config.chunk_overlap = chunk_overlap_input.value
            config.merge_small_chunks = merge_small_chunks_checkbox.value
            config.min_chunk_size = min_chunk_size_input.value
            index_settings_status.value = f"<b>已保存到配置文件:</b> {_escape_html(str(config_path))}"
        except Exception:
            index_settings_status.value = f"<b>失败:</b><pre>{_escape_html(traceback.format_exc())}</pre>"
        finally:
            save_index_settings_button.disabled = False

    async def delete_selected_knowledge_base(_):
        delete_kb_button.disabled = True
        with manage_output:
            manage_output.clear_output()
            try:
                course_id = manage_kb_dropdown.value.strip()
                if not course_id:
                    print("请先选择知识库。")
                    return
                if not delete_kb_confirm_checkbox.value:
                    print("请先勾选“我确认删除当前知识库”。")
                    return
                result = await manager.delete_knowledge_base(course_id)
                delete_kb_confirm_checkbox.value = False
                manage_file_selected_paths.clear()
                _set_manage_file_options([])
                chat_doc_selector.options = []
                chat_doc_selector.value = ()
                doc_selector.options = []
                doc_selector.value = ()
                print(f"知识库已删除: {course_id}")
                print(f"删除原始文件数: {result['deleted_files']}")
                print(f"删除向量文档数: {result['deleted_vector_docs']}")
                print("提示：删除知识库不可回滚。若只是暂时不用，建议先导出或移动文件。")
                clear_manage_detail_cache()
                await refresh_all_views()
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("删除知识库失败:")
                print(traceback.format_exc())
            finally:
                _update_delete_kb_button_state()

    async def send_question(_):
        send_button.disabled = True
        citation_output.value = "<div></div>"
        chat_status.value = "<b>正在准备回答...</b>"
        clear_chat_progress()
        rendered_answer = ""
        rendered_citations = []
        question_text = question_area.value.strip()
        try:
            if not chat_kb_dropdown.value:
                chat_status.value = "<b>失败:</b> 请先选择知识库。"
                return
            if not session_id_input.value.strip():
                chat_status.value = "<b>失败:</b> 请先在左侧选择会话或新建会话。"
                return
            if not question_text:
                chat_status.value = "<b>失败:</b> 请输入问题。"
                return
            await append_chat_progress("开始处理当前问题...")
            await append_chat_progress(f"当前知识库: {chat_kb_dropdown.value.strip()}")
            await append_chat_progress(
                "当前检索范围: 整个知识库"
                if not chat_doc_selector.value
                else f"当前检索范围: 已选 {len(chat_doc_selector.value)} 个文件"
            )
            await append_chat_progress(f"运行日志: {_active_log_path()}")
            if retrieval_fetch_k_input.value < retrieval_top_k_input.value:
                chat_status.value = "<b>失败:</b> 候选数量必须大于或等于返回数量。"
                return
            generation_settings_error = _validate_generation_settings(
                model_context_window=model_context_window_input.value,
                answer_token_reserve=answer_token_reserve_input.value,
            )
            if generation_settings_error:
                chat_status.value = f"<b>失败:</b> {_escape_html(generation_settings_error)}"
                return
            rerank_error = _validate_rerank_settings(
                min_score=rerank_min_score_input.value,
                min_keep=rerank_min_keep_input.value,
                weight_vector=rerank_weight_vector_input.value,
                weight_keyword=rerank_weight_keyword_input.value,
                weight_phrase=rerank_weight_phrase_input.value,
                weight_metadata=rerank_weight_metadata_input.value,
            )
            if rerank_error:
                chat_status.value = f"<b>失败:</b> {_escape_html(rerank_error)}"
                return
            config.model_context_window = model_context_window_input.value
            config.answer_token_reserve = answer_token_reserve_input.value
            config.long_context_window_tokens = long_context_window_tokens_input.value
            config.long_context_window_overlap_tokens = long_context_window_overlap_tokens_input.value
            config.recursive_summary_target_tokens = recursive_summary_target_tokens_input.value
            config.recursive_summary_batch_size = recursive_summary_batch_size_input.value
            config.prompt_compression_turn_token_limit = prompt_compression_turn_token_limit_input.value
            config.recent_history_turns = recent_history_turns_input.value
            config.enable_rerank = enable_rerank_checkbox.value
            config.rerank_min_score = rerank_min_score_input.value
            config.rerank_min_keep = rerank_min_keep_input.value
            config.rerank_weight_vector = rerank_weight_vector_input.value
            config.rerank_weight_keyword = rerank_weight_keyword_input.value
            config.rerank_weight_phrase = rerank_weight_phrase_input.value
            config.rerank_weight_metadata = rerank_weight_metadata_input.value
            history_turns, total_turns, total_pages, current_page = await _read_chat_history_page(
                session_id_input.value.strip(),
                anchor="latest",
            )
            _update_history_pagination_ui(
                total_turns=total_turns,
                total_pages=total_pages,
                current_page=current_page,
                page_size=max(1, int(history_page_size_dropdown.value)),
            )
            display_turns = list(history_turns) + [
                ChatTurn(role="user", content=question_text, created_at=_now_utc()),
                ChatTurn(role="assistant", content="", created_at=_now_utc()),
            ]
            chat_history_output.value = _render_chat_history(display_turns, pending=True)
            async for event in rag_service.stream_answer(
                session_id=session_id_input.value.strip(),
                course_id=chat_kb_dropdown.value.strip(),
                question=question_text,
                memory_mode=memory_mode_toggle.value,
                language=language_toggle.value,
                doc_ids=list(chat_doc_selector.value) or None,
                enable_query_rewrite=enable_rewrite_checkbox.value,
                qa_system_prompt_override={
                    "zh": qa_system_prompt_zh_input.value,
                    "en": qa_system_prompt_en_input.value,
                },
                rewrite_instruction_override={
                    "zh": rewrite_prompt_zh_input.value,
                    "en": rewrite_prompt_en_input.value,
                },
                answer_instruction_override={
                    "zh": answer_instruction_zh_input.value,
                    "en": answer_instruction_en_input.value,
                },
                retrieval_top_k=retrieval_top_k_input.value,
                retrieval_fetch_k=retrieval_fetch_k_input.value,
                citation_limit=citation_limit_input.value,
                streaming_mode=streaming_mode_toggle.value,
            ):
                if event["type"] == "token":
                    rendered_answer += str(event["content"])
                    display_turns[-1] = ChatTurn(role="assistant", content=rendered_answer, created_at=display_turns[-1].created_at)
                    chat_history_output.value = _render_chat_history(display_turns, pending=True)
                elif event["type"] == "status":
                    chat_status.value = f"<b>{_escape_html(str(event['content']))}</b>"
                    await append_chat_progress(str(event["content"]))
                elif event["type"] == "restart_answer":
                    rendered_answer = ""
                    display_turns[-1] = ChatTurn(role="assistant", content="", created_at=display_turns[-1].created_at)
                    chat_history_output.value = _render_chat_history(display_turns, pending=True)
                    await append_chat_progress("检测到流式输出被限流中断，已清空当前回答并准备重试。")
                elif event["type"] == "citation":
                    rendered_citations.append(event["content"])
                    citation_output.value = _render_citation_details(rendered_citations, language_toggle.value)
                elif event["type"] == "done":
                    payload = event["content"]
                    display_turns[-1] = ChatTurn(role="assistant", content=str(payload["answer"]), created_at=display_turns[-1].created_at)
                    chat_history_output.value = _render_chat_history(display_turns, pending=False)
                    citation_output.value = _render_citation_details(
                        payload.get("citations", rendered_citations),
                        payload.get("language", language_toggle.value),
                    )
                    question_area.value = ""
                    await append_chat_progress("本轮回答已完成。")
                    await load_chat_history(anchor="latest", announce=False)
                    chat_status.value = "<b>回答已完成</b>"
                elif event["type"] == "error":
                    display_turns[-1] = ChatTurn(role="assistant", content=f"[Error] {event['content']}", created_at=display_turns[-1].created_at)
                    chat_history_output.value = _render_chat_history(display_turns, pending=False)
                    chat_status.value = f"<b>失败:</b> {_escape_html(str(event['content']))}"
                    await append_chat_progress(f"失败: {event['content']}")
        except OperationCancelledError as exc:
            chat_status.value = f"<b>已停止:</b> {_escape_html(exc.user_message)}"
            await append_chat_progress(exc.user_message)
        except Exception:
            chat_status.value = f"<b>失败:</b><pre>{_escape_html(traceback.format_exc())}</pre>"
            await append_chat_progress("失败: 运行时异常，请查看详细报错。")
        finally:
            send_button.disabled = False

    async def test_model_connection(_):
        test_model_button.disabled = True
        with model_test_output:
            model_test_output.clear_output()
            try:
                print("开始测试聊天模型连接...")
                result = await run_chat_healthcheck(config, timeout_seconds=20.0)
                print(f"base_url: {result.get('base_url', '')}")
                print(f"chat_model: {result.get('chat_model', '')}")
                if "message" in result:
                    print(result["message"])
                    return
                print("")
                print("[非流式]")
                _print_test_result(result["non_stream"])
                print("")
                print("[流式]")
                _print_test_result(result["stream"])
                print("")
                print("结论:", "通过" if result["ok"] else "失败")
            except OperationCancelledError as exc:
                print(exc.user_message)
            except Exception:
                print("测试模型连接失败:")
                print(traceback.format_exc())
            finally:
                test_model_button.disabled = False

    async def run_single_analysis(_):
        single_button.disabled = True
        report_progress_output.value = _render_analysis_progress_panel(
            title="单文档分析",
            summary="正在准备分析任务。",
            detail="开始后会持续显示当前文档、当前步骤和最近一次处理进度。",
        )
        clear_report_markdown_view()
        clear_report_log()
        reset_report_table_view()
        try:
            if not analysis_kb_dropdown.value:
                write_report_log("请先选择知识库。")
                return
            if not doc_selector.value:
                write_report_log("请选择文档。")
                return
            doc_id = list(doc_selector.value)[0]
            doc_title = analysis_doc_records_by_id.get(doc_id).file_name if doc_id in analysis_doc_records_by_id else doc_id
            write_report_log(f"开始分析文档: {doc_title}")
            write_report_log(f"运行日志: {_active_log_path()}")

            async def single_progress(message: str) -> None:
                compact_message = _shorten_report_message(message, limit=160)
                report_progress_output.value = _render_analysis_progress_panel(
                    title="单文档分析进行中",
                    summary=compact_message,
                    detail=f"当前文档: {doc_title}",
                )
                await append_report_message(message)

            analysis = await single_doc_service.analyze_document(
                course_id=analysis_kb_dropdown.value.strip(),
                doc_id=doc_id,
                output_language=language_toggle.value,
                progress_callback=single_progress,
            )
            analysis_markdown = _single_analysis_markdown(analysis)
            render_report_markdown(analysis_markdown)
            field_specs = collect_target_field_specs()
            if field_specs:
                extraction = await single_doc_service.extract_document_fields(
                    course_id=analysis_kb_dropdown.value.strip(),
                    doc_id=doc_id,
                    field_specs=field_specs,
                    output_language=language_toggle.value,
                    progress_callback=single_progress,
                )
                render_report_markdown(
                    analysis_markdown.rstrip()
                    + "\n\n"
                    + _single_doc_extraction_markdown(extraction).lstrip()
                )
                headers, rows = _single_extraction_table_payload(extraction)
                report_table_output.value = _render_report_table(headers, rows, title="字段抽取表")
            else:
                reset_report_table_view("<i>当前没有配置目标字段，所以这里只展示文字分析结果。</i>")
            report_progress_output.value = _render_analysis_progress_panel(
                title="单文档分析完成",
                summary=f"{doc_title} 已完成分析。",
                detail="正文已显示在上方 Markdown 区域。",
            )
        except OperationPausedError as exc:
            write_report_log(exc.user_message)
            report_progress_output.value = "<i>当前分析已暂停。</i>"
        except OperationCancelledError as exc:
            write_report_log(exc.user_message)
            report_progress_output.value = "<i>当前分析已停止。</i>"
        except AppServiceError as exc:
            write_report_log(exc.user_message)
        except Exception:
            write_report_log(traceback.format_exc())
        finally:
            single_button.disabled = False

    async def run_compare(_):
        await _run_compare(resume_mode=False)

    async def resume_compare(_):
        await _run_compare(resume_mode=True)

    async def _run_compare(*, resume_mode: bool):
        compare_button.disabled = True
        resume_compare_button.disabled = True
        report_progress_output.value = _render_analysis_progress_panel(
            title="批量对比",
            summary="正在准备对比任务。",
            detail="处理过程中会持续显示当前文档、当前步骤、滑窗进度和总体进度。",
        )
        clear_report_markdown_view()
        clear_report_log()
        reset_report_table_view()
        try:
            if not analysis_kb_dropdown.value:
                write_report_log("请先选择知识库。")
                return
            if not doc_selector.value:
                write_report_log("请选择至少一篇文档。")
                return
            field_specs = collect_target_field_specs()
            selected_doc_ids = list(doc_selector.value)
            checkpoint_info = await batch_service.inspect_compare_checkpoint(
                course_id=analysis_kb_dropdown.value.strip(),
                doc_ids=selected_doc_ids,
                output_language=language_toggle.value,
                target_fields=field_specs or None,
                export_csv=export_csv_checkbox.value,
            )
            if resume_mode and not checkpoint_info.get("exists"):
                write_report_log("当前选择没有可继续的批量对比进度。")
                report_progress_output.value = "<i>没有找到可继续的批量对比进度。</i>"
                return
            option_map = {value: label for label, value in doc_selector.options}
            selected_titles = [
                _simplify_doc_option_label(str(option_map.get(doc_id, doc_id)))
                for doc_id in selected_doc_ids
            ]
            if resume_mode:
                write_report_log(f"继续上次批量对比，共 {len(selected_doc_ids)} 篇文档。")
                write_report_log(
                    f"当前已完成单文档分析 {checkpoint_info.get('analysis_done', 0)}/{len(selected_doc_ids)}，"
                    f"已完成字段抽取 {checkpoint_info.get('extraction_done', 0)}/{len(selected_doc_ids)}。"
                )
                if checkpoint_info.get("progress_output_path"):
                    write_report_log(f"进度快照: {checkpoint_info.get('progress_output_path')}")
            else:
                write_report_log(f"开始批量对比，共 {len(selected_doc_ids)} 篇文档。")
            if selected_titles:
                preview = "、".join(selected_titles[:5])
                suffix = " 等" if len(selected_titles) > 5 else ""
                write_report_log(f"已选文档预览: {preview}{suffix}")
            progress_rows = _init_batch_progress_rows(selected_titles)
            progress_rows_by_index = {int(item["index"]): item for item in progress_rows}
            latest_summary = "正在准备对比任务。"
            overall_progress = "-"
            global_stage = "准备"
            global_detail = "等待批量流程启动。"

            def render_compare_progress_panel() -> None:
                report_progress_output.value = _render_batch_progress_board(
                    latest_summary=latest_summary,
                    overall_progress=overall_progress,
                    global_stage=global_stage,
                    global_detail=global_detail,
                    rows=progress_rows,
                )

            render_compare_progress_panel()
            if field_specs:
                write_report_log(f"本次还会额外抽取 {len(field_specs)} 个目标字段，并生成对比表和复查证据。")
            else:
                write_report_log("当前没有配置目标字段，这次会直接生成文字对比报告。")
            write_report_log("处理中会持续显示当前文档、当前步骤和总体进度，请耐心等待。")
            write_report_log(f"运行日志: {_active_log_path()}")

            async def compare_progress(message: str) -> None:
                nonlocal latest_summary, overall_progress, global_stage, global_detail
                compact_message = _shorten_report_message(message, limit=160)
                latest_summary = compact_message
                parsed = _parse_batch_progress_message(message)
                if parsed.get("kind") == "doc":
                    index = int(parsed.get("index", 0) or 0)
                    row = progress_rows_by_index.get(index)
                    if row is not None:
                        row["stage"] = str(parsed.get("stage", row.get("stage", "")))
                        row["status"] = str(parsed.get("stage_status", row.get("status", "")))
                        row["detail"] = str(parsed.get("detail", row.get("detail", "")))
                        row["progress"] = (
                            f"{int(parsed.get('completed', 0) or 0)}/{int(parsed.get('total_units', 0) or 0)}，"
                            f"约 {parsed.get('percent', '0%')}"
                        )
                    overall_progress = (
                        f"{int(parsed.get('completed', 0) or 0)}/{int(parsed.get('total_units', 0) or 0)}，"
                        f"约 {parsed.get('percent', '0%')}"
                    )
                elif parsed.get("kind") == "global":
                    global_stage = str(parsed.get("stage", global_stage))
                    global_detail = str(parsed.get("detail", global_detail))
                    overall_progress = (
                        f"{int(parsed.get('completed', 0) or 0)}/{int(parsed.get('total_units', 0) or 0)}，"
                        f"约 {parsed.get('percent', '0%')}"
                    )
                else:
                    global_detail = compact_message
                render_compare_progress_panel()
                await append_report_message(message)

            report = await batch_service.compare_documents(
                course_id=analysis_kb_dropdown.value.strip(),
                doc_ids=selected_doc_ids,
                output_language=language_toggle.value,
                target_fields=field_specs or None,
                export_csv=export_csv_checkbox.value,
                progress_callback=compare_progress,
                resume=resume_mode,
            )
            render_report_markdown(report.markdown)
            write_report_log(f"报告已保存到: {report.output_path}")
            if report.csv_output_path:
                write_report_log(f"CSV 已保存到: {report.csv_output_path}")
            report_table_output.value = _render_report_table(
                report.table_headers,
                report.table_rows,
                title="对比表格视图",
            )
            report_progress_output.value = _render_analysis_progress_panel(
                title="批量对比完成",
                summary=f"已完成 {len(selected_doc_ids)} 篇文档的批量对比。",
                detail="正文已显示在上方 Markdown 区域；结构化结果显示在下方表格视图。",
            )
            await refresh_analysis_checkpoint_status()
        except OperationPausedError as exc:
            write_report_log(exc.user_message)
            info = await batch_service.inspect_compare_checkpoint(
                course_id=analysis_kb_dropdown.value.strip(),
                doc_ids=selected_doc_ids,
                output_language=language_toggle.value,
                target_fields=field_specs or None,
                export_csv=export_csv_checkbox.value,
            )
            if info.get("progress_output_path"):
                write_report_log(f"进度快照: {info.get('progress_output_path')}")
            report_progress_output.value = "<i>当前分析已暂停。下次点击“继续上次批量对比”即可接着跑。</i>"
            await refresh_analysis_checkpoint_status()
        except OperationCancelledError as exc:
            write_report_log(exc.user_message)
            report_progress_output.value = "<i>当前分析已停止。</i>"
            await refresh_analysis_checkpoint_status()
        except AppServiceError as exc:
            write_report_log(exc.user_message)
            await refresh_analysis_checkpoint_status()
        except Exception:
            write_report_log(traceback.format_exc())
            await refresh_analysis_checkpoint_status()
        finally:
            compare_button.disabled = False
            resume_compare_button.disabled = False

    def stop_tracked_task(task_state: dict[str, object], *, message: str) -> None:
        control = task_state.get("control")
        task = task_state.get("task")
        if isinstance(control, TaskControl):
            control.cancel()
        if task is not None and not task.done():
            task.cancel()
        task_state["label"] = message
        if task_state is manage_task_state:
            with manage_output:
                print(message)
        elif task_state is chat_task_state:
            chat_status.value = f"<b>{_escape_html(message)}</b>"
            _schedule(append_chat_progress(message))
        elif task_state is analysis_task_state:
            write_report_log(message)
            report_progress_output.value = "<i>当前操作已停止。</i>"
        _update_stop_buttons()

    def _pause_analysis_task() -> None:
        control = analysis_task_state.get("control")
        if not isinstance(control, TaskControl):
            write_report_log("当前没有正在运行的分析任务。")
            return
        if str(analysis_task_state.get("label", "")) != "批量对比" and str(analysis_task_state.get("label", "")) != "继续批量对比":
            write_report_log("暂停并断点续跑目前只支持批量对比任务。")
            return
        control.request_pause()
        write_report_log("已收到暂停请求。系统会在当前步骤完成后保存检查点并暂停，下次可点击“继续上次批量对比”。")
        report_progress_output.value = "<i>已收到暂停请求，正在等待当前步骤完成并保存进度。</i>"

    def bind_async(button, coroutine_factory, *, task_state: dict[str, object] | None = None, label: str = ""):
        def _handler(_):
            if task_state is None:
                _schedule(coroutine_factory(_))
                return
            running = task_state.get("task")
            if running is not None and not running.done():
                message = "当前已有任务在进行中，请先等待完成或点击停止。"
                if task_state is manage_task_state:
                    with manage_output:
                        print(message)
                elif task_state is chat_task_state:
                    chat_status.value = f"<b>{_escape_html(message)}</b>"
                    _schedule(append_chat_progress(message))
                elif task_state is analysis_task_state:
                    write_report_log(message)
                return
            control = TaskControl(label=label or button.description)

            async def _runner():
                with task_control_context(control):
                    try:
                        await coroutine_factory(_)
                    except asyncio.CancelledError:
                        pass
                    finally:
                        task_state["task"] = None
                        task_state["control"] = None
                        _update_stop_buttons()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(_runner())
                return
            task = loop.create_task(_runner())
            task_state["task"] = task
            task_state["control"] = control
            task_state["label"] = label or button.description
            _update_stop_buttons()
            task.add_done_callback(_log_background_exception)

        button.on_click(_handler)

    def select_all(selector):
        selector.value = tuple(value for _, value in selector.options)

    def clear_selection(selector):
        selector.value = ()

    def on_manage_kb_change(change):
        if change.get("name") == "value":
            reset_manage_detail_page()
            reset_manage_file_page()
            manage_file_selected_paths.clear()
            rename_kb_input.value = ""
            file_action_confirm_checkbox.value = False
            if chunk_search_input.value:
                chunk_search_input.value = ""
            if change["new"]:
                target_kb_dropdown.value = change["new"]
                chunk_search_status.value = f"<i>正在切换到知识库“{_escape_html(str(change['new']))}”...</i>"
                chunk_output.value = _manage_details_placeholder_html("正在刷新文件详情，请稍候。")
            _schedule(refresh_manageable_file_options(change["new"]))
            _schedule(refresh_knowledge_base_status(change["new"]))
            _update_delete_kb_button_state()

    def on_upload_value_change(change):
        value = change["new"]
        ingest_button.disabled = not bool(value)
        upload_status.value = _upload_status_html(value, upload_widget.error)

    def on_upload_error_change(change):
        upload_status.value = _upload_status_html(upload_widget.value, change["new"])

    def _update_delete_kb_button_state() -> None:
        delete_kb_button.disabled = not bool(manage_kb_dropdown.value and delete_kb_confirm_checkbox.value)

    def on_delete_kb_confirm_change(change):
        if change.get("name") == "value":
            _update_delete_kb_button_state()

    def on_chat_kb_change(change):
        if change.get("name") == "value":
            chat_scope_hint.value = "<i>未选择文件时，默认检索整个知识库。</i>"
            _schedule(refresh_document_options(change["new"], chat_doc_selector, empty_hint="该知识库下暂无文件。"))
            _refresh_session_summary()

    def on_analysis_kb_change(change):
        if change.get("name") == "value":
            analysis_doc_search_input.value = ""
            reset_analysis_doc_page()
            _schedule(refresh_analysis_doc_options(change["new"], empty_hint="该知识库下暂无文件。"))

    def on_analysis_doc_change(change):
        if change.get("name") == "value":
            render_analysis_doc_selector()
            _schedule(refresh_analysis_checkpoint_status())

    def on_analysis_doc_search_change(change):
        if change.get("name") == "value":
            reset_analysis_doc_page()
            render_analysis_doc_selector()

    def on_analysis_doc_page_size_change(change):
        if change.get("name") == "value":
            analysis_doc_pagination["page_size"] = max(1, int(change["new"] or 40))
            reset_analysis_doc_page()
            render_analysis_doc_selector()

    def go_to_previous_analysis_doc_page(_=None):
        analysis_doc_pagination["current"] = max(1, int(analysis_doc_pagination["current"]) - 1)
        render_analysis_doc_selector()

    def go_to_next_analysis_doc_page(_=None):
        analysis_doc_pagination["current"] += 1
        render_analysis_doc_selector()

    def on_memory_mode_change(change):
        if change.get("name") == "value":
            _refresh_session_summary()
            _schedule(refresh_session_options(preferred_session_id=session_id_input.value.strip()))
            if session_id_input.value.strip():
                _schedule(load_chat_history())

    def on_session_search_change(change):
        if change.get("name") == "value":
            _schedule(refresh_session_options(preferred_session_id=session_id_input.value.strip()))

    def on_session_id_change(change):
        if change.get("name") == "value":
            clear_chat_display()
            if str(change["new"]).strip():
                new_id = str(change["new"]).strip()
                valid_values = {value for _, value in session_selector.options}
                if new_id in valid_values and session_selector.value != new_id:
                    session_selector.value = new_id
                _refresh_session_summary()
                chat_status.value = "<b>已切换当前会话。</b>"

    def on_session_select(change):
        if change.get("name") != "value":
            return
        selected = change["new"]
        if not selected:
            rename_session_input.value = ""
            _refresh_session_summary()
            return
        if session_id_input.value != selected:
            session_id_input.value = selected
        delete_session_confirm_checkbox.value = False
        summary = session_summaries_by_id.get(selected, {})
        rename_session_input.value = str(summary.get("title", ""))
        _refresh_session_summary()
        _schedule(load_chat_history())

    def on_history_page_size_change(change):
        if change.get("name") != "value":
            return
        history_pagination_state["page_size"] = int(change["new"] or 20)
        if session_id_input.value.strip():
            _schedule(load_chat_history(anchor="latest", announce=False))
        else:
            _update_history_pagination_ui(
                total_turns=0,
                total_pages=1,
                current_page=1,
                page_size=max(1, int(change["new"] or 20)),
            )

    def go_to_history_first(_=None):
        if not session_id_input.value.strip():
            return
        _schedule(load_chat_history(target_page=1, anchor="current", announce=False))

    def go_to_history_previous(_=None):
        if not session_id_input.value.strip():
            return
        _schedule(
            load_chat_history(
                target_page=max(1, int(history_pagination_state["current_page"]) - 1),
                anchor="current",
                announce=False,
            )
        )

    def go_to_history_next(_=None):
        if not session_id_input.value.strip():
            return
        _schedule(
            load_chat_history(
                target_page=min(
                    int(history_pagination_state["total_pages"]),
                    int(history_pagination_state["current_page"]) + 1,
                ),
                anchor="current",
                announce=False,
            )
        )

    def go_to_history_last(_=None):
        if not session_id_input.value.strip():
            return
        _schedule(
            load_chat_history(
                target_page=max(1, int(history_pagination_state["total_pages"])),
                anchor="current",
                announce=False,
            )
        )

    def go_to_history_page(_=None):
        if not session_id_input.value.strip():
            return
        _schedule(
            load_chat_history(
                target_page=max(1, int(history_page_input.value)),
                anchor="current",
                announce=False,
            )
        )

    def on_chat_doc_change(change):
        if change.get("name") == "value":
            _refresh_session_summary()

    def on_manage_doc_change(change):
        if change.get("name") == "value":
            if manage_file_sync_state["updating"]:
                return
            visible_values = {value for _, value in manage_doc_selector.options}
            manage_file_selected_paths.difference_update(visible_values)
            manage_file_selected_paths.update(change["new"])
            _render_manage_file_selector_page()
            reset_manage_detail_page()
            _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def on_manage_file_page_size_change(change):
        if change.get("name") == "value":
            manage_file_pagination["page_size"] = max(1, int(change["new"] or 40))
            reset_manage_file_page()
            _render_manage_file_selector_page()

    def go_to_previous_manage_file_page(_=None):
        manage_file_pagination["current"] = max(1, int(manage_file_pagination["current"]) - 1)
        _render_manage_file_selector_page()

    def go_to_next_manage_file_page(_=None):
        manage_file_pagination["current"] += 1
        _render_manage_file_selector_page()

    def select_all_manage_files(_=None):
        manage_file_selected_paths.clear()
        manage_file_selected_paths.update(value for _, value in manage_file_all_options)
        _render_manage_file_selector_page()
        reset_manage_detail_page()
        _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def clear_manage_file_selection(_=None):
        manage_file_selected_paths.clear()
        _render_manage_file_selector_page()
        reset_manage_detail_page()
        _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def on_chunk_search_change(change):
        if change.get("name") == "value":
            reset_manage_detail_page()
            _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def on_search_selected_only_change(change):
        if change.get("name") == "value":
            reset_manage_detail_page()
            _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def on_detail_page_size_change(change):
        if change.get("name") == "value":
            reset_manage_detail_page()
            _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def on_enable_migration_ui_change(change):
        if change.get("name") == "value":
            _set_migration_section_visibility(bool(change["new"]))

    def go_to_previous_detail_page(_=None):
        manage_detail_page["current"] = max(1, manage_detail_page["current"] - 1)
        _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def go_to_next_detail_page(_=None):
        manage_detail_page["current"] += 1
        _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    def clear_chunk_search(_=None):
        reset_manage_detail_page()
        if chunk_search_input.value:
            chunk_search_input.value = ""
            return
        _schedule(refresh_manage_details_preview(manage_kb_dropdown.value))

    manage_kb_dropdown.observe(on_manage_kb_change, names="value")
    chat_kb_dropdown.observe(on_chat_kb_change, names="value")
    analysis_kb_dropdown.observe(on_analysis_kb_change, names="value")
    memory_mode_toggle.observe(on_memory_mode_change, names="value")
    session_search_input.observe(on_session_search_change, names="value")
    session_id_input.observe(on_session_id_change, names="value")
    session_selector.observe(on_session_select, names="value")
    history_page_size_dropdown.observe(on_history_page_size_change, names="value")
    chat_doc_selector.observe(on_chat_doc_change, names="value")
    manage_doc_selector.observe(on_manage_doc_change, names="value")
    manage_file_page_size_dropdown.observe(on_manage_file_page_size_change, names="value")
    doc_selector.observe(on_analysis_doc_change, names="value")
    analysis_doc_search_input.observe(on_analysis_doc_search_change, names="value")
    analysis_doc_page_size_dropdown.observe(on_analysis_doc_page_size_change, names="value")
    chunk_search_input.observe(on_chunk_search_change, names="value")
    search_selected_only_checkbox.observe(on_search_selected_only_change, names="value")
    detail_page_size_dropdown.observe(on_detail_page_size_change, names="value")
    enable_migration_ui_checkbox.observe(on_enable_migration_ui_change, names="value")
    upload_widget.observe(on_upload_value_change, names="value")
    upload_widget.observe(on_upload_error_change, names="error")
    delete_kb_confirm_checkbox.observe(on_delete_kb_confirm_change, names="value")
    session_history_button.on_click(lambda _: _schedule(load_chat_history()))
    history_first_button.on_click(go_to_history_first)
    history_prev_button.on_click(go_to_history_previous)
    history_next_button.on_click(go_to_history_next)
    history_last_button.on_click(go_to_history_last)
    history_go_button.on_click(go_to_history_page)
    clear_chat_view_button.on_click(clear_chat_display)
    chunk_search_clear_button.on_click(clear_chunk_search)
    detail_prev_button.on_click(go_to_previous_detail_page)
    detail_next_button.on_click(go_to_next_detail_page)
    manage_file_prev_button.on_click(go_to_previous_manage_file_page)
    manage_file_next_button.on_click(go_to_next_manage_file_page)
    analysis_doc_prev_button.on_click(go_to_previous_analysis_doc_page)
    analysis_doc_next_button.on_click(go_to_next_analysis_doc_page)
    manage_select_all_button.on_click(select_all_manage_files)
    manage_clear_button.on_click(clear_manage_file_selection)
    chat_select_all_button.on_click(lambda _: select_all(chat_doc_selector))
    chat_clear_button.on_click(lambda _: clear_selection(chat_doc_selector))
    analysis_select_all_button.on_click(_select_visible_analysis_docs)
    analysis_clear_button.on_click(_clear_analysis_docs)
    bind_async(refresh_kb_button, lambda _: refresh_all_views())
    bind_async(create_kb_button, create_knowledge_base)
    bind_async(refresh_manage_files_button, lambda _: refresh_manageable_file_options(manage_kb_dropdown.value))
    bind_async(ingest_button, ingest_files, task_state=manage_task_state, label="导入并建索引")
    bind_async(save_index_settings_button, save_index_settings)
    bind_async(delete_kb_button, delete_selected_knowledge_base, task_state=manage_task_state, label="删除知识库")
    bind_async(scan_workspace_button, scan_workspace_files)
    bind_async(import_workspace_button, import_workspace_files, task_state=manage_task_state, label="从工作区导入")
    bind_async(export_migration_button, export_project_bundle, task_state=manage_task_state, label="导出迁移包")
    bind_async(import_migration_button, import_project_bundle, task_state=manage_task_state, label="导入迁移包")
    bind_async(vectorize_button, vectorize_selected_files, task_state=manage_task_state, label="向量化处理")
    bind_async(delete_files_button, delete_selected_files, task_state=manage_task_state, label="删除文件")
    bind_async(move_files_button, move_selected_files, task_state=manage_task_state, label="移动文件")
    bind_async(rename_file_button, rename_selected_file, task_state=manage_task_state, label="重命名文件")
    bind_async(rename_kb_button, rename_selected_kb, task_state=manage_task_state, label="重命名知识库")
    bind_async(view_chunks_button, view_selected_chunks)
    bind_async(chunk_search_button, lambda _: refresh_manage_details_preview(manage_kb_dropdown.value))
    bind_async(save_settings_button, save_runtime_settings)
    bind_async(new_session_button, create_new_session)
    bind_async(refresh_sessions_button, lambda _: refresh_session_options(preferred_session_id=session_id_input.value.strip()))
    bind_async(delete_session_button, delete_selected_session, task_state=chat_task_state, label="删除会话")
    bind_async(rename_session_button, rename_selected_session, task_state=chat_task_state, label="重命名会话")
    bind_async(test_model_button, test_model_connection, task_state=chat_task_state, label="测试模型连接")
    bind_async(send_button, send_question, task_state=chat_task_state, label="问答")
    bind_async(single_button, run_single_analysis, task_state=analysis_task_state, label="单文档分析")
    bind_async(compare_button, run_compare, task_state=analysis_task_state, label="批量对比")
    bind_async(resume_compare_button, resume_compare, task_state=analysis_task_state, label="继续批量对比")
    bind_async(clear_checkpoint_button, clear_compare_checkpoint)

    async def clear_selected_file_cache(_=None):
        """Remove the analysis cache for every file currently selected."""
        course_id = analysis_kb_dropdown.value.strip()
        selected_doc_ids = list(doc_selector.value)
        if not course_id:
            clear_report_log()
            write_report_log("请先选择知识库。")
            return
        if not selected_doc_ids:
            clear_report_log()
            write_report_log("请先勾选至少一个文件。")
            return
        clear_report_log()
        write_report_log(f"正在清除选中的 {len(selected_doc_ids)} 个文件的分析缓存...")
        removed_count = 0
        for doc_id in selected_doc_ids:
            try:
                removed = await single_doc_service.clear_analysis_cache(
                    course_id=course_id, doc_id=doc_id,
                )
                if removed:
                    removed_count += 1
                    analysis_doc_cache_status[doc_id] = False
            except Exception:
                pass
        write_report_log(f"已清除 {removed_count}/{len(selected_doc_ids)} 个文件的分析缓存。")
        render_analysis_doc_selector()

    bind_async(clear_selected_cache_button, clear_selected_file_cache)
    bind_async(chat_refresh_files_button, lambda _: refresh_document_options(chat_kb_dropdown.value, chat_doc_selector, empty_hint="该知识库下暂无文件。"))
    bind_async(refresh_docs_button, lambda _: refresh_analysis_doc_options(analysis_kb_dropdown.value, empty_hint="该知识库下暂无文件。"))
    save_template_button.on_click(save_current_template)
    load_template_button.on_click(load_selected_template)
    delete_template_button.on_click(delete_selected_template)
    manage_stop_button.on_click(lambda _: stop_tracked_task(manage_task_state, message="已停止当前知识库操作。"))
    stop_chat_button.on_click(lambda _: stop_tracked_task(chat_task_state, message="已停止当前回答。"))
    stop_analysis_button.on_click(lambda _: stop_tracked_task(analysis_task_state, message="已停止当前分析。"))
    pause_analysis_button.on_click(lambda _: _pause_analysis_task())
    add_field_button.on_click(lambda _: add_target_field_row())
    add_echem_template_button.on_click(
        lambda _: add_field_template(
            [
                ("反应物", "", ""),
                ("产物", "", ""),
                ("氢气产生速率", "优先提取主结果中的数值", "umol/g/h"),
                ("反应温度", "如果没有提及写无", "摄氏度"),
                ("反应光强", "如果没有提及写无", "AM"),
            ]
        )
    )
    add_common_template_button.on_click(
        lambda _: add_field_template(
            [
                ("效率", "提取主结果中的效率数值", "%"),
                ("产率", "提取主结果中的产率", "%"),
                ("温度", "提取实验温度", "°C"),
                ("电压", "提取关键电压条件", "V"),
            ]
        )
    )
    clear_fields_button.on_click(lambda _: clear_target_fields())
    _schedule(refresh_analysis_checkpoint_status())
    render_target_field_rows()

    knowledge_header_box = widgets.HBox(
        [
            widgets.VBox(
                [widgets.HBox([manage_kb_dropdown, refresh_kb_button])],
                layout=widgets.Layout(width="420px"),
            ),
            widgets.VBox(
                [
                    widgets.HBox([create_kb_input, create_kb_button]),
                    widgets.HBox([rename_kb_input, rename_kb_button]),
                ],
                layout=widgets.Layout(width="560px"),
            ),
        ],
        layout=widgets.Layout(justify_content="space-between", align_items="flex-start"),
    )
    migration_section = widgets.VBox(
        [
            widgets.HTML("<b>迁移与备份</b>"),
            widgets.HBox([migration_export_name_input, export_migration_button]),
            migration_bundle_upload,
            widgets.HBox([migration_bundle_path_input, import_migration_button]),
            migration_confirm_checkbox,
            migration_status_html,
        ],
        layout=widgets.Layout(display="flex" if config.enable_migration_ui else "none"),
    )

    knowledge_tab = widgets.VBox(
        [
            widgets.HTML("<b>知识库设置</b>"),
            knowledge_header_box,
            widgets.HTML("<b>导入文件与索引参数</b>"),
            knowledge_help_html,
            widgets.HBox([target_kb_dropdown, source_type_dropdown, rebuild_checkbox]),
            widgets.HBox([chunk_size_input, chunk_overlap_input]),
            widgets.HBox([merge_small_chunks_checkbox, min_chunk_size_input]),
            widgets.HBox([save_index_settings_button]),
            index_settings_status,
            upload_widget,
            upload_status,
            widgets.HBox([ingest_button, manage_stop_button]),
            widgets.HTML("<b>如果浏览器上传不稳定，可以改用工作区导入</b>"),
            widgets.HBox([import_dir_input, scan_workspace_button, import_workspace_button]),
            workspace_file_selector,
            widgets.HTML("<b>文件操作</b>"),
            widgets.HBox([refresh_manage_files_button, vectorize_button, move_files_button, delete_files_button]),
            widgets.HBox([manage_select_all_button, manage_clear_button, file_action_confirm_checkbox]),
            widgets.HBox([manage_file_page_size_dropdown, manage_file_prev_button, manage_file_next_button, manage_file_page_status_html]),
            manage_doc_selector,
            widgets.HBox([move_target_kb_dropdown, move_target_type_dropdown]),
            widgets.HBox([rename_file_input, rename_file_button, view_chunks_button]),
            widgets.HTML("<b>当前知识库状态</b>"),
            knowledge_base_status_html,
            knowledge_base_change_html,
            widgets.HBox([delete_kb_confirm_checkbox, delete_kb_button]),
            widgets.HTML("<b>详情区</b>"),
            manage_detail_hint_html,
            widgets.HBox([chunk_search_input, chunk_search_button, chunk_search_clear_button]),
            widgets.HBox([search_selected_only_checkbox, detail_page_size_dropdown, detail_prev_button, detail_next_button]),
            detail_page_status_html,
            chunk_search_status,
            manage_output_panel,
            chunk_output,
            migration_section,
        ]
    )
    retrieval_settings_box = widgets.VBox(
        [
            widgets.HTML("<b>检索参数</b>"),
            widgets.HBox([retrieval_top_k_input, retrieval_fetch_k_input, citation_limit_input]),
            widgets.HTML("<b>切片评分与过滤</b>"),
            widgets.HBox([enable_rerank_checkbox, rerank_min_score_input, rerank_min_keep_input]),
            widgets.HBox([rerank_weight_vector_input, rerank_weight_keyword_input, rerank_weight_phrase_input]),
            widgets.HBox([rerank_weight_metadata_input]),
            widgets.HBox([model_context_window_input, answer_token_reserve_input, recent_history_turns_input]),
            widgets.HTML("<b>上下文策略参数</b>"),
            widgets.HBox([long_context_window_tokens_input, long_context_window_overlap_tokens_input, recursive_summary_target_tokens_input]),
            widgets.HBox([recursive_summary_batch_size_input, prompt_compression_turn_token_limit_input]),
            widgets.HTML("<b>界面设置</b>"),
            widgets.HBox([enable_rewrite_checkbox, enable_migration_ui_checkbox]),
        ]
    )
    prompt_settings_box = widgets.VBox(
        [
            widgets.HTML("<b>提示词设置</b>"),
            widgets.HTML("<b>中文提示词</b><div style='color:#5b6472;margin-top:4px;'>当回答语言为中文时，会自动使用这一组提示词。</div>"),
            qa_system_prompt_zh_input,
            rewrite_prompt_zh_input,
            answer_instruction_zh_input,
            widgets.HTML("<b>English prompts</b><div style='color:#5b6472;margin-top:4px;'>These prompts are used automatically when the answer language is English.</div>"),
            qa_system_prompt_en_input,
            rewrite_prompt_en_input,
            answer_instruction_en_input,
            widgets.HTML("<b>分析与报告提示词</b><div style='color:#5b6472;margin-top:4px;'>单文档分析与批量对比报告也支持独立的中英文提示词。</div>"),
            widgets.HTML("<b>中文分析/报告</b>"),
            single_analysis_prompt_zh_input,
            compare_report_prompt_zh_input,
            data_extraction_prompt_zh_input,
            table_summary_prompt_zh_input,
            widgets.HTML("<b>English analysis/report prompts</b>"),
            single_analysis_prompt_en_input,
            compare_report_prompt_en_input,
            data_extraction_prompt_en_input,
            table_summary_prompt_en_input,
        ]
    )
    advanced_chat_settings = widgets.Accordion(children=[retrieval_settings_box, prompt_settings_box])
    advanced_chat_settings.set_title(0, "检索参数")
    advanced_chat_settings.set_title(1, "提示词设置")
    session_panel = widgets.VBox(
        [
            widgets.HTML("<b>会话列表</b>"),
            session_search_input,
            widgets.HBox([refresh_sessions_button, new_session_button, stop_chat_button]),
            session_selector,
            widgets.HBox([rename_session_input, rename_session_button]),
            widgets.HBox([delete_session_confirm_checkbox, delete_session_button]),
            session_summary_html,
        ],
        layout=widgets.Layout(width="31%", min_width="320px", padding="0 12px 0 0"),
    )
    conversation_panel = widgets.VBox(
        [
            widgets.HBox([chat_kb_dropdown, chat_refresh_files_button]),
            chat_doc_selector,
            widgets.HBox([chat_select_all_button, chat_clear_button]),
            chat_scope_hint,
            widgets.HBox([session_history_button, clear_chat_view_button, history_page_size_dropdown]),
            widgets.HBox(
                [
                    history_first_button,
                    history_prev_button,
                    history_next_button,
                    history_last_button,
                    history_page_input,
                    history_go_button,
                ]
            ),
            history_page_info,
            widgets.HBox([memory_mode_toggle, language_toggle, streaming_mode_toggle]),
            widgets.HTML("<b>当前进展</b>"),
            chat_progress_output,
            widgets.HTML("<b>对话内容</b>"),
            chat_history_output,
            question_area,
            widgets.HBox([send_button, test_model_button, save_settings_button]),
            advanced_chat_settings,
            settings_status,
            model_test_output,
            chat_status,
            widgets.HTML("<b>引用与原文定位</b>"),
            citation_output,
        ],
        layout=widgets.Layout(width="69%", padding="0 0 0 16px"),
    )
    chat_divider = widgets.Box(
        layout=widgets.Layout(
            width="1px",
            min_width="1px",
            align_self="stretch",
            border_left="1px solid #d0d7de",
            margin="0 4px",
        )
    )
    chat_tab = widgets.VBox(
        [
            widgets.HBox(
                [session_panel, chat_divider, conversation_panel],
                layout=widgets.Layout(width="100%", align_items="flex-start"),
            ),
        ]
    )
    analysis_tab = widgets.VBox(
        [
            widgets.HBox([analysis_kb_dropdown, refresh_docs_button]),
            file_type_legend_html,
            widgets.HTML(
                "<b>分析文件</b><div style='color:#5b6472;margin-top:4px;'>"
                "前面的勾选框用于选择要参与分析的文件，点击文件名会在下方打开该文件的切片详情。"
                "</div>"
            ),
            widgets.HBox([analysis_doc_search_input, analysis_select_all_button, analysis_clear_button]),
            widgets.HBox([analysis_doc_page_size_dropdown, analysis_doc_prev_button, analysis_doc_next_button, analysis_doc_page_status_html]),
            analysis_doc_status_html,
            analysis_doc_list_box,
            widgets.HTML("<b>文件切片详情</b>"),
            analysis_doc_detail_html,
            doc_selector,
            analysis_checkpoint_status_html,
            widgets.HTML(
                "<b>定向字段抽取</b><div style='color:#5b6472;margin-top:4px;'>"
                "按卡片逐项配置字段，比纯文本更直观。若原文单位和期望单位不同，系统会在可换算时自动标准化。"
                "填写后，单文档分析会附加字段抽取结果，批量对比会自动附加对比表、复查证据和 CSV 导出。"
                "</div>"
            ),
            widgets.HBox([add_field_button, add_echem_template_button, add_common_template_button, clear_fields_button]),
            widgets.HBox([template_name_input, save_template_button]),
            widgets.HBox([field_template_dropdown, load_template_button, delete_template_button]),
            target_fields_summary_html,
            target_fields_box,
            export_csv_checkbox,
            widgets.HBox([single_button, compare_button, resume_compare_button, clear_checkpoint_button, pause_analysis_button, stop_analysis_button, clear_selected_cache_button]),
            widgets.HTML("<b>当前进度</b><div style='color:#5b6472;margin-top:4px;'>运行中的阶段、总体进度和文档级进度会显示在这里。</div>"),
            report_progress_output,
            widgets.HTML("<b>表格视图</b>"),
            report_table_output,
            widgets.HTML("<b>Markdown 正文</b><div style='color:#5b6472;margin-top:4px;'>分析结果和批量报告会在这里完整展开显示，不再折叠到日志框里。</div>"),
            report_markdown_output_panel,
            widgets.HTML("<b>运行日志</b>"),
            report_output_panel,
        ]
    )

    tabs = widgets.Tab(children=[knowledge_tab, chat_tab, analysis_tab])
    tabs.set_title(0, "知识库管理")
    tabs.set_title(1, "问答助手")
    tabs.set_title(2, "分析报告")
    _install_log_panel_support(display, HTML, Javascript)
    _schedule(initialize_ui())
    return tabs


def _build_recovery_app(widgets, config: AppConfig):
    restore_bundle_upload = widgets.FileUpload(accept=".zip", multiple=False, description="上传备份包")
    restore_bundle_path_input = widgets.Text(
        value="",
        description="工作区包路径",
        placeholder="例如 release/migration_backups/backup_xxx.zip",
        layout=widgets.Layout(width="520px"),
    )
    restore_button = widgets.Button(description="恢复备份", button_style="warning")
    restore_confirm_checkbox = widgets.Checkbox(
        value=False,
        description="我已确认恢复会覆盖当前配置和数据文件（日志不会改动）",
    )
    restore_status_html = widgets.HTML(
        value=(
            "<div style='padding:10px 12px;border:1px solid #fbbf24;border-radius:8px;background:#fffbeb;'>"
            "<b>当前模型配置未填写完整，已自动进入恢复模式。</b><br>"
            "请导入之前导出的备份 / 迁移包，恢复有效配置后再重启 Notebook / Kernel。"
            "</div>"
        )
    )
    restore_output = widgets.Output(
        layout={
            "width": "100%",
            "height": "220px",
            "max_height": "220px",
            "overflow": "auto",
        }
    )
    restore_output_panel = widgets.Box(
        [restore_output],
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="8px",
            height="220px",
            max_height="220px",
            overflow="auto",
        ),
    )
    restore_output_panel.add_class("assistant-log-panel")

    def _resolve_restore_bundle_path() -> Path:
        uploads = _normalize_upload_value(restore_bundle_upload.value)
        if uploads:
            item = uploads[0]
            bundle_name = Path(str(item.get("name", "migration_bundle.zip"))).name
            destination_dir = config.project_root / "release" / "migration_uploads"
            destination_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = destination_dir / f"{timestamp}_{bundle_name}"
            content = item.get("content", b"")
            payload = bytes(content) if not isinstance(content, bytes) else content
            destination.write_bytes(payload)
            return destination
        raw_path = restore_bundle_path_input.value.strip()
        if not raw_path:
            raise ValueError("请先上传备份包，或填写工作区中的 zip 路径。")
        return _resolve_workspace_bundle_path(config.project_root, raw_path)

    def _render_restore_status(title: str, result: dict[str, object], *, note: str = "") -> str:
        roots = ", ".join(str(item) for item in result.get("roots", [])) or "-"
        file_count = int(result.get("file_count", 0) or 0)
        total_size = _human_readable_bytes(int(result.get("total_size_bytes", 0) or 0))
        bundle_path = str(result.get("bundle_path", ""))
        backup_path = str(result.get("backup_path", ""))
        lines = [
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#f6f8fa;'>",
            f"<b>{_escape_html(title)}</b><br>",
            f"包路径: {_escape_html(bundle_path)}<br>",
            f"包含目录: {_escape_html(roots)}<br>",
            f"文件数: {file_count} | 总大小: {_escape_html(total_size)}",
        ]
        if backup_path:
            lines.append(f"<br>导入前备份: {_escape_html(backup_path)}")
        if note:
            lines.append(f"<br><span style='color:#5b6472;'>{_escape_html(note)}</span>")
        lines.append("</div>")
        return "".join(lines)

    def _render_restore_progress(title: str, detail: str = "") -> str:
        detail_html = (
            "<div style='margin-top:6px;color:#5b6472;'>"
            f"{_escape_html(detail)}"
            "</div>"
            if detail
            else ""
        )
        return (
            "<style>"
            "@keyframes codex-progress-slide {"
            "0% { transform: translateX(-70%); }"
            "50% { transform: translateX(80%); }"
            "100% { transform: translateX(220%); }"
            "}"
            "</style>"
            "<div style='padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#f6f8fa;'>"
            f"<b>{_escape_html(title)}</b>"
            f"{detail_html}"
            "<div style='margin-top:10px;height:8px;border-radius:999px;background:#e5e7eb;overflow:hidden;'>"
            "<div style='width:45%;height:100%;border-radius:999px;background:#2563eb;"
            "animation:codex-progress-slide 1.2s ease-in-out infinite;'></div>"
            "</div>"
            "</div>"
        )

    async def restore_project_bundle(_):
        restore_button.disabled = True
        with restore_output:
            restore_output.clear_output()
            try:
                if not restore_confirm_checkbox.value:
                    print("恢复会覆盖当前配置和运行数据。请先勾选确认框。")
                    return
                bundle_path = _resolve_restore_bundle_path()
                restore_status_html.value = _render_restore_progress(
                    "正在校验备份包",
                    "正在检查 zip 文件和迁移清单。",
                )
                print(f"准备读取备份包: {bundle_path}")
                await asyncio.sleep(0.05)
                manifest = await asyncio.to_thread(inspect_migration_bundle, bundle_path)
                print(f"准备恢复备份包: {bundle_path}")
                print(f"打包时间: {manifest.get('created_at', '-')}")
                print(f"包含目录: {', '.join(str(item) for item in manifest.get('roots', []))}")
                print(f"文件数: {manifest.get('file_count', len(manifest.get('files', [])))}")
                restore_status_html.value = _render_restore_progress(
                    "正在恢复备份",
                    "正在自动备份当前数据，并恢复 zip 中的配置与运行数据。",
                )
                print("正在自动备份当前数据并恢复备份包...")
                await asyncio.sleep(0.05)
                result = await asyncio.to_thread(import_migration_bundle, config, bundle_path)
                print("恢复完成。")
                if result.get("backup_path"):
                    print(f"已自动备份当前数据: {result['backup_path']}")
                print("请立即重启 Notebook / Kernel，然后重新运行第一个单元。")
                restore_status_html.value = _render_restore_status(
                    "备份恢复完成",
                    result,
                    note="为了让恢复后的配置和数据库完全生效，请现在重启当前 Notebook / Kernel。",
                )
                restore_confirm_checkbox.value = False
                restore_bundle_path_input.value = ""
                restore_bundle_upload.value = ()
                restore_bundle_upload.error = ""
            except Exception:
                print("恢复备份失败:")
                print(traceback.format_exc())
                restore_status_html.value = (
                    "<b>恢复失败:</b><pre>"
                    f"{_escape_html(traceback.format_exc())}"
                    "</pre>"
                )
            finally:
                restore_button.disabled = False

    restore_button.on_click(lambda _: _schedule(restore_project_bundle(None)))

    return widgets.VBox(
        [
            widgets.HTML("<h3 style='margin:0 0 8px 0;'>恢复备份</h3>"),
            widgets.HTML(
                "<div style='color:#5b6472;margin-bottom:8px;'>"
                "当前缺少聊天模型或嵌入模型配置，完整功能已暂时隐藏。"
                "你可以先恢复备份包，或者手工填写 config/app_config.json 后重启。"
                "</div>"
            ),
            restore_bundle_upload,
            widgets.HBox([restore_bundle_path_input, restore_button]),
            restore_confirm_checkbox,
            restore_status_html,
            restore_output_panel,
        ]
    )


def _normalize_upload_value(value):
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def _resolve_workspace_bundle_path(project_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    candidate = Path(normalized).expanduser()
    tried: list[Path] = []

    def add_path(path: Path) -> None:
        resolved = path.resolve(strict=False)
        if resolved not in tried:
            tried.append(resolved)

    if candidate.is_absolute():
        add_path(candidate)
    else:
        add_path(project_root / candidate)
        if candidate.parts and candidate.parts[0] == project_root.name and len(candidate.parts) > 1:
            add_path(project_root / Path(*candidate.parts[1:]))
        if len(candidate.parts) == 1:
            add_path(project_root / "release" / "migration_bundles" / candidate.name)
            add_path(project_root / "release" / "migration_backups" / candidate.name)
            add_path(project_root / candidate.name)

    for resolved in tried:
        if resolved.exists():
            return resolved
    return tried[0]


def _schedule(coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coroutine)
        return
    task = loop.create_task(coroutine)
    task.add_done_callback(_log_background_exception)


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _human_readable_bytes(size: int) -> str:
    value = float(max(0, int(size)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{int(value)} B"


def _escape_markdown_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _render_markdown_html(markdown_text: str) -> str:
    text = str(markdown_text or "").strip()
    if not text:
        return (
            "<div style='color:#6b7280;line-height:1.7;'>"
            "当前还没有可展示的 Markdown 正文。"
            "</div>"
        )
    try:
        import markdown as markdown_pkg

        rendered = markdown_pkg.markdown(
            text,
            extensions=["extra", "tables", "fenced_code", "sane_lists", "nl2br", "toc"],
            output_format="html5",
        )
    except Exception:
        rendered = (
            "<pre style='white-space:pre-wrap;word-break:break-word;line-height:1.7;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'>"
            + _escape_html(text)
            + "</pre>"
        )
    # Make [N] citation references clickable anchors that scroll to the reference section.
    rendered = re.sub(
        r'(?<!id=["\'])\[(\d{1,3})\]',
        lambda m: (
            f'<a href="#codex-ref-{m.group(1)}" '
            f'style="color:#2563eb;text-decoration:none;font-weight:600;cursor:pointer;" '
            f'title="跳转到引用 [{m.group(1)}]">[{m.group(1)}]</a>'
        ),
        rendered,
    )
    # Add id anchors to reference list items like "[1] ..."
    rendered = re.sub(
        r'(<li>)\s*\[(\d{1,3})\]',
        lambda m: f'{m.group(1)}<span id="codex-ref-{m.group(2)}"></span>[{m.group(2)}]',
        rendered,
    )
    # Also match paragraphs or lines starting with [N] in reference sections
    rendered = re.sub(
        r'(<p>)\s*\[(\d{1,3})\]',
        lambda m: f'{m.group(1)}<span id="codex-ref-{m.group(2)}"></span>[{m.group(2)}]',
        rendered,
    )
    # Wrap reference/citation sections in collapsible <details> blocks.
    _ref_heading_re = re.compile(
        r'(<h[2-4][^>]*>)\s*(参考文献|引用|引用文献|References?|Bibliography|Works Cited|参考资料|来源|文献引用)\s*(</h[2-4]>)',
        re.IGNORECASE,
    )
    match = _ref_heading_re.search(rendered)
    if match:
        heading_start = match.start()
        heading_html = match.group(0)
        rest_content = rendered[match.end():]
        rendered = (
            rendered[:heading_start]
            + '<details style="margin-top:12px;border:1px solid #d0d7de;border-radius:8px;padding:8px 12px;">'
            + f'<summary style="cursor:pointer;font-weight:600;font-size:15px;">{match.group(2)} (点击展开)</summary>'
            + '<div style="margin-top:8px;">'
            + rest_content
            + '</div></details>'
        )
    return (
        "<style>"
        ".codex-markdown-body{line-height:1.8;color:#111827;font-size:14px;}"
        ".codex-markdown-body h1,.codex-markdown-body h2,.codex-markdown-body h3,.codex-markdown-body h4{margin:16px 0 8px;}"
        ".codex-markdown-body p,.codex-markdown-body ul,.codex-markdown-body ol{margin:8px 0;}"
        ".codex-markdown-body pre{overflow:auto;padding:12px;border-radius:8px;background:#f6f8fa;}"
        ".codex-markdown-body code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
        ".codex-markdown-body table{width:100%;border-collapse:collapse;margin:12px 0;}"
        ".codex-markdown-body th,.codex-markdown-body td{border:1px solid #d0d7de;padding:8px 10px;vertical-align:top;text-align:left;}"
        ".codex-markdown-body th{background:#f6f8fa;}"
        ".codex-markdown-body blockquote{margin:12px 0;padding:8px 12px;border-left:4px solid #d0d7de;background:#f9fafb;color:#4b5563;}"
        ".codex-markdown-body a{color:#2563eb;text-decoration:none;}"
        ".codex-markdown-body a:hover{text-decoration:underline;}"
        "</style>"
        f"<div class='codex-markdown-body'>{rendered}</div>"
    )


def _render_report_log_text(lines: list[str]) -> str:
    if not lines:
        return "当前还没有运行日志。开始分析后，这里会持续追加处理记录。"
    return "\n".join(str(line) for line in lines[-500:])


def _validate_chunk_settings(
    chunk_size: int,
    chunk_overlap: int,
    merge_small_chunks: bool,
    min_chunk_size: int,
) -> str | None:
    if chunk_overlap >= chunk_size:
        return "重叠大小必须小于切片大小。"
    if merge_small_chunks and min_chunk_size > chunk_size:
        return "最小块长不能大于切片大小。"
    return None


def _validate_generation_settings(model_context_window: int, answer_token_reserve: int) -> str | None:
    if answer_token_reserve >= model_context_window:
        return "回答预留必须小于上下文上限。"
    if model_context_window - answer_token_reserve < 512:
        return "可用于输入的上下文预算过小，请调大上下文上限或调小回答预留。"
    return None


def _validate_context_strategy_settings(
    window_tokens: int,
    overlap_tokens: int,
    summary_target_tokens: int,
    summary_batch_size: int,
    prompt_compression_turn_token_limit: int,
) -> str | None:
    if overlap_tokens >= window_tokens:
        return "滑窗重叠必须小于滑窗大小。"
    if summary_target_tokens >= window_tokens * 2:
        return "摘要目标建议显著小于滑窗总量，请适当调小摘要目标。"
    if summary_batch_size < 2:
        return "摘要批大小至少为 2。"
    if prompt_compression_turn_token_limit < 24:
        return "历史压缩上限不能低于 24。"
    return None


def _validate_rerank_settings(
    min_score: float,
    min_keep: int,
    weight_vector: float,
    weight_keyword: float,
    weight_phrase: float,
    weight_metadata: float,
) -> str | None:
    if not 0.0 <= min_score <= 1.0:
        return "最低分数必须在 0 到 1 之间。"
    if min_keep < 0:
        return "最低保留不能小于 0。"
    if (weight_vector + weight_keyword + weight_phrase + weight_metadata) <= 0:
        return "切片评分权重总和必须大于 0。"
    return None


def _render_chat_history(turns: list[ChatTurn], pending: bool = False) -> str:
    if not turns:
        return (
            '<div style="border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; min-height: 220px; background: #fafbfc;">'
            "<i>这里会显示当前会话的完整对话。发送消息后，后续追问会自动沿用这段会话的上下文。</i>"
            "</div>"
        )
    bubbles = [_render_chat_turn(turn, pending=(pending and index == len(turns) - 1)) for index, turn in enumerate(turns)]
    return (
        '<div style="border: 1px solid #d0d7de; border-radius: 10px; padding: 14px; min-height: 260px; '
        'max-height: 480px; overflow-y: auto; background: linear-gradient(180deg, #fbfcfe 0%, #f5f7fa 100%);">'
        + "".join(bubbles)
        + "</div>"
    )


def _truncate_middle(text: str, limit: int = 88) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    head = max(20, limit // 2 - 6)
    tail = max(16, limit - head - 3)
    return f"{normalized[:head]}...{normalized[-tail:]}"


def _shorten_report_message(message: str, limit: int = 180) -> str:
    """Keep progress logs readable when file names are very long."""

    normalized = " ".join(str(message).split())
    parts = re.split(r"([：:])", normalized, maxsplit=1)
    if len(parts) == 3:
        prefix, separator, suffix = parts
        shortened = f"{prefix}{separator}{_truncate_middle(suffix, 88)}"
        if len(shortened) <= limit:
            return shortened
        return _truncate_middle(shortened, limit)
    return _truncate_middle(normalized, limit)


def _shorten_manage_message(message: str, limit: int = 170) -> str:
    """Keep knowledge-base progress lines compact inside the scrolling log box."""

    normalized = " ".join(str(message).split())
    patterns = [
        r"^(正在读取文件\s+\d+/\d+:\s+)(.+)$",
        r"^(正在切片文件\s+\d+/\d+:\s+)(.+)$",
        r"^(已完成文件\s+\d+/\d+:\s+)(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if not match:
            continue
        prefix, suffix = match.groups()
        shortened = prefix + _truncate_middle(suffix, 90)
        return shortened if len(shortened) <= limit else _truncate_middle(shortened, limit)
    return _truncate_middle(normalized, limit)


_BATCH_DOC_PROGRESS_RE = re.compile(
    r"^\[(?P<stage>[^\]|]+?)\s+(?P<index>\d+)\s*/\s*(?P<total_docs>\d+)\s*\|\s*"
    r"(?P<stage_status>[^|]+?)\s*\|\s*总进度\s*(?P<completed>\d+)\s*/\s*(?P<total_units>\d+)\s*，约\s*"
    r"(?P<percent>[^\]]+?)\]\s*(?P<payload>.+)$"
)
_BATCH_GLOBAL_PROGRESS_RE = re.compile(
    r"^\[(?P<stage>[^\]|]+?)\s*\|\s*总进度\s*(?P<completed>\d+)\s*/\s*(?P<total_units>\d+)\s*，约\s*"
    r"(?P<percent>[^\]]+?)\]\s*(?P<detail>.+)$"
)


def _parse_batch_progress_message(message: str) -> dict[str, object]:
    normalized = " ".join(str(message).split())
    doc_match = _BATCH_DOC_PROGRESS_RE.match(normalized)
    if doc_match:
        payload = str(doc_match.group("payload")).strip()
        title_text = ""
        detail_text = payload
        if "：" in payload:
            title_text, detail_text = payload.split("：", 1)
        elif ":" in payload:
            title_text, detail_text = payload.split(":", 1)
        return {
            "kind": "doc",
            "stage": str(doc_match.group("stage")).strip(),
            "index": int(doc_match.group("index")),
            "total_docs": int(doc_match.group("total_docs")),
            "stage_status": str(doc_match.group("stage_status")).strip(),
            "completed": int(doc_match.group("completed")),
            "total_units": int(doc_match.group("total_units")),
            "percent": str(doc_match.group("percent")).strip(),
            "title": title_text.strip(),
            "detail": detail_text.strip() or "正在处理。",
        }
    global_match = _BATCH_GLOBAL_PROGRESS_RE.match(normalized)
    if global_match:
        return {
            "kind": "global",
            "stage": str(global_match.group("stage")).strip(),
            "completed": int(global_match.group("completed")),
            "total_units": int(global_match.group("total_units")),
            "percent": str(global_match.group("percent")).strip(),
            "detail": str(global_match.group("detail")).strip() or "正在处理。",
        }
    return {"kind": "raw", "detail": normalized}


def _init_batch_progress_rows(selected_titles: list[str]) -> list[dict[str, str]]:
    return [
        {
            "index": str(index),
            "title": title,
            "stage": "未开始",
            "status": "等待中",
            "detail": "尚未进入处理队列。",
            "progress": "-",
        }
        for index, title in enumerate(selected_titles, start=1)
    ]


def _render_batch_progress_board(
    *,
    latest_summary: str,
    overall_progress: str,
    global_stage: str,
    global_detail: str,
    rows: list[dict[str, str]],
) -> str:
    header = (
        "<div style='padding:12px;border:1px solid #d0d7de;border-radius:10px;background:#f6f8fa;line-height:1.6;'>"
        "<b>批量对比进行中</b>"
        f"<div style='margin-top:6px;'>最近进度：{_escape_html(latest_summary)}</div>"
        f"<div style='margin-top:4px;'>总体进度：{_escape_html(overall_progress)}</div>"
        "<div style='margin-top:8px;color:#5b6472;font-size:12px;line-height:1.5;'>"
        f"全局阶段：{_escape_html(global_stage)} | {_escape_html(global_detail)}"
        "</div>"
    )
    table_header = (
        "<div style='margin-top:10px;border:1px solid #d8dee4;border-radius:8px;overflow:hidden;background:#ffffff;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
        "<thead><tr style='background:#f1f5f9;color:#334155;'>"
        "<th style='padding:8px 10px;text-align:left;border-bottom:1px solid #d8dee4;'>文档</th>"
        "<th style='padding:8px 10px;text-align:left;border-bottom:1px solid #d8dee4;'>阶段</th>"
        "<th style='padding:8px 10px;text-align:left;border-bottom:1px solid #d8dee4;'>状态</th>"
        "<th style='padding:8px 10px;text-align:left;border-bottom:1px solid #d8dee4;'>详情</th>"
        "<th style='padding:8px 10px;text-align:left;border-bottom:1px solid #d8dee4;'>总进度</th>"
        "</tr></thead><tbody>"
    )
    table_rows = []
    for row in rows:
        title = _truncate_middle(str(row.get("title", "")), 78)
        detail = _truncate_middle(str(row.get("detail", "")), 92)
        index = _escape_html(str(row.get("index", "-")))
        table_rows.append(
            "<tr>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #eef2f7;'>{index}. {_escape_html(title)}</td>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #eef2f7;'>{_escape_html(str(row.get('stage', '-')))}</td>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #eef2f7;'>{_escape_html(str(row.get('status', '-')))}</td>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #eef2f7;'>{_escape_html(detail)}</td>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #eef2f7;'>{_escape_html(str(row.get('progress', '-')))}</td>"
            "</tr>"
        )
    if not table_rows:
        table_rows.append(
            "<tr><td colspan='5' style='padding:10px;border-bottom:1px solid #eef2f7;color:#5b6472;'>暂无文档状态。</td></tr>"
        )
    return header + table_header + "".join(table_rows) + "</tbody></table></div></div>"


def _render_chat_turn(turn: ChatTurn, pending: bool = False) -> str:
    is_user = turn.role == "user"
    role_label = "你" if is_user else "助手"
    bubble_style = (
        "background:#0f766e;color:#ffffff;margin-left:64px;"
        if is_user
        else "background:#ffffff;color:#111827;margin-right:64px;border:1px solid #dbe3ea;"
    )
    status_label = " <span style='font-size:12px;opacity:0.75;'>(生成中)</span>" if pending else ""
    content_html = _escape_html(turn.content).replace("\n", "<br>")
    if not content_html and pending:
        content_html = "<i>...</i>"
    return (
        '<div style="margin: 10px 0;">'
        f'<div style="font-size:12px;color:#5b6472;margin-bottom:4px;">{role_label}{status_label}</div>'
        f'<div style="border-radius:12px;padding:12px 14px;white-space:pre-wrap;line-height:1.6;{bubble_style}">{content_html}</div>'
        "</div>"
    )


def _render_citation_details(citations: list[dict[str, object]], language: str) -> str:
    if not citations:
        return "<div>暂无引用。</div>"
    source_label = "来源文件"
    source_path_label = "文件路径"
    source_type_label = "文件类型"
    locator_label = "定位"
    chunk_label = "切片ID"
    term_label = "命中关键词"
    content_label = "完整切片内容"
    score_label = "检索评分"
    breakdown_label = "评分构成"
    cards: list[str] = []
    for item in citations:
        citation_id = _escape_html(str(item.get("citation_id", "")))
        file_name = _escape_html(str(item.get("file_name", "")))
        file_path = _escape_html(str(item.get("file_path", "")))
        source_type = _escape_html(_translate_source_type(str(item.get("source_type", ""))))
        page_label = str(item.get("page_label") or "").strip()
        section_label = str(item.get("section_label") or "").strip()
        locator = page_label or section_label or "-"
        chunk_id = _escape_html(str(item.get("chunk_id", "")))
        matched_terms = [str(term).strip() for term in item.get("matched_terms", []) if str(term).strip()]
        content = _highlight_terms_html(str(item.get("quote", "")), matched_terms)
        score_value = item.get("retrieval_score")
        score_text = "-" if score_value in (None, "") else f"{float(score_value):.4f}"
        raw_breakdown = item.get("score_breakdown")
        breakdown = raw_breakdown if isinstance(raw_breakdown, dict) else {}
        breakdown_text = " | ".join(
            f"{_escape_html(str(key))}: {float(value):.4f}"
            for key, value in breakdown.items()
            if isinstance(value, (int, float))
        ) or "-"
        summary = f'[{citation_id}] {file_name} | {locator_label}: {_escape_html(locator)}'
        cards.append(
            "".join(
                [
                    '<details style="margin: 8px 0; border: 1px solid #d0d7de; border-radius: 6px; padding: 8px;">',
                    f'<summary style="cursor: pointer; font-weight: 600;">{summary}</summary>',
                    '<div style="margin-top: 8px; line-height: 1.5;">',
                    f'<div><b>{source_label}:</b> {file_name}</div>',
                    f'<div><b>{source_type_label}:</b> {source_type or "-"}</div>',
                    f'<div><b>{source_path_label}:</b> {file_path or "-"}</div>',
                    f'<div><b>{locator_label}:</b> {_escape_html(locator)}</div>',
                    f'<div><b>{chunk_label}:</b> {chunk_id or "-"}</div>',
                    f'<div><b>{term_label}:</b> {_escape_html(" / ".join(matched_terms) if matched_terms else "-")}</div>',
                    f'<div><b>{score_label}:</b> {_escape_html(score_text)}</div>',
                    f'<div><b>{breakdown_label}:</b> {_escape_html(breakdown_text)}</div>',
                    f'<div style="margin-top: 8px;"><b>{content_label}:</b></div>',
                    f'<pre style="white-space: pre-wrap; overflow-x: auto; margin-top: 6px;">{content}</pre>',
                    "</div>",
                    "</details>",
                ]
            )
        )
    return "".join(cards)


def _highlight_terms_html(text: str, terms: list[str]) -> str:
    escaped = _escape_html(text)
    for term in sorted({item for item in terms if item}, key=len, reverse=True):
        pattern = re.compile(re.escape(_escape_html(term)), re.IGNORECASE)
        escaped = pattern.sub(
            lambda match: (
                "<mark style='background:#fef08a;padding:0 2px;border-radius:3px;'>"
                f"{match.group(0)}</mark>"
            ),
            escaped,
        )
    return escaped


def _format_session_option(summary) -> str:
    title = str(summary.title).strip() or summary.session_id
    if len(title) > 22:
        title = title[:22] + "..."
    if summary.last_updated:
        timestamp = summary.last_updated.astimezone().strftime("%m-%d %H:%M")
    else:
        timestamp = "--"
    return f"{title} ({summary.turn_count}条, {timestamp})"


def _translate_source_type(value: str) -> str:
    return {
        "lecture": "讲义",
        "assignment": "作业",
        "paper": "论文",
    }.get(value, value)


def _source_type_icon(value: str) -> str:
    return {
        "lecture": "🟦",
        "assignment": "🟨",
        "paper": "🟩",
    }.get(value, "⬜")


def _source_type_badge_html(value: str) -> str:
    styles = {
        "lecture": ("讲义", "#dbeafe", "#1d4ed8"),
        "assignment": ("作业", "#fef3c7", "#b45309"),
        "paper": ("论文", "#dcfce7", "#15803d"),
    }
    label, bg, fg = styles.get(value, (value, "#e5e7eb", "#374151"))
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:12px;font-weight:600;'>{_escape_html(label)}</span>"
    )


def _translate_memory_mode(value: str) -> str:
    return {
        "session": "会话记忆",
        "persistent": "持久记忆",
    }.get(value, value)


def _now_utc():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _format_doc_option(record: DocumentRecord) -> str:
    return f"{_source_type_icon(record.source_type)} {record.file_name} ({_translate_source_type(record.source_type)}, {record.chunk_count} 个切片)"


def _format_manage_doc_option(record: DocumentRecord) -> str:
    status = "已向量化" if record.is_vectorized else "未向量化"
    chunk_info = f"{record.chunk_count} 个切片" if record.is_vectorized else "0 个切片"
    return f"{_source_type_icon(record.source_type)} {record.file_name} ({_translate_source_type(record.source_type)}, {status}, {chunk_info})"


def _simplify_doc_option_label(label: str) -> str:
    """Strip UI decorations from one document option label for progress preview text."""

    simplified = label.strip()
    if " " in simplified:
        simplified = simplified.split(" ", 1)[1]
    if " (" in simplified:
        simplified = simplified.split(" (", 1)[0]
    return simplified


def _single_analysis_markdown(analysis) -> str:
    lines = [
        f"# {analysis.title}",
        "",
        f"- 语言: {analysis.language}",
        f"- 情感: {analysis.sentiment}",
        "",
        "## 摘要",
        analysis.summary,
        "",
        "## 关键词",
    ]
    lines.extend(f"- {item}" for item in analysis.keywords)
    lines.extend(["", "## 主题"])
    lines.extend(f"- {item}" for item in analysis.main_topics)
    lines.extend(["", "## 风险点"])
    lines.extend(f"- {item}" for item in analysis.risk_points)
    return "\n".join(lines)


def _single_doc_extraction_markdown(extraction) -> str:
    lines = ["## 定向字段抽取", ""]
    lines.append("| 字段 | 结果 | 状态 | 来源 |")
    lines.append("| --- | --- | --- | --- |")
    for field in extraction.fields:
        locator = field.page_label or field.section_label or "-"
        source = f"{field.source_file or extraction.title} {locator}".strip()
        value = field.normalized_value or field.value or "未提及"
        if field.converted and field.source_unit:
            value = f"{value} (原始单位: {field.source_unit})"
        lines.append(
            "| {field_name} | {value} | {status} | {source} |".format(
                field_name=_escape_markdown_cell(field.field_name),
                value=_escape_markdown_cell(value),
                status=_escape_markdown_cell(field.status),
                source=_escape_markdown_cell(source),
            )
        )
    lines.append("")
    lines.append("### 复查证据")
    lines.append("")
    for field in extraction.fields:
        summary = f"{field.field_name} | {field.status} | {field.normalized_value or field.value or '未提及'}"
        lines.append(f"#### {summary}")
        lines.append("")
        lines.append(f"- 来源文件: {field.source_file or extraction.title}")
        lines.append(f"- 定位: {field.page_label or field.section_label or '-'}")
        lines.append(f"- 原始单位: {field.source_unit or '-'}")
        lines.append(f"- 换算后单位: {field.unit or '-'}")
        lines.append(f"- 说明: {field.notes or '-'}")
        if field.evidence_quote:
            lines.append("")
            lines.append("```text")
            lines.append(field.evidence_quote.strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _chunk_details_markdown(record: DocumentRecord, chunk_details: list[dict[str, object]]) -> str:
    lines = [
        f"## {_source_type_icon(record.source_type)} {record.file_name}",
        "",
        f"- 文档ID: `{record.doc_id}`",
        f"- 文件类型: `{_translate_source_type(record.source_type)}`",
        f"- 已向量化: `{record.is_vectorized}`",
        f"- 切片数: `{record.chunk_count}`",
        "",
    ]
    if not chunk_details:
        lines.extend(
            [
                "> 该文件尚未向量化，当前没有 chunk 详情。",
                "",
            ]
        )
        return "\n".join(lines)
    for chunk in chunk_details:
        locator = chunk.get("page_label") or chunk.get("section_label") or "-"
        merged_from = int(chunk.get("merged_from_count", 1))
        merge_label = f" | 合并自{merged_from}个小块" if merged_from > 1 else ""
        lines.extend(
            [
                f"### 切片 {chunk['chunk_index']} | {locator} | 长度={chunk['length']}{merge_label}",
                "```text",
                str(chunk["content"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _knowledge_base_overview_markdown(course_id: str, records: list[DocumentRecord]) -> str:
    lines = [
        f"## 知识库概览: {course_id}",
        "",
        f"- 文件总数: `{len(records)}`",
        f"- 已向量化文件数: `{sum(1 for record in records if record.is_vectorized)}`",
        f"- 切片总数: `{sum(int(record.chunk_count or 0) for record in records)}`",
        "",
    ]
    if not records:
        lines.extend(
            [
                "> 当前知识库还没有文件。",
                "",
            ]
        )
        return "\n".join(lines)
    lines.append("### 文件列表")
    lines.append("")
    for record in records:
        status = "已向量化" if record.is_vectorized else "未向量化"
        lines.append(
            f"- {_source_type_icon(record.source_type)} `{record.file_name}` | 类型: `{_translate_source_type(record.source_type)}` | 状态: `{status}` | 切片数: `{record.chunk_count}`"
        )
    lines.extend(
        [
            "",
            "> 选中文件后，这里会自动展开对应的切片详情。",
        ]
    )
    return "\n".join(lines)


async def _empty_chunk_details() -> list[dict[str, object]]:
    return []


def _manage_details_placeholder_html(message: str) -> str:
    return (
        '<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px;background:#fafbfc;line-height:1.6;color:#5b6472;">'
        f"{message}"
        "</div>"
    )


def _render_manage_details_html(
    course_id: str,
    records: list[DocumentRecord],
    detail_lists: list[list[dict[str, object]]],
    search_query: str,
    selected_file_count: int,
    total_file_count: int,
    details_loaded: bool,
    total_chunk_count: int,
    current_page: int,
    page_size: int,
) -> tuple[str, str, int, int, int]:
    normalized_query = search_query.strip().lower()
    scope_text = (
        f"当前仅展示已选 {selected_file_count} 个文件。"
        if selected_file_count
        else f"当前展示整个知识库，共 {total_file_count} 个文件。"
    )
    loaded_file_count = len(records)
    loaded_chunk_count = (
        sum(len(chunk_details) for chunk_details in detail_lists)
        if details_loaded
        else total_chunk_count
    )
    matched_file_count = 0
    matched_chunk_count = 0
    cards: list[str] = []
    for record, chunk_details in zip(records, detail_lists):
        rendered = _render_manage_detail_card(record, chunk_details, normalized_query, details_loaded=details_loaded)
        if not rendered["include"]:
            continue
        matched_file_count += 1
        matched_chunk_count += int(rendered["matched_chunk_count"])
        cards.append(str(rendered["html"]))
    if not cards:
        status = (
            f"<i>已加载 {loaded_file_count} 个文件 / {loaded_chunk_count} 个切片。"
            f"{_escape_html(scope_text)} 当前搜索词“{_escape_html(search_query)}”没有命中任何文件或切片。</i>"
            if normalized_query
            else f"<i>已加载 {loaded_file_count} 个文件 / {loaded_chunk_count} 个切片。{_escape_html(scope_text)} 当前没有可展示的文件详情。</i>"
        )
        return _manage_details_placeholder_html("没有符合条件的文件或切片。"), status, 0, 1, 1
    overview = _render_manage_overview_card(course_id, records, loaded_chunk_count, open_by_default=not normalized_query)
    total_items = len(cards)
    total_pages = max(1, ceil(total_items / max(1, page_size)))
    resolved_page = max(1, min(current_page, total_pages))
    page_start = (resolved_page - 1) * max(1, page_size)
    page_end = page_start + max(1, page_size)
    paged_cards = cards[page_start:page_end]
    if normalized_query:
        status = (
            f"<i>已加载 {loaded_file_count} 个文件 / {loaded_chunk_count} 个切片。"
            f"{_escape_html(scope_text)} 搜索词“{_escape_html(search_query)}”命中 {matched_file_count} 个文件，"
            f"{matched_chunk_count} 个切片。当前第 {resolved_page} / {total_pages} 页。</i>"
        )
    elif details_loaded:
        status = (
            f"<i>已加载 {loaded_file_count} 个文件 / {loaded_chunk_count} 个切片。"
            f"{_escape_html(scope_text)} 当前已按所选文件加载切片详情。当前第 {resolved_page} / {total_pages} 页。</i>"
        )
    else:
        status = (
            f"<i>已加载 {loaded_file_count} 个文件 / {loaded_chunk_count} 个切片。"
            f"{_escape_html(scope_text)} 已自动展开知识库概览。建议先在上面的文件列表中选择需要加载的文件，再按需查看切片正文，因此切换知识库会更快。当前第 {resolved_page} / {total_pages} 页。</i>"
        )
    header = (
        '<div style="margin-bottom:10px;padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#f6f8fa;">'
        f"<b>知识库详情: {_escape_html(course_id)}</b>"
        "</div>"
    )
    return header + overview + "".join(paged_cards), status, total_items, total_pages, resolved_page


def _render_manage_overview_card(
    course_id: str,
    records: list[DocumentRecord],
    loaded_chunk_count: int,
    *,
    open_by_default: bool,
) -> str:
    vectorized_files = sum(1 for record in records if record.is_vectorized)
    unvectorized_files = len(records) - vectorized_files
    open_attr = " open" if open_by_default else ""
    body_parts = [
        '<div style="margin-top:10px;line-height:1.7;">',
        f"<div><b>知识库:</b> {_escape_html(course_id)}</div>",
        f"<div><b>文件总数:</b> {len(records)}</div>",
        f"<div><b>已向量化文件:</b> {vectorized_files}</div>",
        f"<div><b>未向量化文件:</b> {unvectorized_files}</div>",
        f"<div><b>当前已加载切片:</b> {loaded_chunk_count}</div>",
    ]
    if records:
        body_parts.append("<div style='margin-top:8px;'><b>文件列表:</b></div>")
        body_parts.append("<div style='margin-top:6px;'>")
        for record in records:
            status = "已向量化" if record.is_vectorized else "未向量化"
            body_parts.append(
                "<div style='margin:4px 0;'>"
                f"{_source_type_badge_html(record.source_type)} "
                f"<span style='margin-left:6px;'>{_escape_html(record.file_name)}</span>"
                f"<span style='color:#5b6472;margin-left:8px;'>{status} / {int(record.chunk_count)} 个切片</span>"
                "</div>"
            )
        body_parts.append("</div>")
    else:
        body_parts.append("<div style='margin-top:8px;color:#5b6472;'><i>当前知识库还没有文件。</i></div>")
    body_parts.append("</div>")
    return (
        '<details style="margin:10px 0;border:1px solid #d0d7de;border-radius:8px;padding:10px 12px;background:#f8fafc;"'
        f"{open_attr}>"
        f"<summary style='cursor:pointer;font-weight:600;'>知识库概览 | {_escape_html(course_id)}</summary>"
        + "".join(body_parts)
        + "</details>"
    )


def _render_manage_detail_card(
    record: DocumentRecord,
    chunk_details: list[dict[str, object]],
    normalized_query: str,
    *,
    details_loaded: bool,
) -> dict[str, object]:
    file_name_match = normalized_query in record.file_name.lower() if normalized_query else False
    visible_chunks = chunk_details
    if normalized_query:
        matched_chunks: list[dict[str, object]] = []
        for chunk in chunk_details:
            haystack = " ".join(
                [
                    str(chunk.get("chunk_id", "")),
                    str(chunk.get("page_label") or ""),
                    str(chunk.get("section_label") or ""),
                    str(chunk.get("content", "")),
                ]
            ).lower()
            if normalized_query in haystack:
                matched_chunks.append(chunk)
        if file_name_match and not matched_chunks:
            visible_chunks = chunk_details
        else:
            visible_chunks = matched_chunks
        if not file_name_match and not visible_chunks:
            return {"include": False, "matched_chunk_count": 0, "html": ""}
    status = "已向量化" if record.is_vectorized else "未向量化"
    type_label = _translate_source_type(record.source_type)
    matched_chunk_count = len(visible_chunks) if normalized_query or details_loaded else int(record.chunk_count or 0)
    summary_parts = [
        _escape_html(record.file_name),
        status,
        f"{record.chunk_count} 个切片",
    ]
    if normalized_query:
        summary_parts.append(f"命中 {matched_chunk_count} 个切片")
    summary_text = " | ".join(summary_parts)
    open_attr = " open" if normalized_query else ""
    body_parts = [
        '<div style="margin-top:10px;line-height:1.6;">',
        f"<div><b>文件类型:</b> {_source_type_badge_html(record.source_type)}</div>",
        f"<div><b>向量化状态:</b> {_escape_html(status)}</div>",
        f"<div><b>切片数:</b> {int(record.chunk_count)}</div>",
    ]
    if not details_loaded:
        body_parts.append(
            "<div style='margin-top:8px;color:#5b6472;'><i>当前为了加快切换速度，尚未加载该文件的切片正文。"
            "请先在上面的文件列表中选中需要查看的文件，详情区才会按需加载。</i></div>"
        )
    elif not visible_chunks:
        body_parts.append("<div style='margin-top:8px;color:#5b6472;'><i>该文件当前没有可展示的切片内容。</i></div>")
    else:
        for chunk in visible_chunks:
            locator = str(chunk.get("page_label") or chunk.get("section_label") or "-")
            merged_from = int(chunk.get("merged_from_count", 1))
            merge_label = f" | 合并自{merged_from}个小块" if merged_from > 1 else ""
            body_parts.extend(
                [
                    '<div style="margin-top:10px;padding:10px 12px;border:1px solid #e5e7eb;border-radius:8px;background:#ffffff;">',
                    f"<div style='font-weight:600;'>切片 {int(chunk.get('chunk_index', 0))} | {_escape_html(locator)} | 长度={int(chunk.get('length', 0))}{_escape_html(merge_label)}</div>",
                    f"<div style='margin-top:4px;color:#5b6472;'><b>切片ID:</b> {_escape_html(str(chunk.get('chunk_id', '')) or '-')}</div>",
                    f"<pre style='white-space:pre-wrap;overflow-x:auto;margin-top:8px;background:#f8fafc;padding:10px;border-radius:6px;border:1px solid #e5e7eb;'>{_escape_html(str(chunk.get('content', '')))}</pre>",
                    "</div>",
                ]
            )
    body_parts.append("</div>")
    html = (
        '<details style="margin:10px 0;border:1px solid #d0d7de;border-radius:8px;padding:10px 12px;background:#fff;"'
        f"{open_attr}>"
        f"<summary style='cursor:pointer;font-weight:600;'>{_source_type_badge_html(record.source_type)} <span style='margin-left:6px;'>{summary_text}</span></summary>"
        + "".join(body_parts)
        + "</details>"
    )
    return {
        "include": True,
        "matched_chunk_count": matched_chunk_count,
        "html": html,
    }


def _log_background_exception(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        print(traceback.format_exc())


def _upload_status_html(value, error: str) -> str:
    if error:
        return f"<b>上传错误:</b> {_escape_html(str(error))}"
    items = _normalize_upload_value(value)
    if not items:
        return "<i>还没有选中文件。提示：在系统文件选择框中按住 Cmd/Ctrl 或 Shift 可以多选；如果按钮上显示已选文件，但这里仍显示未选择，说明 JupyterHub 没把文件同步到 kernel，请改用下方“从工作区导入”。</i>"
    file_names = []
    for item in items:
        name = getattr(item, "name", None) or item.get("name", "")
        file_names.append(str(name))
    preview = "<br>".join(_escape_html(name) for name in file_names[:8])
    extra = "" if len(file_names) <= 8 else f"<br>... 还有 {len(file_names) - 8} 个文件"
    return f"<b>已选择 {len(file_names)} 个文件：</b><br>{preview}{extra}"


def _print_test_result(result: dict) -> None:
    print("ok:", result.get("ok"))
    print("elapsed_seconds:", result.get("elapsed_seconds"))
    if result.get("preview"):
        print("preview:", result["preview"])
    if result.get("error"):
        print("error:", result["error"])


def _resolve_workspace_dir(value: str) -> Path:
    directory = Path(value.strip() or ".")
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    return directory.resolve()


def _install_log_panel_support(display, HTML, Javascript) -> None:
    """Install one small front-end helper so log panels stay scrolled to the latest entry."""

    display(
        HTML(
            """
            <style>
            .assistant-log-panel {
              overflow: auto !important;
            }
            .assistant-log-panel .widget-output,
            .assistant-log-panel .jupyter-widgets-output-area,
            .assistant-log-panel .output,
            .assistant-log-panel .output_scroll {
              max-height: inherit !important;
              overflow: auto !important;
            }
            </style>
            """
        )
    )
    display(
        Javascript(
            """
            (function () {
              if (window.__assistantLogPanelAutoScrollInstalled) {
                return;
              }
              window.__assistantLogPanelAutoScrollInstalled = true;

              function resolveScrollTarget(root) {
                return (
                  root.querySelector('textarea') ||
                  root.querySelector('.output_scroll') ||
                  root.querySelector('.output') ||
                  root.querySelector('.jupyter-widgets-output-area') ||
                  root.querySelector('.widget-output') ||
                  root
                );
              }

              function attachObserver(root) {
                if (!root || root.__assistantLogObserverAttached) {
                  return;
                }
                root.__assistantLogObserverAttached = true;
                const target = resolveScrollTarget(root);
                const scrollToLatest = () => {
                  target.scrollTop = target.scrollHeight;
                };
                new MutationObserver(scrollToLatest).observe(root, {
                  childList: true,
                  subtree: true,
                  characterData: true,
                });
                scrollToLatest();
              }

              function scanPanels() {
                document.querySelectorAll('.assistant-log-panel').forEach(attachObserver);
              }

              new MutationObserver(scanPanels).observe(document.body, {
                childList: true,
                subtree: true,
              });
              scanPanels();
            })();
            """
        )
    )


def _scan_supported_files(root: Path) -> list[Path]:
    supported = {".pdf", ".md", ".txt", ".docx"}
    ignored_roots = {".venv", "storage", "reports", "data/raw", "src", "tests", "notebooks", ".idea", "__pycache__"}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue
        relative = str(path.relative_to(root))
        if any(part in ignored_roots for part in Path(relative).parts):
            continue
        files.append(path)
    return sorted(files)


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1

"""Transient and persistent chat memory backends."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from .models import ChatSessionSummary, ChatTurn


class BaseMemoryStore(Protocol):
    """Common async protocol shared by in-memory and SQLite-backed stores."""

    async def append_turn(self, session_id: str, turn: ChatTurn) -> None: ...

    async def get_recent_turns(self, session_id: str, limit: int = 12) -> list[ChatTurn]: ...

    async def get_turn_count(self, session_id: str) -> int: ...

    async def get_turns_page(self, session_id: str, limit: int = 20, offset: int = 0) -> list[ChatTurn]: ...

    async def get_session_profile(self, session_id: str) -> dict[str, Any]: ...

    async def set_session_profile(self, session_id: str, profile: dict[str, Any]) -> None: ...

    async def list_sessions(self, limit: int = 100) -> list[ChatSessionSummary]: ...

    async def delete_session(self, session_id: str) -> None: ...


class SessionMemoryStore:
    """Keep all session data in the current Python process only."""

    def __init__(self) -> None:
        self._turns: dict[str, list[ChatTurn]] = defaultdict(list)
        self._profiles: dict[str, dict[str, Any]] = defaultdict(dict)

    async def append_turn(self, session_id: str, turn: ChatTurn) -> None:
        self._turns[session_id].append(turn)

    async def get_recent_turns(self, session_id: str, limit: int = 12) -> list[ChatTurn]:
        return list(self._turns.get(session_id, []))[-limit:]

    async def get_turn_count(self, session_id: str) -> int:
        return len(self._turns.get(session_id, []))

    async def get_turns_page(self, session_id: str, limit: int = 20, offset: int = 0) -> list[ChatTurn]:
        if limit <= 0:
            return []
        start = max(0, int(offset))
        end = start + int(limit)
        return list(self._turns.get(session_id, []))[start:end]

    async def get_session_profile(self, session_id: str) -> dict[str, Any]:
        return dict(self._profiles.get(session_id, {}))

    async def set_session_profile(self, session_id: str, profile: dict[str, Any]) -> None:
        current = self._profiles.setdefault(session_id, {})
        current.update(profile)

    async def list_sessions(self, limit: int = 100) -> list[ChatSessionSummary]:
        session_ids = set(self._turns) | set(self._profiles)
        summaries: list[ChatSessionSummary] = []
        for session_id in session_ids:
            turns = self._turns.get(session_id, [])
            profile = self._profiles.get(session_id, {})
            last_updated = turns[-1].created_at if turns else None
            title = str(profile.get("session_title") or _default_session_title(session_id))
            summaries.append(
                ChatSessionSummary(
                    session_id=session_id,
                    title=title,
                    last_updated=last_updated,
                    turn_count=len(turns),
                    memory_mode=str(profile.get("memory_mode", "session")),
                    last_prompt_token_estimate=int(profile.get("last_prompt_token_estimate", 0) or 0),
                    last_context_compressed=bool(profile.get("last_context_compressed", False)),
                    last_context_doc_count=int(profile.get("last_context_doc_count", 0) or 0),
                    last_context_strategies=[
                        str(item)
                        for item in profile.get("last_context_strategies", [])
                        if str(item).strip()
                    ],
                    last_candidate_doc_count=int(profile.get("last_candidate_doc_count", 0) or 0),
                    last_rerank_kept_count=int(profile.get("last_rerank_kept_count", 0) or 0),
                    last_rerank_filtered_count=int(profile.get("last_rerank_filtered_count", 0) or 0),
                    last_low_score_filtered=bool(profile.get("last_low_score_filtered", False)),
                )
            )
        summaries.sort(
            key=lambda item: (
                item.last_updated.isoformat() if item.last_updated else "",
                item.session_id,
            ),
            reverse=True,
        )
        return summaries[:limit]

    async def delete_session(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
        self._profiles.pop(session_id, None)


def _default_session_title(session_id: str) -> str:
    return f"新会话 {session_id[-6:]}" if len(session_id) >= 6 else session_id


class SQLiteMemoryStore:
    """Persist chat turns and session profiles in a local SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_profiles (
                session_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    async def append_turn(self, session_id: str, turn: ChatTurn) -> None:
        """Append one chat turn into SQLite."""

        await asyncio.to_thread(self._append_turn_sync, session_id, turn)

    def _append_turn_sync(self, session_id: str, turn: ChatTurn) -> None:
        self._connection.execute(
            """
            INSERT INTO chat_turns(session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, turn.role, turn.content, turn.created_at.isoformat()),
        )
        self._connection.commit()

    async def get_recent_turns(self, session_id: str, limit: int = 12) -> list[ChatTurn]:
        """Return the latest ``limit`` turns in chronological order."""

        return await asyncio.to_thread(self._get_recent_turns_sync, session_id, limit)

    def _get_recent_turns_sync(self, session_id: str, limit: int) -> list[ChatTurn]:
        rows = self._connection.execute(
            """
            SELECT role, content, created_at
            FROM chat_turns
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            ChatTurn(role=row["role"], content=row["content"], created_at=_parse_datetime(row["created_at"]))
            for row in reversed(rows)
        ]

    async def get_turn_count(self, session_id: str) -> int:
        return await asyncio.to_thread(self._get_turn_count_sync, session_id)

    def _get_turn_count_sync(self, session_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM chat_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["count"] if row else 0)

    async def get_turns_page(self, session_id: str, limit: int = 20, offset: int = 0) -> list[ChatTurn]:
        return await asyncio.to_thread(
            self._get_turns_page_sync,
            session_id,
            max(0, int(limit)),
            max(0, int(offset)),
        )

    def _get_turns_page_sync(self, session_id: str, limit: int, offset: int) -> list[ChatTurn]:
        if limit <= 0:
            return []
        rows = self._connection.execute(
            """
            SELECT role, content, created_at
            FROM chat_turns
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT ?
            OFFSET ?
            """,
            (session_id, limit, offset),
        ).fetchall()
        return [
            ChatTurn(role=row["role"], content=row["content"], created_at=_parse_datetime(row["created_at"]))
            for row in rows
        ]

    async def get_session_profile(self, session_id: str) -> dict[str, Any]:
        """Load one session profile from SQLite."""

        return await asyncio.to_thread(self._get_session_profile_sync, session_id)

    def _get_session_profile_sync(self, session_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT profile_json FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return {}
        return json.loads(row["profile_json"])

    async def set_session_profile(self, session_id: str, profile: dict[str, Any]) -> None:
        """Upsert one session profile into SQLite."""

        await asyncio.to_thread(self._set_session_profile_sync, session_id, profile)

    def _set_session_profile_sync(self, session_id: str, profile: dict[str, Any]) -> None:
        current = self._get_session_profile_sync(session_id)
        current.update(profile)
        self._connection.execute(
            """
            INSERT INTO session_profiles(session_id, profile_json)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET profile_json = excluded.profile_json
            """,
            (session_id, json.dumps(current, ensure_ascii=False)),
        )
        self._connection.commit()

    async def list_sessions(self, limit: int = 100) -> list[ChatSessionSummary]:
        """List recent sessions for the left-side session selector."""

        return await asyncio.to_thread(self._list_sessions_sync, limit)

    def _list_sessions_sync(self, limit: int) -> list[ChatSessionSummary]:
        rows = self._connection.execute(
            """
            WITH turn_stats AS (
                SELECT session_id, MAX(created_at) AS last_updated, COUNT(*) AS turn_count
                FROM chat_turns
                GROUP BY session_id
            ),
            all_sessions AS (
                SELECT session_id FROM turn_stats
                UNION
                SELECT session_id FROM session_profiles
            )
            SELECT
                all_sessions.session_id AS session_id,
                turn_stats.last_updated AS last_updated,
                COALESCE(turn_stats.turn_count, 0) AS turn_count,
                session_profiles.profile_json AS profile_json
            FROM all_sessions
            LEFT JOIN turn_stats ON turn_stats.session_id = all_sessions.session_id
            LEFT JOIN session_profiles ON session_profiles.session_id = all_sessions.session_id
            ORDER BY COALESCE(turn_stats.last_updated, '') DESC, all_sessions.session_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        summaries: list[ChatSessionSummary] = []
        for row in rows:
            profile = json.loads(row["profile_json"]) if row["profile_json"] else {}
            session_id = str(row["session_id"])
            summaries.append(
                ChatSessionSummary(
                    session_id=session_id,
                    title=str(profile.get("session_title") or _default_session_title(session_id)),
                    last_updated=_parse_datetime(row["last_updated"]) if row["last_updated"] else None,
                    turn_count=int(row["turn_count"] or 0),
                    memory_mode=str(profile.get("memory_mode", "persistent")),
                    last_prompt_token_estimate=int(profile.get("last_prompt_token_estimate", 0) or 0),
                    last_context_compressed=bool(profile.get("last_context_compressed", False)),
                    last_context_doc_count=int(profile.get("last_context_doc_count", 0) or 0),
                    last_context_strategies=[
                        str(item)
                        for item in profile.get("last_context_strategies", [])
                        if str(item).strip()
                    ],
                    last_candidate_doc_count=int(profile.get("last_candidate_doc_count", 0) or 0),
                    last_rerank_kept_count=int(profile.get("last_rerank_kept_count", 0) or 0),
                    last_rerank_filtered_count=int(profile.get("last_rerank_filtered_count", 0) or 0),
                    last_low_score_filtered=bool(profile.get("last_low_score_filtered", False)),
                )
            )
        return summaries

    async def delete_session(self, session_id: str) -> None:
        """Delete all turns and profile data for one session."""

        await asyncio.to_thread(self._delete_session_sync, session_id)

    def _delete_session_sync(self, session_id: str) -> None:
        self._connection.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
        self._connection.execute("DELETE FROM session_profiles WHERE session_id = ?", (session_id,))
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()


def _parse_datetime(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)

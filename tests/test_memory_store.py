from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.memory_store import SessionMemoryStore, SQLiteMemoryStore
from src.models import ChatTurn


class SessionMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_memory_round_trip(self) -> None:
        store = SessionMemoryStore()
        turn = ChatTurn(role="user", content="hello", created_at=datetime.now(timezone.utc))
        await store.append_turn("session-1", turn)
        turns = await store.get_recent_turns("session-1")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].content, "hello")

        await store.set_session_profile("session-1", {"memory_mode": "session"})
        profile = await store.get_session_profile("session-1")
        self.assertEqual(profile["memory_mode"], "session")

    async def test_session_memory_lists_and_deletes_sessions(self) -> None:
        store = SessionMemoryStore()
        now = datetime.now(timezone.utc)
        await store.append_turn("session-a", ChatTurn(role="user", content="hello", created_at=now))
        await store.set_session_profile("session-a", {"session_title": "标题A", "memory_mode": "session"})

        sessions = await store.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].title, "标题A")

        await store.delete_session("session-a")
        self.assertEqual(await store.list_sessions(), [])


class SQLiteMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_memory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            store = SQLiteMemoryStore(db_path)
            turn = ChatTurn(role="assistant", content="hi", created_at=datetime.now(timezone.utc))
            await store.append_turn("session-2", turn)
            await store.set_session_profile("session-2", {"preferred_language": "zh"})

            turns = await store.get_recent_turns("session-2")
            profile = await store.get_session_profile("session-2")

            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0].role, "assistant")
            self.assertEqual(profile["preferred_language"], "zh")
            store.close()

    async def test_sqlite_memory_lists_and_deletes_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            store = SQLiteMemoryStore(db_path)
            turn = ChatTurn(role="user", content="hello world", created_at=datetime.now(timezone.utc))
            await store.append_turn("session-z", turn)
            await store.set_session_profile("session-z", {"session_title": "测试会话", "memory_mode": "persistent"})

            sessions = await store.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].title, "测试会话")

            await store.delete_session("session-z")
            self.assertEqual(await store.list_sessions(), [])
            store.close()


if __name__ == "__main__":
    unittest.main()

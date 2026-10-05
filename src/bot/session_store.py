"""Persistent, channel-scoped conversation sessions."""
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).resolve().parents[2] / "storage" / "sessions.sqlite3"


@dataclass
class Session:
    id: int
    channel_id: int
    guild_id: int
    summary: str
    summary_through: int


class SessionStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._initialized = False
        self._locks: dict[int, asyncio.Lock] = {}
        self._cancel: dict[int, set[asyncio.Task]] = {}
        self._closed: set[int] = set()

    async def initialize(self):
        if self._initialized:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA foreign_keys=ON")
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL UNIQUE,
                    guild_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_through INTEGER NOT NULL DEFAULT 0,
                    next_seq INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    subject_id TEXT,
                    source_metadata TEXT NOT NULL DEFAULT '{}',
                    event_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'complete',
                    UNIQUE(session_id, seq), UNIQUE(session_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS messages_session_seq ON messages(session_id, seq);
            """)
            await db.execute("UPDATE messages SET status='interrupted' WHERE status='processing'")
            await db.commit()
        self._initialized = True

    async def get(self, channel_id: int) -> Session | None:
        await self.initialize()
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM sessions WHERE channel_id=?", (channel_id,))
            row = await cur.fetchone()
        return Session(row["id"], row["channel_id"], row["guild_id"], row["summary"], row["summary_through"]) if row else None

    async def start(self, channel_id: int, guild_id: int) -> tuple[Session | None, bool]:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("INSERT OR IGNORE INTO sessions(channel_id,guild_id,created_at) VALUES(?,?,?)",
                             (channel_id, guild_id, datetime.now(timezone.utc).isoformat()))
            await db.commit()
        session = await self.get(channel_id)
        created = bool(session and session.guild_id == guild_id)
        # Determine whether INSERT OR IGNORE created it from timestamps is unreliable; serialize starts.
        return session, created

    async def create(self, channel_id: int, guild_id: int) -> tuple[Session | None, bool]:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("INSERT OR IGNORE INTO sessions(channel_id,guild_id,created_at) VALUES(?,?,?)",
                                   (channel_id, guild_id, datetime.now(timezone.utc).isoformat()))
            made = cur.rowcount == 1
            await db.commit()
        return await self.get(channel_id), made

    def lock_for(self, channel_id: int) -> asyncio.Lock:
        return self._locks.setdefault(channel_id, asyncio.Lock())

    def register_task(self, channel_id: int, task: asyncio.Task):
        self._cancel.setdefault(channel_id, set()).add(task)

    def unregister_task(self, channel_id: int, task: asyncio.Task):
        tasks = self._cancel.get(channel_id)
        if tasks:
            tasks.discard(task)
            if not tasks:
                self._cancel.pop(channel_id, None)

    async def is_active(self, session_id: int, channel_id: int) -> bool:
        session = await self.get(channel_id)
        return bool(session and session.id == session_id and session_id not in self._closed)

    async def context(self, session_id: int) -> tuple[str, list[dict]]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT summary,summary_through FROM sessions WHERE id=?", (session_id,))
            row = await cur.fetchone()
            if not row:
                return "", []
            cur = await db.execute("SELECT speaker,role,content,subject_id,seq FROM messages WHERE session_id=? AND seq>? AND status='complete' ORDER BY seq",
                                  (session_id, row["summary_through"]))
            messages = [dict(x) for x in await cur.fetchall()]
        return row["summary"], messages

    async def append(self, session_id: int, channel_id: int, speaker: str, role: str, content: str,
                     *, subject_id: str | None = None, event_id: int | None = None,
                     status: str = "complete", source_metadata: dict | None = None) -> bool:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA foreign_keys=ON")
            cur = await db.execute("SELECT id,next_seq FROM sessions WHERE id=? AND channel_id=?", (session_id, channel_id))
            row = await cur.fetchone()
            if not row or session_id in self._closed:
                return False
            try:
                await db.execute("INSERT INTO messages(session_id,seq,speaker,role,content,subject_id,source_metadata,event_id,status) VALUES(?,?,?,?,?,?,?,?,?)",
                                 (session_id, row[1], speaker, role, content, subject_id, json.dumps(source_metadata or {}), event_id, status))
                await db.execute("UPDATE sessions SET next_seq=next_seq+1 WHERE id=?", (session_id,))
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                await db.rollback()
                return False

    async def end(self, channel_id: int) -> bool:
        lock = self.lock_for(channel_id)
        self._closed.update(s.id for s in [await self.get(channel_id)] if s)
        current = asyncio.current_task()
        for task in tuple(self._cancel.get(channel_id, set())):
            if task is not current and not task.done():
                task.cancel()
        async with lock:
            async with aiosqlite.connect(self.path) as db:
                # SQLite foreign-key enforcement is per connection. Enable it
                # here so deleting a session also deletes its transcript.
                await db.execute("PRAGMA foreign_keys=ON")
                cur = await db.execute("DELETE FROM sessions WHERE channel_id=?", (channel_id,))
                await db.commit()
            session = await self.get(channel_id)
            if session:
                self._closed.discard(session.id)
            return cur.rowcount > 0

    async def replace_summary(self, session_id: int, summary: str, through: int) -> bool:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("UPDATE sessions SET summary=?,summary_through=? WHERE id=? AND summary_through<=?",
                                   (summary, through, session_id, through))
            await db.commit()
            return cur.rowcount == 1


session_store = SessionStore()

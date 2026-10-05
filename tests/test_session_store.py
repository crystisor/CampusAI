import aiosqlite
import pytest

from src.bot.session_store import SessionStore


@pytest.mark.asyncio
async def test_end_deletes_transcript_and_summary_only_for_its_channel(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    store = SessionStore(path)
    await store.initialize()
    ended, _ = await store.create(100, 1)
    retained, _ = await store.create(200, 1)
    for session in (ended, retained):
        assert await store.append(session.id, session.channel_id, "Student", "user", "Remember this", event_id=session.id)
        assert await store.append(session.id, session.channel_id, "CampusAI", "assistant", "Understood")
        assert await store.replace_summary(session.id, "A stored summary", 2)

    assert await store.end(100)
    assert not await store.end(100)

    # Inspect the messages table directly: a join would hide orphaned messages.
    async with aiosqlite.connect(path) as db:
        rows = await (await db.execute("SELECT session_id FROM messages ORDER BY seq")).fetchall()
        assert rows == [(retained.id,), (retained.id,)]
        assert await (await db.execute("PRAGMA foreign_key_check")).fetchall() == []
        assert await (await db.execute("SELECT id,summary FROM sessions")).fetchall() == [(retained.id, "A stored summary")]

    restored = SessionStore(path)
    await restored.initialize()
    assert await restored.get(100) is None
    fresh, created = await restored.create(100, 1)
    assert created and fresh.id != ended.id
    assert await restored.context(fresh.id) == ("", [])

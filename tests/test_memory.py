from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.memory import Store, format_time, relative_day


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "t.db")


def test_recall_returns_last_asks_even_without_a_match(tmp_path):
    store = make_store(tmp_path)
    thread = store.create_thread("one")
    store.add_turn(thread["id"], "user", "What is the capital of France?")
    store.add_turn(thread["id"], "user", "What is the capital of Japan?")
    store.add_turn(thread["id"], "user", "What did I ask about capitals?")

    block, items = store.memory_block("what did I ask earlier", thread_id=None)
    assert "capital of France" in block
    assert "capital of Japan" in block
    assert len(items) == 3


def test_fts_recall_finds_an_older_topic(tmp_path):
    store = make_store(tmp_path)
    thread = store.create_thread("t")
    store.add_turn(thread["id"], "user", "How do I install LangGraph?")
    for index in range(6):
        store.add_turn(thread["id"], "user", f"unrelated question number {index}")

    matches = store.matching_asks("langgraph install", limit=5)
    assert any("LangGraph" in item["text"] for item in matches)


def test_memory_block_carries_a_local_timestamp_and_relative_day(tmp_path):
    store = make_store(tmp_path)
    thread = store.create_thread("t")
    turn = store.add_turn(thread["id"], "user", "remember this question")

    block, items = store.memory_block("something else", thread_id=None)
    assert items[0]["created_at"] == turn.created_at
    formatted = format_time(turn.created_at)
    assert formatted in block
    assert relative_day(turn.created_at) in block
    assert "remember this question" in block


def test_timeline_is_newest_first_and_keeps_thread_titles(tmp_path):
    store = make_store(tmp_path)
    thread = store.create_thread("Budget trip")
    first = store.add_turn(thread["id"], "user", "cheap flights to Lisbon")
    second = store.add_turn(thread["id"], "user", "what about Porto")

    rows = store.timeline()
    assert [row["id"] for row in rows] == [second.id, first.id]
    assert rows[0]["thread_title"] == "Budget trip"


def test_delete_thread_removes_its_asks_from_memory(tmp_path):
    store = make_store(tmp_path)
    thread = store.create_thread("t")
    store.add_turn(thread["id"], "user", "delete me from recall")
    assert store.delete_thread(thread["id"]) is True
    assert store.matching_asks("recall") == []
    assert store.list_threads() == []


def test_relative_day_names_yesterday(tmp_path):
    moment = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    assert relative_day(moment) == "yesterday"

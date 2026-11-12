from pathlib import Path

from audiobookshelf_rag_ingester.heartbeat import is_fresh, touch


def test_heartbeat(tmp_path: Path) -> None:
    beat = tmp_path / "beat"
    assert not is_fresh(beat)
    touch(beat)
    assert is_fresh(beat)
    mtime = beat.stat().st_mtime
    assert is_fresh(beat, max_age=60, now=mtime + 59)
    assert not is_fresh(beat, max_age=60, now=mtime + 61)

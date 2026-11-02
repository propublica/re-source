import pytest

import re_source.sync_state as sync_state


def test_missing_state_is_empty(tmp_path):
    assert sync_state.load_sync_state(tmp_path / "missing.json").indexed == {}


def test_corrupt_state_fails_closed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{")

    with pytest.raises(RuntimeError, match="Could not load sync state"):
        sync_state.load_sync_state(path)


def test_failed_write_preserves_previous_state(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    original = '{"indexed": {"old": "hash"}}'
    path.write_text(original)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sync_state.json, "dump", fail)

    with pytest.raises(OSError, match="disk full"):
        sync_state.save_sync_state(
            sync_state.SyncState(indexed={"new": "hash"}), path
        )

    assert path.read_text() == original
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]

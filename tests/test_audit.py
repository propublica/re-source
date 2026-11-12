"""Audit trail event shape, S3 key layout, and failure behavior."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from re_source import audit, config


@dataclass
class FakeResult:
    """Enough of re_source.search_types.SearchResult for record_ref."""

    source_file: str = "tips_2024"
    doc_id: str = "doc-1"
    fields: dict = field(default_factory=dict)


@pytest.fixture
def file_sink(tmp_path, monkeypatch):
    """Route the audit log to a temp file and hand back its path."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(config, "AUDIT_SINK", "file")
    monkeypatch.setattr(config, "AUDIT_SPOOL_PATH", path)
    return path


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ----- record identity -----------------------------------------------------

def test_record_ref_uses_upstream_record_id():
    r = FakeResult(fields={"_record_id": "recAbc123"})
    assert audit.record_ref(r) == "tips_2024::recAbc123"


def test_record_ref_falls_back_to_doc_id():
    r = FakeResult(doc_id="tips_2024-chunk-7", fields={})
    assert audit.record_ref(r) == "tips_2024::tips_2024-chunk-7"


def test_record_refs_preserves_order():
    rs = [FakeResult(fields={"_record_id": f"rec{i}"}) for i in range(3)]
    assert audit.record_refs(rs) == [
        "tips_2024::rec0", "tips_2024::rec1", "tips_2024::rec2"
    ]


# ----- event shape ---------------------------------------------------------

def test_build_event_has_every_field():
    evt = audit.build_event(
        audit.EVENT_EXPORT,
        actor="dana@example.com",
        record_ids=["tips_2024::recA", "tips_2024::recB"],
        session_id="sess-1",
        strategy="Content",
        filename="results.xlsx",
    )
    expected = {
        "schema_version", "event_id", "ts", "event", "actor", "session_id",
        "surface", "strategy", "record_ids", "record_count",
        "filename", "page",
    }
    assert set(evt) == expected
    assert evt["record_count"] == 2
    assert evt["actor"] == "dana@example.com"


def test_build_event_timestamp_is_utc_iso8601():
    evt = audit.build_event(
        audit.EVENT_RESULTS_VIEWED,
        actor="a@x.com",
        record_ids=[],
        now=datetime(2026, 9, 10, 18, 22, 4, 113000, tzinfo=timezone.utc),
    )
    assert evt["ts"] == "2026-09-10T18:22:04.113Z"


def test_build_event_converts_naive_local_time_to_utc():
    from datetime import timedelta

    east = timezone(timedelta(hours=-4))
    evt = audit.build_event(
        audit.EVENT_RESULTS_VIEWED,
        actor="a@x.com",
        record_ids=[],
        now=datetime(2026, 9, 10, 14, 22, 4, 0, tzinfo=east),
    )
    assert evt["ts"] == "2026-09-10T18:22:04.000Z"


def test_event_ids_are_unique():
    ids = {
        audit.build_event(audit.EVENT_EXPORT, actor="a", record_ids=[])["event_id"]
        for _ in range(50)
    }
    assert len(ids) == 50


def test_record_ids_are_not_truncated():
    ids = [f"tips::rec{i}" for i in range(5000)]
    evt = audit.build_event(audit.EVENT_EXPORT, actor="a", record_ids=ids)
    assert evt["record_count"] == 5000
    assert evt["record_ids"][-1] == "tips::rec4999"


def test_build_event_copies_record_ids():
    ids = ["tips::recA"]
    evt = audit.build_event(audit.EVENT_EXPORT, actor="a", record_ids=ids)
    ids.append("tips::recB")
    assert evt["record_ids"] == ["tips::recA"]


# ----- S3 key layout -------------------------------------------------------

def test_s3_key_is_hive_partitioned():
    evt = audit.build_event(
        audit.EVENT_EXPORT,
        actor="a@x.com",
        record_ids=[],
        now=datetime(2026, 9, 10, 18, 22, 4, 113000, tzinfo=timezone.utc),
    )
    key = audit.s3_key(evt, "audit")
    assert key.startswith("audit/dt=2026-09-10/event=export/")
    assert key.endswith(".jsonl")
    # Object name is epoch-ms first, so keys sort chronologically within a day.
    epoch_ms = int(
        datetime(2026, 9, 10, 18, 22, 4, 113000, tzinfo=timezone.utc).timestamp() * 1000
    )
    assert key.rsplit("/", 1)[1].startswith(f"{epoch_ms}-")


def test_s3_key_carries_no_actor_identity():
    evt = audit.build_event(
        audit.EVENT_EXPORT, actor="dana@example.com", record_ids=[]
    )
    assert "dana" not in audit.s3_key(evt, "audit")


def test_s3_keys_do_not_collide_within_the_same_millisecond():
    now = datetime(2026, 9, 10, 18, 22, 4, 113000, tzinfo=timezone.utc)
    keys = {
        audit.s3_key(
            audit.build_event(
                audit.EVENT_EXPORT, actor="a", record_ids=[], now=now
            ),
            "audit",
        )
        for _ in range(50)
    }
    assert len(keys) == 50


def test_s3_key_tolerates_empty_prefix():
    evt = audit.build_event(audit.EVENT_EXPORT, actor="a", record_ids=[])
    assert audit.s3_key(evt, "").startswith("dt=")


# ----- log_event dispatch --------------------------------------------------

def test_log_event_writes_to_file_sink(file_sink):
    audit.log_event(
        audit.EVENT_EXPORT,
        actor="dana@example.com",
        record_ids=["tips_2024::recA"],
        session_id="sess-1",
        strategy="Content",
        filename="out.xlsx",
    )
    (written,) = _lines(file_sink)
    assert written["event"] == "export"
    assert written["record_ids"] == ["tips_2024::recA"]
    assert written["filename"] == "out.xlsx"


def test_log_event_appends_rather_than_overwrites(file_sink):
    audit.log_event(audit.EVENT_RESULTS_VIEWED, actor="a", record_ids=["x::1"], page=1)
    audit.log_event(audit.EVENT_RESULTS_VIEWED, actor="a", record_ids=["x::2"], page=2)
    assert [e["page"] for e in _lines(file_sink)] == [1, 2]


def test_log_event_off_writes_nothing(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(config, "AUDIT_SINK", "off")
    monkeypatch.setattr(config, "AUDIT_SPOOL_PATH", path)

    assert audit.log_event(audit.EVENT_EXPORT, actor="a", record_ids=["x::1"]) is None
    assert not path.exists()


def test_log_event_creates_missing_parent_directory(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "deeper" / "audit.jsonl"
    monkeypatch.setattr(config, "AUDIT_SINK", "file")
    monkeypatch.setattr(config, "AUDIT_SPOOL_PATH", path)

    audit.log_event(audit.EVENT_EXPORT, actor="a", record_ids=["x::1"])
    assert len(_lines(path)) == 1


# ----- failure behavior ----------------------------------------------------

def test_sink_failure_does_not_propagate_and_spools(tmp_path, monkeypatch):
    spool = tmp_path / "spool.jsonl"
    monkeypatch.setattr(config, "AUDIT_SINK", "s3")
    monkeypatch.setattr(config, "AUDIT_S3_BUCKET", "some-bucket")
    monkeypatch.setattr(config, "AUDIT_S3_PREFIX", "audit")
    monkeypatch.setattr(config, "AUDIT_SPOOL_PATH", spool)

    class ExplodingSink:
        def write(self, evt):
            raise RuntimeError("S3 is down")

    monkeypatch.setattr(audit, "_active_sink", lambda: ExplodingSink())

    evt = audit.log_event(audit.EVENT_EXPORT, actor="a@x.com", record_ids=["x::1"])

    assert evt is not None
    (spooled,) = _lines(spool)
    assert spooled["event_id"] == evt["event_id"]


def test_s3_sink_without_bucket_spools_instead_of_dropping(tmp_path, monkeypatch):
    spool = tmp_path / "spool.jsonl"
    monkeypatch.setattr(config, "AUDIT_SINK", "s3")
    monkeypatch.setattr(config, "AUDIT_S3_BUCKET", "")
    monkeypatch.setattr(config, "AUDIT_SPOOL_PATH", spool)

    audit.log_event(audit.EVENT_EXPORT, actor="a@x.com", record_ids=["x::1"])
    assert len(_lines(spool)) == 1


def test_spool_failure_still_does_not_propagate(monkeypatch, caplog):
    monkeypatch.setattr(config, "AUDIT_SINK", "s3")
    monkeypatch.setattr(config, "AUDIT_S3_BUCKET", "b")
    monkeypatch.setattr(audit, "_active_sink", lambda: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(audit, "_spool", lambda evt: (_ for _ in ()).throw(OSError("also nope")))

    # Must not raise; the log stream becomes the only surviving copy.
    audit.log_event(audit.EVENT_EXPORT, actor="a@x.com", record_ids=["x::1"])
    assert "unrecorded event" in caplog.text


# ----- S3 sink mechanics (no network) --------------------------------------

def test_s3_sink_puts_ndjson_object():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def put_object(self, **kwargs):
            self.calls.append(kwargs)

    sink = audit._S3Sink("my-bucket", "audit", "us-east-1")
    fake = FakeClient()
    sink._client = fake

    evt = audit.build_event(audit.EVENT_EXPORT, actor="a@x.com", record_ids=["x::1"])
    sink.write(evt)

    (call,) = fake.calls
    assert call["Bucket"] == "my-bucket"
    assert call["Key"].startswith("audit/dt=")
    assert call["ContentType"] == "application/x-ndjson"
    # One event per object, newline-terminated so objects concatenate into JSONL.
    body = call["Body"].decode("utf-8")
    assert body.endswith("\n")
    assert json.loads(body)["event_id"] == evt["event_id"]

"""Record-format v1 validator (docs/record-format.md).

validate_record checks one record's schema; validate_probe_stream checks the stream-level rules
of probe.jsonl; validate_run_dir checks a whole run directory. Every function returns a list of
human-readable error strings; an empty list means the input is valid.

The rules encode the doc, not the probes: if a probe starts emitting a new record type, the doc
gets a row and TYPE_FIELDS below gets an entry, and the validator follows.
"""

from __future__ import annotations

import json
from pathlib import Path

FORMAT_VERSION = 1

# Every record, whatever its source.
COMMON = ("v", "t_ns", "src", "type")

SOURCES = ("probe", "runner", "sampler", "kernel", "load", "crosscheck")

# type -> required fields (beyond v/t_ns/src/type).
TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    # probe records
    "probe_start": ("params",),
    "declared": ("key", "value", "source"),
    "measure": ("name", "value", "unit", "tags", "verified"),
    "check": ("name", "pass", "detail"),
    "feature": ("feature", "requested", "granted", "effect_observed", "evidence"),
    "note": ("message",),
    "error": ("message",),
    "probe_end": ("status", "failed_checks"),
    # runner records
    "step_start": ("step", "argv", "drop_caches"),
    "step_end": ("step", "status", "returncode", "duration_s", "timed_out", "unkillable",
                 "interrupted"),
    "abort": (),
    "stdout_text": ("text",),
    # sampler records
    "sample": (),
    # kernel records
    "kernel_event": ("category", "priority", "seq", "message"),
    "kernel_event_gap": ("lost", "between_seq"),
    # load-generator records (load.jsonl)
    "load_start": ("load",),
    "load_ready": (),
    "heartbeat": (),
    "load_end": (),
    "load_error": (),
    # crosscheck verdict records (crosscheck.jsonl, written by ./echolock crosscheck --check)
    "crosscheck": ("probe", "name", "a", "b", "verdict", "tolerance", "detail"),
}

PROBE_END_STATUSES = ("ok", "verify_failed", "error")
STEP_STATUSES = ("ok", "verify_failed", "error", "timeout", "incomplete")
CROSSCHECK_VERDICTS = ("agrees", "diverges", "within-tolerance", "missing")

MANIFEST_FIELDS = ("v", "run_id", "probe", "backend", "label", "status", "started_utc", "steps")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_record(rec: object) -> list[str]:
    """Schema errors for one record; empty list when the record is well formed."""
    if not isinstance(rec, dict):
        return [f"record is a {type(rec).__name__}, not an object"]
    errs = []
    for f in COMMON:
        if f not in rec:
            errs.append(f"missing field '{f}'")
    if "v" in rec and rec["v"] != FORMAT_VERSION:
        errs.append(f"v is {rec['v']!r}, expected {FORMAT_VERSION}")
    if "t_ns" in rec and not _is_int(rec["t_ns"]):
        errs.append(f"t_ns is {rec['t_ns']!r}, expected an integer (ns)")
    if "src" in rec and rec["src"] not in SOURCES:
        errs.append(f"src is {rec['src']!r}, expected one of {', '.join(SOURCES)}")
    t = rec.get("type")
    if "type" in rec and t not in TYPE_FIELDS:
        errs.append(f"type is {t!r}, unknown")
        return errs
    for f in TYPE_FIELDS.get(t, ()):
        if f not in rec:
            errs.append(f"{t}: missing field '{f}'")
    if rec.get("src") == "probe":
        for f in ("probe", "backend"):
            if f not in rec:
                errs.append(f"probe record missing field '{f}'")
    if t == "measure" and "verified" in rec and not isinstance(rec["verified"], bool):
        errs.append("measure: verified must be a bool")
    if t == "probe_end" and "status" in rec and rec["status"] not in PROBE_END_STATUSES:
        errs.append(f"probe_end: status is {rec['status']!r}, expected one of "
                    f"{', '.join(PROBE_END_STATUSES)}")
    if t == "step_end" and "status" in rec and rec["status"] not in STEP_STATUSES:
        errs.append(f"step_end: status is {rec['status']!r}, expected one of "
                    f"{', '.join(STEP_STATUSES)}")
    if t == "crosscheck":
        if "verdict" in rec and rec["verdict"] not in CROSSCHECK_VERDICTS:
            errs.append(f"crosscheck: verdict is {rec['verdict']!r}, expected one of "
                        f"{', '.join(CROSSCHECK_VERDICTS)}")
        for side in ("a", "b"):
            s = rec.get(side)
            if isinstance(s, dict) and "run_id" not in s:
                errs.append(f"crosscheck: {side} missing 'run_id'")
    return errs


def _parse_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    recs, errs = [], []
    if not path.is_file():
        return recs, []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"{path.name}:{n}: not JSON ({e.msg})")
            continue
        if not isinstance(rec, dict):
            errs.append(f"{path.name}:{n}: a {type(rec).__name__}, not an object")
            continue
        recs.append(rec)
    return recs, errs


def validate_probe_stream(records: list[dict]) -> list[str]:
    """Stream-level rules for probe.jsonl: probe_start first, probe_end last, and the end
    record's status consistent with the checks and errors the probe reported."""
    errs = []
    probe_recs = [r for r in records if r.get("src") == "probe"
                  and r.get("type") in TYPE_FIELDS]
    if not probe_recs:
        return errs  # the step died before the probe wrote anything
    first, last = probe_recs[0], probe_recs[-1]
    if first.get("type") != "probe_start":
        errs.append(f"first probe record is {first.get('type')!r}, expected probe_start")
    if last.get("type") != "probe_end":
        errs.append(f"last probe record is {last.get('type')!r}, expected probe_end")
        return errs
    failed = [r for r in probe_recs if r.get("type") == "check" and not r.get("pass")]
    has_error = any(r.get("type") == "error" for r in probe_recs)
    expect = "error" if has_error else "verify_failed" if failed else "ok"
    if last.get("status") != expect:
        errs.append(f"probe_end status is {last.get('status')!r}, but the stream says "
                    f"{len(failed)} failed check(s) and {int(has_error)} error record(s), "
                    f"which requires {expect!r}")
    if last.get("failed_checks") != len(failed):
        errs.append(f"probe_end failed_checks is {last.get('failed_checks')!r}, "
                    f"the stream has {len(failed)} failed check(s)")
    return errs


def _check_light(recs: list[dict], path_name: str) -> list[str]:
    """Lighter validation for the side streams (samples/kernel/load/crosscheck): v, t_ns, a
    known src and a known type. Their per-type payloads are the producers' concern."""
    errs = []
    for n, r in enumerate(recs, 1):
        if r.get("v") != FORMAT_VERSION:
            errs.append(f"{path_name}:{n}: v is {r.get('v')!r}, expected {FORMAT_VERSION}")
        if not _is_int(r.get("t_ns")):
            errs.append(f"{path_name}:{n}: t_ns missing or not an integer")
        if r.get("src") not in SOURCES:
            errs.append(f"{path_name}:{n}: src is {r.get('src')!r}, unknown")
        if r.get("type") not in TYPE_FIELDS:
            errs.append(f"{path_name}:{n}: type is {r.get('type')!r}, unknown")
    return errs


def validate_run_dir(run_dir: Path) -> list[str]:
    """Validate one run directory: manifest, probe stream, and the lighter streams."""
    errs = []
    if not run_dir.is_dir():
        return [f"{run_dir}: not a directory"]
    mpath = run_dir / "manifest.json"
    try:
        manifest = json.loads(mpath.read_text())
    except OSError as e:
        return [f"manifest.json: {e.strerror}"]
    except json.JSONDecodeError as e:
        return [f"manifest.json: not JSON ({e.msg})"]
    if not isinstance(manifest, dict):
        return [f"manifest.json: a {type(manifest).__name__}, not an object"]
    for f in MANIFEST_FIELDS:
        if f not in manifest:
            errs.append(f"manifest.json: missing field '{f}'")
    if "v" in manifest and manifest["v"] != FORMAT_VERSION:
        errs.append(f"manifest.json: v is {manifest['v']!r}, expected {FORMAT_VERSION}")
    if "status" in manifest and manifest["status"] not in STEP_STATUSES:
        errs.append(f"manifest.json: status is {manifest['status']!r}, expected one of "
                    f"{', '.join(STEP_STATUSES)}")
    steps = manifest.get("steps")
    if isinstance(steps, list) and steps:
        worst = min((s.get("status") for s in steps if isinstance(s, dict)),
                    key=lambda s: STEP_STATUSES.index(s) if s in STEP_STATUSES else len(STEP_STATUSES),
                    default=None)
        if worst is not None and manifest.get("status") not in (None, worst):
            errs.append(f"manifest.json: status is {manifest['status']!r}, the steps' worst is "
                        f"{worst!r}")
    if "env.json" in _names(run_dir):
        try:
            env = json.loads((run_dir / "env.json").read_text())
            if not isinstance(env, dict) or env.get("v") != FORMAT_VERSION:
                errs.append("env.json: not a v1 object")
        except (OSError, json.JSONDecodeError) as e:
            errs.append(f"env.json: {e}")

    records, e = _parse_jsonl(run_dir / "probe.jsonl")
    errs.extend(e)
    for n, r in enumerate(records, 1):
        for err in validate_record(r):
            errs.append(f"probe.jsonl:{n}: {err}")
    errs.extend(validate_probe_stream(records))

    for name in ("samples.jsonl", "kernel.jsonl", "load.jsonl", "crosscheck.jsonl"):
        recs, e = _parse_jsonl(run_dir / name)
        errs.extend(e)
        errs.extend(_check_light(recs, name))
    return errs


def _names(run_dir: Path) -> set[str]:
    try:
        return {p.name for p in run_dir.iterdir()}
    except OSError:
        return set()


def validate_all(run_root: Path) -> dict[str, list[str]]:
    """Every run directory under run_root -> its validation errors (empty when clean)."""
    out = {}
    for child in sorted(run_root.iterdir()):
        if child.is_dir() and (child / "manifest.json").is_file():
            errs = validate_run_dir(child)
            out[child.name] = errs
    return out

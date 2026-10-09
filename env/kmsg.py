"""Kernel log watcher: GPU driver faults (NVIDIA Xid and others), OOM kills, hangs.

Reads /dev/kmsg, whose timestamps are CLOCK_MONOTONIC in microseconds -- the same clock as every
record's t_ns -- so a kernel event lines up with the probe sample and the 1 s environment sample
around it. Lines logged before the watcher started are summarized separately as history, so an
Xid from an earlier run is not attributed to this one.
"""

from __future__ import annotations

import errno
import json
import os
import re
import select
import threading
import time
from dataclasses import dataclass
from typing import IO, Callable

# (category, pattern). The first match wins; unmatched lines are ignored.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("xid", re.compile(r"NVRM: Xid \((?P<device>[^)]*)\): (?P<xid>\d+),\s*(?P<detail>.*)")),
    ("sxid", re.compile(r"NVRM: SXid \((?P<device>[^)]*)\): (?P<xid>\d+),\s*(?P<detail>.*)")),
    # A reset means the driver threw the work away, which no number taken around it survives.
    ("reset", re.compile(r"GPU reset|gpu reset begin|gpu reset (end|succeeded|failed)|"
                         r"TDR:.*reset|bus resets?")),
    ("nvrm", re.compile(r"NVRM: .*(error|fault|failed|timeout|GPU has fallen off|reset)", re.I)),
    ("gpu", re.compile(r"(amdgpu|i915|xe|panfrost|panthor|msm|nouveau|nova)[^;]*?"
                       r"(error|fault|hang|timeout|reset|ring \w+ trigger|MODE: \(\d+\))", re.I)),
    ("oom", re.compile(r"(invoked oom-killer|Out of memory|oom-kill:|Killed process)")),
    ("hang", re.compile(r"(blocked for more than \d+ seconds|soft lockup|hard LOCKUP|"
                        r"rcu_.*(stall|self-detected))")),
    ("iommu", re.compile(r"(arm-smmu|iommu|SMMU).*(fault|error)", re.I)),
    ("mce", re.compile(r"(Machine check|mce:|EDAC|RAS: .*error|\{\d+\}\[Hardware Error\])")),
]


@dataclass
class KernelEvent:
    ts_ns: int
    seq: int
    priority: int
    category: str
    message: str
    xid: int | None = None
    device: str | None = None

    def record(self) -> dict:
        rec = {"v": 1, "t_ns": self.ts_ns, "src": "kernel", "type": "kernel_event",
               "category": self.category, "priority": self.priority, "seq": self.seq,
               "message": self.message}
        if self.xid is not None:
            rec["xid"] = self.xid
            rec["device"] = self.device
        return rec


def classify(message: str) -> tuple[str, re.Match[str]] | None:
    for category, pattern in PATTERNS:
        m = pattern.search(message)
        if m:
            return category, m
    return None


def parse_kmsg_record(raw: str) -> KernelEvent | None:
    """Parses one /dev/kmsg record ("prio,seq,ts_us,flags;message\\n key=value...")."""
    head, sep, rest = raw.partition(";")
    if not sep:
        return None
    fields = head.split(",")
    if len(fields) < 3:
        return None
    try:
        prio = int(fields[0]) & 7
        seq = int(fields[1])
        ts_us = int(fields[2])
    except ValueError:
        return None
    message = rest.split("\n", 1)[0]
    hit = classify(message)
    if hit is None:
        return None
    category, m = hit
    xid = int(m.group("xid")) if category in ("xid", "sxid") else None
    device = m.group("device") if category in ("xid", "sxid") else None
    return KernelEvent(ts_us * 1000, seq, prio, category, message, xid, device)


class KernelLogWatcher:
    """Background thread that writes matching kernel log events as JSONL records to `out`."""

    def __init__(self, out: IO[str] | None = None,
                 on_event: Callable[[KernelEvent], None] | None = None,
                 path: str = "/dev/kmsg"):
        self.out = out
        self.on_event = on_event
        self.path = path
        self.events: list[KernelEvent] = []
        self.history: list[KernelEvent] = []
        self.available = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None

    def start(self) -> "KernelLogWatcher":
        try:
            self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            self.error = f"{self.path}: {e.strerror}"
            return self
        self.available = True
        # Everything readable now predates the run.
        self._drain(history=True)
        self._thread = threading.Thread(target=self._loop, name="kmsg", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._fd is not None:
            self._drain(history=False)
            os.close(self._fd)
            self._fd = None

    def summary(self) -> dict:
        def count(events: list[KernelEvent]) -> dict:
            out: dict[str, int] = {}
            for e in events:
                out[e.category] = out.get(e.category, 0) + 1
            return out

        return {
            "available": self.available,
            "error": self.error,
            "during_run": count(self.events),
            "xids_during_run": sorted({e.xid for e in self.events if e.xid is not None}),
            "before_run": count(self.history),
            "before_run_last": [e.record() for e in self.history[-20:]],
        }

    def _loop(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.2)
            if ready:
                self._drain(history=False)

    def _drain(self, history: bool) -> None:
        assert self._fd is not None
        while True:
            try:
                raw = os.read(self._fd, 16384)
            except BlockingIOError:
                return
            except OSError as e:
                if e.errno == errno.EPIPE:  # ring buffer overran our position; resume
                    continue
                if e.errno == errno.EINVAL:
                    return
                self.error = f"read {self.path}: {e.strerror}"
                return
            if not raw:
                return
            event = parse_kmsg_record(raw.decode("utf-8", errors="replace"))
            if event is None:
                continue
            if history:
                self.history.append(event)
                continue
            self.events.append(event)
            if self.out is not None:
                self.out.write(json.dumps(event.record()) + "\n")
                self.out.flush()
            if self.on_event:
                self.on_event(event)


def main() -> None:
    import sys

    w = KernelLogWatcher(out=sys.stdout).start()
    if not w.available:
        print(json.dumps({"v": 1, "t_ns": time.monotonic_ns(), "src": "kernel",
                          "type": "error", "message": w.error}), flush=True)
        raise SystemExit(1)
    print(json.dumps({"v": 1, "t_ns": time.monotonic_ns(), "src": "kernel",
                      "type": "watch_start", "history": w.summary()["before_run"]}), flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        w.stop()


if __name__ == "__main__":
    main()

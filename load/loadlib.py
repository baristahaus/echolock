"""Shared plumbing for the load generators: JSONL records on stdout, SIGTERM handling, and a
parent-death watchdog, so a load never outlives the runner that started it.

Record protocol (all src "load"):
  load_start  {load, intended, args, deadline_min}      first record
  load_ready  {achieved, detail}                         steady state reached
  heartbeat    {achieved, detail}                        every few seconds
  load_end    {achieved, detail}                         last record, on SIGTERM or deadline
  load_error  {message}                                   then exit
"""

import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def emit(type_: str, **fields) -> None:
    rec = {"v": 1, "t_ns": time.monotonic_ns(), "src": "load", "type": type_, **fields}
    print(json.dumps(rec), flush=True)


class Stop(Exception):
    """Raised in the main thread on SIGTERM; worker threads watch their stop event."""


def install_stop(deadline_min: float = 120.0) -> float:
    """SIGTERM raises Stop in the main thread; returns the deadline in monotonic seconds."""
    def handler(_sig, _frame):
        raise Stop()

    signal.signal(signal.SIGTERM, handler)
    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        pass
    return time.monotonic() + deadline_min * 60.0


def make_parent_check(parent_pid: int | None):
    """A load whose runner died must not keep holding the machine.

    With an explicit --parent-pid the check is /proc/<pid> liveness: the runner is alive when it
    starts the load, so there is no window where a dead parent goes unnoticed (getppid() alone
    fails exactly there -- a load that starts after its parent died captures the reaper's pid
    and never notices)."""
    if parent_pid is None:
        parent = os.getppid()

        def check() -> None:
            if os.getppid() != parent:
                raise Stop()
    else:
        def check() -> None:
            if not os.path.exists(f"/proc/{parent_pid}"):
                raise Stop()
    return check


def parent_pid_of(args) -> int | None:
    """The --parent-pid argument, if the runner passed one."""
    return int(args["parent-pid"]) if "parent-pid" in args else None


def parse_args(argv, usage: str):
    """--key value pairs with a usage message on error."""
    out = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--help", "-h"):
            print(usage)
            raise SystemExit(0)
        if not a.startswith("--") or i + 1 >= len(argv):
            print(f"{usage}\nunknown or incomplete argument: {a}", file=sys.stderr)
            raise SystemExit(2)
        out[a[2:]] = argv[i + 1]
        i += 2
    return out

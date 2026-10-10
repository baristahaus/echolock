"""Memory fill: touch a fraction of physical RAM and hold it, so "90% fill" is measured
resident pages, not an allocation the kernel could reclaim.

The touch pass writes one byte per 4 KiB page, then the process holds the mapping and reports
its resident size from /proc/self/statm until stopped. Ready is only emitted once the touch
pass has actually completed -- the grid runner waits for it, so a probe never starts against a
half-filled machine.

Usage: memfill.py [--fraction F (0.9)] [--max-gib N (none)] [--deadline-min 120]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadlib import (Stop, emit, install_stop, make_parent_check, parent_pid_of,
                         parse_args)

USAGE = "usage: memfill.py [--fraction F (0.9)] [--max-gib N (none)] [--deadline-min 120]"
PAGE = 4096


def memtotal_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def statm_resident_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE


def main() -> None:
    args = parse_args(sys.argv[1:], USAGE)
    fraction = float(args.get("fraction", 0.9))
    max_gib = float(args["max-gib"]) if "max-gib" in args else None
    deadline_min = float(args.get("deadline-min", 120))
    if not (0 < fraction <= 1):
        print(USAGE + "\nfraction must be in (0, 1]", file=sys.stderr)
        raise SystemExit(2)

    emit("load_start", load="mem", intended={"fraction": fraction, "max_gib": max_gib},
         args=sys.argv[1:], deadline_min=deadline_min)
    parent_pid = parent_pid_of(args)

    total = memtotal_bytes()
    size = int(total * fraction)
    if max_gib is not None:
        size = min(size, int(max_gib * (1 << 30)))
    pages = size // PAGE
    deadline = install_stop(deadline_min)
    check_parent = make_parent_check(parent_pid)

    buf = bytearray(size)
    t0 = time.monotonic()
    touched = 0
    try:
        for i in range(0, size, PAGE):
            buf[i] = (i // PAGE) & 0xFF
            touched += 1
            if touched % (1 << 20) == 0:  # every 4 GiB of pages
                check_parent()  # a long touch pass must not outlive a dead runner either
                emit("heartbeat", achieved={"touched_gib": round(touched * PAGE / 2**30, 1),
                                            "of_gib": round(size / 2**30, 1)})
    except Stop:
        emit("load_error", message="stopped during touch pass")
        raise SystemExit(0)

    fill_s = time.monotonic() - t0
    emit("load_ready", achieved={"resident_bytes": statm_resident_bytes(), "size_bytes": size,
                                 "touch_seconds": round(fill_s, 1)})
    try:
        while True:
            time.sleep(2.0)
            check_parent()
            if time.monotonic() > deadline:
                raise Stop()
            emit("heartbeat", achieved={"resident_bytes": statm_resident_bytes()})
    except Stop:
        pass
    finally:
        # Measure before freeing: the achieved fill is what was resident while holding.
        resident_at_stop = statm_resident_bytes()
        del buf
        emit("load_end", achieved={"resident_bytes": resident_at_stop,
                                   "size_bytes": size, "touch_seconds": round(fill_s, 1)})


if __name__ == "__main__":
    main()

"""CPU load: N threads doing GIL-releasing hash work at a duty cycle, so "50% CPU" is a real
occupancy measured from /proc, not an intention.

The busy work is hashlib.sha256 over a resident buffer: it releases the GIL for large inputs,
so N threads really do occupy N cores during their busy windows. Achieved duty comes from
/proc/self/stat (utime+stime over wall time, per thread).

Usage: cpuload.py [--threads N (all)] [--duty P (50)] [--buffer-mib 64] [--deadline-min 120]
"""

import hashlib
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadlib import (Stop, emit, install_stop, make_parent_check, parent_pid_of,
                         parse_args)

USAGE = ("usage: cpuload.py [--threads N (all CPUs)] [--duty P (50)] "
         "[--buffer-mib 64] [--deadline-min 120]")


def read_stat_seconds() -> float:
    with open("/proc/self/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    # utime, stime are fields 14, 15 (1-based after the comm field): ticks at 100 Hz.
    return (int(parts[11]) + int(parts[12])) / 100.0


def main() -> None:
    args = parse_args(sys.argv[1:], USAGE)
    threads = int(args.get("threads", os.cpu_count() or 1))
    duty = float(args.get("duty", 50)) / 100.0
    buffer_mib = int(args.get("buffer-mib", 64))
    deadline_min = float(args.get("deadline-min", 120))
    if not (0 < duty <= 1) or threads < 1 or buffer_mib < 1:
        print(USAGE + "\nvalues must be positive", file=sys.stderr)
        raise SystemExit(2)

    emit("load_start", load="cpu", intended={"threads": threads, "duty": duty},
         args=sys.argv[1:], deadline_min=deadline_min)
    parent_pid = parent_pid_of(args)

    buf = bytes(buffer_mib << 20)
    mv = memoryview(buf)
    chunk = 4 << 20
    on_s = 0.2
    off_s = 0.2 * (1 - duty) / duty if duty < 1 else 0.0
    stop = threading.Event()

    def worker() -> None:
        i = 0
        while not stop.is_set():
            t0 = time.monotonic()
            while time.monotonic() - t0 < on_s and not stop.is_set():
                # sha256 over a 4 MiB view releases the GIL, so N threads really occupy N cores;
                # the small chunk keeps the busy window close to on_s.
                hashlib.sha256(mv[i:i + chunk]).digest()
                i = (i + chunk) % len(mv)
            if off_s > 0 and not stop.is_set():
                stop.wait(off_s)

    pool = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for t in pool:
        t.start()

    deadline = install_stop(deadline_min)
    check_parent = make_parent_check(parent_pid)
    emit("load_ready", achieved={"threads": threads, "duty": duty})

    cpu0, t0 = read_stat_seconds(), time.monotonic()
    last_cpu, last_t = cpu0, t0
    try:
        while True:
            time.sleep(2.0)
            check_parent()
            if time.monotonic() > deadline:
                raise Stop()
            cpu, now = read_stat_seconds(), time.monotonic()
            emit("heartbeat", achieved={"duty": round((cpu - last_cpu) / (now - last_t) / threads, 4)})
            last_cpu, last_t = cpu, now
    except Stop:
        pass
    finally:
        stop.set()
        for t in pool:
            t.join(timeout=on_s + 1.0)
        cpu1, t1 = read_stat_seconds(), time.monotonic()
        emit("load_end", achieved={"duty": round((cpu1 - cpu0) / (t1 - t0) / threads, 4),
                                   "cpu_seconds": round(cpu1 - cpu0, 1)})


if __name__ == "__main__":
    main()

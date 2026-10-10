"""IO load: sequential file reads throttled to a target rate, so page-cache pressure and disk
traffic are a chosen MB/s rather than "as fast as possible".

Whether the reads hit the disk depends on the file size against the page cache: a file larger
than RAM re-reads from disk every pass, a smaller one lives in the cache and the load becomes
pure cache pressure. The achieved rate is measured, and the record says which case this is.

Usage: ioload.py [--mbps N (500)] [--file PATH] [--gib 8] [--deadline-min 120]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadlib import (Stop, emit, install_stop, make_parent_check, parent_pid_of,
                         parse_args)

USAGE = "usage: ioload.py [--mbps N (500)] [--file PATH] [--gib 8] [--deadline-min 120]"
CHUNK = 1 << 20


def prepare_file(path: str, gib: float) -> None:
    if os.path.exists(path) and os.path.getsize(path) >= gib * (1 << 30):
        return
    with open(path, "wb") as f:
        block = bytes(1 << 20)
        for _ in range(int(gib)):
            f.write(block)
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    args = parse_args(sys.argv[1:], USAGE)
    mbps = float(args.get("mbps", 500))
    path = args.get("file", "ioload.scratch")
    gib = float(args.get("gib", 8))
    deadline_min = float(args.get("deadline-min", 120))
    if mbps <= 0 or gib <= 0:
        print(USAGE + "\nvalues must be positive", file=sys.stderr)
        raise SystemExit(2)

    emit("load_start", load="io", intended={"mbps": mbps, "file": path},
         args=sys.argv[1:], deadline_min=deadline_min)
    parent_pid = parent_pid_of(args)

    prepare_file(path, gib)
    mem_mib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20
    cache_resident = gib * 1024 < mem_mib
    deadline = install_stop(deadline_min)
    check_parent = make_parent_check(parent_pid)

    read_bytes = 0
    t0 = time.monotonic()
    emit("load_ready", achieved={"file_gib": gib, "cache_resident_likely": cache_resident})
    try:
        with open(path, "rb") as f:
            while True:
                data = f.read(CHUNK)
                if not data:
                    f.seek(0)
                    continue
                read_bytes += len(data)
                # Throttle: sleep so the average since start matches the target rate.
                target_s = read_bytes / (mbps * 2**20)
                ahead = target_s - (time.monotonic() - t0)
                if ahead > 0:
                    time.sleep(min(ahead, 0.5))
                if read_bytes % (CHUNK * 512) < CHUNK:  # ~every 512 MiB
                    emit("heartbeat", achieved={"mbps": round(read_bytes / 2**20 /
                                                              (time.monotonic() - t0), 1)})
                check_parent()
                if time.monotonic() > deadline:
                    raise Stop()
    except Stop:
        pass
    finally:
        elapsed = time.monotonic() - t0
        emit("load_end", achieved={"bytes": read_bytes,
                                   "mbps": round(read_bytes / 2**20 / elapsed, 1),
                                   "seconds": round(elapsed, 1)})


if __name__ == "__main__":
    main()

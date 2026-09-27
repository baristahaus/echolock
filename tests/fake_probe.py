"""Stand-in probe for runner tests: emits the record format without a GPU."""

import json
import sys
import time


def emit(type_, **fields):
    print(json.dumps({"v": 1, "t_ns": time.monotonic_ns(), "src": "probe", "probe": "fake",
                      "backend": "test", "type": type_, **fields}), flush=True)


mode = sys.argv[1]
emit("probe_start", params={"mode": mode})
emit("declared", key="device.sm_count", value=48, source="fake")
if mode == "hang":
    time.sleep(60)
print("a line that is not JSON", flush=True)
print("diagnostic", file=sys.stderr, flush=True)
for rep in range(3):
    emit("measure", name="bw", value=100.0 + rep, unit="GB/s", tags={"source": "x"},
         verified=mode != "wrong", rep=rep)
emit("measure", name="bw", value=101.0, unit="GB/s",
     tags={"source": "x", "reps": 3, "per_rep": {"min": 100.0, "max": 102.0}},
     verified=mode != "wrong")
emit("check", name="bw.xor", **{"pass": mode != "wrong"}, detail={})
emit("feature", feature="compression", requested=True, granted=True, effect_observed=False,
     evidence={"gain": 1.0})
if mode == "crash":
    sys.exit(1)
emit("probe_end", status="verify_failed" if mode == "wrong" else "ok")
sys.exit(3 if mode == "wrong" else 0)

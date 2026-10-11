#!/usr/bin/env python3
"""constants: require device facts from independent sources to agree.

The one probe that needs no GPU API: it reads what each source on the machine says about the
device and checks the sources against each other. The sources share no code path, so a
disagreement means something is misreporting -- the closest thing to a pure oracle here.

Sources per vendor, best-effort: a fact only one source reports is declared, not checked; a
fact no source reports is noted. Spec: probes/constants/spec.md.

  NVIDIA: NVML, nvidia-smi -q -x, /proc/driver/nvidia/gpus/*/information, vulkaninfo --summary
  AMD:    amdgpu sysfs, KFD nodes, vulkaninfo --summary
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from env import amdgpu, capture                      # noqa: E402
from env.nvml import Nvml                            # noqa: E402


def emit(rec: dict) -> None:
    rec.update({"v": 1, "t_ns": time.monotonic_ns(), "src": "probe",
                "probe": "constants", "backend": "host"})
    print(json.dumps(rec), flush=True)


def declared(source: str, key: str, value, device: str) -> None:
    emit({"type": "declared", "source": source, "key": f"{device}.{key}", "value": value})


def check(name: str, ok: bool, detail: dict) -> None:
    emit({"type": "check", "name": name, "pass": ok, "detail": detail})


def note(message: str) -> None:
    emit({"type": "note", "message": message})


def probe_start() -> None:
    emit({"type": "probe_start", "params": {}})


def probe_end(status: str = "ok") -> None:
    emit({"type": "probe_end", "status": status})


# ---- sources ------------------------------------------------------------------------------------

def nvidia_sources() -> dict[str, dict[str, dict]]:
    """per-device: source name -> {key: value}. Devices keyed by PCI id where possible."""
    out: dict[str, dict[str, dict]] = {}
    nv = Nvml()
    if nv.available:
        for h in nv.handles:
            d = nv.describe(h)
            pci = d.get("uuid", f"nvml{len(out)}")
            out.setdefault(pci, {})["nvml"] = {
                "name": d.get("name"), "uuid": d.get("uuid"), "vbios": d.get("vbios"),
                "max_sm_clock_mhz": d.get("max_sm_clock_mhz"),
                "max_mem_clock_mhz": d.get("max_mem_clock_mhz"),
                "power_limit_w": d.get("power_limit_w"),
            }
    smi = capture.nvidia_smi()
    if smi:
        # Key by UUID, as the NVML branch above does. Bus ids are not comparable across sources on
        # this hardware (nvidia-smi reports `0000000F:01:00.0`, NVML `0001:01:00.0` for one GB10),
        # so keying per source files one device into several buckets and compares nothing silently.
        for gpu in smi.get("gpus") or []:  # capture returns "gpus"; "gpu" silently yields nothing
            pci = gpu.get("uuid") or gpu.get("pci_bus_id") or f"smi{len(out)}"
            # Driver 580 nests clocks as {graphics_clock, sm_clock, mem_clock} with units and "N/A".
            # Reading max_clocks["sm"] was an older schema and silently yielded None here, so the
            # clock facts were never compared on this machine.
            mc, ac = gpu.get("max_clocks") or {}, gpu.get("applications_clocks") or {}
            smi_sm = _to_int(_first_of(mc, ("sm_clock", "sm", "graphics_clock", "graphics")))
            out.setdefault(pci, {})["nvidia_smi"] = {
                "name": gpu.get("product_name"),
                "uuid": gpu.get("uuid"),
                "vbios": gpu.get("vbios_version"),
                "max_sm_clock_mhz": smi_sm if smi_sm is not None else _to_int(
                    _first_of(ac, ("sm_clock", "sm"))),
                "max_mem_clock_mhz": _to_int(_first_of(mc, ("mem_clock", "memory"))),
                "power_limit_w": _to_float(_first_of(gpu.get("power") or {},
                                                    ("power_limit", "current_power_limit"))),
                "memory_bytes": _to_int(_first_of(gpu.get("fb_memory")
                                                  or gpu.get("fb_memory_usage") or {}, ("total",))),
            }
    for info in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")) or []:
        text = info.read_text()
        vals = dict(re.findall(r"^(\w+)\s*: (.*)$", text, re.M))
        # Same reason: prefer the UUID, which all three sources carry when the driver reports it.
        # On GB10 this file says `Model: Unknown` with no Bus Location, so this branch contributes
        # nothing here -- which is exactly what the source_coverage check must report.
        pci = vals.get("GPU UUID", "") or vals.get("Bus Location", "").lower().replace("0000:", "").replace(":", "")
        if not pci:
            continue
        out.setdefault(pci, {})["proc_information"] = {
            "name": vals.get("Model"), "uuid": vals.get("GPU UUID"),
            "vbios": vals.get("Video BIOS"),
        }
    return out


def _first_of(d: dict, names):
    """First present, non-`N/A` value among `names`. nvidia-smi's tag spelling and unit suffixes
    vary by driver branch, and `N/A` means unavailable rather than zero -- reading it as 0 would
    turn a missing fact into a wrong one."""
    for n in names:
        v = d.get(n)
        if v is not None and str(v).strip().upper() != "N/A":
            return v
    return None


def _to_int(v):
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.search(r"[\d.]+", v.replace(",", ""))
        if m:
            f = float(m.group())
            if "MiB" in v or "GiB" in v:
                f *= 2**20 if "MiB" in v else 2**30
            return int(f)
    return None


def _to_float(v):
    m = re.search(r"[\d.]+", str(v).replace(",", "")) if v is not None else None
    return float(m.group()) if m else None


def amd_sources() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for card in amdgpu.cards():
        d = amdgpu.describe(card)
        key = f"{d.get('domain')}:{d.get('bus')}" or card.name
        sclk = amdgpu.dpm_table(card / "device", "pp_dpm_sclk")
        mclk = amdgpu.dpm_table(card / "device", "pp_dpm_mclk")
        mon = amdgpu.hwmon(card / "device")
        out.setdefault(key, {})["amdgpu_sysfs"] = {
            "name": f"0x{d.get('device_id')}", "vbios": d.get("vbios_version"),
            "memory_bytes": d.get("vram_total_bytes"),
            "max_sm_clock_mhz": max(sclk.values()) if sclk else None,
            "max_mem_clock_mhz": max(mclk.values()) if mclk else None,
            "power_limit_w": mon.get("power_limit"),
        }
    for node in amdgpu._kfd_nodes():
        key = f"kfd{node.get('node_id')}"
        out.setdefault(key, {})["kfd"] = {
            "name": node.get("name"),
            "memory_bytes": (node.get("mem") or {}).get("vram_total_bytes")
                            if isinstance(node.get("mem"), dict) else None,
        }
    return out


def vulkaninfo_devices() -> list[dict]:
    """Name, memory and driver-reported limits from the Vulkan stack itself."""
    try:
        p = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    text = p.stdout
    out = []
    for block in re.split(r"GPU\d", text)[1:]:
        name = re.search(r"deviceName\s*=\s*(.+)", block)
        mem = re.search(r"deviceSize\s*=\s*(\d+)", block)
        out.append({"name": name.group(1).strip() if name else None,
                    "memory_bytes": int(mem.group(1)) if mem else None})
    return out


# ---- comparison ---------------------------------------------------------------------------------

NUMERIC_TOLERANCE = {  # relative tolerance for values that legitimately differ slightly
    "max_sm_clock_mhz": 0.02, "max_mem_clock_mhz": 0.02, "power_limit_w": 0.02,
    "memory_bytes": 0.02,
}


def compare(devices: dict[str, dict[str, dict]], vk_devices: list[dict]) -> int:
    """Emits one check per fact held by 2+ sources; returns the number of failed checks."""
    failed = 0
    for dev, sources in sorted(devices.items()):
        keys = sorted({k for s in sources.values() for k in s})
        for key in keys:
            values = {src: s[key] for src, s in sources.items() if s.get(key) is not None}
            if len(values) < 2:
                continue  # declared above by the source loop; nothing to compare
            uniq = {json.dumps(v, sort_keys=True) for v in values.values()}
            if len(uniq) == 1:
                continue  # agreement is the expected case; the declared records show it
            vals = list(values.values())
            tol = NUMERIC_TOLERANCE.get(key)
            if tol is not None and all(isinstance(v, (int, float)) for v in vals):
                lo, hi = min(vals), max(vals)
                ok = hi <= lo * (1 + tol) or lo == 0
            else:
                ok = False
            failed += not ok
            check(f"{dev}.{key}", ok, {
                "per_source": {src: v for src, v in values.items()},
                "sources": len(values),
            })
        # Vulkan stack as one more voice on name and memory. The Vulkan stack enumerates every
        # GPU on the machine, so a fact passes when any of its devices matches.
        for key in ("name", "memory_bytes"):
            ours = [s.get(key) for s in sources.values() if s.get(key) is not None]
            if not ours or not vk_devices:
                continue
            want = ours[0]
            candidates = [v.get(key) for v in vk_devices]
            if key == "name":
                a = str(want).lower()
                ok = any(a in str(g).lower() or str(g).lower() in a for g in candidates if g)
            else:
                ok = any(g is not None and abs(g - want) <= max(want, 1) * NUMERIC_TOLERANCE[key]
                         for g in candidates)
            check(f"{dev}.{key}.vs_vulkan", ok, {"ours": want, "vulkan_devices": candidates})
            failed += not ok
    return failed


def main() -> int:
    probe_start()
    try:
        devices = nvidia_sources() if capture.run_cmd(["nvidia-smi", "-L"]) else amd_sources()
    except Exception as e:  # noqa: BLE001 -- every source is best-effort
        note(f"source collection failed: {type(e).__name__}: {e}")
        devices = {}
    if not devices:
        note("no GPU sources found (no NVIDIA, no amdgpu); nothing to compare")
        probe_end("error")
        return 1

    # Declare every fact first, so single-source facts still appear in the report's Declared table.
    for dev, sources in sorted(devices.items()):
        for src, facts in sorted(sources.items()):
            for key, value in facts.items():
                if value is not None:
                    declared(src, key, value, dev)

    vk = vulkaninfo_devices()
    if not vk:
        note("vulkaninfo unavailable or empty: the Vulkan stack is not part of this comparison")
    failed = compare(devices, vk)

    # An `agrees` check needs two sources to agree. A machine whose driver exposes fewer than the
    # probe assumes therefore emits no checks at all, and probe_end("ok") reads as a clean bill of
    # health: observed on GB10, where nvidia-smi's device list is keyed "gpus" (not "gpu"), /proc's
    # per-GPU information file reports `Model: Unknown` with no Bus Location, and no Vulkan ICD is
    # installed -- 1 source, 0 checks, status ok. Absence of evidence must not print as a pass, the
    # same rule that puts a ✗ on an unverified measurement.
    answering = {dev: sorted(s for s, facts in srcs.items()
                              if any(v is not None for v in facts.values()))
                 for dev, srcs in devices.items()}
    thin = {dev: srcs for dev, srcs in answering.items() if len(srcs) < 2}
    check("source_coverage", not thin and bool(answering),
          {"per_device": {dev: len(s) for dev, s in answering.items()},
           "answering": answering,
           "need": "at least 2 sources per device, or there is nothing to compare",
           "thin": thin or None})
    if thin or not answering:
        failed += 1

    probe_end("ok" if failed == 0 else "verify_failed")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())

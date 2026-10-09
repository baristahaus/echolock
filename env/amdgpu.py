"""AMD GPUs through the amdgpu kernel driver: no ROCm, no AMD-SMI, no vendor library.

Two layers, in order of trustworthiness:

1. **sysfs text attributes** (`gpu_busy_percent`, `pp_dpm_sclk`, `mem_info_vram_used`, hwmon
   `temp1_input`/`power1_input`, `power_state`, ...). Documented in
   `Documentation/ABI/testing/sysfs-driver-amdgpu`, so they are the ABI AMD promises to keep. This
   is where every number in a sample comes from.
2. **`gpu_metrics`**, a versioned binary struct the driver copies out of the SMU. It adds the
   throttling status word, the energy counter and per-engine activity. Its meaning is stable but
   its layout depends on `(struct_version, struct_revision)`, and fields a given GPU cannot report
   are filled with all-ones, so it is read as a supplement, never as the source of a number.

Two things that make or break a measurement are recorded as declared state, not silently absorbed:
`power_dpm_force_performance_level` (a locked "level N" makes every clock the GPU reports a
constant of your own choosing) and `pp_od_clk_voltage` (an overclock in place).

The kernel's `clock_event_reasons` equivalent, `throttle_status`, is stored raw: which bit means
what is asic-specific, and guessing would be worse than saying nothing. See docs/amd.md.
"""

from __future__ import annotations

import glob
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Callable

# gpu_metrics: {name: (offset, struct char, count)} for the start of the struct that every
# gpu_metrics_v1_* revision shares, read off the field list in
# drivers/gpu/drm/amd/include/kgd_pp_interface.h and verified on gpu_metrics_v1_3 hardware.
# AMD appends fields at the end, so this prefix is offered to any version and then cross-checked
# against the text attributes: a layout that disagrees with hwmon is the wrong layout, not a
# broken GPU. See gpu_metrics().
GPU_METRICS_LAYOUT: dict[str, tuple[int, str, int]] = {
    "temperature_edge": (4, "H", 1), "temperature_hotspot": (6, "H", 1),
    "temperature_mem": (8, "H", 1), "average_gfx_activity": (16, "H", 1),
    "average_umc_activity": (18, "H", 1), "average_mm_activity": (20, "H", 1),
    "average_socket_power": (22, "H", 1), "energy_accumulator": (24, "Q", 1),
    "average_gfxclk_frequency": (40, "H", 1), "average_uclk_frequency": (44, "H", 1),
    "current_gfxclk": (54, "H", 1), "current_uclk": (58, "H", 1),
    "throttle_status": (68, "I", 1), "pcie_link_width": (74, "H", 1),
    "pcie_link_speed": (76, "H", 1),
}

# The driver writes all-ones for a metric this GPU does not have.
UNSUPPORTED = {0xFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

# Module parameters that change what a probe measures, out of /sys/module/amdgpu/parameters.
PARAMS_WORTH_RECORDING = (
    "cg_mask", "ppfeature", "runpm", "vrampage", "vm_fragment_size", "vm_system_mem",
    "compute_scheduler", "debug_disable", "gem_debugfs", "gpu_timeout", "mmhub_pg",
    "noretry", "sg_display", "si_support", "amdkfd_support", "ps_report_interval",
)


def _read(path: str | Path, cast: Callable[[str], object] = str):
    """Reads one sysfs file, or None. Missing is normal: it depends on the asic and driver."""
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return cast(text)
    except ValueError:
        return text            # tables and versions come back as text whatever the cast asked for


def _read_int(path: str | Path) -> int | None:
    v = _read(path, int)
    return v if isinstance(v, int) else None


def cards() -> list[Path]:
    """drm card directories bound to amdgpu, in card-number order. Connectors (card1-HDMI-A-1)
    live in the same directory and are not cards."""
    out = []
    found = [p for p in glob.glob("/sys/class/drm/card*") if re.fullmatch(r"card\d+", Path(p).name)]
    for card in sorted(found, key=lambda p: int(Path(p).name[4:])):
        link = Path(card, "device/driver")
        if not link.exists():
            continue
        if Path(link.resolve()).name == "amdgpu":
            out.append(Path(card))
    return out


def dpm_active(dev: Path, attr: str) -> tuple[int, int] | None:
    """The (index, MHz) of the `*`-marked state in a pp_dpm_* table, or None.

    `pp_dpm_sclk` lists every clock the part can run at with the currently selected one starred;
    the same table also shows whether a level has been forced, which is why the whole table is
    kept in the environment snapshot.
    """
    table = dpm_table(dev, attr)
    active = table.get("active")
    return (active, int(table["states"][active])) if active is not None else None


def dpm_table(dev: Path, attr: str) -> dict:
    """The whole pp_dpm_* table: {level: MHz} plus which level is active. Levels are usually
    numbers; some parts also list a named one (`S`), so the label is kept as written."""
    text = _read(dev / attr)
    if not isinstance(text, str):
        return {}
    states, active = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        starred = line.endswith("*")
        parts = line.rstrip("* ").split()
        if len(parts) < 2 or not parts[1].lower().endswith("mhz"):
            continue
        label = parts[0].rstrip(":")
        try:
            states[label] = float(parts[1][:-3])
        except ValueError:
            continue
        if starred:
            active = label
    return {"states": states, "active": active, "forced_level": _read(
        dev / "power_dpm_force_performance_level")}


_HWMON_DIR: dict[str, str | None] = {}      # resolved device path -> its hwmon instance
_LAYOUT_CHECKED: dict[tuple[str, int, int], dict] = {}   # layout verdict, once per process


def _hwmon_dir(dev: Path) -> str | None:
    """The hwmon instance belonging to this drm device, found once and remembered: the sampler
    asks every second and /sys/class/hwmon has a dozen entries to resolve."""
    key = str(dev.resolve())
    if key not in _HWMON_DIR:
        found = None
        for mon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
            if _read(f"{mon}/name") != "amdgpu":
                continue
            if str(Path(mon, "device").resolve()) == key:
                found = mon
                break
        _HWMON_DIR[key] = found
    return _HWMON_DIR[key]


def hwmon(dev: Path) -> dict:
    """The amdgpu hwmon instance for this card: temperature in C, power in W. Some parts report
    power as an average rather than an instantaneous input, so both are read."""
    mon = _hwmon_dir(dev)
    if mon is None:
        return {}
    out: dict = {}
    for kind, scale, unit in (("temp", 1e-3, "c"), ("power", 1e-6, "w")):
        for f in sorted(glob.glob(f"{mon}/{kind}*_input") + glob.glob(f"{mon}/{kind}*_average")):
            v = _read_int(f)
            if v is None:
                continue
            base = Path(f).name.rsplit("_", 1)[0]
            label = _read(f"{mon}/{base}_label") or base
            out[f"{kind}_{label}_{unit}"] = v * scale
    return out


def gpu_metrics(dev: Path) -> dict:
    """Decodes gpu_metrics into a dict, or says why it could not. Never the source of a headline
    number: it supplies the throttle word, the energy counter and per-engine activity."""
    path = dev / "gpu_metrics"
    try:
        blob = path.read_bytes()
    except OSError as e:
        return {"error": e.strerror}
    if len(blob) < 4:
        return {"error": f"too short ({len(blob)} bytes)"}
    size, version, revision = struct.unpack_from("<HBB", blob, 0)
    out: dict = {"struct_version": version, "struct_revision": revision, "size": size}
    for name, (offset, char, count) in GPU_METRICS_LAYOUT.items():
        if offset + struct.calcsize(char) * count > len(blob):
            continue                                # this struct is shorter than the layout
        raw = struct.unpack_from("<" + char * count, blob, offset)
        value = raw[0] if count == 1 else list(raw)
        if isinstance(value, int) and value in UNSUPPORTED:
            continue                                # this GPU cannot report it
        out[name] = value
    out["cross_check"] = _cross_check(dev, (version, revision))
    if not any(out["cross_check"].values()):
        return {"error": f"layout v{version}.{revision} did not cross-check, not trusting it",
                "struct_version": version, "struct_revision": revision, "size": size,
                "cross_check": out["cross_check"]}
    return out


def _cross_check(dev: Path, version: tuple[int, int]) -> dict:
    """Agreement with two independent sources, required before any decoded value is trusted: the
    clock must be one the part can run at, and the edge temperature must be what hwmon says. Done
    once per process and layout -- checking every second would make the verdict wobble when the
    part idles into a state the tables do not list."""
    key = (str(dev), version)
    if key not in _LAYOUT_CHECKED:
        checks: dict[str, bool] = {}
        states = dpm_table(dev, "pp_dpm_sclk").get("states") or {}
        blob = {}
        try:
            raw = (dev / "gpu_metrics").read_bytes()
            for name, (offset, char, count) in GPU_METRICS_LAYOUT.items():
                if offset + struct.calcsize(char) * count <= len(raw):
                    blob[name] = struct.unpack_from("<" + char * count, raw, offset)[0]
        except OSError:
            blob = {}
        for field in ("current_gfxclk", "average_gfxclk_frequency"):
            clk = blob.get(field)
            if clk is not None and states and clk not in UNSUPPORTED:
                checks[field + "_in_dpm_table"] = any(abs(clk - v) <= max(15.0, 0.1 * v)
                                                      for v in states.values())
        edge, hw = blob.get("temperature_edge"), hwmon(dev).get("temp_edge_c")
        if edge is not None and hw is not None and edge not in UNSUPPORTED:
            checks["temperature_edge_matches_hwmon"] = abs(edge - hw) <= 5
        _LAYOUT_CHECKED[key] = checks
    return _LAYOUT_CHECKED[key]


def _kfd_nodes() -> list[dict]:
    """The KFD topology: what the driver says about CUs, waves, LDS and memory per GPU node."""
    out = []
    for node in sorted(glob.glob("/sys/devices/virtual/kfd/kfd/topology/nodes/[0-9]*"),
                       key=lambda p: int(Path(p).name)):
        props = {}
        try:
            for line in Path(node, "properties").read_text().splitlines():
                k, _, v = line.partition(" ")
                props[k] = int(v)
        except (OSError, ValueError):
            continue
        if props.get("cpu_cores_count"):
            continue                      # a CPU node; the CPU side belongs to env.capture.host
        banks = []
        for bank in sorted(glob.glob(f"{node}/mem_banks/[0-9]*"), key=lambda p: int(Path(p).name)):
            info = {}
            try:
                for line in Path(bank, "properties").read_text().splitlines():
                    k, _, v = line.partition(" ")
                    info[k] = int(v)
            except (OSError, ValueError):
                continue
            banks.append({"size_bytes": info.get("size_in_bytes"),
                          "mem_clk_max": info.get("mem_clk_max")})
        out.append({"node": Path(node).name, **{
            k: props.get(k) for k in
            ("simd_count", "simd_id_base", "max_waves_per_simd", "lds_size_in_kb", "array_pref",
             "num_gws", "io_links_count", "mem_banks_count", "caches_count")},
            "mem_banks": banks})
    return out


def sample(card: Path, index: int) -> dict:
    """One sample in the same shape env/nvml.py produces, so report/ needs no AMD branch."""
    dev = card / "device"
    sclk = dpm_active(dev, "pp_dpm_sclk")
    mclk = dpm_active(dev, "pp_dpm_mclk")
    mon = hwmon(dev)
    temps = [v for k, v in mon.items() if k.startswith("temp_")]
    powers = [v for k, v in mon.items() if k.startswith("power_")]
    rec = {
        "index": index,
        "card": card.name,
        "sm_clock_mhz": sclk[1] if sclk else None,
        "mem_clock_mhz": mclk[1] if mclk else None,
        "temp_c": max(temps) if temps else None,          # hotspot, if there is one
        "power_w": max(powers) if powers else None,       # socket power
        "util_gpu_pct": _read_int(dev / "gpu_busy_percent"),
        "util_mem_pct": _read_int(dev / "mem_busy_percent"),
        "pstate": _read(dev / "power_state"),             # D0 active, D3 cold
        "vram_used_bytes": _read_int(dev / "mem_info_vram_used"),
        "gtt_used_bytes": _read_int(dev / "mem_info_gtt_used"),
    }
    m = gpu_metrics(dev)
    if "throttle_status" in m:
        rec["clock_event_reasons"] = m["throttle_status"]     # raw; bit meanings are asic-specific
        rec["clock_event_names"] = None
    if "energy_accumulator" in m:
        rec["energy_mj"] = m["energy_accumulator"]
    for k in ("average_gfx_activity", "average_umc_activity", "average_mm_activity"):
        if k in m:
            rec["metrics_" + k] = m[k]
    return rec


def sample_all() -> list[dict]:
    return [sample(dev, i) for i, dev in enumerate(cards())]


def describe(card: Path) -> dict:
    """What this card declares about itself: ids, firmware, and the settings that would change
    what a probe measures."""
    dev = card / "device"
    pci = dev.resolve().name                              # 0000:03:00.0
    out = {
        "card": card.name,
        "pci": pci,
        "vendor_id": _read(dev / "vendor"), "device_id": _read(dev / "device"),
        "revision": _read(dev / "revision"), "subsystem_id": _read(dev / "subsystem_device"),
        "driver": Path(dev / "driver").resolve().name if (dev / "driver").exists() else None,
        "vbios_version": _read(dev / "vbios_version"),
        "vbios_build": _read(dev / "vbios_build"),
        "vram_total_bytes": _read_int(dev / "mem_info_vram_total"),
        "gtt_total_bytes": _read_int(dev / "mem_info_gtt_total"),
        "vram_vendor": _read(dev / "mem_info_vram_vendor"),
        "pcie_link": {"current_speed": _read(f"/sys/bus/pci/devices/{pci}/current_link_speed"),
                      "current_width": _read(f"/sys/bus/pci/devices/{pci}/current_link_width"),
                      "max_speed": _read(f"/sys/bus/pci/devices/{pci}/max_link_speed"),
                      "max_width": _read(f"/sys/bus/pci/devices/{pci}/max_link_width")},
        "clocks": {"sclk": dpm_table(dev, "pp_dpm_sclk"),
                   "mclk": dpm_table(dev, "pp_dpm_mclk"),
                   "fclk": dpm_table(dev, "pp_dpm_fclk"),
                   "socclk": dpm_table(dev, "pp_dpm_socclk")},
        "performance_level": _read(dev / "power_dpm_force_performance_level"),
        # pp_od_clk_voltage lists the table and its range whether or not anyone moved it, so the
        # question is whether an offset is in it, not whether the file has text.
        "clock_limits": _read(dev / "pp_od_clk_voltage"),
        "overclock_applied": bool(
            re.search(r"[+-]\s*\d+\s*[Mm][Hh]z", _read(dev / "pp_od_clk_voltage") or "")),
        "power_profile": _read(dev / "pp_power_profile_mode"),
        "features_enabled": _features(dev / "pp_features"),
        "gpu_metrics": gpu_metrics(dev),
        "hwmon": hwmon(dev),
    }
    return out


def _features(path: Path) -> dict:
    """`pp_features` / `pp_feature_status` as {name: bool}. GFXOFF, DS_* and THROTTLERS in here
    explain most surprises in an AMD timing series."""
    text = _read(path) or _read(path.parent / "pp_feature_status")
    if not isinstance(text, str):
        return {}
    out = {}
    for line in text.splitlines():
        name, _, state = line.partition("(")
        if not state:
            continue
        name = name.split(".", 1)[-1].strip()
        if name:
            out[name] = "enabled" in state
    return out


def driver_version() -> str | None:
    """amdgpu's own version, from modinfo when available (built-in modules expose no sysfs
    version file, in which case the kernel version says it well enough)."""
    v = _read("/sys/module/amdgpu/version") or _read("/sys/module/drm/version")
    if v:
        return v
    if shutil.which("modinfo"):
        try:
            p = subprocess.run(["modinfo", "-F", "version", "amdgpu"], capture_output=True,
                               text=True, timeout=10)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout.splitlines()[0].strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def describe_all() -> dict:
    """Everything AMD, for env.json: cards, KFD topology, and the module parameters that matter."""
    params = {p: _read(f"/sys/module/amdgpu/parameters/{p}") for p in PARAMS_WORTH_RECORDING}
    return {
        "driver_version": driver_version(),
        "module_params": {k: v for k, v in params.items() if v is not None},
        "kfd_nodes": _kfd_nodes(),
        "cards": [describe(c) for c in cards()],
    }


def main() -> None:
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "describe":
        print(json.dumps(describe_all(), indent=1))
    else:
        for rec in sample_all():
            print(json.dumps(rec))


if __name__ == "__main__":
    main()

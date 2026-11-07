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

The kernel's `clock_event_reasons` equivalent, `throttle_status`, uses per-ASIC bit
positions, so it is decoded only for asics with a table transcribed from the amdgpu driver
(gfx10.3 and gfx12 on this machine, cited in the bit tables below); for every other asic the
word stays raw and `clock_event_names` is null — guessing would be worse than saying nothing.
See docs/amd.md.
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
# drivers/gpu/drm/amd/include/kgd_pp_interface.h and verified on gpu_metrics v1.3 (dGPU) and
# v2.1 (iGPU) hardware.
# AMD appends fields at the end, so this prefix is offered to any family-1 version and then
# cross-checked against the text attributes: a layout that disagrees with hwmon is the wrong
# layout, not a broken GPU. See gpu_metrics().
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

# Family 2 (APU): the scalar fields of gpu_metrics_v2_1 in kgd_pp_interface.h, same source.
# The v2_1 struct is 120 bytes end to end, which is exactly what the v2.1 card on this
# machine reports. AMD appends at the end (v2_2 adds indep_throttle_status, v2_3 the average
# temperatures), so the prefix stands for every v2_* revision. Family 2 has no energy
# accumulator and no PCIe link fields; its clocks are gfxclk/uclk/fclk, not gfx/uclk only.
GPU_METRICS_LAYOUT_V2: dict[str, tuple[int, str, int]] = {
    "temperature_gfx": (4, "H", 1), "temperature_soc": (6, "H", 1),
    "average_gfx_activity": (28, "H", 1), "average_mm_activity": (30, "H", 1),
    "average_socket_power": (40, "H", 1), "average_cpu_power": (42, "H", 1),
    "average_soc_power": (44, "H", 1), "average_gfx_power": (46, "H", 1),
    "average_gfxclk_frequency": (64, "H", 1), "average_uclk_frequency": (68, "H", 1),
    "average_fclk_frequency": (70, "H", 1), "current_gfxclk": (76, "H", 1),
    "current_uclk": (80, "H", 1), "current_fclk": (82, "H", 1),
    "throttle_status": (108, "I", 1), "fan_pwm": (112, "H", 1),
}


def _layout_for(version: int) -> dict[str, tuple[int, str, int]] | None:
    """The verified layout for a gpu_metrics struct family, or None: a family with no
    offsets read off the kernel header is not decoded at all, rather than guessed."""
    if version == 1:
        return GPU_METRICS_LAYOUT
    if version == 2:
        return GPU_METRICS_LAYOUT_V2
    return None

# The driver writes all-ones for a metric this GPU does not have.
UNSUPPORTED = {0xFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

# Module parameters that change what a probe measures, out of /sys/module/amdgpu/parameters.
PARAMS_WORTH_RECORDING = (
    "cg_mask", "ppfeature", "runpm", "vrampage", "vm_fragment_size", "vm_system_mem",
    "compute_scheduler", "debug_disable", "gem_debugfs", "gpu_timeout", "mmhub_pg",
    "noretry", "sg_display", "si_support", "amdkfd_support", "ps_report_interval",
)

# The gpu_metrics `throttle_status` word is built by the driver from SMU metrics with
# per-ASIC bit positions (the struct comment in v2_2 and later calls it
# "Throttle status (ASIC dependent)"), so a bit's meaning without the asic's table is
# noise. A word is decoded only for a family with a table transcribed bit-for-bit from
# the amdgpu driver (Linux master, read 2026-11-06); the port covers exactly the asics
# on this machine:
#
#   gfx10.3 (Renoir/Raphael APU class, SMU v12):
#     drivers/gpu/drm/amd/pm/swsmu/smu12/renoir_ppt.c   — renoir_throttler_map[]
#     drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu12_driver_if.h
#                                                        — THROTTLER_STATUS_BIT_*
#   gfx12 (RDNA4, MP1 v14.0.2/3):
#     drivers/gpu/drm/amd/pm/swsmu/smu14/smu_v14_0_2_ppt.c
#                                                        — smu_v14_0_2_throttler_map[],
#                                                          smu_v14_0_2_get_throttler_status()
#     drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu14_driver_if_v14_0.h
#                                                        — THROTTLER_*_BIT, THROTTLER_COUNT
#
# Other families have no table here and stay raw with clock_event_names null; adding one
# means transcribing its table with the same citation, not guessing positions.
THROTTLE_BITS_GFX10_3: dict[int, str] = {
    0: "SPL", 1: "FPPT", 2: "SPPT", 3: "SPPT_APU", 4: "THM_CORE", 5: "THM_GFX",
    6: "THM_SOC", 7: "TDC_VDD", 8: "TDC_SOC", 9: "PROCHOT_CPU", 10: "PROCHOT_GFX",
    11: "EDC_CPU", 12: "EDC_GFX",
}
THROTTLE_BITS_GFX12: dict[int, str] = {
    0: "TEMP_EDGE", 1: "TEMP_HOTSPOT", 2: "TEMP_HOTSPOT_GFX", 3: "TEMP_HOTSPOT_SOC",
    4: "TEMP_MEM", 5: "TEMP_VR_GFX", 6: "TEMP_VR_SOC", 7: "TEMP_VR_MEM0",
    8: "TEMP_VR_MEM1", 9: "TEMP_LIQUID0", 10: "TEMP_LIQUID1", 11: "TEMP_PLX",
    12: "TDC_GFX", 13: "TDC_SOC", 14: "PPT0", 15: "PPT1", 16: "PPT2", 17: "PPT3",
    18: "FIT", 19: "GFX_APCC_PLUS", 20: "GFX_DVO",
}


def _throttle_table(gfx_version: int | None) -> dict[int, str] | None:
    """The bit table for a KFD gfx target version, or None when this family has none.
    The version is the driver's decimal gfx target (kfd_device.c: major*10000 +
    minor*100 + step — 100306 = gfx10.3.6, 120001 = gfx12.0.1)."""
    if gfx_version is None:
        return None
    major, minor = gfx_version // 10000, (gfx_version // 100) % 100
    if (major, minor) == (10, 3):
        return THROTTLE_BITS_GFX10_3
    if major == 12:
        return THROTTLE_BITS_GFX12
    return None


def throttle_names(word: int, gfx_version: int | None) -> list[str] | None:
    """The active throttler bits of a `throttle_status` word, as names — or None when the
    asic has no citable bit table, in which case the word stays raw. An empty list is a
    real answer: a known asic, nothing active."""
    table = _throttle_table(gfx_version)
    if table is None:
        return None
    return [name for bit, name in sorted(table.items()) if (word >> bit) & 1]


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
    """Decodes gpu_metrics into a dict, or says why it could not. Never the source of a
    headline number: it supplies the throttle word, the energy counter and per-engine
    activity. The layout is chosen by struct family (v1_* and v2_* have different field
    orders); a family with no verified layout is not decoded at all."""
    path = dev / "gpu_metrics"
    try:
        blob = path.read_bytes()
    except OSError as e:
        return {"error": e.strerror}
    if len(blob) < 4:
        return {"error": f"too short ({len(blob)} bytes)"}
    size, version, revision = struct.unpack_from("<HBB", blob, 0)
    layout = _layout_for(version)
    if layout is None:
        return {"error": f"no verified layout for gpu_metrics family v{version}.{revision}",
                "struct_version": version, "struct_revision": revision, "size": size}
    out: dict = {"struct_version": version, "struct_revision": revision, "size": size}
    for name, (offset, char, count) in layout.items():
        if offset + struct.calcsize(char) * count > len(blob):
            continue                                # this struct is shorter than the layout
        raw = struct.unpack_from("<" + char * count, blob, offset)
        value = raw[0] if count == 1 else list(raw)
        if isinstance(value, int) and value in UNSUPPORTED:
            continue                                # this GPU cannot report it
        out[name] = value
    if version == 2:
        # Family 2 is filled by the SMU v12 ppt (renoir class), where the temperatures
        # are centi-Celsius (smu12_driver_if.h), unlike family 1's plain degrees. The
        # cross-check below re-verifies the reading against hwmon before it is trusted.
        for f in ("temperature_gfx", "temperature_soc"):
            if f in out:
                out[f + "_c"] = out.pop(f) / 100.0
    if "throttle_status" in out:
        out["throttle_names"] = throttle_names(out["throttle_status"],
                                               card_gfx_version(dev.parent))
    out["cross_check"] = _cross_check(dev, (version, revision))
    if not any(out["cross_check"].values()):
        return {"error": f"layout v{version}.{revision} did not cross-check, not trusting it",
                "struct_version": version, "struct_revision": revision, "size": size,
                "cross_check": out["cross_check"]}
    return out


def _cross_check(dev: Path, version: tuple[int, int]) -> dict:
    """Agreement with two independent sources, required before any decoded value is trusted: the
    clock must be one the part can run at, and the temperature must be what hwmon says (family 1's
    edge, family 2's gfx — on the APU that is what the hwmon edge sensor reports). Done
    once per process and layout -- checking every second would make the verdict wobble when the
    part idles into a state the tables do not list."""
    key = (str(dev), version)
    if key not in _LAYOUT_CHECKED:
        layout = _layout_for(version[0]) or {}
        checks: dict[str, bool] = {}
        states = dpm_table(dev, "pp_dpm_sclk").get("states") or {}
        blob = {}
        try:
            raw = (dev / "gpu_metrics").read_bytes()
            for name, (offset, char, count) in layout.items():
                if offset + struct.calcsize(char) * count <= len(raw):
                    blob[name] = struct.unpack_from("<" + char * count, raw, offset)[0]
        except OSError:
            blob = {}
        for field in ("current_gfxclk", "average_gfxclk_frequency"):
            clk = blob.get(field)
            if clk is not None and states and clk not in UNSUPPORTED:
                checks[field + "_in_dpm_table"] = any(abs(clk - v) <= max(15.0, 0.1 * v)
                                                      for v in states.values())
        if version[0] == 1:
            edge, check_name = blob.get("temperature_edge"), "temperature_edge_matches_hwmon"
        else:
            raw_gfx = blob.get("temperature_gfx")
            edge = raw_gfx / 100.0 if raw_gfx is not None else None
            check_name = "temperature_gfx_matches_hwmon"
        hw = hwmon(dev).get("temp_edge_c")
        if edge is not None and hw is not None and edge not in UNSUPPORTED:
            checks[check_name] = abs(edge - hw) <= 5
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


_KFD_LOCATIONS: dict[tuple[int, int], int] | None = None   # scanned once per process


def _kfd_gpu_locations() -> dict[tuple[int, int], int]:
    """KFD's GPU nodes as {(domain, pci_dev_id): gfx target version}, read straight off
    the node properties. location_id is the kernel's pci_dev_id() (kfd_topology.c sets it
    from the PCI address), which is PCI_DEVID (include/linux/pci.h): (bus << 8) |
    (slot << 3) | func, 16 bits; the node's `domain` property carries the PCI domain
    number separately. Matching on (domain, devid) is the card's own address, repacked
    the same way. A node with cpu_cores_count set is the CPU and has no gfx version."""
    global _KFD_LOCATIONS
    if _KFD_LOCATIONS is None:
        out: dict[tuple[int, int], int] = {}
        for node in glob.glob("/sys/class/kfd/kfd/topology/nodes/[0-9]*"):
            props: dict[str, str] = {}
            try:
                for line in Path(node, "properties").read_text().splitlines():
                    k, _, v = line.partition(" ")
                    props[k] = v.strip()
            except OSError:
                continue
            if int(props.get("cpu_cores_count") or 0):
                continue
            gfx = props.get("gfx_ip_version") or props.get("gfx_target_version")
            try:
                out[(int(props.get("domain", 0)), int(props["location_id"]))] = int(gfx)
            except (KeyError, TypeError, ValueError):
                continue
        _KFD_LOCATIONS = out
    return _KFD_LOCATIONS


_KFD_GFX: dict[str, int | None] = {}    # resolved card device path -> gfx target version


def card_gfx_version(card: Path) -> int | None:
    """The card's gfx target version (e.g. 120001 = gfx12.0.1, the driver's decimal
    gfx target), from KFD's node properties matched by PCI address, resolved once per
    card. None when KFD does not expose it (KFD off, or a card outside the topology) —
    the asic-specific counters then stay raw rather than guessed."""
    key = str(Path(card, "device").resolve())
    if key not in _KFD_GFX:
        loc: tuple[int, int] | None = None
        try:
            dombus, _, devfunc = Path(card, "device").resolve().name.rpartition(":")
            dom, _, bus = dombus.partition(":")
            dev, _, func = devfunc.partition(".")
            # PCI_DEVID (include/linux/pci.h), the packing KFD's location_id uses.
            loc = (int(dom, 16), (int(bus, 16) << 8) | (int(dev, 16) << 3)
                   | (int(func, 16) if func else 0))
        except ValueError:
            loc = None
        _KFD_GFX[key] = _kfd_gpu_locations().get(loc) if loc is not None else None
    return _KFD_GFX[key]


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
        rec["clock_event_reasons"] = m["throttle_status"]
        # A list when the asic has a citable bit table (decoded names, possibly empty);
        # null when the word is raw because no table covers this asic.
        rec["clock_event_names"] = m.get("throttle_names")
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
        "gfx_version": card_gfx_version(card),
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

"""Minimal NVML binding through ctypes (no pynvml dependency).

Only what the sampler and the environment capture read. Every getter returns None when the call
is unsupported on this device (GB10 reports several memory queries as not supported), so a
missing value never stops a sample.
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_char, c_int, c_uint, c_ulonglong, c_void_p

NVML_SUCCESS = 0
CLOCK_GRAPHICS, CLOCK_SM, CLOCK_MEM, CLOCK_VIDEO = 0, 1, 2, 3
TEMPERATURE_GPU = 0

# nvmlClocksEventReasons (formerly ClocksThrottleReasons) bits.
CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


def decode_reasons(mask: int | None) -> list[str] | None:
    if mask is None:
        return None
    names = [name for bit, name in CLOCK_EVENT_REASONS.items() if mask & bit]
    unknown = mask & ~sum(CLOCK_EVENT_REASONS)
    if unknown:
        names.append(f"unknown_0x{unknown:x}")
    return names


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class Nvml:
    """Loaded NVML, or `available == False` with the reason in `error`."""

    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self.handles: list[c_void_p] = []
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as e:
            self.error = f"libnvidia-ml.so.1: {e}"
            return
        rc = self._call("nvmlInit_v2")
        if rc != NVML_SUCCESS:
            self.error = f"nvmlInit_v2 returned {rc}"
            return
        count = c_uint()
        if self._call("nvmlDeviceGetCount_v2", byref(count)) != NVML_SUCCESS:
            self.error = "nvmlDeviceGetCount_v2 failed"
            return
        for i in range(count.value):
            h = c_void_p()
            if self._call("nvmlDeviceGetHandleByIndex_v2", c_uint(i), byref(h)) == NVML_SUCCESS:
                self.handles.append(h)
        self.available = True

    def _call(self, name: str, *args) -> int:
        fn = getattr(self.lib, name, None)
        if fn is None:
            return -1
        return int(fn(*args))

    def _uint(self, name: str, *args) -> int | None:
        v = c_uint()
        return v.value if self._call(name, *args, byref(v)) == NVML_SUCCESS else None

    def _ull(self, name: str, *args) -> int | None:
        v = c_ulonglong()
        return v.value if self._call(name, *args, byref(v)) == NVML_SUCCESS else None

    def _str(self, name: str, *args, size: int = 96) -> str | None:
        buf = (c_char * size)()
        if self._call(name, *args, buf, c_uint(size)) != NVML_SUCCESS:
            return None
        return buf.value.decode(errors="replace")

    def sample(self, h: c_void_p) -> dict:
        util = _Utilization()
        util_ok = self._call("nvmlDeviceGetUtilizationRates", h, byref(util)) == NVML_SUCCESS
        pstate = c_int()
        pstate_ok = self._call("nvmlDeviceGetPerformanceState", h, byref(pstate)) == NVML_SUCCESS
        reasons = self._ull("nvmlDeviceGetCurrentClocksEventReasons", h)
        if reasons is None:
            reasons = self._ull("nvmlDeviceGetCurrentClocksThrottleReasons", h)
        power_mw = self._uint("nvmlDeviceGetPowerUsage", h)
        return {
            "sm_clock_mhz": self._uint("nvmlDeviceGetClockInfo", h, c_int(CLOCK_SM)),
            "graphics_clock_mhz": self._uint("nvmlDeviceGetClockInfo", h, c_int(CLOCK_GRAPHICS)),
            "mem_clock_mhz": self._uint("nvmlDeviceGetClockInfo", h, c_int(CLOCK_MEM)),
            "temp_c": self._uint("nvmlDeviceGetTemperature", h, c_int(TEMPERATURE_GPU)),
            "power_w": None if power_mw is None else power_mw / 1000.0,
            "energy_mj": self._ull("nvmlDeviceGetTotalEnergyConsumption", h),
            "util_gpu_pct": util.gpu if util_ok else None,
            "util_mem_pct": util.memory if util_ok else None,
            "pstate": pstate.value if pstate_ok else None,
            "clock_event_reasons": reasons,
            "clock_event_names": decode_reasons(reasons),
        }

    def describe(self, h: c_void_p) -> dict:
        return {
            "name": self._str("nvmlDeviceGetName", h),
            "uuid": self._str("nvmlDeviceGetUUID", h),
            "vbios": self._str("nvmlDeviceGetVbiosVersion", h),
            "max_sm_clock_mhz": self._uint("nvmlDeviceGetMaxClockInfo", h, c_int(CLOCK_SM)),
            "max_mem_clock_mhz": self._uint("nvmlDeviceGetMaxClockInfo", h, c_int(CLOCK_MEM)),
            "power_limit_w": (lambda v: None if v is None else v / 1000.0)(
                self._uint("nvmlDeviceGetEnforcedPowerLimit", h)),
        }

    def driver_version(self) -> str | None:
        return self._str("nvmlSystemGetDriverVersion", size=96)

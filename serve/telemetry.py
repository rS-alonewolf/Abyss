"""serve/telemetry.py - hardware readings for the web app's Monitor tab and Abyss Terminal (idea from PR #22 by
code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: each card read through backends tried in order, each filling what the ones before it left None, and "source"
  in each card's entry says which answered ("nvml", "sysfs+gpu_metrics", "amd-smi", ...):
  NVIDIA - its NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no pip
  package is needed, else `nvidia-smi --query-gpu`; AMD (Linux) - the amdgpu driver's sysfs files (the PCIe link
  traced up to the root port, as `--bench` does) and its binary gpu_metrics table (throttle reasons, an APU's RAM
  traffic), else `amd-smi` / `rocm-smi`; Windows, any vendor - the performance counters Task Manager reads (load
  and VRAM only).  The tools run only for a card no library or sysfs file can read, at most once a second.
  "gpus" lists every card with its role: "main" / "split" (the model's layers: the gpu_* totals), "helper" (a
  decode expert helper) and "igpu" (the integrated Radeon the iGPU lane computes on, in shared memory).
- CPU (serve/telemetry_cpu.py): load per core, clocks per core and cluster, temperature, power, instruction sets.
- RAM: `psutil` when it is installed (setup installs it), else the OS (Windows GlobalMemoryStatusEx, Linux
  /proc); swap, the page cache and memory pressure (Linux PSI).
- Drives: the disk rate of every drive (psutil), and each model / mirror drive's own (Linux /proc/diskstats).
Anything that cannot be read is None (shown as "--", never 0); nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import glob
import os
import platform
import re
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
def gpu_pids():
    """Every process with a compute context on any NVIDIA GPU (NVML), or None when that can't be read (no NVIDIA driver,
    an AMD card, an old driver)."""
    first = _Nvml(0)
    if not first.ok():
        return None
    count = ctypes.c_uint()
    try:
        fn = getattr(first.lib, "nvmlDeviceGetCount_v2", None) or first.lib.nvmlDeviceGetCount
        if fn(ctypes.byref(count)) != 0:
            return None
    except (AttributeError, OSError):
        return None
    out = set()
    for i in range(count.value):
        p = (first if i == 0 else _Nvml(i)).pids()
        if p is None:
            return None
        out |= p
    return out


# nvmlClocksEventReasons bits -> the names /metrics reports ("idle" is no slowdown: the card has nothing to do)
NVML_THROTTLE = ((0x1, "idle"), (0x2, "app_clocks"), (0x4, "power_cap"), (0x8, "hw_slowdown"), (0x10, "sync_boost"),
                 (0x20, "sw_thermal"), (0x40, "hw_thermal"), (0x80, "power_brake"), (0x100, "display_clocks"))


class _Nvml:
    source = "nvml"

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    class Proc1(ctypes.Structure):                       # nvmlProcessInfo_v1_t
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong)]

    class Proc2(ctypes.Structure):                       # nvmlProcessInfo_v2_t (the _v2 and _v3 calls)
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong), ("gi", ctypes.c_uint), ("ci", ctypes.c_uint)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def pids(self):
        """The processes with a compute context on this GPU, or None when NVML can't say."""
        for fn, st in (("nvmlDeviceGetComputeRunningProcesses_v3", self.Proc2),
                       ("nvmlDeviceGetComputeRunningProcesses_v2", self.Proc2),
                       ("nvmlDeviceGetComputeRunningProcesses", self.Proc1)):
            f = getattr(self.lib, fn, None)
            if f is None:
                continue
            n, buf = ctypes.c_uint(256), (st * 256)()
            try:
                if f(self.dev, ctypes.byref(n), buf) == 0:
                    return {buf[i].pid for i in range(n.value)}
            except OSError:
                pass
        return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def throttle(self):
        """Why the clocks are held down right now (NVML's clock event reasons, ex-"throttle reasons") as names, [] when
        nothing holds them, None when NVML can't say."""
        v = ctypes.c_ulonglong()
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            f = getattr(self.lib, fn, None)
            try:
                if f is not None and f(self.dev, ctypes.byref(v)) == 0:
                    return [name for bit, name in NVML_THROTTLE if v.value & bit]
            except OSError:
                pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        out["pcie_width_max"] = self._uint("nvmlDeviceGetMaxPcieLinkWidth")
        out["clock_mhz"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(0))       # NVML_CLOCK_GRAPHICS
        out["mem_clock_mhz"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(2))   # NVML_CLOCK_MEM
        out["throttle"] = self.throttle()
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# ------------------------------------------------------------------------------------------------ AMD (sysfs)
PCIE_GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}   # GT/s -> PCIe generation


class _AmdSysfs:
    """An AMD card's readings from the amdgpu driver's sysfs files (Linux, no ROCm library needed), read like _Nvml:
    a HIP run has to read the card it runs on, not NVML's card 0 (issue #14: an NVIDIA display card beside an
    R9700 was reported as the GPU, and the stall watchdog read its load).  `pci`: the card's address
    ("0000:c7:00.0"); `arch`: its gfx target, for the name when the driver has no product_name.

    The PCIe link is the narrowest one from the card up to its root port, as `--bench` reports it (abyss.py
    pcie_chain): an R9700's own switch says Gen5 x16 inside the card while its OCuLink cable runs Gen4 x4.  The
    driver has no PCIe throughput counter for RDNA 3/4 (pcie_bw is gone, and where it exists a read blocks for a
    second), so pcie_rx_mb / pcie_tx_mb stay None - unknown, not idle."""
    source = "sysfs"

    def __init__(self, pci, arch="", base="/sys/bus/pci/devices"):
        self.dev = os.path.join(base, pci) if pci else None
        self.pci, self.arch = pci, arch
        if self.dev and not os.path.isfile(os.path.join(self.dev, "gpu_busy_percent")):
            self.dev = None
        hw = sorted(glob.glob(os.path.join(self.dev, "hwmon", "hwmon*"))) if self.dev else []
        self.hwmon = hw[0] if hw else None
        # hwmon's sensors by their label (edge / junction / mem, sclk / mclk): the numbers differ between chips
        self.sensor = {}
        for f in glob.glob(os.path.join(self.hwmon, "*_label")) if self.hwmon else []:
            label = self._read(f)
            if label:
                self.sensor[label] = f[:-len("_label")] + "_input"

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _read(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None

    def _int(self, folder, name):
        v = self._read(os.path.join(folder, name)) if folder else None
        try:
            return int(v.split()[0]) if v else None
        except ValueError:
            return None

    def _sensor(self, label, scale):
        path = self.sensor.get(label)
        v = self._int(os.path.dirname(path), os.path.basename(path)) if path else None
        return v / scale if v is not None else None

    def _dpm_mhz(self, name):
        """The current level of a pp_dpm_* table ("1: 3426Mhz *"): the clock when hwmon has no freqN_input."""
        for line in (self._read(os.path.join(self.dev, name)) or "").splitlines():
            if line.rstrip().endswith("*"):
                try:
                    return int(line.split(":", 1)[1].strip().lower().split("mhz")[0])
                except (IndexError, ValueError):
                    return None
        return None

    def pids(self):
        return None

    def name(self):
        return (self._read(os.path.join(self.dev, "product_name")) if self.dev else None) or \
            f"AMD Radeon ({self.arch or '?'})"

    def pcie(self):
        """(generation, width, max generation, max width) of the narrowest link from the card to its root port: each
        the smallest on the way (links train down at idle, so the current values drop between requests).  None
        without the link files."""
        try:
            d = os.path.realpath(self.dev)
        except OSError:
            return None
        chain = []
        while os.path.isfile(os.path.join(d, "current_link_speed")):
            speed, mspeed = self._read(os.path.join(d, "current_link_speed")), self._read(os.path.join(d, "max_link_speed"))
            chain.append((PCIE_GEN.get(speed.split()[0]) if speed else None, self._int(d, "current_link_width"),
                          PCIE_GEN.get(mspeed.split()[0]) if mspeed else None, self._int(d, "max_link_width")))
            d = os.path.dirname(d)
        if not chain:
            return None

        def least(k):
            v = [c[k] for c in chain if c[k] is not None]
            return min(v) if v else None
        return least(0), least(1), least(2), least(3)

    def read(self):
        if not self.dev:
            return {}
        uw = self._int(self.hwmon, "power1_average")
        if uw is None:
            uw = self._int(self.hwmon, "power1_input")
        cap, mc = self._int(self.hwmon, "power1_cap"), self._int(self.hwmon, "temp1_input")
        sclk, mclk = self._sensor("sclk", 1e6), self._sensor("mclk", 1e6)
        gen, width, gen_max, width_max = self.pcie() or (None, None, None, None)
        return {"util": self._int(self.dev, "gpu_busy_percent"),
                "mem_used": self._int(self.dev, "mem_info_vram_used"),
                "mem_total": self._int(self.dev, "mem_info_vram_total"),
                "temp": mc / 1000.0 if mc is not None else None,
                "temp_junction": self._sensor("junction", 1000.0), "temp_mem": self._sensor("mem", 1000.0),
                "power": uw / 1e6 if uw is not None else None,
                "power_limit": cap / 1e6 if cap is not None else None,
                "clock_mhz": round(sclk) if sclk is not None else self._dpm_mhz("pp_dpm_sclk"),
                "mem_clock_mhz": round(mclk) if mclk is not None else self._dpm_mhz("pp_dpm_mclk"),
                "throttle": None,             # the reasons are only in the binary gpu_metrics table: not read
                "pcie_gen": gen, "pcie_gen_max": gen_max, "pcie_width": width, "pcie_width_max": width_max,
                "pcie_rx_mb": None, "pcie_tx_mb": None}


# ------------------------------------------------------------------------------------------------ AMD gpu_metrics
# indep_throttle_status (gpu_metrics v1.3, ASIC independent): bit ranges -> the names /metrics reports
AMD_THROTTLE = ((range(0, 8), "power_cap"), (range(16, 24), "current_limit"), (range(32, 44), "thermal"),
                (range(44, 48), "vr_hot"), (range(56, 58), "other"))
# gpu_metrics v3.0 (APUs: Strix Point / Halo) throttle residency counters at their offsets -> names (one that grew
# since the last read is active); the cores' own thermal limit is the CPU lane's, kept apart
APU_THROTTLE = ((228, "prochot"), (232, "power_cap"), (236, "power_cap"), (240, "power_cap"), (244, "core_thermal"),
                (248, "thermal"), (252, "thermal"))
PCIE_RATE = {25: 1, 50: 2, 80: 3, 160: 4, 320: 5, 640: 6}             # gpu_metrics' 0.1 GT/s -> PCIe generation


def parse_gpu_metrics(b, prev=None):
    """The amdgpu driver's binary gpu_metrics table -> /metrics' card fields, for the layouts read here: v1.1-1.3
    (discrete RDNA 2-4: an R9700 has v1.3) and v3.0 (APUs: Strix Point / Halo).  Fields the firmware does not fill
    (0xFFFF) are None; another format gives {}.  `prev`: the last table, for the v3.0 throttle counters.  The layouts
    are the kernel's structs (kgd_pp_interface.h, natural alignment), checked against an R9700 and a Ryzen AI Max+
    395's Radeon 8060S beside their sysfs readings."""
    import struct
    if len(b) < 4:
        return {}
    size, fmt, rev = struct.unpack_from("<HBB", b, 0)

    def u16(o, scale=1):
        if o + 2 > len(b):
            return None
        v = struct.unpack_from("<H", b, o)[0]
        return None if v == 0xFFFF else (v / scale if scale != 1 else v)

    if fmt == 1 and rev >= 1 and len(b) >= 80:
        out = {"temp": u16(4), "temp_junction": u16(6), "temp_mem": u16(8), "util": u16(16),
               "power": u16(22), "clock_mhz": u16(54), "mem_clock_mhz": u16(58)}
        width, rate = u16(74), u16(76)
        out["pcie_width"] = width if width and width <= 32 else None
        out["pcie_gen"] = PCIE_RATE.get(rate)
        if rev >= 3 and len(b) >= 120:
            v = struct.unpack_from("<Q", b, 112)[0]
            out["throttle"] = None if v == 2**64 - 1 else \
                sorted({name for bits, name in AMD_THROTTLE for i in bits if v >> i & 1})
        return out
    if fmt == 3 and len(b) >= 260:
        temp = u16(4, 100.0)
        out = {"temp": temp or None, "util": u16(42), "clock_mhz": u16(174), "mem_clock_mhz": u16(186),
               "power": struct.unpack_from("<I", b, 112)[0] / 1000.0,       # the whole APU's (CPU + GPU) socket power
               "dram_read_mb": u16(94), "dram_write_mb": u16(96)}         # every reader of the shared RAM
        if prev is not None and len(prev) >= 260:
            grew = lambda o: struct.unpack_from("<I", b, o)[0] > struct.unpack_from("<I", prev, o)[0]  # noqa: E731
            out["throttle"] = sorted({name for o, name in APU_THROTTLE if grew(o)})
        return out
    return {}


def apu_core_power(b):
    """The CPU cores' power (W) from an APU's gpu_metrics v3.0 table - the CPU lane's draw where RAPL can't be read."""
    import struct
    if len(b) >= 260 and struct.unpack_from("<HBB", b, 0)[1] == 3:
        return struct.unpack_from("<I", b, 132)[0] / 1000.0
    return None


class _AmdGpuMetrics:
    """An AMD card's gpu_metrics table (Linux sysfs): what the text files lack - the throttle reasons, an APU's RAM
    traffic, and the temperatures and clocks where hwmon has no label for them."""
    source = "gpu_metrics"

    def __init__(self, pci, base="/sys/bus/pci/devices"):
        self.path = os.path.join(base, pci, "gpu_metrics") if pci else None
        self.prev = None
        if self.path and not parse_gpu_metrics(self._raw() or b""):
            self.path = None

    def _raw(self):
        try:
            with open(self.path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def ok(self):
        return self.path is not None

    def name(self):
        return None

    def pids(self):
        return None

    def read(self):
        b = self._raw() if self.path else None
        if not b:
            return {}
        out, self.prev = parse_gpu_metrics(b, self.prev), b
        return out


def _read_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""


def _apu_metrics(base="/sys/class/drm"):
    """An AMD APU's gpu_metrics v3.0 table (its CPU cores' power), or None."""
    for p in sorted(glob.glob(os.path.join(base, "card[0-9]*", "device", "gpu_metrics"))):
        b = _read_bytes(p)
        if len(b) >= 4 and b[2] == 3:
            return p
    return None


# ------------------------------------------------------------------------------------------------ command fallbacks
_said = set()


def _once(msg):
    """A fallback that failed in a way worth knowing about: said once, in the server's console."""
    if msg not in _said:
        _said.add(msg)
        print(f"[abyss] telemetry: {msg}", flush=True)


class _Command:
    """A GPU tool's output for every card at once (nvidia-smi, amd-smi, rocm-smi), run at most every `every` seconds
    however many cards ask: the fallbacks for a PC whose driver library or sysfs files can't be read."""

    def __init__(self, argv, parse, every=1.0):
        import shutil
        self.exe = shutil.which(argv[0]) or (os.path.join("/opt/rocm/bin", argv[0])
                                             if os.path.isfile(os.path.join("/opt/rocm/bin", argv[0])) else None)
        self.argv, self.parse, self.every = argv, parse, every
        self.t, self.data = 0.0, None

    def get(self):
        if not self.exe:
            return None
        if time.time() - self.t >= self.every:
            import subprocess
            self.t = time.time()
            try:
                r = subprocess.run([self.exe, *self.argv[1:]], capture_output=True, text=True, timeout=10)
                self.data = self.parse(r.stdout) if r.returncode == 0 else None
                if self.data is None:
                    _once(f"{self.argv[0]} gave no readings (exit code {r.returncode})")
            except (OSError, subprocess.TimeoutExpired, ValueError) as e:
                self.data = None
                _once(f"{self.argv[0]} could not be read ({type(e).__name__})")
        return self.data


def _num(v):
    """A tool's number ("274.0", "[N/A]", {"value": 61, "unit": "C"}, "(3191Mhz)") -> float, None when it has none."""
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v or ""))
    return float(m.group()) if m and "n/a" not in str(v).lower() else None


NVIDIA_SMI_FIELDS = ("index", "name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw",
                     "power.limit", "clocks.gr", "clocks.mem", "pcie.link.gen.current", "pcie.link.gen.max",
                     "pcie.link.width.current", "pcie.link.width.max", "clocks_throttle_reasons.active")


def parse_nvidia_smi(text):
    """`nvidia-smi --query-gpu=<NVIDIA_SMI_FIELDS> --format=csv,noheader,nounits` -> {index: card fields}."""
    out = {}
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != len(NVIDIA_SMI_FIELDS) or not f[0].isdigit():
            continue
        n = [_num(x) for x in f]
        mib = lambda v: int(v * 2**20) if v is not None else None   # noqa: E731
        whole = lambda v: int(v) if v is not None else None         # noqa: E731
        try:
            bits = int(f[14], 16)
        except ValueError:
            bits = None
        out[int(f[0])] = {"name": f[1], "util": n[2], "mem_used": mib(n[3]), "mem_total": mib(n[4]), "temp": n[5],
                          "power": n[6], "power_limit": n[7], "clock_mhz": whole(n[8]), "mem_clock_mhz": whole(n[9]),
                          "pcie_gen": whole(n[10]), "pcie_gen_max": whole(n[11]), "pcie_width": whole(n[12]),
                          "pcie_width_max": whole(n[13]),
                          "throttle": [name for bit, name in NVML_THROTTLE if bits & bit] if bits is not None else None}
    return out


def parse_amd_smi(text):
    """`amd-smi metric --json` -> {gpu number: card fields} (ROCm 6.1+; amd-smi's own numbering, matched by PCI
    address through `amd-smi list --json`)."""
    import json
    data = json.loads(text)
    out = {}
    for g in data.get("gpu_data", []) if isinstance(data, dict) else data:
        if not isinstance(g, dict) or "gpu" not in g:
            continue
        use, pw, clk, t, mem, pcie = (g.get(k) or {} for k in ("usage", "power", "clock", "temperature", "mem_usage",
                                                                 "pcie"))
        mb = lambda v: int(v * 2**20) if v is not None else None   # noqa: E731
        whole = lambda v: int(v) if v is not None else None         # noqa: E731
        gfx, mclk = clk.get("gfx_0") or {}, clk.get("mem_0") or {}
        speed = _num(pcie.get("speed"))
        out[g["gpu"]] = {"util": _num(use.get("gfx_activity")), "power": _num(pw.get("socket_power")),
                         "temp": _num(t.get("edge")), "temp_junction": _num(t.get("hotspot")),
                         "temp_mem": _num(t.get("mem")), "mem_used": mb(_num(mem.get("used_vram"))),
                         "mem_total": mb(_num(mem.get("total_vram"))),
                         "clock_mhz": whole(_num(gfx.get("clk"))), "mem_clock_mhz": whole(_num(mclk.get("clk"))),
                         "pcie_width": whole(_num(pcie.get("width"))),
                         "pcie_gen": PCIE_GEN.get(f"{speed:.1f}") if speed else None,
                         "pcie_rx_mb": _num(pcie.get("current_bandwidth_received")),
                         "pcie_tx_mb": _num(pcie.get("current_bandwidth_sent"))}
    return out


def parse_amd_smi_list(text):
    """`amd-smi list --json` -> {PCI address: amd-smi's gpu number}."""
    import json
    return {str(g.get("bdf", "")).lower(): g.get("gpu") for g in json.loads(text) if isinstance(g, dict)}


def parse_rocm_smi(text):
    """`rocm-smi --showuse --showmeminfo vram --showtemp --showpower --showclocks --showbus --json` (the older tool)
    -> {PCI address: card fields}."""
    import json
    out = {}
    for card in json.loads(text).values():
        if not isinstance(card, dict) or "PCI Bus" not in card:
            continue
        get = lambda *keys: next((_num(card[k]) for k in keys if k in card), None)   # noqa: E731
        whole = lambda v: int(v) if v is not None else None                          # noqa: E731
        link = re.search(r"\(([\d.]+)GT/s x(\d+)\)", card.get("pcie clock level", ""))
        out[card["PCI Bus"].lower()] = {
            "util": get("GPU use (%)"), "temp": get("Temperature (Sensor edge) (C)"),
            "temp_junction": get("Temperature (Sensor junction) (C)"),
            "temp_mem": get("Temperature (Sensor memory) (C)"),
            "power": get("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)"),
            "clock_mhz": whole(get("sclk clock speed:")), "mem_clock_mhz": whole(get("mclk clock speed:")),
            "mem_used": whole(get("VRAM Total Used Memory (B)")), "mem_total": whole(get("VRAM Total Memory (B)")),
            "pcie_gen": PCIE_GEN.get(link.group(1)) if link else None,
            "pcie_width": int(link.group(2)) if link else None}
    return out


class _FromCommand:
    """One card's share of a _Command's readings: `key` is its number (nvidia-smi) or PCI address (rocm-smi);
    `lookup(data)` turns the whole output into this card's fields."""

    def __init__(self, source, command, lookup):
        self.source, self.command, self.lookup = source, command, lookup
        self.last = None

    def ok(self):
        data = self.command.get()
        return bool(data) and self.lookup(data) is not None

    def name(self):
        r = self.lookup(self.command.get() or {}) or {}
        return r.get("name")

    def pids(self):
        return None

    def read(self):
        r = self.lookup(self.command.get() or {})
        return {k: v for k, v in (r or {}).items() if k != "name"}


_COMMANDS = {}


def _command(name):
    """The shared tools: one nvidia-smi / amd-smi / rocm-smi run a second serves every card."""
    if name not in _COMMANDS:
        if name == "nvidia-smi":
            _COMMANDS[name] = _Command(["nvidia-smi", "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS),
                                        "--format=csv,noheader,nounits"], parse_nvidia_smi)
        elif name == "amd-smi":
            _COMMANDS[name] = _Command(["amd-smi", "metric", "--json"], parse_amd_smi)
        elif name == "amd-smi list":
            _COMMANDS[name] = _Command(["amd-smi", "list", "--json"], parse_amd_smi_list, every=3600.0)
        elif name == "rocm-smi":
            _COMMANDS[name] = _Command(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower",
                                        "--showclocks", "--showbus", "--json"], parse_rocm_smi)
    return _COMMANDS[name]


def _nvidia_smi(index):
    return _FromCommand("nvidia-smi", _command("nvidia-smi"), lambda d: d.get(index))


def _amd_smi(pci):
    def lookup(d):
        num = (_command("amd-smi list").get() or {}).get(str(pci).lower())
        return d.get(num) if num is not None else None
    return _FromCommand("amd-smi", _command("amd-smi"), lookup)


def _rocm_smi(pci):
    return _FromCommand("rocm-smi", _command("rocm-smi"), lambda d: d.get(str(pci).lower()))


# ------------------------------------------------------------------------------------------------ Windows PDH
def pdh_adapters(engines, memory):
    """Task Manager's GPU figures from the "GPU Engine" / "GPU Adapter Memory" performance counters (any vendor):
    {adapter LUID: {"util", "mem_used"}}.  `engines`: {instance: utilization %} - instances are
    "pid_<n>_luid_<a>_<b>_phys_0_eng_<k>_engtype_<type>"; an adapter's load is its busiest engine type (each type
    the sum over the processes using it), as Task Manager shows it.  `memory`: {"luid_<a>_<b>_phys_0": bytes}."""
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    for inst, v in engines.items():
        m = re.search(r"(luid_0x[0-9a-f]+_0x[0-9a-f]+).*engtype_(\w*)", inst, re.I)
        if m and v is not None:
            per[m.group(1).lower()][m.group(2)] += v
    out = {luid: {"util": min(100.0, max(types.values())) if types else None, "mem_used": None}
           for luid, types in per.items()}
    for inst, v in memory.items():
        m = re.search(r"(luid_0x[0-9a-f]+_0x[0-9a-f]+)", inst, re.I)
        if m:
            out.setdefault(m.group(1).lower(), {"util": None, "mem_used": None})["mem_used"] = int(v)
    return out


class _PdhQuery:
    """The two counter sets through pdh.dll (Windows' performance counters, no extra package)."""

    class Item(ctypes.Structure):          # PDH_FMT_COUNTERVALUE_ITEM_W with a double value
        _fields_ = [("name", ctypes.c_wchar_p), ("status", ctypes.c_ulong), ("value", ctypes.c_double)]

    def __init__(self):
        self.pdh = ctypes.windll.pdh
        self.q = ctypes.c_void_p()
        self.engine, self.mem = ctypes.c_void_p(), ctypes.c_void_p()
        if self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.q)) != 0:
            raise OSError("PdhOpenQuery")
        for path, h in ((r"\GPU Engine(*)\Utilization Percentage", self.engine),
                        (r"\GPU Adapter Memory(*)\Dedicated Usage", self.mem)):
            if self.pdh.PdhAddEnglishCounterW(self.q, path, None, ctypes.byref(h)) != 0:
                raise OSError("PdhAddEnglishCounter")
        self.pdh.PdhCollectQueryData(self.q)
        self.t, self.data = 0.0, {}

    def _array(self, h):
        size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
        self.pdh.PdhGetFormattedCounterArrayW(h, 0x200, ctypes.byref(size), ctypes.byref(count), None)
        if not size.value:
            return {}
        buf = ctypes.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(h, 0x200, ctypes.byref(size), ctypes.byref(count), buf) != 0:
            return {}
        items = ctypes.cast(buf, ctypes.POINTER(self.Item))
        return {items[i].name: items[i].value for i in range(count.value) if items[i].status == 0}

    def get(self):
        if time.time() - self.t >= 1.0:
            self.t = time.time()
            self.pdh.PdhCollectQueryData(self.q)
            self.data = pdh_adapters(self._array(self.engine), self._array(self.mem))
        return self.data


class _Pdh:
    """A card from the Windows performance counters - load and VRAM in use only, matched by position: the n-th
    adapter (by LUID) that has a 3D engine.  The last resort when no vendor library or tool answers."""
    source = "pdh"
    query = None

    def __init__(self, position):
        self.position = position
        try:
            if _Pdh.query is None:
                _Pdh.query = _PdhQuery()
        except (OSError, AttributeError) as e:
            _once(f"the Windows GPU counters could not be read ({e})")

    def _mine(self):
        data = _Pdh.query.get() if _Pdh.query else {}
        luids = sorted(data)
        return data[luids[self.position]] if self.position < len(luids) else None

    def ok(self):
        return self._mine() is not None

    def name(self):
        return None

    def pids(self):
        return None

    def read(self):
        return dict(self._mine() or {})


# ------------------------------------------------------------------------------------------------ one card
class _Card:
    """One card read through backends tried in order, each filling the fields the ones before it left None, and the
    command-line tools (`fallback`, made on first need) only when none of them can read it at all - no tool runs
    each second beside a working library.  read() says in "source" which backends the figures came from."""

    def __init__(self, backends, fallback=lambda: []):
        self.backends = [b for b in backends if b.ok()]
        if not self.backends:
            try:
                self.backends = [b for b in fallback() if b.ok()]
            except Exception:  # noqa: BLE001 - a fallback that can't even start is no reading
                self.backends = []
        self.pci = next((getattr(b, "pci", None) for b in backends if getattr(b, "pci", None)), None)

    def ok(self):
        return bool(self.backends)

    def name(self):
        return next((n for n in (b.name() for b in self.backends) if n), None)

    def pids(self):
        return self.backends[0].pids() if self.backends else None

    def read(self):
        out, used = {}, []
        for b in self.backends:
            try:
                r = b.read()
            except Exception:  # noqa: BLE001 - one backend's hiccup must not lose the others' figures
                continue
            filled = [k for k, v in r.items() if v is not None and out.get(k) is None]
            for k in filled:
                out[k] = r[k]
            for k in r:
                out.setdefault(k, None)
            if filled:
                used.append(b.source)
        out["source"] = "+".join(used) or None
        return out


def nvidia_card(index):
    """An NVIDIA card: NVML, else nvidia-smi, else (Windows) the performance counters."""
    return _Card([_Nvml(index)], lambda: [_nvidia_smi(index)] + ([_Pdh(index)] if os.name == "nt" else []))


def amd_card(index, pci, arch=""):
    """An AMD card: sysfs text files then its gpu_metrics table (Linux), else amd-smi / rocm-smi, else (Windows) the
    performance counters."""
    if os.name == "nt":
        return _Card([], lambda: [_Pdh(index or 0)])
    return _Card([_AmdSysfs(pci, arch), _AmdGpuMetrics(pci)],
                 lambda: [_amd_smi(pci), _rocm_smi(pci)] if pci else [])


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


def memory_pressure(path="/proc/pressure/memory"):
    """(some, full): the share of the last 10 s that some / all running tasks waited for memory (Linux PSI, %), the
    sign that the RAM tier or the page cache is being squeezed before anything is killed.  (None, None) elsewhere."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                kind, *fields = line.split()
                kv = dict(x.split("=", 1) for x in fields if "=" in x)
                out[kind] = float(kv["avg10"])
    except (OSError, ValueError, KeyError):
        return None, None
    return out.get("some"), out.get("full")


def _meminfo(path="/proc/meminfo"):
    """/proc/meminfo in bytes ({} when it can't be read: not Linux)."""
    try:
        with open(path) as f:
            return {k: int(v.split()[0]) * 1024 for k, v in (line.split(":", 1) for line in f if ":" in line)}
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------------ drives
def block_devices(path, sys_block="/sys/class/block", dev_block="/sys/dev/block", mountinfo="/proc/self/mountinfo"):
    """The whole disks a file or folder is stored on (Linux): ["nvme0n1"] - a partition's disk, an LVM volume's or a
    RAID's member disks under it, so the reading is the drive's own (what limits expert reads), not one volume's.  []
    when it can't be told (not Linux, a network share, a path that does not exist)."""
    try:
        st = os.stat(path)
    except OSError:
        return []
    dev = os.path.join(dev_block, f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}") if os.name != "nt" else ""
    name = os.path.basename(os.path.realpath(dev)) if dev and os.path.exists(dev) else None
    if name is None:                    # btrfs and friends give an anonymous st_dev: the mount's source device
        best = ""
        try:
            with open(mountinfo) as f:
                for line in f:
                    left, _, right = line.partition(" - ")
                    mnt, src = left.split()[4], (right.split() + ["", ""])[1]
                    here = os.path.realpath(path)
                    if src.startswith("/dev/") and (here == mnt or here.startswith(mnt.rstrip("/") + "/")) and \
                            len(mnt) >= len(best):
                        best, name = mnt, os.path.basename(os.path.realpath(src))
        except (OSError, IndexError):
            return []
    if not name:
        return []

    def disks(n, depth=0):
        d = os.path.join(sys_block, n)
        if depth > 8 or not os.path.exists(d):
            return []
        if os.path.exists(os.path.join(d, "partition")):         # nvme0n1p2 -> nvme0n1
            return disks(os.path.basename(os.path.dirname(os.path.realpath(d))), depth + 1)
        try:
            slaves = sorted(os.listdir(os.path.join(d, "slaves")))
        except OSError:
            slaves = []
        if slaves:                                                 # dm-0 (LVM, LUKS), md0: the disks under it
            return sorted({x for sl in slaves for x in disks(sl, depth + 1)})
        return [n]
    return disks(name)


def disk_counters(path="/proc/diskstats"):
    """{disk: (bytes read, bytes written)} since boot (Linux; sectors are 512 bytes here whatever the drive's)."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 10:
                    out[p[2]] = (int(p[5]) * 512, int(p[9]) * 512)
    except (OSError, ValueError):
        return {}
    return out


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    MODEL_ROLES = ("main", "split")      # the cards the model's layers run on: what the gpu_* totals are made of

    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=None, roles=None, igpu=None, drives=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature).  `amd`: a HIP run's cards as [(HIP number, PCI
        address, arch)], read from sysfs instead of NVML (issue #14).  `roles`: {number: "main" | "split" |
        "helper"} (the first is "main", the rest "split" when it names none) - a decode expert helper is listed but
        left out of the totals.  `igpu`: (HIP number or None, PCI address, arch) of the integrated Radeon the iGPU
        lane computes on, listed as role "igpu" and never in the totals.  `drives`: [(role, path)] - the model's
        shards ("model") and their STRATA_GLM_MIRROR copies ("mirror"): each one's drive is read on its own.
        "gpus" lists every card read, one card too."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        cards = [(i, amd_card(i, pci, arch)) for i, pci, arch in amd] if amd is not None else \
            [(i, nvidia_card(i)) for i in idx]
        cards = [(i, g) for i, g in cards if g.ok()] or cards[:1]
        roles = dict(roles or {})
        self.cards = [(i, g, roles.get(i) or ("main" if n == 0 else "split")) for n, (i, g) in enumerate(cards)]
        if igpu is not None:
            lane = amd_card(igpu[0], igpu[1], igpu[2])
            if lane.ok():
                self.cards.append((igpu[0], lane, "igpu"))
        self.gpus = [(i, g) for i, g, r in self.cards if r in self.MODEL_ROLES] or [(i, g) for i, g, _ in self.cards[:1]]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        from serve.telemetry_cpu import CpuProbe
        apu = _apu_metrics()
        self.cpu = CpuProbe(ps=self.ps, apu_power=(lambda: apu_core_power(_read_bytes(apu))) if apu else None)
        self.names = {id(g): (g.name() if g.ok() else None) for _, g, _ in self.cards}
        # the drives: one entry a disk, with the roles and paths stored there (a mirror on the model's own drive is one)
        self.drives = []
        for role, path in drives or []:
            disks = tuple(block_devices(path))
            d = next((d for d in self.drives if d["disks"] == disks and disks), None)
            if d is None:
                d = {"name": "+".join(disks) or None, "disks": disks, "roles": [], "paths": []}
                self.drives.append(d)
            if role not in d["roles"]:
                d["roles"].append(role)
            d["paths"].append(str(path))
        self.static = {
            "gpu_name": " + ".join(self.names[id(g)] or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            "gpus": [{"index": i, "role": r, "name": self.names[id(g)], "pci": getattr(g, "pci", None),
                      "shared_memory": r == "igpu"} for i, g, r in self.cards if g.ok()],
            "drives": [{"name": d["name"], "roles": d["roles"], "paths": d["paths"]} for d in self.drives],
            "cpu_name": _cpu_name() or self.cpu.static.get("cpu_model"),
            **self.cpu.static,
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        self._drives_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def _drive_rates(self):
        """hardware.drives: each model / mirror drive's MB/s since the last sample (None until two samples, and for a
        drive that could not be told or counted)."""
        if not self.drives:
            return None
        t, c = time.time(), disk_counters()
        prev, self._drives_prev = self._drives_prev, (t, c)
        out = []
        for d in self.drives:
            rd = wr = None
            if prev is not None and t > prev[0] and d["disks"] and all(k in c and k in prev[1] for k in d["disks"]):
                dt = t - prev[0]
                rd = sum(c[k][0] - prev[1][k][0] for k in d["disks"]) / dt / 2**20
                wr = sum(c[k][1] - prev[1][k][1] for k in d["disks"]) / dt / 2**20
            out.append({"name": d["name"], "roles": list(d["roles"]), "read_mb": rd, "write_mb": wr})
        return out

    def _memory(self, s):
        """Swap, the page cache and memory pressure (PSI) into the sample s."""
        info = _meminfo()
        if self.ps:
            try:
                sw = self.ps.swap_memory()
                s["swap_used"], s["swap_total"] = sw.used, sw.total
            except (OSError, RuntimeError):
                pass
        elif "SwapTotal" in info:
            s["swap_used"], s["swap_total"] = info["SwapTotal"] - info.get("SwapFree", 0), info["SwapTotal"]
        s["ram_cached"] = info.get("Cached")      # the page cache: shards the RAM tier does not hold come from here
        s["psi_mem_some"], s["psi_mem_full"] = memory_pressure()

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = {id(g): g.read() for _, g, _ in self.cards if g.ok()}
            model = [reads[id(g)] for _, g in self.gpus]
            g = dict(model[0])
            if len(model) > 1:
                def vals(k):
                    return [r[k] for r in model if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                for k in ("temp", "temp_junction", "temp_mem"):
                    t = vals(k)
                    g[k] = max(t) if t else None
            s.update({f"gpu_{k}": v for k, v in g.items()})
            s["gpus"] = [{"index": i, "role": r, "name": self.names[id(c)], "shared_memory": r == "igpu", **reads[id(c)]}
                         for i, c, r in self.cards if id(c) in reads]
        try:
            s.update(self.cpu.read())
        except Exception:  # noqa: BLE001 - telemetry must never take the server down
            pass
        if self.ps:
            try:
                if s.get("cpu") is None:
                    s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            if s.get("cpu") is None:
                s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        self._memory(s)
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        drives = self._drive_rates()
        if drives is not None:
            s["drives"] = drives
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _loop(self):
        while True:
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}

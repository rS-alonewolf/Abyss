"""serve/telemetry_cpu.py - the CPU's readings for /metrics (serve/telemetry.py samples them once a second).

The CPU lane computes the experts the GPU does not hold, so its load, clocks, heat and power belong beside the GPU's.
Each reading is tried from the sources a PC may have, in order, and "cpu_sources" says which one answered:
- load, whole and per core: Linux /proc/stat, else psutil, else the OS (Windows GetSystemTimes: the whole only);
- clocks: Linux cpufreq per core and per cluster (cpufreq policies with the same top clock: an Arm big.LITTLE
  chip's clusters, a Ryzen's two CCDs), its driver (amd-pstate / intel_pstate / cppc), governor and EPP, else psutil;
- temperature: hwmon k10temp / zenpower (AMD), coretemp (Intel), an Arm SoC's cpu thermal sensor, else the thermal
  zones (x86_pkg_temp, cpu-thermal, ...).  Windows has no reliable source without a driver: None;
- power: RAPL (powercap intel-rapl:0, AMD too on recent kernels - root-only to read on most distributions since
  2020's PLATYPUS fix, so usually None), amd_energy, a CPU power sensor in hwmon (Arm), else an AMD APU's own power
  table (gpu_metrics v3.0: the cores' power);
- the instruction sets the CPU lane's kernels pick from (AVX-512 / VBMI / AVX-VNNI on x86, NEON / dotprod / SVE on
  Arm) and Intel's thermal throttle counters.
Anything that cannot be read is None (the Terminal shows "--"); nothing here can stop the server.
"""
from __future__ import annotations

import glob
import os
import re
import time

# the instruction sets worth showing: /proc/cpuinfo's spelling -> the name /metrics uses
X86_ISA = (("avx2", "avx2"), ("fma", "fma"), ("avx512f", "avx512f"), ("avx512bw", "avx512bw"),
           ("avx512vl", "avx512vl"), ("avx512vbmi", "avx512vbmi"), ("avx512_vnni", "avx512_vnni"),
           ("avx512_bf16", "avx512_bf16"), ("avx_vnni", "avx_vnni"), ("amx_tile", "amx"))
ARM_ISA = (("asimd", "neon"), ("asimddp", "dotprod"), ("i8mm", "i8mm"), ("bf16", "bf16"), ("sve", "sve"),
           ("sve2", "sve2"))
ARM_VENDORS = {"0x41": "Arm", "0x42": "Broadcom", "0x43": "Cavium", "0x48": "HiSilicon", "0x4e": "NVIDIA",
               "0x51": "Qualcomm", "0x61": "Apple", "0xc0": "Ampere"}
# hwmon drivers of CPU sensors -> the label that is the package temperature (None: the first temperature)
CPU_HWMON = (("k10temp", "Tctl"), ("zenpower", "Tdie"), ("coretemp", "Package id 0"), ("cpu_thermal", None),
             ("cpu-thermal", None), ("soc_thermal", None))
CPU_ZONES = ("x86_pkg_temp", "cpu-thermal", "cpu_thermal", "cpu0-thermal", "soc_thermal", "cpu-0-0-thermal")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _int(path):
    v = _read(path)
    try:
        return int(v.split()[0]) if v else None
    except ValueError:
        return None


def _cpus(text):
    """"0-3,8" -> [0, 1, 2, 3, 8]."""
    out = []
    for part in (text or "").replace(" ", ",").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                out += range(int(a), int(b) + 1)
        elif part.isdigit():
            out.append(int(part))
    return out


class CpuProbe:
    """The CPU's readings.  `sys`, `proc`: the folders to read (a test points them at a fake tree); `ps`: psutil or
    None; `apu_power()`: the cores' power from an AMD APU's gpu_metrics (W) or None."""

    def __init__(self, sys_root="/sys", proc="/proc", ps=None, apu_power=None):
        self.sys, self.proc, self.ps, self.apu_power = sys_root, proc, ps, apu_power
        self.prev_stat = None
        self.prev_energy = None
        self.power_blocked = None
        self.prev_throttle = None
        self.cpuinfo = self._cpuinfo()
        self.static = self._static()

    # ------------------------------------------------------------------------------------------------ what it is
    def _cpuinfo(self):
        """The first processor's /proc/cpuinfo fields (+ the device tree's model on an Arm board)."""
        info = {}
        for line in (_read(os.path.join(self.proc, "cpuinfo")) or "").splitlines():
            if not line.strip():
                if info.get("flags") or info.get("Features"):
                    break
                continue
            k, _, v = line.partition(":")
            info.setdefault(k.strip(), v.strip())
        return info

    def _static(self):
        ci = self.cpuinfo
        flags = set((ci.get("flags") or ci.get("Features") or "").split())
        arm = "Features" in ci or "CPU implementer" in ci
        if flags:
            isa = [name for f, name in (ARM_ISA if arm else X86_ISA) if f in flags]
        else:
            isa = self._windows_isa()
        vendor = {"AuthenticAMD": "AMD", "GenuineIntel": "Intel"}.get(ci.get("vendor_id"), ci.get("vendor_id")) or \
            (ARM_VENDORS.get(ci.get("CPU implementer", "").lower()) if arm else None)
        if vendor is None and os.name == "nt":
            import platform
            p = platform.processor()
            vendor = "AMD" if "AMD" in p else "Intel" if "Intel" in p else "Arm" if "ARM" in p.upper() else None
        model = ci.get("model name") or (_read(os.path.join(self.sys, "firmware/devicetree/base/model")) or "")\
            .rstrip("\x00") or None
        pol = os.path.join(self.sys, "devices/system/cpu/cpufreq/policy0")
        return {"cpu_vendor": vendor, "cpu_model": model, "cpu_isa": isa,
                "cpu_freq_driver": _read(os.path.join(pol, "scaling_driver")),
                "cpu_governor": _read(os.path.join(pol, "scaling_governor")),
                "cpu_epp": _read(os.path.join(pol, "energy_performance_preference"))}

    @staticmethod
    def _windows_isa():
        """Windows: what IsProcessorFeaturePresent can tell (AVX2, AVX-512F; NEON on Arm) - VBMI and VNNI it can't."""
        if os.name != "nt":
            return None
        try:
            import ctypes
            has = ctypes.windll.kernel32.IsProcessorFeaturePresent
            return [name for pf, name in ((40, "avx2"), (41, "avx512f"), (19, "neon")) if has(pf)]
        except (OSError, AttributeError):
            return None

    # ------------------------------------------------------------------------------------------------ load
    def _usage(self):
        """(whole %, [per core %], source)."""
        text = _read(os.path.join(self.proc, "stat"))
        if text:
            cur = {}
            for line in text.splitlines():
                f = line.split()
                if f and f[0].startswith("cpu"):
                    try:
                        v = [int(x) for x in f[1:9]]
                    except ValueError:
                        continue
                    cur[f[0]] = (v[3] + v[4], sum(v))           # (idle + iowait, all)
            prev, self.prev_stat = self.prev_stat, cur

            def pct(k):
                if not prev or k not in prev or k not in cur or cur[k][1] <= prev[k][1]:
                    return None
                return round(max(0.0, min(100.0, 100.0 * (1 - (cur[k][0] - prev[k][0]) /
                                                          (cur[k][1] - prev[k][1])))), 1)
            cores = sorted((k for k in cur if k[3:].isdigit()), key=lambda k: int(k[3:]))
            return pct("cpu"), [pct(k) for k in cores] if prev else None, "proc"
        if self.ps:
            try:
                return self.ps.cpu_percent(interval=None), self.ps.cpu_percent(interval=None, percpu=True), "psutil"
            except (OSError, RuntimeError):
                pass
        return None, None, None

    # ------------------------------------------------------------------------------------------------ clocks
    def _freq(self):
        """([per core MHz], [clusters], source): a cluster = the cpufreq policies with the same top clock."""
        base = os.path.join(self.sys, "devices/system/cpu")
        per = {}
        for d in glob.glob(os.path.join(base, "cpu[0-9]*")):
            n = os.path.basename(d)[3:]
            khz = _int(os.path.join(d, "cpufreq", "scaling_cur_freq"))
            if n.isdigit() and khz:
                per[int(n)] = round(khz / 1000)
        if per:
            groups = {}
            for pol in glob.glob(os.path.join(base, "cpufreq", "policy*")):
                top = _int(os.path.join(pol, "cpuinfo_max_freq"))
                cpus = _cpus(_read(os.path.join(pol, "related_cpus")) or _read(os.path.join(pol, "affected_cpus")))
                if top and cpus:
                    groups.setdefault(top, set()).update(cpus)
            clusters = []
            for top, cpus in sorted(groups.items(), key=lambda kv: min(kv[1])):
                now = [per[c] for c in sorted(cpus) if c in per]
                clusters.append({"cpus": sorted(cpus), "max_mhz": round(top / 1000),
                                 "mhz": round(sum(now) / len(now)) if now else None})
            return [per.get(i) for i in range(max(per) + 1)], clusters or None, "cpufreq"
        if self.ps:
            try:
                f = self.ps.cpu_freq(percpu=True) or []
                if f:
                    return [round(x.current) if x.current else None for x in f], None, "psutil"
            except (OSError, RuntimeError, NotImplementedError, AttributeError):
                pass
        return None, None, None

    # ------------------------------------------------------------------------------------------------ heat
    def _temp(self):
        """(package °C, {sensor label: °C}, source)."""
        mons = {}
        for h in glob.glob(os.path.join(self.sys, "class/hwmon/hwmon*")):
            mons.setdefault(_read(os.path.join(h, "name")) or "", h)
        for name, main in CPU_HWMON:
            h = mons.get(name)
            if not h:
                continue
            temps = {}
            for f in sorted(glob.glob(os.path.join(h, "temp*_input")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
                v = _int(f)
                if v is not None:
                    label = _read(f[:-len("_input")] + "_label") or os.path.basename(f)[:-len("_input")]
                    temps[label] = v / 1000.0
            if temps:
                pkg = temps.get(main) if main else None
                return (pkg if pkg is not None else next(iter(temps.values()))), temps, name
        for z in sorted(glob.glob(os.path.join(self.sys, "class/thermal/thermal_zone*"))):
            kind = _read(os.path.join(z, "type"))
            v = _int(os.path.join(z, "temp"))
            if kind in CPU_ZONES and v is not None:
                return v / 1000.0, {kind: v / 1000.0}, "thermal_zone"
        return None, None, None

    # ------------------------------------------------------------------------------------------------ power
    def _energy(self):
        """(µJ counter, its wrap, source) of the package energy: RAPL, then amd_energy's socket counter."""
        rapl = os.path.join(self.sys, "class/powercap/intel-rapl:0")
        if os.path.isdir(rapl):
            try:
                with open(os.path.join(rapl, "energy_uj")) as f:
                    return int(f.read()), _int(os.path.join(rapl, "max_energy_range_uj")), "rapl"
            except PermissionError:
                self.power_blocked = "rapl"      # root-only (docs/MANUAL: how to allow it)
            except (OSError, ValueError):
                pass
        for h in glob.glob(os.path.join(self.sys, "class/hwmon/hwmon*")):
            if _read(os.path.join(h, "name")) == "amd_energy":
                for f in sorted(glob.glob(os.path.join(h, "energy*_label"))):
                    if (_read(f) or "").lower().startswith("esocket"):
                        v = _int(f[:-len("_label")] + "_input")
                        if v is not None:
                            return v, None, "amd_energy"
        return None, None, None

    def _power(self):
        """(W, source)."""
        e, wrap, src = self._energy()
        t = time.monotonic()
        if e is not None:
            prev, self.prev_energy = self.prev_energy, (t, e, src)
            if prev and prev[2] == src and t > prev[0]:
                d = e - prev[1]
                if d < 0 and wrap:
                    d += wrap
                if d >= 0:
                    return round(d / 1e6 / (t - prev[0]), 1), src
            return None, src
        for h in glob.glob(os.path.join(self.sys, "class/hwmon/hwmon*")):
            if (_read(os.path.join(h, "name")) or "") in ("amdgpu", "nouveau", "nvme"):
                continue
            for f in sorted(glob.glob(os.path.join(h, "power*_label"))):
                if re.search(r"cpu|core|package|big|little", _read(f) or "", re.I):
                    for suffix in ("_input", "_average"):
                        v = _int(f[:-len("_label")] + suffix)
                        if v is not None:
                            return round(v / 1e6, 1), "hwmon"
        if self.apu_power:
            try:
                w = self.apu_power()
            except Exception:  # noqa: BLE001
                w = None
            if w is not None:
                return round(w, 1), "gpu_metrics"
        return None, None

    def _throttle(self):
        """(Intel's thermal throttle events since boot, whether it rose since the last sample) - None elsewhere."""
        base = os.path.join(self.sys, "devices/system/cpu")
        core = [_int(p) for p in glob.glob(os.path.join(base, "cpu[0-9]*", "thermal_throttle", "core_throttle_count"))]
        pkg = [_int(p) for p in glob.glob(os.path.join(base, "cpu[0-9]*", "thermal_throttle", "package_throttle_count"))]
        if not core and not pkg:
            return None, None
        n = sum(v for v in core if v) + max([v for v in pkg if v] or [0])
        prev, self.prev_throttle = self.prev_throttle, n
        return n, (n > prev) if prev is not None else None

    def read(self):
        usage, cores, s_use = self._usage()
        mhz, clusters, s_freq = self._freq()
        temp, temps, s_temp = self._temp()
        power, s_pow = self._power()
        count, rising = self._throttle()
        return {"cpu": usage, "cpu_cores": cores, "cpu_mhz": mhz, "cpu_clusters": clusters, "cpu_temp": temp,
                "cpu_temps": temps, "cpu_power": power, "cpu_throttle_count": count, "cpu_throttling": rising,
                "cpu_sources": {"usage": s_use, "freq": s_freq, "temp": s_temp, "power": s_pow if power is not None
                                else None},
                "cpu_power_blocked": self.power_blocked if power is None else None}

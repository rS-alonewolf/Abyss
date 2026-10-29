"""serve/test_telemetry.py - the hardware readings behind /metrics, against fake sysfs trees and recorded tool output (no
GPU needed): the AMD PCIe link up to the root port, gpu_metrics tables (recorded from an AI PRO R9700 and a Ryzen AI
Max+ 395's Radeon 8060S), the backends tried in order per card, nvidia-smi / amd-smi / rocm-smi / Windows counter
output, the cards' roles, the model and mirror drives, memory pressure, and the CPU on AMD, Intel and Arm.

    python -m unittest serve.test_telemetry -v
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import telemetry as T  # noqa: E402
from serve.telemetry_cpu import CpuProbe  # noqa: E402

# gpu_metrics as read on this project's test PC (2026-10-30, an imatrix job on the R9700): v1.3 and v3.0
R9700_V13 = bytes.fromhex(
    "78000103470050004b004700440046004a00040000007f00ffffffffffffffff3002897b0cf90100d909ffff410818001800000000"
    "00d909690464041900190019001900020000004f040400a000ffffffffffffffffffffffffffffffffffffffffffffffffffff3103"
    "94032905ffff0000000010000000")
GFX1151_V30 = bytes.fromhex(
    "08010300940ff2100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "000000000000000000002f000400060005000f0005003b001600060003000300060004000200070005004d15db0b00000000ffff"
    "9220d2af28f90100698000000000ffff698000000000000000000000f83a00003e042100fb10210035003300d22b2a03861d5a02"
    "1400140014001c001e002b0227f8ffffffff58025802200300009b072003e80300001e14790b1e14790b790b790b1e14790b4213"
    "4213960b960b960b960b960b42131d14540bffff00000000f026a500938a5700a331e60399390604000000000000000040420f00"
    "ffffffff")


def write(root, files):
    for name, v in files.items():
        p = Path(root, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(v + "\n")


class GpuMetrics(unittest.TestCase):
    def test_r9700_v13(self):
        r = T.parse_gpu_metrics(R9700_V13)
        self.assertEqual((r["temp"], r["temp_junction"], r["temp_mem"]), (71, 80, 75))
        self.assertEqual((r["util"], r["power"], r["clock_mhz"]), (74, 127, 2521))
        self.assertEqual((r["pcie_gen"], r["pcie_width"]), (4, 4))      # the card's view of its link: OCuLink Gen4 x4
        self.assertEqual(r["throttle"], ["thermal"])                     # indep bit 36: the hotspot limit

    def test_missing_fields_are_none(self):
        b = bytearray(R9700_V13)
        b[16:18] = b"\xff\xff"                                           # activity not filled by the firmware
        b[112:120] = b"\xff" * 8
        r = T.parse_gpu_metrics(bytes(b))
        self.assertIsNone(r["util"])
        self.assertIsNone(r["throttle"])

    def test_apu_v30_and_its_throttle_counters(self):
        r = T.parse_gpu_metrics(GFX1151_V30)
        self.assertAlmostEqual(r["temp"], 39.88)
        self.assertEqual((r["util"], r["clock_mhz"], r["dram_read_mb"]), (0, 600, 5453))
        self.assertNotIn("throttle", r)                                  # one table: no counter has grown yet
        prev = bytearray(GFX1151_V30)
        prev[244:248] = struct.pack("<I", struct.unpack_from("<I", GFX1151_V30, 244)[0] - 5)
        self.assertEqual(T.parse_gpu_metrics(GFX1151_V30, bytes(prev))["throttle"], ["core_thermal"])
        self.assertAlmostEqual(T.apu_core_power(GFX1151_V30), struct.unpack_from("<I", GFX1151_V30, 132)[0] / 1000)

    def test_unknown_format(self):
        self.assertEqual(T.parse_gpu_metrics(struct.pack("<HBB", 64, 2, 1) + bytes(60)), {})
        self.assertEqual(T.parse_gpu_metrics(b""), {})


class AmdSysfs(unittest.TestCase):
    def card(self, t):
        """An R9700 behind its own switch on OCuLink: root port Gen4 x4 (x8 capable), switch up Gen4 x4, switch down and
        the card Gen5 x16."""
        root = Path(t, "pci0000:00", "0000:00:03.1")
        up = root / "0000:c5:00.0"
        down = up / "0000:c6:00.0"
        dev = down / "0000:c7:00.0"
        for d, cur, w, mx, mw in ((root, "16.0", 4, "16.0", 8), (up, "16.0", 4, "32.0", 16), (down, "32.0", 16, "32.0", 16),
                                  (dev, "32.0", 16, "32.0", 16)):
            write(d, {"current_link_speed": f"{cur} GT/s PCIe", "current_link_width": str(w),
                      "max_link_speed": f"{mx} GT/s PCIe", "max_link_width": str(mw)})
        write(dev, {"gpu_busy_percent": "92", "mem_info_vram_used": str(26 << 30), "mem_info_vram_total": str(32 << 30),
                    "hwmon/hwmon4/temp1_input": "70000", "hwmon/hwmon4/temp1_label": "edge",
                    "hwmon/hwmon4/temp2_input": "77000", "hwmon/hwmon4/temp2_label": "junction",
                    "hwmon/hwmon4/temp3_input": "72000", "hwmon/hwmon4/temp3_label": "mem",
                    "hwmon/hwmon4/freq1_input": "3426000000", "hwmon/hwmon4/freq1_label": "sclk",
                    "hwmon/hwmon4/power1_average": "274000000", "hwmon/hwmon4/power1_cap": "300000000",
                    "pp_dpm_mclk": "0: 96Mhz\n1: 456Mhz\n5: 1258Mhz *"})
        bus = Path(t, "bus")
        bus.mkdir()
        os.symlink(dev, bus / "0000:c7:00.0")
        return str(bus), dev

    def test_pcie_is_the_narrowest_link_to_the_root_port(self):
        with tempfile.TemporaryDirectory() as t:
            base, dev = self.card(t)
            r = T._AmdSysfs("0000:c7:00.0", "gfx1201", base=base).read()
            self.assertEqual((r["pcie_gen"], r["pcie_width"], r["pcie_gen_max"], r["pcie_width_max"]), (4, 4, 4, 8))
            self.assertIsNone(r["pcie_rx_mb"])                    # no counter on RDNA 3/4: unknown, never 0
            self.assertEqual((r["temp"], r["temp_junction"], r["temp_mem"]), (70.0, 77.0, 72.0))
            self.assertEqual((r["clock_mhz"], r["mem_clock_mhz"]), (3426, 1258))   # hwmon sclk; mclk from pp_dpm

    def test_card_backends_fill_in_order_and_say_so(self):
        with tempfile.TemporaryDirectory() as t:
            base, dev = self.card(t)
            (dev / "gpu_metrics").write_bytes(R9700_V13)
            card = T._Card([T._AmdSysfs("0000:c7:00.0", "gfx1201", base=base), T._AmdGpuMetrics("0000:c7:00.0", base=base)],
                           lambda: self.fail("no tool while sysfs answers"))
            r = card.read()
            self.assertEqual(r["source"], "sysfs+gpu_metrics")
            self.assertEqual(r["temp"], 70.0)                     # sysfs first: gpu_metrics only fills what it lacks
            self.assertEqual(r["throttle"], ["thermal"])
            self.assertEqual(card.pci, "0000:c7:00.0")


class FakeBackend:
    def __init__(self, source, values, ok=True):
        self.source, self.values, self._ok = source, values, ok

    def ok(self):
        return self._ok

    def name(self):
        return self.values.get("name")

    def pids(self):
        return None

    def read(self):
        return {k: v for k, v in self.values.items() if k != "name"}


class Cards(unittest.TestCase):
    def test_fallback_only_when_nothing_else_reads_the_card(self):
        card = T._Card([FakeBackend("nvml", {}, ok=False)], lambda: [FakeBackend("nvidia-smi", {"util": 50, "name": "X"})])
        self.assertEqual((card.read()["source"], card.read()["util"], card.name()), ("nvidia-smi", 50, "X"))
        self.assertFalse(T._Card([], lambda: []).ok())

    def test_a_backend_that_adds_nothing_is_not_named(self):
        card = T._Card([FakeBackend("sysfs", {"util": 1, "temp": None}), FakeBackend("gpu_metrics", {"util": 9})])
        r = card.read()
        self.assertEqual((r["util"], r["temp"], r["source"]), (1, None, "sysfs"))

    def test_roles_totals_and_the_igpu(self):
        cards = {(0, "a"): FakeBackend("sysfs", {"util": 80, "mem_used": 10, "temp": 70, "name": "R9700"}),
                 (1, "b"): FakeBackend("sysfs", {"util": 20, "mem_used": 5, "temp": 50, "name": "helper"}),
                 (2, "c"): FakeBackend("sysfs", {"util": 99, "mem_used": 1, "temp": 90, "name": "8060S"})}
        with mock.patch.object(T, "amd_card", side_effect=lambda i, pci, arch="": T._Card([cards[(i, pci)]])), \
                mock.patch.object(T.threading, "Thread"):
            tel = T.Telemetry(amd=[(0, "a", ""), (1, "b", "")], roles={1: "helper"}, igpu=(2, "c", "gfx1151"))
            s = tel.sample()
        self.assertEqual([(g["index"], g["role"], g["shared_memory"]) for g in s["gpus"]],
                         [(0, "main", False), (1, "helper", False), (2, "igpu", True)])
        self.assertEqual((s["gpu_util"], s["gpu_mem_used"], s["gpu_temp"]), (80, 10, 70))   # the model's card only
        self.assertEqual((tel.static["gpu_name"], tel.static["gpu_count"]), ("R9700", 1))

    def test_one_card_is_listed_too(self):
        with mock.patch.object(T, "nvidia_card", return_value=T._Card([FakeBackend("nvml", {"util": 5, "name": "5090"})])), \
                mock.patch.object(T.threading, "Thread"):
            s = T.Telemetry(gpu_index=0).sample()
        self.assertEqual([(g["role"], g["source"]) for g in s["gpus"]], [("main", "nvml")])


class Tools(unittest.TestCase):
    def test_nvidia_smi(self):
        out = T.parse_nvidia_smi("0, NVIDIA GeForce RTX 5090, 97, 30000, 32607, 66, 455.20, 575.00, 2730, 14001, 4, 5, "
                                 "4, 16, 0x0000000000000004\n1, NVIDIA GeForce RTX 3090, [N/A], 1, 24576, 40, [N/A], "
                                 "350.00, 210, 405, 1, 4, 16, 16, 0x0000000000000001\n")
        a, b = out[0], out[1]
        self.assertEqual((a["util"], a["mem_total"], a["pcie_gen"], a["pcie_width"], a["throttle"]),
                         (97.0, 32607 << 20, 4, 4, ["power_cap"]))
        self.assertEqual((b["util"], b["power"], b["throttle"]), (None, None, ["idle"]))

    def test_amd_smi(self):
        text = json.dumps({"gpu_data": [{"gpu": 0, "usage": {"gfx_activity": {"value": 50, "unit": "%"}},
                                         "power": {"socket_power": {"value": 60, "unit": "W"}},
                                         "clock": {"gfx_0": {"clk": {"value": 1976, "unit": "MHz"}}},
                                         "temperature": {"edge": {"value": 70, "unit": "C"},
                                                         "hotspot": {"value": 76, "unit": "C"}, "mem": "N/A"},
                                         "pcie": {"width": 4, "speed": {"value": 16, "unit": "GT/s"},
                                                  "current_bandwidth_received": "N/A"},
                                         "mem_usage": {"used_vram": {"value": 26454, "unit": "MB"}}}]})
        r = T.parse_amd_smi(text)[0]
        self.assertEqual((r["util"], r["power"], r["clock_mhz"], r["temp_junction"], r["temp_mem"]),
                         (50.0, 60.0, 1976, 76.0, None))
        self.assertEqual((r["pcie_gen"], r["pcie_width"], r["pcie_rx_mb"], r["mem_used"]), (4, 4, None, 26454 << 20))
        self.assertEqual(T.parse_amd_smi_list('[{"gpu": 0, "bdf": "0000:C7:00.0"}]'), {"0000:c7:00.0": 0})

    def test_rocm_smi(self):
        text = ('{"card0": {"Temperature (Sensor edge) (C)": "71.0", "Temperature (Sensor junction) (C)": "83.0", '
                '"mclk clock speed:": "(875Mhz)", "sclk clock speed:": "(3191Mhz)", "pcie clock level": '
                '"20 (16.0GT/s x4)", "Average Graphics Package Power (W)": "274.0", "GPU use (%)": "100", '
                '"PCI Bus": "0000:C7:00.0", "VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": '
                '"27739648000"}, "system": {"Driver version": "6.19"}}')
        r = T.parse_rocm_smi(text)["0000:c7:00.0"]
        self.assertEqual((r["util"], r["temp_junction"], r["clock_mhz"], r["power"]), (100.0, 83.0, 3191, 274.0))
        self.assertEqual((r["pcie_gen"], r["pcie_width"], r["mem_total"], r["temp_mem"]), (4, 4, 34208743424, None))

    def test_command_runs_once_a_second_for_every_card(self):
        calls = []
        cmd = T._Command(["sh"], lambda text: {0: {"util": 1}, 1: {"util": 2}})
        with mock.patch("subprocess.run", side_effect=lambda *a, **k: calls.append(a) or
                        mock.Mock(returncode=0, stdout="")):
            a = T._FromCommand("nvidia-smi", cmd, lambda d: d.get(0))
            b = T._FromCommand("nvidia-smi", cmd, lambda d: d.get(1))
            self.assertEqual((a.read()["util"], b.read()["util"]), (1, 2))
        self.assertEqual(len(calls), 1)

    def test_windows_counters(self):
        engines = {"pid_100_luid_0x00000000_0x0000d1f5_phys_0_eng_0_engtype_3D": 30.0,
                   "pid_200_luid_0x00000000_0x0000d1f5_phys_0_eng_0_engtype_3D": 25.0,
                   "pid_200_luid_0x00000000_0x0000d1f5_phys_0_eng_3_engtype_Compute_0": 70.0,
                   "pid_300_luid_0x00000000_0x0000a000_phys_0_eng_0_engtype_3D": 2.0}
        out = T.pdh_adapters(engines, {"luid_0x00000000_0x0000d1f5_phys_0": 8 << 30})
        self.assertEqual(out["luid_0x00000000_0x0000d1f5"], {"util": 70.0, "mem_used": 8 << 30})
        self.assertEqual(out["luid_0x00000000_0x0000a000"]["mem_used"], None)


class Drives(unittest.TestCase):
    def test_partitions_and_lvm_resolve_to_their_disks(self):
        with tempfile.TemporaryDirectory() as t:
            blk = Path(t)
            (blk / "nvme1n1" / "nvme1n1p2").mkdir(parents=True)
            (blk / "nvme1n1" / "nvme1n1p2" / "partition").write_text("2")
            (blk / "dm-0" / "slaves" / "nvme1n1p2").mkdir(parents=True)
            cls = blk / "class"
            cls.mkdir()
            for n, target in (("nvme1n1", blk / "nvme1n1"), ("nvme1n1p2", blk / "nvme1n1" / "nvme1n1p2"),
                              ("dm-0", blk / "dm-0")):
                os.symlink(target, cls / n)
            st = os.stat(t)
            sysdev = Path(t, "dev")
            sysdev.mkdir()
            os.symlink(cls / "dm-0", sysdev / f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
            self.assertEqual(T.block_devices(t, sys_block=str(cls), dev_block=str(sysdev)), ["nvme1n1"])
        self.assertEqual(T.block_devices("/no/such/path"), [])

    def test_disk_counters_and_rates(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t, "diskstats")
            p.write_text(" 259 0 nvme1n1 100 0 2048 0 10 0 1024 0 0 0 0\n")
            self.assertEqual(T.disk_counters(str(p)), {"nvme1n1": (2048 * 512, 1024 * 512)})

    def test_two_paths_on_one_drive_are_one_entry(self):
        with mock.patch.object(T, "block_devices", side_effect=lambda p: ["sda"] if p == "/usb" else ["nvme0n1"]), \
                mock.patch.object(T.threading, "Thread"), \
                mock.patch.object(T, "nvidia_card", return_value=T._Card([FakeBackend("nvml", {"name": "x"})])):
            tel = T.Telemetry(drives=[("model", "/m"), ("mirror", "/m2"), ("mirror", "/usb")])
            self.assertEqual([(d["name"], d["roles"]) for d in tel.static["drives"]],
                             [("nvme0n1", ["model", "mirror"]), ("sda", ["mirror"])])
            stats = iter([{"nvme0n1": (0, 0), "sda": (0, 0)}, {"nvme0n1": (2 << 20, 0), "sda": (0, 0)}])
            with mock.patch.object(T, "disk_counters", side_effect=lambda: next(stats)):
                self.assertEqual([d["read_mb"] for d in tel._drive_rates()], [None, None])   # one sample: no rate yet
                r = tel._drive_rates()
            self.assertGreater(r[0]["read_mb"], 0)
            self.assertEqual(r[1]["read_mb"], 0.0)


class Memory(unittest.TestCase):
    def test_pressure(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t, "memory")
            p.write_text("some avg10=17.73 avg60=12.00 avg300=4.00 total=1\nfull avg10=17.63 avg60=1 avg300=1 total=1\n")
            self.assertEqual(T.memory_pressure(str(p)), (17.73, 17.63))
        self.assertEqual(T.memory_pressure("/no/such"), (None, None))


class Cpu(unittest.TestCase):
    def stat(self, root, second):
        """/proc/stat (user ... idle): then cpu0 runs flat out for 1000 ticks while cpu1 idles."""
        c0, c1 = ((1000, 1000), (0, 2000)) if second else ((0, 1000), (0, 1000))
        write(root, {"proc/stat": f"cpu  {c0[0] + c1[0]} 0 0 {c0[1] + c1[1]} 0 0 0 0 0 0\n"
                                  f"cpu0 {c0[0]} 0 0 {c0[1]} 0 0 0 0 0 0\ncpu1 {c1[0]} 0 0 {c1[1]} 0 0 0 0 0 0\n"})

    def test_amd(self):
        with tempfile.TemporaryDirectory() as t:
            write(t, {"proc/cpuinfo": "vendor_id\t: AuthenticAMD\nmodel name\t: AMD RYZEN AI MAX+ 395\n"
                                      "flags\t\t: fpu avx2 fma avx512f avx512bw avx512vbmi avx512_vnni avx_vnni\n\n",
                      "sys/class/hwmon/hwmon6/name": "k10temp", "sys/class/hwmon/hwmon6/temp1_input": "56000",
                      "sys/class/hwmon/hwmon6/temp1_label": "Tctl", "sys/class/hwmon/hwmon6/temp3_input": "51000",
                      "sys/class/hwmon/hwmon6/temp3_label": "Tccd1",
                      "sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "amd-pstate-epp",
                      "sys/devices/system/cpu/cpufreq/policy0/scaling_governor": "performance",
                      "sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference": "performance",
                      "sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq": "5187500",
                      "sys/devices/system/cpu/cpufreq/policy0/related_cpus": "0",
                      "sys/devices/system/cpu/cpufreq/policy1/cpuinfo_max_freq": "5000000",
                      "sys/devices/system/cpu/cpufreq/policy1/related_cpus": "1",
                      "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq": "5137733",
                      "sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq": "2000000",
                      "sys/class/powercap/intel-rapl:0/energy_uj": "1000"})
            rapl = Path(t, "sys/class/powercap/intel-rapl:0/energy_uj")
            rapl.chmod(0)                                         # root-only, as most distributions ship it
            self.stat(t, False)
            p = CpuProbe(sys_root=str(Path(t, "sys")), proc=str(Path(t, "proc")), apu_power=lambda: 17.6)
            self.assertEqual((p.static["cpu_vendor"], p.static["cpu_freq_driver"], p.static["cpu_epp"]),
                             ("AMD", "amd-pstate-epp", "performance"))
            self.assertEqual(p.static["cpu_isa"], ["avx2", "fma", "avx512f", "avx512bw", "avx512vbmi", "avx512_vnni",
                                                   "avx_vnni"])
            self.assertIsNone(p.read()["cpu"])                    # one /proc/stat sample: no load yet
            self.stat(t, True)
            r = p.read()
            self.assertEqual((r["cpu"], r["cpu_cores"]), (50.0, [100.0, 0.0]))
            self.assertEqual((r["cpu_mhz"], [c["max_mhz"] for c in r["cpu_clusters"]]), ([5138, 2000], [5188, 5000]))
            self.assertEqual((r["cpu_temp"], r["cpu_temps"]["Tccd1"]), (56.0, 51.0))
            if os.geteuid() != 0:                                 # root reads it anyway
                self.assertEqual((r["cpu_power"], r["cpu_sources"]["power"]), (17.6, "gpu_metrics"))
                self.assertEqual(p.power_blocked, "rapl")
            self.assertEqual(r["cpu_sources"]["temp"], "k10temp")
            self.assertIsNone(r["cpu_throttle_count"])

    def test_intel(self):
        with tempfile.TemporaryDirectory() as t:
            write(t, {"proc/cpuinfo": "vendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) Ultra 9 285K\n"
                                      "flags\t\t: avx2 fma avx_vnni\n\n",
                      "sys/class/hwmon/hwmon2/name": "coretemp", "sys/class/hwmon/hwmon2/temp1_input": "64000",
                      "sys/class/hwmon/hwmon2/temp1_label": "Package id 0", "sys/class/hwmon/hwmon2/temp2_input": "60000",
                      "sys/class/hwmon/hwmon2/temp2_label": "Core 0",
                      "sys/class/powercap/intel-rapl:0/energy_uj": "1000000",
                      "sys/class/powercap/intel-rapl:0/max_energy_range_uj": "262143328850",
                      "sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count": "3",
                      "sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count": "7"})
            p = CpuProbe(sys_root=str(Path(t, "sys")), proc=str(Path(t, "proc")))
            self.assertEqual((p.static["cpu_vendor"], p.static["cpu_isa"]), ("Intel", ["avx2", "fma", "avx_vnni"]))
            r = p.read()
            self.assertEqual((r["cpu_temp"], r["cpu_sources"]["temp"]), (64.0, "coretemp"))
            self.assertEqual((r["cpu_throttle_count"], r["cpu_throttling"]), (10, None))
            self.assertIsNone(r["cpu_power"])                     # one energy reading: no rate yet
            Path(t, "sys/class/powercap/intel-rapl:0/energy_uj").write_text("3000000\n")
            Path(t, "sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count").write_text("4\n")
            r = p.read()
            self.assertGreater(r["cpu_power"], 0)
            self.assertEqual((r["cpu_sources"]["power"], r["cpu_throttling"]), ("rapl", True))

    def test_arm_clusters(self):
        with tempfile.TemporaryDirectory() as t:
            files = {"proc/cpuinfo": "processor\t: 0\nFeatures\t: fp asimd asimddp i8mm sve sve2\n"
                                     "CPU implementer\t: 0x41\n\n",
                     "sys/firmware/devicetree/base/model": "Radxa Orion O6",
                     "sys/class/thermal/thermal_zone0/type": "cpu-thermal", "sys/class/thermal/thermal_zone0/temp": "48500",
                     "sys/class/hwmon/hwmon1/name": "scmi_sensors", "sys/class/hwmon/hwmon1/power1_label": "big cpu",
                     "sys/class/hwmon/hwmon1/power1_input": "4500000"}
            for pol, cpus, top in (("policy0", "0-3", "1800000"), ("policy4", "4-7", "2600000")):
                files[f"sys/devices/system/cpu/cpufreq/{pol}/related_cpus"] = cpus
                files[f"sys/devices/system/cpu/cpufreq/{pol}/cpuinfo_max_freq"] = top
            for c in range(8):
                files[f"sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"] = "1800000" if c < 4 else "2400000"
            write(t, files)
            p = CpuProbe(sys_root=str(Path(t, "sys")), proc=str(Path(t, "proc")))
            self.assertEqual((p.static["cpu_vendor"], p.static["cpu_model"]), ("Arm", "Radxa Orion O6"))
            self.assertEqual(p.static["cpu_isa"], ["neon", "dotprod", "i8mm", "sve", "sve2"])
            r = p.read()
            self.assertEqual([(c["cpus"], c["max_mhz"], c["mhz"]) for c in r["cpu_clusters"]],
                             [([0, 1, 2, 3], 1800, 1800), ([4, 5, 6, 7], 2600, 2400)])
            self.assertEqual((r["cpu_temp"], r["cpu_sources"]["temp"]), (48.5, "thermal_zone"))
            self.assertEqual((r["cpu_power"], r["cpu_sources"]["power"]), (4.5, "hwmon"))
            self.assertIsNone(r["cpu"])                           # no /proc/stat in this tree, no psutil given


if __name__ == "__main__":
    unittest.main()

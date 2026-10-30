"""Abyss Terminal's test fixtures: GET /health and GET /metrics as an Abyss server sends them (serve/server.py metrics,
with the phase-1.5 fields), engine logs, and a fake server that serves them - for the tests (test_abyss_terminal.py)
and the screenshots (abyss_terminal.py --demo).  Numbers from real runs: DeepSeek V4.1 Flash 1017 UD-IQ2_XXS on an
AI PRO R9700 with the Ryzen AI Max's iGPU lane and two SSDs (abyss-dsv41-iq2xxs-r9700l.log, 2026-10-29), a
16 GB + 64 GB PC, an RTX 5090 + RTX 3090 layer split, four mixed NVIDIA cards (5090 + 3090 + 2x V100) with two
SSDs, eight / ten RTX 3090s on risers (three drives with eight), a server that is loading, one that is down.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

GIB = 2**30
SCENARIOS = ("r9700", "lowspec", "split", "multi4", "multi8", "multi10", "loading", "starting", "down")
# the phase-2 engine's STAT keys (tools/terminal_model reads them): the token's breakdown live
STAT_KEYS = ("tok_s", "ms_tok", "vram_hit", "ram_fetch", "disk", "promo", "disk_ms", "disk_waits", "miss_ms", "lane",
             "lane_ms", "tail_skip", "mirror_mb0", "mirror_mb1", "w_ms_tok", "w_disk_ms", "w_lane_ms", "spec_acc",
             "spec_offered", "spec_rounds")


def _p2(disk_ms, waits, miss_ms, lane, lane_ms, skip=0.0, mirror=None, w=None, spec=None):
    """A STAT line's phase-2 keys: the SSD waits, the lane, the mirror's MB (shards' drive, mirror), the last ~64
    tokens' window (ms, disk ms, lane ms), the drafter (accepted, offered, rounds)."""
    out = {"disk_ms": disk_ms, "disk_waits": waits, "miss_ms": miss_ms, "lane": lane, "lane_ms": lane_ms,
           "tail_skip": skip}
    if mirror:
        out["mirror_mb0"], out["mirror_mb1"] = mirror
    if w:
        out["w_ms_tok"], out["w_disk_ms"], out["w_lane_ms"] = w
    if spec:
        out["spec_acc"], out["spec_offered"], out["spec_rounds"] = spec
    return out



def _wave(n, lo, hi, phase=0.0):
    import math
    return [round(lo + (hi - lo) * (0.5 + 0.5 * math.sin((i + phase) / 4.0)) * (0.85 + 0.15 * ((i * 7) % 5) / 4), 1)
            for i in range(n)]


def _cpu(n, base, temp, power, isa, name, cores, threads, mhz=4800, max_mhz=5187):
    per = [round(base + (i * 37 % 40) - 20, 1) for i in range(threads)]
    half = threads // 2
    hw = {"cpu": base, "cpu_cores": per, "cpu_mhz": [mhz - (i * 53 % 300) for i in range(threads)],
          "cpu_clusters": [{"cpus": list(range(0, half)), "max_mhz": max_mhz, "mhz": mhz},
                           {"cpus": list(range(half, threads)), "max_mhz": max_mhz - 400, "mhz": mhz - 250}],
          "cpu_temp": temp, "cpu_power": power, "cpu_throttling": False}
    st = {"cpu_name": name, "cpu_isa": isa, "cores": cores, "threads": threads, "cpu_governor": "performance",
          "cpu_epp": "performance", "psutil": True}
    return hw, st


def metrics(name: str, now: float | None = None) -> dict | None:
    """GET /metrics of a scenario (None: the server does not answer)."""
    now = time.time() if now is None else now
    if name in ("loading", "starting", "down"):
        return None
    if name == "r9700":
        engine = {"model": "deepseek-v4.1-flash-1017-iq2xxs", "max_context": 65536, "images": False,
                  "context": 65536, "kv": "fp16", "expert_slots": 2386, "engine_kind": "glm-fast", "experts": 11008,
                  "vram_slots": 2386, "vram_gb": 21.9, "ram_slots": 6301, "ram_gb": 66.1, "mtp": 1, "cpu_threads": 16,
                  "pcie_share": "0.00", "uncensor": 1, "version": "1.3.77", "kv_ctx": 65536, "kv_gb": 1.0,
                  "maya_version": "1.3.77", "trained_context": 1048576,
                  "display_name": "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "quant": "UD-IQ2_XXS",
                  "config": "abyss-dsv41-iq2xxs-r9700l.json", "backend": "hip",
                  "lane": "igpu", "drafter": "mtp", "expert_mib": 9.4, "tail_skip": 0, "defer": 0, "mirror": 1,
                  "helper": 0}
        live = {"state": "generating", "queued": 1, "phase": "answering", "prompt_tokens": 5210, "prompt_read": None,
                "prompt_total": None, "generated": 412, "max_tokens": 2048, "elapsed_s": 21.3, "tok_s": 24.6,
                "tok_s_mean": 23.9, "prefill_tok_s_mean": 781.0, "tok_s_window_s": 3}
        tn = {"tok_s": 24.1, "ms_tok": 41.5, "vram_hit": 0.6356, "ram_fetch": 0.0, "disk": 5.84, "promo": 6.02,
              "vram_used": 2271, "vram_slots": 2386, "ram_used": 6258, "ram_slots": 6301, "ram_gb": 66.1,
              "vram_gb": 21.9, "time": now - 1.2,
              **_p2(18.2, 4.1, 19.0, 3.0, 1.9, mirror=(1210.5, 952.0), w=(40.1, 17.5, 1.8), spec=(301, 410, 160))}
        main = {"index": 0, "role": "main", "name": "AMD Radeon AI PRO R9700", "shared_memory": False, "util": 93,
                "mem_used": int(30.4 * GIB), "mem_total": int(31.86 * GIB), "temp": 71, "temp_junction": 88,
                "temp_mem": 76, "power": 254.0, "power_limit": 300.0, "clock_mhz": 2475, "mem_clock_mhz": 1258,
                "throttle": [], "pcie_gen": 4, "pcie_gen_max": 4, "pcie_width": 4, "pcie_width_max": 4,
                "pcie_rx_mb": None, "pcie_tx_mb": None, "source": ["sysfs", "gpu_metrics"]}
        igpu = {"index": 1, "role": "igpu", "name": "AMD Radeon 8060S Graphics", "shared_memory": True, "util": 64,
                "mem_used": None, "mem_total": None, "temp": 58, "temp_junction": None, "temp_mem": None,
                "power": 82.0, "power_limit": None, "clock_mhz": 2900, "mem_clock_mhz": 1000, "throttle": [],
                "pcie_gen": None, "pcie_width": None, "dram_read_mb": 41200, "source": ["sysfs", "gpu_metrics"]}
        hw = {"gpu_util": 93, "gpu_mem_used": main["mem_used"], "gpu_mem_total": main["mem_total"], "gpu_temp": 71,
              "gpu_temp_junction": 88, "gpu_power": 254.0, "gpu_power_limit": 300.0, "gpu_pcie_gen": 4,
              "gpu_pcie_gen_max": 4, "gpu_pcie_width": 4, "gpu_pcie_rx_mb": None, "gpus": [main, igpu],
              "ram_used": int(88.2 * GIB), "ram_total": int(94.0 * GIB), "ram_cached": int(3.1 * GIB),
              "swap_used": int(0.4 * GIB), "swap_total": 8 * GIB, "psi_mem_some": 0.8, "psi_mem_full": 0.1,
              "disk_read_mb": 5920.0, "disk_write_mb": 1.2,
              "drives": [{"name": "nvme0n1", "roles": ["model"], "read_mb": 3260.0, "write_mb": 1.2},
                         {"name": "nvme1n1", "roles": ["mirror"], "read_mb": 2660.0, "write_mb": 0.0}],
              "tok_s": 24.6, "tok_s_mean": 23.9, "prefill_tok_s_mean": None}
        c_hw, c_st = _cpu(32, 38.0, 72.0, 45.0, ["avx2", "avx512f", "avx512vbmi", "avx_vnni"],
                          "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", 16, 32)
        hw.update(c_hw)
        static = {"gpu_name": "AMD Radeon AI PRO R9700", "gpu_count": 1,
                  "gpus": [{"index": 0, "role": "main", "name": main["name"], "pci": "0000:c7:00.0"},
                           {"index": 1, "role": "igpu", "name": igpu["name"], "pci": "0000:c8:00.0",
                            "shared_memory": True}],
                  "drives": [{"name": "nvme0n1", "roles": ["model"],
                              "paths": ["~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS"]},
                             {"name": "nvme1n1", "roles": ["mirror"], "paths": ["/mnt/ssd2/V4.1-UD-IQ2_XXS"]}], **c_st}
        reqs = [_req(now - 40, 18.2, "stop", "openai", 6020, 5120, 431, 22.7, 790.0, 0.64, 9, 16),
                _req(now - 95, 21.0, "stop", "anthropic", 5210, 0, 478, 24.3, 812.0, 0.62, 12, 18),
                _req(now - 160, 9.6, "length", "web", 1890, 1650, 256, 26.5, 640.0, 0.66, 4, 7),
                _req(now - 300, 33.4, "stop", "openai", 12044, 0, 512, 18.5, 781.0, 0.55, 15, 31),
                _req(now - 420, 2.4, "cancel", "web", 880, 880, 31, 25.1, None, 0.60, 0, 1),
                _req(now - 600, 14.1, "stop", "openai", 3300, 2900, 290, 25.4, 702.0, 0.63, 6, 9)]
        totals = {"since": now - 3600, "requests": 23, "prompt_tokens": 98200, "reused": 41200, "output_tokens": 8120,
                  "prompt_ms": 74000.0, "decode_ms": 351000.0, "drafts_accepted": 214, "drafts_offered": 402}
        hist = {"tok_s": [None] * 6 + _wave(54, 18, 27), "prefill_tok_s_mean": [None] * 40 + [781.0] * 8 + [None] * 12,
                "gpu_util": _wave(60, 60, 97, 1), "cpu": _wave(60, 20, 45, 2), "ram_used": [88 * GIB] * 60,
                "disk_read_mb": _wave(60, 2500, 6400, 3), "gpu_mem_used": [main["mem_used"]] * 60}
        storage = {"path": "~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS",
                   "free": int(1.2e12), "total": int(4.0e12)}
    elif name == "lowspec":
        engine = {"model": "deepseek-v4.1-flash-1017-iq1m", "max_context": 32768, "experts": 11008,
                  "vram_slots": 610, "vram_gb": 6.2, "ram_slots": 4310, "ram_gb": 41.9, "mtp": 0, "cpu_threads": 8,
                  "pcie_share": "0.25", "uncensor": 0, "version": "1.3.77", "kv_ctx": 32768, "kv_gb": 0.5,
                  "maya_version": "1.3.77", "display_name": "DeepSeek-V4.1-Flash-1017-UD-IQ1_M", "quant": "UD-IQ1_M",
                  "config": "abyss-dsv41-iq1m-16g.json", "backend": "cuda", "engine_kind": "glm-fast"}
        live = {"state": "idle", "queued": 0, "phase": None, "prompt_tokens": None, "prompt_read": None,
                "prompt_total": None, "generated": None, "max_tokens": None, "elapsed_s": None, "tok_s": None,
                "tok_s_mean": None, "prefill_tok_s_mean": None, "tok_s_window_s": None}
        tn = {"tok_s": 20.4, "ms_tok": 49.0, "vram_hit": 0.31, "ram_fetch": 6.1, "disk": 9.2, "promo": 3.3,
              "vram_used": 610, "vram_slots": 610, "ram_used": 4310, "ram_slots": 4310, "ram_gb": 41.9, "vram_gb": 6.2,
              "time": now - 95, **_p2(24.6, 6.8, 26.0, 4.0, 5.1, w=(49.5, 25.0, 5.0), spec=(0, 0, 0))}
        main = {"index": 0, "role": "main", "name": "NVIDIA GeForce RTX 4060 Ti", "util": 3, "mem_used": int(15.1 * GIB),
                "mem_total": 16 * GIB, "temp": 46, "temp_junction": None, "temp_mem": None, "power": 19.0,
                "power_limit": 165.0, "clock_mhz": 210, "mem_clock_mhz": 405, "throttle": ["idle"], "pcie_gen": 1,
                "pcie_gen_max": 4, "pcie_width": 8, "pcie_width_max": 8, "pcie_rx_mb": 0.0, "pcie_tx_mb": 0.0,
                "source": ["nvml"]}
        hw = {"gpu_util": 3, "gpu_mem_used": main["mem_used"], "gpu_mem_total": main["mem_total"], "gpu_temp": 46,
              "gpu_power": 19.0, "gpu_power_limit": 165.0, "gpu_pcie_gen": 1, "gpu_pcie_gen_max": 4,
              "gpu_pcie_width": 8, "gpu_pcie_rx_mb": 0.0, "gpus": [main],
              "ram_used": int(59.8 * GIB), "ram_total": 64 * GIB, "ram_cached": int(1.1 * GIB),
              "swap_used": int(3.4 * GIB), "swap_total": 8 * GIB, "psi_mem_some": 12.5, "psi_mem_full": 1.2,
              "disk_read_mb": 0.4, "disk_write_mb": 0.1,
              "drives": [{"name": "nvme0n1", "roles": ["model"], "read_mb": 0.4, "write_mb": 0.1}]}
        c_hw, c_st = _cpu(16, 4.0, 44.0, None, ["avx2"], "AMD Ryzen 7 5700X 8-Core Processor", 8, 16, 3400, 4650)
        hw.update(c_hw)
        hw["cpu_power_blocked"] = "RAPL is root-only"
        static = {"gpu_name": main["name"], "gpu_count": 1,
                  "gpus": [{"index": 0, "role": "main", "name": main["name"]}],
                  "drives": [{"name": "nvme0n1", "roles": ["model"], "paths": ["~/models/V4.1-UD-IQ1_M"]}], **c_st}
        reqs = [_req(now - 95, 31.0, "stop", "openai", 2410, 0, 512, 20.4, 233.0, 0.31, None, None),
                _req(now - 400, 25.7, "stop", "web", 1200, 1100, 488, 21.0, 180.0, 0.33, None, None)]
        totals = {"since": now - 1800, "requests": 2, "prompt_tokens": 3610, "reused": 1100, "output_tokens": 1000,
                  "prompt_ms": 10900.0, "decode_ms": 48300.0, "drafts_accepted": 0, "drafts_offered": 0}
        hist = {"tok_s": [None] * 60, "gpu_util": _wave(60, 1, 6), "cpu": _wave(60, 2, 8), "ram_used": [60 * GIB] * 60,
                "disk_read_mb": _wave(60, 0, 2)}
        storage = {"path": "~/models/V4.1-UD-IQ1_M", "free": int(0.3e12), "total": int(1.0e12)}
    elif name == "split":
        engine = {"model": "glm-5.3-flash", "max_context": 131072, "experts": 19456, "vram_slots": 7210,
                  "vram_gb": 52.0, "ram_slots": 9986, "ram_gb": 66.2, "mtp": 1, "cpu_threads": 16, "pcie_share": "1.00",
                  "uncensor": 0, "version": "1.3.77", "kv_ctx": 131072, "kv_gb": 4.0, "maya_version": "1.3.77",
                  "display_name": "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS", "quant": "IQ2_XXS",
                  "config": "abyss-glm-s-v2-split.json", "backend": "cuda", "engine_kind": "glm-fast"}
        live = {"state": "reading", "queued": 0, "phase": "prompt", "prompt_tokens": 16384, "prompt_read": 6144,
                "prompt_reused": 2048, "prompt_disk_ms": 1840.0,
                "prompt_total": 16384, "generated": None, "max_tokens": 4096, "elapsed_s": 10.1, "tok_s": None,
                "tok_s_mean": None, "prefill_tok_s_mean": 612.0, "tok_s_window_s": None}
        tn = {"tok_s": 61.0, "ms_tok": 16.4, "vram_hit": 0.88, "ram_fetch": 3.9, "disk": 0.0, "promo": 1.2,
              "vram_used": 7210, "vram_slots": 7210, "ram_used": 9986, "ram_slots": 9986, "ram_gb": 66.2,
              "vram_gb": 52.0, "time": now - 40}
        a = {"index": 0, "role": "main", "name": "NVIDIA GeForce RTX 5090", "util": 97, "mem_used": int(30.9 * GIB),
             "mem_total": int(31.8 * GIB), "temp": 68, "temp_junction": None, "temp_mem": 80, "power": 452.0,
             "power_limit": 575.0, "clock_mhz": 2730, "mem_clock_mhz": 14001, "throttle": [], "pcie_gen": 4,
             "pcie_gen_max": 5, "pcie_width": 4, "pcie_width_max": 16, "pcie_rx_mb": 5900.0, "pcie_tx_mb": 120.0,
             "source": ["nvml"]}
        b = {"index": 1, "role": "split", "name": "NVIDIA GeForce RTX 3090", "util": 81, "mem_used": int(23.1 * GIB),
             "mem_total": 24 * GIB, "temp": 83, "temp_junction": None, "temp_mem": 96, "power": 341.0,
             "power_limit": 350.0, "clock_mhz": 1695, "mem_clock_mhz": 9751, "throttle": ["sw_power_cap"],
             "pcie_gen": 3, "pcie_gen_max": 4, "pcie_width": 4, "pcie_width_max": 16, "pcie_rx_mb": 3100.0,
             "pcie_tx_mb": 80.0, "source": ["nvml"]}
        hw = {"gpu_util": 89, "gpu_mem_used": a["mem_used"] + b["mem_used"], "gpu_mem_total": a["mem_total"] +
              b["mem_total"], "gpu_temp": 83, "gpu_power": 793.0, "gpu_power_limit": 925.0, "gpu_pcie_gen": 3,
              "gpu_pcie_gen_max": 5, "gpu_pcie_width": 4, "gpu_pcie_rx_mb": 9000.0, "gpus": [a, b],
              "ram_used": int(101 * GIB), "ram_total": 128 * GIB, "ram_cached": int(9.5 * GIB), "swap_used": 0,
              "swap_total": 4 * GIB, "psi_mem_some": 0.0, "psi_mem_full": 0.0, "disk_read_mb": 12.0,
              "disk_write_mb": 0.4, "drives": [{"name": "nvme0n1", "roles": ["model"], "read_mb": 12.0,
                                                 "write_mb": 0.4}]}
        c_hw, c_st = _cpu(32, 61.0, 78.0, 142.0, ["avx2", "avx512f", "avx512vbmi"], "AMD Ryzen 9 9950X 16-Core Processor",
                          16, 32, 5300, 5752)
        hw.update(c_hw)
        static = {"gpu_name": "NVIDIA GeForce RTX 5090 + NVIDIA GeForce RTX 3090", "gpu_count": 2,
                  "gpus": [{"index": 0, "role": "main", "name": a["name"]}, {"index": 1, "role": "split",
                                                                           "name": b["name"]}],
                  "drives": [{"name": "nvme0n1", "roles": ["model"], "paths": ["/models/Maya-S-v2-IQ2_XXS"]}], **c_st}
        reqs = [_req(now - 70, 12.0, "stop", "anthropic", 30120, 29800, 612, 61.2, 1190.0, 0.89, 301, 410),
                _req(now - 200, 40.3, "stop", "openai", 22014, 0, 1024, 58.0, 1011.0, 0.88, 520, 702)]
        totals = {"since": now - 7200, "requests": 41, "prompt_tokens": 512000, "reused": 400100, "output_tokens": 30100,
                  "prompt_ms": 101000.0, "decode_ms": 520000.0, "drafts_accepted": 15100, "drafts_offered": 20400}
        hist = {"tok_s": _wave(40, 52, 64) + [None] * 20, "prefill_tok_s_mean": [None] * 50 + _wave(10, 580, 640),
                "gpu_util": _wave(60, 70, 99), "cpu": _wave(60, 40, 70), "ram_used": [101 * GIB] * 60,
                "disk_read_mb": _wave(60, 0, 20)}
        storage = {"path": "/models/Maya-S-v2-IQ2_XXS", "free": int(2.0e12), "total": int(4.0e12)}
    elif name in ("multi4", "multi8", "multi10"):
        return _multi(name, now)
    else:
        raise KeyError(name)
    return {"engine": engine, "live": live, "tiers": {"now": tn, "history": {k: [tn.get(k)] * 20 for k in STAT_KEYS
                                                                            if k in tn}},
            "requests": reqs, "requests_kept": len(reqs), "totals": totals, "hardware": hw, "hardware_static": static,
            "history": hist, "time": now, "reload": None, "storage": storage}


def _gpu(i, role, name, util, used, total, temp, power, plim, clk, mclk, link, link_max, rx, tmem=None, throttle=()):
    return {"index": i, "role": role, "name": name, "util": util, "mem_used": int(used * GIB),
            "mem_total": int(total * GIB), "temp": temp, "temp_junction": None, "temp_mem": tmem, "power": power,
            "power_limit": plim, "clock_mhz": clk, "mem_clock_mhz": mclk, "throttle": list(throttle),
            "pcie_gen": link[0], "pcie_width": link[1], "pcie_gen_max": link_max[0], "pcie_width_max": link_max[1],
            "pcie_rx_mb": rx, "pcie_tx_mb": round(rx * 0.03, 1) if rx is not None else None, "source": ["nvml"]}


def _multi(name: str, now: float) -> dict:
    """Many cards: four mixed NVIDIA (a 5090 on OCuLink, a 3090, two V100s - one on an x8 slot) with the shards on
    two SSDs, or eight RTX 3090s on risers (one fell to Gen1 x1) with a third, unrelated drive."""
    if name == "multi4":
        engine = {"model": "deepseek-v4-flash-0731-q2kxl", "max_context": 65536, "experts": 11008, "vram_slots": 8650,
                  "vram_gb": 98.4, "ram_slots": 2358, "ram_gb": 24.7, "mtp": 1, "cpu_threads": 16,
                  "pcie_share": "1.00", "uncensor": 0, "version": "1.3.77", "kv_ctx": 65536, "kv_gb": 1.2,
                  "maya_version": "1.3.77", "display_name": "DeepSeek-V4-Flash-0731-UD-Q2_K_XL", "quant": "UD-Q2_K_XL",
                  "config": "abyss-dsv4-q2kxl-4gpu.json", "backend": "cuda", "engine_kind": "glm-fast"}
        gpus = [_gpu(0, "main", "NVIDIA GeForce RTX 5090", 91, 30.8, 31.8, 66, 430.0, 575.0, 2700, 14001, (4, 4),
                     (5, 16), 4100.0, 78),
                _gpu(1, "split", "NVIDIA GeForce RTX 3090", 78, 23.0, 24.0, 79, 330.0, 350.0, 1740, 9751, (4, 16),
                     (4, 16), 2100.0, 94),
                _gpu(2, "split", "Tesla V100-PCIE-32GB", 97, 31.2, 32.0, 74, 240.0, 250.0, 1380, 877, (3, 16),
                     (3, 16), 1500.0, 70, ["sw_power_cap"]),
                _gpu(3, "split", "Tesla V100-PCIE-32GB", 88, 30.1, 32.0, 71, 221.0, 250.0, 1380, 877, (3, 8),
                     (3, 16), 900.0, 68)]
        drives = [{"name": "nvme0n1", "roles": ["model"], "read_mb": 3120.0, "write_mb": 0.8},
                  {"name": "nvme1n1", "roles": ["mirror"], "read_mb": 2710.0, "write_mb": 0.0}]
        paths = {"nvme0n1": ["/models/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL"], "nvme1n1": ["/mnt/ssd2/V4-Q2_K_XL"]}
        live = {"state": "generating", "queued": 0, "phase": "answering", "prompt_tokens": 8120, "generated": 913,
                "max_tokens": 4096, "elapsed_s": 41.0, "tok_s": 31.4, "tok_s_mean": 30.2, "prefill_tok_s_mean": 655.0}
        tn = {"tok_s": 31.0, "ms_tok": 32.3, "vram_hit": 0.86, "ram_fetch": 1.9, "disk": 2.1, "promo": 1.4,
              "vram_used": 8650, "vram_slots": 8650, "ram_used": 2358, "ram_slots": 2358, "ram_gb": 24.7,
              "vram_gb": 98.4, "time": now - 1.0,
              **_p2(5.1, 1.2, 6.0, 2.2, 7.9, mirror=(640.0, 590.0), w=(31.0, 4.4, 7.5), spec=(520, 700, 290))}
        c_hw, c_st = _cpu(32, 34.0, 69.0, 121.0, ["avx2", "avx512f", "avx512vbmi"], "AMD Ryzen 9 7950X 16-Core Processor",
                          16, 32, 5100, 5881)
        ram = (61.0, 128.0, 22.1, 0.0, 0.4, 0.0)
        reqs = [_req(now - 60, 28.0, "stop", "openai", 8120, 7900, 840, 30.8, 655.0, 0.86, 410, 590),
                _req(now - 180, 44.1, "stop", "anthropic", 21000, 0, 1210, 29.9, 702.0, 0.85, 600, 880),
                _req(now - 400, 12.3, "length", "web", 900, 0, 256, 32.0, 512.0, 0.87, 130, 180)]
        totals = {"since": now - 5400, "requests": 17, "prompt_tokens": 140200, "reused": 81000, "output_tokens": 14100,
                  "prompt_ms": 92000.0, "decode_ms": 470000.0, "drafts_accepted": 7000, "drafts_offered": 10100}
        hist = {"tok_s": [None] * 4 + _wave(56, 26, 33), "gpu_util": _wave(60, 70, 98, 2), "cpu": _wave(60, 25, 45),
                "ram_used": [61 * GIB] * 60, "disk_read_mb": _wave(60, 3000, 6200, 1)}
        storage = {"path": paths["nvme0n1"][0], "free": int(0.9e12), "total": int(2.0e12)}
    else:
        engine = {"model": "glm-5.3-flash-q4kxl", "max_context": 131072, "experts": 19456, "vram_slots": 17800,
                  "vram_gb": 168.0, "ram_slots": 1656, "ram_gb": 15.6, "mtp": 1, "cpu_threads": 24,
                  "pcie_share": "1.00", "uncensor": 0, "version": "1.3.77", "kv_ctx": 131072, "kv_gb": 6.0,
                  "maya_version": "1.3.77", "display_name": "GLM-5.3-Flash-UD-Q4_K_XL", "quant": "UD-Q4_K_XL",
                  "config": "abyss-glm-q4kxl-8gpu.json", "backend": "cuda", "engine_kind": "glm-fast"}
        n = 8 if name == "multi8" else 10
        utils = (84, 77, 81, 95, 72, 79, 83, 41, 80, 76)[:n]
        temps = (71, 74, 76, 84, 69, 73, 75, 58, 72, 70)[:n]
        links = [(4, 16)] + [(3, 4)] * (n - 2) + [(1, 1) if n == 8 else (2, 1)]   # (risers; the last one x1)
        gpus = [_gpu(i, "main" if i == 0 else "split", "NVIDIA GeForce RTX 3090", utils[i], 22.6 + (i % 3) * 0.3, 24.0,
                     temps[i], 240.0 + utils[i], 350.0, 1695 + (i * 15) % 90, 9751, links[i], (4, 16),
                     None if i == n - 1 else 1200.0 - i * 40, 88 + i % 4) for i in range(n)]
        if n == 10:
            engine.update(config="abyss-glm-q4kxl-10gpu.json", vram_gb=212.0, vram_slots=19456, ram_slots=0, ram_gb=0.0)
        drives = [{"name": "nvme0n1", "roles": ["model"], "read_mb": 160.0, "write_mb": 0.4},
                  {"name": "nvme1n1", "roles": ["mirror"], "read_mb": 120.0, "write_mb": 0.0},
                  {"name": "sda", "roles": [], "read_mb": 0.6, "write_mb": 2.0}]
        paths = {"nvme0n1": ["/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL"], "nvme1n1": ["/mnt/ssd2/GLM-Q4_K_XL"],
                 "sda": ["/"]}
        live = {"state": "generating", "queued": 2, "phase": "answering", "prompt_tokens": 3100, "generated": 220,
                "max_tokens": 2048, "elapsed_s": 6.0, "tok_s": 38.5, "tok_s_mean": 37.9, "prefill_tok_s_mean": 910.0}
        tn = {"tok_s": 38.2, "ms_tok": 26.2, "vram_hit": 0.97, "ram_fetch": 0.4, "disk": 0.05, "promo": 0.3,
              "vram_used": 17800, "vram_slots": 17800, "ram_used": 1656, "ram_slots": 1656, "ram_gb": 15.6,
              "vram_gb": 168.0, "time": now - 0.8,
              **_p2(0.1, 0.02, 0.3, 0.6, 0.4, w=(26.0, 0.1, 0.4), spec=(90, 110, 60))}
        c_hw, c_st = _cpu(48, 22.0, 58.0, 160.0, ["avx2", "avx512f"], "AMD EPYC 7413 24-Core Processor", 24, 48,
                          3200, 3630)
        ram = (180.0, 256.0, 40.0, 0.0, 0.0, 0.0)
        reqs = [_req(now - 30, 18.0, "stop", "openai", 3100, 2800, 690, 38.0, 910.0, 0.97, 330, 420),
                _req(now - 90, 25.0, "stop", "openai", 12000, 0, 900, 36.5, 1020.0, 0.96, 420, 560)]
        totals = {"since": now - 86400, "requests": 1204, "prompt_tokens": 9.1e6, "reused": 6.0e6,
                  "output_tokens": 8.0e5, "prompt_ms": 3.0e6, "decode_ms": 2.1e7, "drafts_accepted": 3.0e5,
                  "drafts_offered": 4.0e5}
        hist = {"tok_s": _wave(60, 33, 40), "gpu_util": _wave(60, 60, 96, 1), "cpu": _wave(60, 15, 30),
                "ram_used": [180 * GIB] * 60, "disk_read_mb": _wave(60, 80, 320, 2)}
        storage = {"path": paths["nvme0n1"][0], "free": int(3.0e12), "total": int(8.0e12)}
    base = {"prompt_read": None, "prompt_total": None, "tok_s_window_s": 3}
    live = {**base, **live}
    used = sum(g["mem_used"] for g in gpus)
    total = sum(g["mem_total"] for g in gpus)
    hw = {"gpu_util": max(g["util"] for g in gpus), "gpu_mem_used": used, "gpu_mem_total": total,
          "gpu_temp": max(g["temp"] for g in gpus), "gpu_power": sum(g["power"] for g in gpus),
          "gpu_power_limit": sum(g["power_limit"] for g in gpus), "gpus": gpus,
          "ram_used": int(ram[0] * GIB), "ram_total": int(ram[1] * GIB), "ram_cached": int(ram[2] * GIB),
          "swap_used": int(ram[3] * GIB), "swap_total": 8 * GIB, "psi_mem_some": ram[4], "psi_mem_full": ram[5],
          "disk_read_mb": sum(d["read_mb"] for d in drives), "disk_write_mb": sum(d["write_mb"] for d in drives),
          "drives": drives}
    hw.update(c_hw)
    static = {"gpu_name": " + ".join(g["name"] for g in gpus), "gpu_count": len(gpus),
              "gpus": [{"index": g["index"], "role": g["role"], "name": g["name"]} for g in gpus],
              "drives": [{"name": d["name"], "roles": d["roles"], "paths": paths[d["name"]]} for d in drives], **c_st}
    return {"engine": engine, "live": live, "tiers": {"now": tn, "history": {k: [tn.get(k)] * 20 for k in STAT_KEYS
                                                                            if k in tn}},
            "requests": reqs, "requests_kept": len(reqs), "totals": totals, "hardware": hw, "hardware_static": static,
            "history": hist, "time": now, "reload": None, "storage": storage}


def _req(t, dur, finish, api, prompt, reused, out, tg, pp, hit, da, do, rounds=None):
    return {"projection": None, "time": t, "duration_s": dur, "finish": finish, "api": api, "prefill_tok_s": pp,
            "prompt_tokens": prompt, "reused": reused, "output_tokens": out, "engine_generated": out,
            "prompt_ms": (prompt - reused) / pp * 1000 if pp else 120.0,
            "decode_ms": out / tg * 1000 if tg else None, "decode_tok_s": tg, "hit_rate": hit,
            "drafts_accepted": da, "drafts_offered": do,
            "spec_rounds": rounds if rounds is not None else (max(1, out - da) if do else None)}


def health(name: str) -> dict | None:
    if name == "starting":
        return {"status": "loading", "max_context": None, "model": "deepseek-v4.1-flash-1017-iq2xxs", "images": False,
                "api_key": True, "defaults": {}, "version": "1.3.80", "mcp_tools": False, "service": "abyss",
                "config": "abyss-dsv41-iq2xxs-r9700l.json",
                "load": {"stage": "warming", "text": "CUDA0 warming the expert tiers 42% (37.0 of 88.0 GB, 7 s)",
                         "pct": 42.0, "done_gb": 37.0, "total_gb": 88.0, "elapsed_s": 61.0}}
    m = metrics(name)
    if m is None:
        return None
    return {"status": "ok", "max_context": m["engine"]["max_context"], "model": m["engine"]["model"], "images": False,
            "api_key": name in ("split", "multi8"), "defaults": {}, "version": "1.3.77", "mcp_tools": False, "service": "abyss"}


def _stat(ms, hit, fetch, reads, waits, disk_ms, lane, lane_ms, skip=0.0, prompt_ms=45.0):
    return (f"glm stat: decode {ms:.2f} ms/tok ({1000 / ms:.2f} tok/s) | vram hit {hit:.2f}% | ram fetches "
            f"{fetch:.2f}/tok | disk {reads:.2f} reads/tok in {waits:.2f} waits/tok, {disk_ms:.2f} ms/tok | promotions "
            f"{reads:.2f}/tok | cpu lane {lane:.2f}/tok in {lane_ms:.2f} ms/tok | tail skip {skip:.2f}/tok | prompt "
            f"{prompt_ms:.1f} ms/tok | vram 2270/2386 ram 6265/6301")


LOG_R9700 = [
    "glm fast: CUDA0 VRAM before the expert pool: 9.71 of 31.86 GB in use - dense weights 7.98, state/KV (65536 ctx) "
    "1.00, activations 0.06, other 0.67",
    "glm fast: CUDA0 layers [0,43) expert pool 21.90 GB, 55-56 slots/layer (2386 total)",
    "glm fast: CUDA0 RAM tier 66.10 GB pinned, 6301 slots",
    "glm fast: CUDA0 warming the expert tiers 50% (44.0 of 88.0 GB, 9 s)",
    "glm fast: CUDA0 tiers warm: 8687 experts (88.0 GB) in 18 s (4.89 GB/s)",
    "glm fast: CUDA0 iGPU lane on (~/llm/apps/Abyss/build-hip/libabyss_igpu.so): the RAM tier's 66.10 GB registered",
    "glm fast: CUDA0 iGPU lane: an expert 0.042 ms vs 0.085 on the CPU's 16 threads -> the iGPU computes the lane",
    "glm fast: CUDA0 CPU lane: 16 threads, an expert 0.042 ms on the CPU vs 1.053 ms over PCIe -> of 0..8 RAM-tier "
    "experts the CPU takes 0 1 2 3 4 5 6 7 8",
    "glm fast: mirror on: /mnt/ssd2/V4.1-UD-IQ2_XXS (reads split between the two drives)",
    _stat(44.11, 61.70, 0.0, 7.56, 6.95, 14.23, 84.02, 5.59), "glm spec: 2 of 3 drafts accepted (66.7%)",
    _stat(42.05, 64.43, 0.0, 6.88, 6.17, 13.10, 76.83, 5.22),
    _stat(39.37, 63.70, 0.0, 5.75, 5.45, 10.66, 79.77, 5.33), "glm spec: 1 of 2 drafts accepted (50.0%)",
    _stat(41.24, 63.56, 0.0, 5.84, 5.42, 13.13, 79.95, 5.54),
    "WARNING: STRATA_GLM_RAM_GB above 73% of RAM: the page cache has 3 GB left",
]
LOG_LOWSPEC = [
    "glm fast: CUDA0 VRAM before the expert pool: 9.20 of 16.0 GB in use",
    "glm fast: CUDA0 RAM tier 41.90 GB pinned, 4310 slots",
    "glm fast: CUDA0 CPU lane: 8 threads, an expert 0.31 ms on the CPU vs 0.9 ms over PCIe -> of 0..8 RAM-tier experts "
    "the CPU takes 0 1 1 2 3 4 4 5 6",
    _stat(48.8, 31.2, 6.1, 9.3, 8.7, 21.6, 210.0, 11.8, prompt_ms=4.3),
    _stat(49.2, 30.8, 6.0, 9.1, 8.5, 21.0, 205.0, 11.5, prompt_ms=5.5),
]
LOG_LOADING = [
    "glm fast: CUDA0 VRAM before the expert pool: 9.71 of 31.86 GB in use - dense weights 7.98",
    "glm fast: CUDA0 RAM tier 66.10 GB pinned, 6301 slots",
    "glm fast: CUDA0 warming the expert tiers 42% (37.0 of 88.0 GB, 7 s)",
]
LOG_MULTI4 = [
    "glm fast: CUDA0 VRAM before the expert pool: 6.10 of 31.8 GB in use - dense weights 4.2",
    "glm fast: CUDA1 VRAM before the expert pool: 3.90 of 24.0 GB in use - dense weights 2.9",
    "glm fast: CUDA0 RAM tier 24.70 GB pinned, 2358 slots",
    "glm fast: CUDA0 CPU lane: 16 threads, an expert 0.09 ms on the CPU vs 0.5 ms over PCIe -> of 0..8 RAM-tier "
    "experts the CPU takes 0 1 2 2 3 4 5 5 6",
    "glm fast: mirror on: /mnt/ssd2/V4-Q2_K_XL (reads split between the two drives)",
    "glm fast: CUDA0 layers [0,14) expert pool 24.60 GB, 190-201 slots/layer (2740 total)",
    "glm fast: CUDA1 layers [14,25) expert pool 18.90 GB, 191-196 slots/layer (2110 total)",
    "glm fast: CUDA2 layers [25,36) expert pool 27.40 GB, 270-281 slots/layer (3050 total)",
    "glm fast: CUDA3 layers [36,43) expert pool 27.50 GB, 390-400 slots/layer (2750 total)",
    _stat(32.6, 86.1, 1.9, 2.2, 2.0, 4.9, 22.0, 3.1), "glm spec: 3 of 4 drafts accepted (75.0%)",
    _stat(32.0, 85.8, 1.8, 2.0, 1.9, 4.6, 21.0, 3.0),
]
LOG_MULTI10 = ["glm fast: CUDA0 VRAM before the expert pool: 4.2 of 24.0 GB in use - dense weights 3.1"] + [
    f"glm fast: CUDA{i} layers [{i * 9 // 2},{(i + 1) * 9 // 2}) expert pool 19.{i} GB, 420-460 slots/layer "
    f"(1946 total)" for i in range(10)] + [
    "glm fast: CUDA0 CPU lane off: every expert fits in VRAM",
    _stat(26.4, 99.6, 0.1, 0.02, 0.02, 0.05, 0.0, 0.0), "glm spec: 3 of 4 drafts accepted (75.0%)",
    _stat(26.1, 99.7, 0.1, 0.01, 0.01, 0.03, 0.0, 0.0)]
LOGS = {"r9700": LOG_R9700, "lowspec": LOG_LOWSPEC, "loading": LOG_LOADING, "multi4": LOG_MULTI4,
        "multi10": LOG_MULTI10}


def local(name: str, log_path=None, now: float | None = None) -> dict:
    """terminal_local's facts of a scenario: the config (its display name, quant, env), the PID file, the bench line."""
    now = time.time() if now is None else now
    if name in ("split", "multi8", "starting"):
        return {"remote": True, "is_local": False, "pid_alive": None}
    cfg = {"r9700": ("abyss-dsv41-iq2xxs-r9700l", "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS", 65536, "hip",
                     {"STRATA_GLM_MIRROR": "/mnt/ssd2/V4.1-UD-IQ2_XXS", "GPU_MAX_HW_QUEUES": "2"}, 8082),
           "lowspec": ("abyss-dsv41-iq1m-16g", "DeepSeek-V4.1-Flash-1017-UD-IQ1_M", "UD-IQ1_M", 32768, "cuda", {}, 8080),
           "loading": ("abyss-dsv41-iq2xxs-r9700l", "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS", 65536, "hip",
                       {"STRATA_GLM_MIRROR": "/mnt/ssd2/V4.1-UD-IQ2_XXS"}, 8082),
           "multi4": ("abyss-dsv4-q2kxl-4gpu", "DeepSeek-V4-Flash-0731-UD-Q2_K_XL", "UD-Q2_K_XL", 65536, "cuda",
                      {"STRATA_GLM_MIRROR": "/mnt/ssd2/V4-Q2_K_XL"}, 8082),
           "multi10": ("abyss-glm-q4kxl-10gpu", "GLM-5.3-Flash-UD-Q4_K_XL", "UD-Q4_K_XL", 131072, "cuda", {},
                       8082),
           "down": ("abyss-ornith-r9700l", "Ornith-1.5-122B-IQ3_XXS-B", "IQ3_XXS", 131072, "hip", {}, 8082)}[name]
    stem, disp, quant, ctx, back, env, port = cfg
    bench = {"line": "Abyss v1.3.77 | DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS | AMD Radeon AI PRO R9700 (gfx1201) 32 GB | "
                     "PCIe Gen4 x4 (~7.9 GB/s) | RAM 94 GB | SSD 6.2 GB/s | decode 25.1 tok/s | prompt 8k 781 tok/s | "
                     "quality ok | bottleneck: ssd", "decode": 25.1, "prompt8k": 781.0, "ssd": 6.2,
             "bottleneck": "ssd", "model": disp, "version": "v1.3.77", "config": stem + ".json"} \
        if name in ("r9700", "loading") else None
    return {"remote": False, "is_local": True, "pid_alive": name == "loading" or (None if name != "down" else False),
            "started": now - 23 if name == "loading" else None, "bench": bench, "version": "1.3.77",
            "config": {"file": stem + ".json", "stem": stem, "display": disp, "quant": quant, "ctx": ctx,
                       "backend": back, "env": env, "model_name": stem, "port": port,
                       "gpus": list(range({"multi4": 4, "multi10": 10}.get(name, 1)))}}


CONFIGS = [  # the Models view's list (terminal_local.config_info's fields)
    {"file": "abyss-dsv41-iq2xxs-r9700l.json", "stem": "abyss-dsv41-iq2xxs-r9700l", "display":
     "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "quant": "UD-IQ2_XXS", "ctx": 65536, "gpus": [0], "port": 8082,
     "backend": "hip", "missing": [], "serving": True},
    {"file": "abyss-dsv41-iq1m-r9700l.json", "stem": "abyss-dsv41-iq1m-r9700l", "display":
     "DeepSeek-V4.1-Flash-1017-UD-IQ1_M", "quant": "UD-IQ1_M", "ctx": 65536, "gpus": [0], "port": 8082,
     "backend": "hip", "missing": []},
    {"file": "abyss-ornith-r9700l.json", "stem": "abyss-ornith-r9700l", "display": "Ornith-1.5-122B-IQ3_XXS-B",
     "quant": "IQ3_XXS", "ctx": 131072, "gpus": [0], "port": 8082, "backend": "hip", "missing": []},
    {"file": "abyss-q38ista-r9700l.json", "stem": "abyss-q38ista-r9700l", "display": "Qwen3.8-Flash-Next-IQ3_S",
     "quant": "IQ3_S", "ctx": 65536, "gpus": [0], "port": 8082, "backend": "hip", "missing": []},
    {"file": "abyss-glm-s-v2-r9700l.json", "stem": "abyss-glm-s-v2-r9700l", "display":
     "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS", "quant": "IQ2_XXS", "ctx": 65536, "gpus": [0], "port": 8080,
     "backend": "hip", "missing": ["pack"]},
]


class FakeServer:
    """An Abyss server's monitoring side on 127.0.0.1 (a free port): /health, /metrics (the Bearer `key` when set),
    /v1/models and POST /api/shutdown (recorded, 202).  `scenario` can change between polls; down() closes it."""

    def __init__(self, scenario: str = "r9700", key: str | None = None, service: str = "abyss", log_text=None):
        self.scenario, self.key, self.service = scenario, key, service
        self.log_text = log_text                   # GET /api/log's file (None: an older server, 404)
        self.shutdowns: list = []
        self.hits: list = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                if not outer.key:
                    return True
                if self.headers.get("Authorization", "") == f"Bearer {outer.key}":
                    return True
                self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
                return False

            def do_GET(self):
                path = self.path.split("?")[0]
                outer.hits.append(self.path)
                h = health(outer.scenario) or {}
                if path == "/health":
                    self._json(200, dict(h, service=outer.service, api_key=bool(outer.key)))
                elif outer.scenario == "starting":     # (the phase-2 server before its engine is ready)
                    self.send_response(503)
                    self.send_header("Retry-After", "15")
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"{}")
                elif path == "/metrics":
                    if self._auth():
                        self._json(200, metrics(outer.scenario))
                elif path == "/api/log":
                    if self._auth():
                        if outer.log_text is None:
                            self._json(404, {"error": {"message": "not found"}})
                        else:
                            q = {k: v[-1] for k, v in parse_qs(self.path.partition("?")[2]).items()}
                            since = int(q["since"]) if q.get("since", "").lstrip("-").isdigit() else None
                            mx = int(q.get("max") or 65536)
                            data = outer.log_text.encode()
                            start = max(0, len(data) - mx) if since is None or since < 0 else min(since, len(data))
                            piece = data[start:start + mx]
                            cut = piece.rfind(b"\n") + 1 or len(piece)
                            self._json(200, {"path": "engine.log", "size": len(data), "since": start,
                                             "next": start + cut, "text": piece[:cut].decode(),
                                             "more": start + cut < len(data)})
                elif path == "/v1/models":
                    if self._auth():
                        self._json(200, {"object": "list", "data": [{"id": h.get("model"), "object": "model"}]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/shutdown" and self._auth():
                    outer.shutdowns.append(body)
                    self._json(202, {"stopping": True, "force": bool(body.get("force"))})
                elif self.path != "/api/shutdown":
                    self._json(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def down(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    __enter__ = lambda self: self  # noqa: E731

    def __exit__(self, *a):
        try:
            self.down()
        except OSError:
            pass

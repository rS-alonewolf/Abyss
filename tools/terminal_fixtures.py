"""Abyss Terminal's test fixtures: GET /health and GET /metrics as an Abyss server sends them (serve/server.py metrics,
with the phase-1.5 fields), engine logs, and a fake server that serves them - for the tests (test_abyss_terminal.py)
and the screenshots (abyss_terminal.py --demo).  Numbers from real runs: DeepSeek V4.1 Flash 1017 UD-IQ2_XXS on an
AI PRO R9700 with the Ryzen AI Max's iGPU lane and two SSDs (abyss-dsv41-iq2xxs-r9700l.log, 2026-10-29), a
16 GB + 64 GB PC, an RTX 5090 + RTX 3090 layer split, a server that is loading, one that is down.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GIB = 2**30
SCENARIOS = ("r9700", "lowspec", "split", "loading", "down")


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
    if name in ("loading", "down"):
        return None
    if name == "r9700":
        engine = {"model": "deepseek-v4.1-flash-1017-iq2xxs", "max_context": 65536, "images": False,
                  "context": 65536, "kv": "fp16", "expert_slots": 2386, "engine_kind": "glm-fast", "experts": 11008,
                  "vram_slots": 2386, "vram_gb": 21.9, "ram_slots": 6301, "ram_gb": 66.1, "mtp": 1, "cpu_threads": 16,
                  "pcie_share": "0.00", "uncensor": 1, "version": "1.3.77", "kv_ctx": 65536, "kv_gb": 1.0,
                  "maya_version": "1.3.77", "trained_context": 1048576,
                  "display_name": "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "quant": "UD-IQ2_XXS",
                  "config": "abyss-dsv41-iq2xxs-r9700l.json", "backend": "hip"}
        live = {"state": "generating", "queued": 1, "phase": "answering", "prompt_tokens": 5210, "prompt_read": None,
                "prompt_total": None, "generated": 412, "max_tokens": 2048, "elapsed_s": 21.3, "tok_s": 24.6,
                "tok_s_mean": 23.9, "prefill_tok_s_mean": 781.0, "tok_s_window_s": 3}
        tn = {"tok_s": 24.1, "ms_tok": 41.5, "vram_hit": 0.6356, "ram_fetch": 0.0, "disk": 5.84, "promo": 6.02,
              "vram_used": 2271, "vram_slots": 2386, "ram_used": 6258, "ram_slots": 6301, "ram_gb": 66.1,
              "vram_gb": 21.9, "time": now - 1.2}
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
              "time": now - 95}
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
    else:
        raise KeyError(name)
    return {"engine": engine, "live": live, "tiers": {"now": tn, "history": {k: [tn.get(k)] * 20 for k in
                                                                            ("tok_s", "ms_tok", "vram_hit",
                                                                             "ram_fetch", "disk", "promo")}},
            "requests": reqs, "requests_kept": len(reqs), "totals": totals, "hardware": hw, "hardware_static": static,
            "history": hist, "time": now, "reload": None, "storage": storage}


def _req(t, dur, finish, api, prompt, reused, out, tg, pp, hit, da, do):
    return {"projection": None, "time": t, "duration_s": dur, "finish": finish, "api": api, "prefill_tok_s": pp,
            "prompt_tokens": prompt, "reused": reused, "output_tokens": out, "engine_generated": out,
            "prompt_ms": (prompt - reused) / pp * 1000 if pp else 120.0,
            "decode_ms": out / tg * 1000 if tg else None, "decode_tok_s": tg, "hit_rate": hit,
            "drafts_accepted": da, "drafts_offered": do}


def health(name: str) -> dict | None:
    m = metrics(name)
    if m is None:
        return None
    return {"status": "ok", "max_context": m["engine"]["max_context"], "model": m["engine"]["model"], "images": False,
            "api_key": name == "split", "defaults": {}, "version": "1.3.77", "mcp_tools": False, "service": "abyss"}


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
LOGS = {"r9700": LOG_R9700, "lowspec": LOG_LOWSPEC, "loading": LOG_LOADING}


def local(name: str, log_path=None, now: float | None = None) -> dict:
    """terminal_local's facts of a scenario: the config (its display name, quant, env), the PID file, the bench line."""
    now = time.time() if now is None else now
    if name == "split":
        return {"remote": True, "is_local": False, "pid_alive": None}
    cfg = {"r9700": ("abyss-dsv41-iq2xxs-r9700l", "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS", 65536, "hip",
                     {"STRATA_GLM_MIRROR": "/mnt/ssd2/V4.1-UD-IQ2_XXS", "GPU_MAX_HW_QUEUES": "2"}, 8082),
           "lowspec": ("abyss-dsv41-iq1m-16g", "DeepSeek-V4.1-Flash-1017-UD-IQ1_M", "UD-IQ1_M", 32768, "cuda", {}, 8080),
           "loading": ("abyss-dsv41-iq2xxs-r9700l", "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS", 65536, "hip",
                       {"STRATA_GLM_MIRROR": "/mnt/ssd2/V4.1-UD-IQ2_XXS"}, 8082),
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
                       "backend": back, "env": env, "model_name": stem, "port": port}}


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

    def __init__(self, scenario: str = "r9700", key: str | None = None, service: str = "abyss"):
        self.scenario, self.key, self.service = scenario, key, service
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
                elif path == "/metrics":
                    if self._auth():
                        self._json(200, metrics(outer.scenario))
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

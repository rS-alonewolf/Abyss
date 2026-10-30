"""Where a token's time goes and what slows it: the engine log's per-request lines ("glm stat", "glm spec", "glm gap
prof"), the rules that name the bottleneck and the advice for each - one place, so `./abyss.sh --bench` (abyss.py) and
Abyss Terminal (tools/abyss_terminal.py) never disagree.  Standard library only; nothing runs on import.
"""
from __future__ import annotations

import os
import re

ME = "START-ABYSS.bat" if os.name == "nt" else "./abyss.sh"   # how Abyss is started, for the advice
IGPU_LANE = "the iGPU computes the lane"                       # the engine's start line when the iGPU takes the lane


# the engine's per-request line after each answer (src/program/generate.cpp, "glm stat"), its drafter's and - with
# STRATA_GLM_GAP_PROF=1 - its GPU timeline's (src/core/glm_fast_path.cu gap_report)
STAT_LINE = re.compile(
    r"glm stat: decode (?P<ms>[\d.]+) ms/tok \([\d.]+ tok/s\) \| vram hit (?P<vram_hit>[\d.]+)% \| "
    r"ram fetches (?P<ram_fetch>[\d.]+)/tok \| disk (?P<disk_reads>[\d.]+) reads/tok in (?P<disk_waits>[\d.]+) "
    r"waits/tok, (?P<disk_ms>[\d.]+) ms/tok \| promotions (?P<promotions>[\d.]+)/tok \| cpu lane "
    r"(?P<lane>[\d.]+)/tok in (?P<lane_ms>[\d.]+) ms/tok \| tail skip (?P<tail_skip>[\d.]+)/tok \| "
    r"prompt (?P<prompt_ms>[\d.]+) ms/tok")
SPEC_LINE = re.compile(r"glm spec: (\d+) of (\d+) drafts accepted")
GAP_LINE = re.compile(r"glm gap prof .*GPU busy (?P<busy>[\d.]+), idle between tokens (?P<idle>[\d.]+)")


def bench_records(lines) -> list:
    """One record per request, in order: its "glm stat" numbers, with the drafter's and the GPU timeline's lines that
    follow it attached (a prompt-only request has its stat line too: one token out)."""
    recs = []
    for x in lines:
        m = STAT_LINE.search(x)
        if m:
            recs.append({k: float(v) for k, v in m.groupdict().items()})
            continue
        if not recs:
            continue
        m = SPEC_LINE.search(x)
        if m:
            recs[-1]["spec"] = (int(m.group(1)), int(m.group(2)))
        m = GAP_LINE.search(x)
        if m:
            recs[-1]["gpu_busy"], recs[-1]["gpu_idle"] = float(m.group("busy")), float(m.group("idle"))
    return recs


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# what each bottleneck means and what moves it - the bench's verdict, and docs/BENCH(.ja).md's table
BENCH_ADVICE = {
    "ssd": ("the SSD: the experts that fit neither in GPU memory nor in the RAM tier are read from disk every token",
            "a smaller quant that fits GPU + RAM, more RAM for the RAM tier (STRATA_GLM_RAM_GB), a faster SSD; or, "
            "changing the answers a little, the disk tail skip (STRATA_GLM_TAIL_SKIP=4 and STRATA_GLM_TAIL_SKIP_DISK=1 "
            "in the config's \"env\": DeepSeek V4.1 UD-IQ2_XXS on an R9700 +34%, KL +0.007)"),
    "cpu": ("the CPU / iGPU lane: the experts it computes take as long as the GPU's whole token",
            f"{ME} --calibrate (the lane's share and threads), the CPU's power mode; on a PC with a Radeon iGPU the "
            "iGPU lane"),
    "pcie": ("PCIe: experts copied from RAM to the GPU every token",
             "the GPU's slot (Gen4 x16 moves 4x what an x4 / OCuLink link does), "
             f"{ME} --calibrate (moves the share to the CPU lane)"),
    "gpu": ("the GPU itself: the memory tiers keep up",
            f"{ME} --gpu-tune, and a report with this file: the kernels for this GPU may have room"),
}
# --calibrate measures the CUDA engine's lane only (issue #14): on HIP the advice names the setting itself
BENCH_ADVICE_HIP = {
    "cpu": f"STRATA_GLM_PCIE_SHARE in the config's \"env\" (--calibrate is CUDA-only; keep a share only if a plain "
           f"{ME} --bench gets faster with it), the CPU's power mode; on a PC with a Radeon iGPU the iGPU lane",
    "pcie": "the GPU's slot (Gen4 x16 moves 4x what an x4 / OCuLink link does), STRATA_GLM_PCIE_SHARE in the config's "
            "\"env\" (moves the share to the CPU lane; keep it only if a plain --bench gets faster with it)",
}


def bench_verdict(recs, lane_name: str = "CPU lane") -> tuple:
    """(the likely bottleneck - a BENCH_ADVICE key, or "" - and the evidence lines) from the decode records' means.
    Rules of thumb, checked on the R9700 / RTX 5090 runs of docs/MODELS.md: SSD waits a quarter of a token or more
    -> the SSD; the CPU lane busy 80% of a token or more -> the CPU lane; the GPU busy 85% of its timeline (--deep,
    without a drafter: its windows' timeline does not split that way) -> the GPU; ten or more experts a token over PCIe
    -> PCIe; else the GPU.  `lane_name`: "iGPU lane" where the engine's log says the iGPU computes it."""
    if not recs:
        return "", ["the engine printed no per-token statistics (not the fast path?)"]
    ms = _mean(r["ms"] for r in recs)
    disk, lane = _mean(r["disk_ms"] for r in recs), _mean(r["lane_ms"] for r in recs)
    fetch, hit = _mean(r["ram_fetch"] for r in recs), _mean(r["vram_hit"] for r in recs)
    ev = [f"a token: {ms:.1f} ms ({1000.0 / max(ms, 1e-6):.1f} tokens/s)",
          f"SSD waits: {disk:.1f} ms a token ({disk / ms:.0%}), {_mean(r['disk_reads'] for r in recs):.1f} expert "
          "reads a token",
          f"{lane_name}: busy {lane:.1f} ms a token ({lane / ms:.0%}), {_mean(r['lane'] for r in recs):.0f} experts a "
          "token",
          f"GPU memory: holds {hit:.0f}% of the experts a token uses; {fetch:.1f} a token come over PCIe from RAM"]
    spec = [r["spec"] for r in recs if "spec" in r]
    if spec:
        ev.append(f"drafter: {sum(a for a, _ in spec) / max(1, sum(b for _, b in spec)):.0%} of its drafts accepted")
    skip = _mean(r["tail_skip"] for r in recs)
    if skip > 0:
        ev.append(f"tail skip on: {skip:.1f} experts a token skipped (lossy - its KL is in docs/MODELS.md)")
    busy = [r["gpu_busy"] / max(1e-6, r["gpu_busy"] + r["gpu_idle"]) for r in recs if "gpu_busy" in r]
    if busy and spec:
        ev.append(f"GPU timeline: busy {_mean(busy):.0%} (with the drafter's windows: shown, not used for the verdict)")
        busy = []
    elif busy:
        ev.append(f"GPU timeline: busy {_mean(busy):.0%} of it, idle between tokens the rest")
    if disk >= 0.25 * ms:
        v = "ssd"
    elif lane >= 0.8 * ms:
        v = "cpu"
    elif busy and _mean(busy) >= 0.85:
        v = "gpu"
    elif fetch >= 10.0:
        v = "pcie"
    else:
        v = "gpu"
    return v, ev


def lane_name(log_lines) -> str:
    """"iGPU lane" where the engine's log says the iGPU computes the CPU lane's experts, else "CPU lane"."""
    return "iGPU lane" if any(IGPU_LANE in x for x in log_lines) else "CPU lane"


def advice(verdict: str, backend: str | None = None) -> tuple:
    """(what the bottleneck is, what moves it) for a bench_verdict key - HIP's own advice where it has one; ("", "")
    for none."""
    if verdict not in BENCH_ADVICE:
        return "", ""
    what, moves = BENCH_ADVICE[verdict]
    if backend == "hip":
        moves = BENCH_ADVICE_HIP.get(verdict, moves)
    return what, moves

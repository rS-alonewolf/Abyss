#!/usr/bin/env python3
"""mtp_greedy_check.py <config.json> [--exe PATH] [--max-new N] [--prompts FILE] [--out JSON]

Greedy answers with the model's drafter on and off must be the same tokens: a draft only decides how many tokens a
verify window checks at once, never which token comes next.  This starts the engine twice from one server config
(the config's env, then the same with STRATA_GLM_NO_MTP=1), asks each the same greedy prompts over the --serve wire
(GEN temperature=0) and prints, per prompt, the first position where the two answers part (or "same").  Exit 0 when
every answer matches, 1 otherwise.  --self: one engine, each answer against its own GEN spec=0 answer (token by token,
no drafts or windows).  A verify window sums in another order than the token path, so a near-tie can part two right
answers late (after the fix below: Qwen3.8 from token 33, V4.1 + DSpark from 16); a bug parts them early.  (2026-10-31: the
qwen4exp MTP drafter on CUDA looped and lost facts while the drafter-off answers were right - the route kernel zeroed
the window columns' expert weights; the answers parted at tokens 2-29, 1-3 without the CPU lane.)"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

PROMPTS = [
    "A train travels 180 km in 2.5 hours. At the same speed, how many km does it travel in 4 hours? "
    "Give the final answer on the last line as 'Answer: <number>'.",
    "A store sells pencils at 3 for 75 cents. How much do 18 pencils cost, in cents? "
    "Give the final answer on the last line as 'Answer: <number>'.",
    "What is the sum of all integers from 1 to 200 that are divisible by 7? "
    "Give the final answer on the last line as 'Answer: <number>'.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
    "How do I find which process is listening on port 8080 on Linux?",
    "What is the capital of Bhutan? Answer in one sentence.",
    "日本の四季それぞれの特徴を簡単に説明してください。",
    "Write a detailed essay of about 300 words on the history of the printing press.",
]


def engine_run(exe, args, cwd, env, prompts_ids, max_new, label, ref=False):
    p = subprocess.Popen([exe, "--serve", *args], cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(f"/tmp/mtpcheck-{label}.engine.log", "w"), text=True, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"{label}: the engine ended before READY (see /tmp/mtpcheck-{label}.engine.log)")
    def gen(ids, keys=""):
        t0 = time.time()
        p.stdin.write(f"GEN {max_new} temperature=0 {keys}{','.join(map(str, ids))}\n")
        p.stdin.flush()
        toks, done = [], ""
        for line in p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                done = line.strip()
                break
        return {"tokens": toks, "done": done, "secs": round(time.time() - t0, 2)}

    outs = []
    for ids in prompts_ids:
        o = gen(ids)
        if ref:   # the same engine token by token (GEN spec=0): the reference these tokens must equal
            o["ref"] = gen(ids, "spec=0 ")
        outs.append(o)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--exe")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--only", choices=["on", "off", "none"])
    ap.add_argument("--variant", action="append", default=[],
                    help="name:KEY=V,KEY=V - another drafter-on run with these env changes, compared with off")
    ap.add_argument("--off-env", default="",
                    help="KEY=V,KEY= - more env changes for the drafter-off engine (KEY= removes KEY), e.g. "
                         "STRATA_DS4_SPEC=0,STRATA_DS4_DSPARK= for no windows at all (an engine without GEN spec=0)")
    ap.add_argument("--n", type=int, default=0, help="the first n prompts only")
    ap.add_argument("--self", action="store_true",
                    help="no drafter-off engine: each run's answers against its own GEN spec=0 answers")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    exe = a.exe or cfg["exe"]
    tdir = Path(cfg["tokenizer"])
    tok = ST.Tokenizer.from_dir(tdir)
    tpl = ChatTemplate(tdir / "chat_template.jinja")
    prompts = json.loads(Path(a.prompts).read_text()) if a.prompts else PROMPTS
    if a.n:
        prompts = prompts[:a.n]
    ids = [tok.encode(tpl.render([{"role": "user", "content": q}], enable_thinking=False), parse_special=True)
           for q in prompts]
    base = dict(os.environ)
    base.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    if cfg.get("lib_dirs"):
        base["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [base.get("LD_LIBRARY_PATH", "")])
    if cfg.get("gpu"):
        base.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, cfg["gpu"])))
    on = dict(base)
    on.pop("STRATA_GLM_NO_MTP", None)
    off = dict(base, STRATA_GLM_NO_MTP="1")
    for x in filter(None, a.off_env.split(",")):
        k, _, val = x.partition("=")
        if val:
            off[k] = val
        else:
            off.pop(k, None)
    res = {}
    runs = [("on", on)] if a.self else [("off", off), ("on", on)]
    for v in a.variant:
        name, _, kv = v.partition(":")
        e = dict(on)
        for x in filter(None, kv.split(",")):
            k, _, val = x.partition("=")
            e[k] = val
        runs.append((name, e))
    for label, env in runs:
        if a.only and label in ("on", "off") and a.only != label:
            continue
        res[label] = engine_run(exe, cfg["args"], cfg.get("cwd"), env, ids, a.max_new, label, ref=a.self)
    bad = 0
    for i, q in enumerate(prompts):
        line = f"[{i}] {q[:50]!r}"
        print(line)
        for lab in (l for l in res if l != "off" and ("off" in res or a.self)):
            refo = res[lab][i]["ref"] if a.self else res["off"][i]
            x, y = res[lab][i]["tokens"], refo["tokens"]
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), None)
            if k is None and len(x) != len(y):
                k = min(len(x), len(y))
            if k is None:
                print(f"  {lab}: same ({len(x)} tokens; {res[lab][i]['secs']} s vs off {refo['secs']} s)")
            else:
                bad += 1
                print(f"  {lab}: DIFFER at {k} of {len(x)}/{len(y)}: {tok.decode(x[max(0, k - 8):k + 8])!r} vs off "
                      f"{tok.decode(y[max(0, k - 8):k + 8])!r}")
        for label in res:
            print(f"    {label}: {tok.decode(res[label][i]['tokens'])[:200]!r}  {res[label][i]['done'][:60]}")
    if a.out:
        Path(a.out).write_text(json.dumps({"prompts": prompts, "ids": ids, "res": res}))
    print(f"{bad} answers differ from off" if ("off" in res and len(res) > 1) or a.self else "one side only")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

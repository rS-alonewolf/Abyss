"""The prompts the engine's --abl-build measures and checks (uncensoring at run time), as token ids.

usage: abl_prompts.py --pack <pack dir> --out <prompts.txt> [--data <safety datasets>] [--train N] [--check N]
                      [--exclude FILE ...]
  Harmful: an even mix of AdvBench, MaliciousInstruct, ForbiddenQuestions, SimpleSafetyTests, JailbreakBench's harmful
  behaviors and XSTest's unsafe prompts (AdvBench alone left the softer refusals - piracy, medicine, fraud - in GLM;
  --harmful advbench for AdvBench only).  Harmless: JailbreakBench's benign behaviors and XSTest's safe
  prompts (the same topics, asked harmlessly - what the direction must leave alone).  Each prompt is rendered by the
  pack's chat template as one user turn, ending where the answer starts, thinking off; --train of each class go to
  the measurement (H / B lines), the next --check to the candidates' check (h / b).  The R line: the first token of
  the usual refusal openings ("I'm sorry", "I can't", ...); the W lines: refusal words (" can't", " cannot",
  " unable", ...) as token runs --abl-gen looks for in the answers' first tokens.  The picks are the same every run (sorted by a hash of
  the text); --exclude drops prompts an evaluation uses (lines of text, or JSON lines with a "prompt").
  The datasets (Abyss-data/datasets/safety, MANIFEST.txt) are untrusted data: run this with python -I.
then: strata --glm-pack <pack> --abl-build <out.abl> --abl-prompts <prompts.txt>, and --uncensor <out.abl>.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

DATA = ROOT.parent / "Abyss-data" / "datasets" / "safety"
# the refusal words anywhere in an answer's opening (the W line: --abl-gen's check - GLM opens with a heading, then refuses)
WORDS = (" can't", " cannot", " can not", " won't", " unable", " Sorry", " sorry", " apologize", " refuse", " decline",
         "I can't", "I cannot", "Sorry", "申し訳", "できません", "お手伝いできません", "控え")
REFUSALS = ("I'm sorry", "I am sorry", "I can't", "I cannot", "I won't", "I apologize", "Sorry", "I'm unable",
            "Unfortunately", "As an AI", "申し訳", "できません")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(texts, exclude, n):
    """n texts, the same every run: sorted by a hash of the text, the excluded ones and duplicates left out"""
    seen, out = set(), []
    for t in sorted(texts, key=lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()):
        t = t.strip()
        if t and t not in exclude and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--train", type=int, default=64)
    ap.add_argument("--check", type=int, default=24)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--harmful", choices=["mix", "advbench"], default="mix")
    a = ap.parse_args()
    data = pathlib.Path(a.data)
    exclude = set()
    for f in a.exclude:
        for line in pathlib.Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    line = str(json.loads(line).get("prompt", "")).strip()
                except ValueError:
                    continue
            if line:
                exclude.add(line)
    srcs = [[r["goal"] for r in read_csv(data / "AdvBench" / "harmful_behaviors.csv")]]
    if a.harmful == "mix":
        srcs += [
            [l.strip() for l in (data / "MaliciousInstruct" / "MaliciousInstruct.txt").read_text(encoding="utf-8").splitlines()],
            [r["question"] for r in read_csv(data / "ForbiddenQuestions" / "forbidden_question_set.csv")],
            [r["prompt"] for r in read_csv(data / "SimpleSafetyTests" / "sst_test_cases.csv")],
            [r["Goal"] for r in read_csv(data / "JBB-Behaviors" / "harmful-behaviors.csv")],
            [r["prompt"] for r in read_csv(data / "XSTest" / "xstest_prompts.csv") if r.get("label") == "unsafe"]]
    harmless = [r["Goal"] for r in read_csv(data / "JBB-Behaviors" / "benign-behaviors.csv")]
    harmless += [r["prompt"] for r in read_csv(data / "XSTest" / "xstest_prompts.csv") if r.get("label") == "safe"]
    n = a.train + a.check
    # the harmful prompts round-robin over the sources (each in its hash order), so every kind is measured and checked
    per = [pick(x, exclude, n) for x in srcs]
    hp, seen = [], set()
    for i in range(n):
        for x in per:
            if i < len(x) and x[i] not in seen and len(hp) < n:
                seen.add(x[i])
                hp.append(x[i])
    hp = sorted(hp, key=lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())   # (train / check: the same mix)
    bp = pick(harmless, exclude, n)
    if len(hp) < n or len(bp) < n:
        print(f"only {len(hp)} harmful / {len(bp)} harmless prompts for {a.train} + {a.check} of each")
        return 1
    tdir = pathlib.Path(a.pack) / "tokenizer"
    tok = ST.Tokenizer.from_dir(tdir)
    bos_id = (getattr(tok, "special_ids", None) or {}).get("tokenizer.ggml.bos_token_id")
    bos = tok.tokens[bos_id] if isinstance(bos_id, int) and 0 <= bos_id < len(tok.tokens) else ""
    tpl = tdir / "chat_template.jinja"
    template = ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja", bos_token=bos,
                            dsml41=(pathlib.Path(a.pack) / "engram.txt").exists())

    def ids(text):
        prompt = template.render([{"role": "user", "content": text}], enable_thinking=False)
        return ",".join(str(i) for i in tok.encode(prompt, parse_special=True))

    refusal = sorted({tok.encode(r, parse_special=False)[0] for r in REFUSALS})
    with open(a.out, "w", encoding="utf-8") as f:
        for kind, texts in (("H", hp[:a.train]), ("B", bp[:a.train]), ("h", hp[a.train:]), ("b", bp[a.train:])):
            for t in texts:
                f.write(f"{kind} {ids(t)}\n")
        f.write("R " + ",".join(str(i) for i in refusal) + "\n")
        words = []   # one W line a word: its tokens, looked for as a run anywhere in an answer's opening
        for w in WORDS:
            t = tok.encode(w, parse_special=False)
            if t and t not in words:
                words.append(t)
                f.write("W " + ",".join(str(i) for i in t) + "\n")
    print(f"{a.out}: {a.train} + {a.check} harmful, {a.train} + {a.check} harmless, {len(refusal)} refusal tokens "
          f"({', '.join(repr(tok.decode([i])) for i in refusal)}), {len(words)} refusal words")
    return 0


if __name__ == "__main__":
    sys.exit(main())

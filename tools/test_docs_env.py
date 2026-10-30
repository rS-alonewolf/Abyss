"""Docs lint: every STRATA_* / ABYSS_* / MAYA_* setting the code reads is either in the manual (docs/MANUAL.md or
docs/SETTINGS.md) or on the debug / experimental list tools/env_undocumented.txt; the English and Japanese manuals
name the same settings and config keys; every config key the server or setup reads is in docs/SETTINGS.md.

A new setting fails here until it is documented or listed (the review of 2026-10-30 found ~250 names in the code
and not in the manual; the list holds those, so it only ever shrinks).  `python tools/test_docs_env.py --write`
rewrites the list from the code as it stands (after documenting what should be).
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tools" / "env_undocumented.txt"
CODE_DIRS = ("src", "include", "serve", "igpu")
CODE_FILES = ("abyss.py", "setup.py", "abyss.sh")
NAME = r"((?:STRATA|ABYSS|MAYA)_[A-Z0-9_]+)"
READS = (re.compile(r'getenv\(\s*"' + NAME),                       # C++ getenv("X") / std::getenv
         re.compile(r'env_(?:num|seconds|on|int|flag)\(\s*"' + NAME),   # the knob helpers
         re.compile(r'environ(?:\.get)?\s*[\(\[]\s*"' + NAME),      # Python os.environ.get("X") / os.environ["X"]
         re.compile(r'\$\{?' + NAME))                               # shell $X / ${X}
DOC_NAME = re.compile(r"`" + NAME)
CFG_KEY = re.compile(r'cfg(?:\.get\(|\[)"([a-z_]+)"')
DOC_KEY = re.compile(r"`([a-z_]+)`")
# names a regex picks up that are not settings (a name used as a prefix, the version stamp)
NOT_SETTINGS = {"STRATA_", "ABYSS_", "MAYA_", "STRATA_VERSION", "STRATA_HIP_ARCHS", "STRATA_USE_HIP",
                "STRATA_NATIVE_EXPERTS", "STRATA_PREFILL_MMQ", "STRATA_MMQ_GLU", "STRATA_SELECT_OLD_"}


def code_names() -> dict:
    """{setting: first file that reads it}"""
    out = {}
    files = [p for d in CODE_DIRS for p in (ROOT / d).rglob("*")
             if p.is_file() and p.suffix in (".cu", ".cuh", ".cpp", ".hpp", ".h", ".py", ".sh")]
    files += [ROOT / f for f in CODE_FILES if (ROOT / f).exists()]
    for p in sorted(files):
        if "test_" in p.name or "/tests/" in str(p):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for rx in READS:
            for m in rx.finditer(text):
                n = m.group(1)
                if n not in NOT_SETTINGS and not n.endswith("_"):
                    out.setdefault(n, str(p.relative_to(ROOT)))
    return out


def doc_names(*names) -> set:
    s = set()
    for n in names:
        s |= set(DOC_NAME.findall((ROOT / "docs" / n).read_text(encoding="utf-8")))
    return s


def allowlist() -> set:
    if not ALLOWLIST.exists():
        return set()
    return {ln.split("#")[0].strip() for ln in ALLOWLIST.read_text().splitlines() if ln.split("#")[0].strip()}


class EnvDocs(unittest.TestCase):
    def test_every_setting_the_code_reads_is_documented_or_listed(self):
        code, docs, listed = code_names(), doc_names("MANUAL.md", "SETTINGS.md"), allowlist()
        new = sorted(n for n in code if n not in docs and n not in listed)
        self.assertEqual(new, [], "settings read by the code but in neither docs/MANUAL.md, docs/SETTINGS.md nor "
                                  "tools/env_undocumented.txt (document them, or list them as debug knobs):\n  " +
                         "\n  ".join(f"{n}  ({code[n]})" for n in new))

    def test_the_list_holds_only_what_is_still_undocumented(self):
        code, docs, listed = code_names(), doc_names("MANUAL.md", "SETTINGS.md"), allowlist()
        gone = sorted(n for n in listed if n not in code)
        self.assertEqual(gone, [], "no code reads these any more: remove them from tools/env_undocumented.txt")
        documented = sorted(n for n in listed if n in docs)
        self.assertEqual(documented, [], "documented now: remove them from tools/env_undocumented.txt")

    def test_english_and_japanese_name_the_same_settings(self):
        for en, ja in (("MANUAL.md", "MANUAL.ja.md"), ("SETTINGS.md", "SETTINGS.ja.md")):
            a, b = doc_names(en), doc_names(ja)
            self.assertEqual(sorted(a - b), [], f"in {en} but not {ja}")
            self.assertEqual(sorted(b - a), [], f"in {ja} but not {en}")

    def test_config_keys_are_in_settings_md(self):
        keys = set()
        for f in ("serve/server.py", "abyss.py"):
            keys |= set(CFG_KEY.findall((ROOT / f).read_text(encoding="utf-8")))
        for name in ("SETTINGS.md", "SETTINGS.ja.md"):
            documented = set(DOC_KEY.findall((ROOT / "docs" / name).read_text(encoding="utf-8")))
            self.assertEqual(sorted(keys - documented), [], f"config keys the code reads and docs/{name} does not name")


if __name__ == "__main__":
    if "--write" in sys.argv:
        code, docs = code_names(), doc_names("MANUAL.md", "SETTINGS.md")
        names = sorted(n for n in code if n not in docs)
        ALLOWLIST.write_text("# Settings the code reads that docs/MANUAL.md and docs/SETTINGS.md do not describe (debug and\n"
                             "# experimental knobs, checked by tools/test_docs_env.py). Document a setting and remove it here;\n"
                             "# `python tools/test_docs_env.py --write` rewrites the list from the code.\n" +
                             "".join(f"{n}  # {code[n]}\n" for n in names))
        print(f"{len(names)} names written to {ALLOWLIST}")
    else:
        unittest.main()

"""tools/abl_prompts.py: the picks are the same every run and leave the excluded prompts out"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    import abl_prompts as AP
except ImportError as e:   # (serve.frontend needs jinja2)
    AP = None
    _err = e


@unittest.skipIf(AP is None, "abl_prompts did not import")
class TestPick(unittest.TestCase):
    def test_same_every_run(self):
        t = [f"prompt {i}" for i in range(50)]
        self.assertEqual(AP.pick(t, set(), 10), AP.pick(list(reversed(t)), set(), 10))

    def test_exclude_and_duplicates(self):
        t = ["a", "b", "a", " c ", "d"]
        got = AP.pick(t, {"b"}, 10)
        self.assertNotIn("b", got)
        self.assertEqual(sorted(got), ["a", "c", "d"])

    def test_words_cover_refusals(self):
        for w in (" can't", " cannot", "申し訳", "できません"):
            self.assertIn(w, AP.WORDS)


    def test_read_prompts_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "own.txt"
            p.write_text("# my prompts\n\nexample prompt 1\n  example prompt 2  \n"
                         '{"prompt": "example prompt 3"}\n{not json\n', encoding="utf-8")
            self.assertEqual(AP.read_prompts(p), ["example prompt 1", "example prompt 2", "example prompt 3"])

    def test_force_keeps_extra_through_trim(self):
        hp = [f"mix {i}" for i in range(10)]
        got = AP.force(hp, ["example prompt 1", "mix 3"], 5)
        self.assertEqual(len(got), 5)
        self.assertEqual(got[:2], ["example prompt 1", "mix 3"])
        self.assertEqual(len(set(got)), 5)

if __name__ == "__main__":
    unittest.main()

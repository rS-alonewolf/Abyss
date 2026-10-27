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


if __name__ == "__main__":
    unittest.main()

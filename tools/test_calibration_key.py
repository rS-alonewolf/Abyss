"""The calibration store's key (abyss.py hardware_key): the GPUs' PCIe link and driver are part of it, so a card moved
between an x16 slot and an OCuLink x4, or a driver update, does not reuse a tuning measured on the other (Maya #419's
idea); an older store without them is read and treated as stale, with one line.  No GPU needed.

    python -m unittest tools.test_calibration_key
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import abyss as maya  # noqa: E402

GPU = {"index": 0, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.8, "arch": "120", "driver": "580.65.06"}
CFG = {"gpu": [0], "args": ["--max-context", "65536"], "model_name": "glm-5.3-flash", "installer": {"quant": "maya-s-v2"}}


class CalibrationKey(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.links = {0: [(1, 16, 5, 16)]}             # idle: trained down to Gen1; the slot can do Gen5 x16
        self.gpus = [dict(GPU)]
        self.warned = []
        for ctx in (
            patch.object(maya, "CAL_STORE", Path(self.tmp.name) / "calibration.json"),
            patch.object(maya.S, "gpus", side_effect=lambda: self.gpus),
            patch.object(maya.S, "cpu_info", return_value=("AMD Ryzen 9 7950X", True, True)),
            patch.object(maya.S, "ram_gb", return_value=94.0),
            patch.object(maya.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(maya, "nvidia_links", side_effect=lambda sel: self.links),
            patch.object(maya, "warn", side_effect=self.warned.append),
            patch.object(maya, "_STALE_SAID", set()),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)

    def store(self, d: dict):
        maya.CAL_STORE.write_text(json.dumps(d))

    def test_key_has_the_slot_link_and_the_driver(self):
        key = maya.hardware_key(CFG)
        self.assertEqual(key.split("|")[-2:], ["Gen5 x16", "580.65.06"])
        self.assertEqual(len(key.split("|")), maya.KEY_FIELDS)
        self.links = {0: [(5, 16, 5, 16)]}             # the same slot, busy: the same key
        self.assertEqual(maya.hardware_key(CFG), key)

    def test_a_tuning_on_this_slot_and_driver_is_used(self):
        self.store({maya.hardware_key(CFG): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.25"}, "date": "2026-10-30"}})
        self.assertEqual(maya.saved_calibration(CFG)["settings"], {"STRATA_GLM_PCIE_SHARE": "0.25"})
        self.assertEqual(self.warned, [])

    def test_the_card_moved_to_an_oculink_slot(self):
        self.store({maya.hardware_key(CFG): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.25"}, "date": "2026-10-30"}})
        self.links = {0: [(4, 4, 4, 4)]}
        self.assertIsNone(maya.saved_calibration(CFG))
        self.assertIsNone(maya.saved_calibration(CFG, any_context=True))
        self.assertEqual(len(self.warned), 1)          # said once, however often the setup asks
        self.assertIn("on Gen5 x16, driver 580.65.06 - now Gen4 x4, driver 580.65.06", self.warned[0])
        self.assertIn("--calibrate", self.warned[0])
        self.assertIn("(2026-10-30)", self.warned[0])

    def test_a_driver_update(self):
        self.store({maya.hardware_key(CFG): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.25"}}})
        self.gpus = [dict(GPU, driver="590.12.01")]
        self.assertIsNone(maya.saved_calibration(CFG, any_context=True))
        self.assertIn("driver 580.65.06 - now Gen5 x16, driver 590.12.01", self.warned[0])

    def test_an_old_store_without_link_and_driver_is_stale_not_a_crash(self):
        old_key = "|".join(maya.hardware_key(CFG).split("|")[:7])
        self.store({old_key: {"settings": {"STRATA_GLM_PCIE_SHARE": "0.25"}, "date": "2026-10-09"}})
        self.assertIsNone(maya.saved_calibration(CFG))
        self.assertIsNone(maya.saved_calibration(CFG, any_context=True))
        self.assertEqual(len(self.warned), 1)
        self.assertIn("before the GPU's PCIe link and driver were part of the record", self.warned[0])
        # odd contents never stop a setup either
        for junk in ('{"a|b": 3, "x": null}', "[1, 2]", "not json", ""):
            maya.CAL_STORE.write_text(junk)
            self.assertIsNone(maya.saved_calibration(CFG, any_context=True))

    def test_the_nearest_context_still_matches_on_the_same_link_and_driver(self):
        key = maya.hardware_key(CFG)
        self.store({key.replace("|65536|", "|32768|"): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.30"}},
                    key.replace("|65536|", "|32768|").replace("Gen5 x16", "Gen4 x4"): {"settings": {"STRATA_GLM_PCIE_SHARE": "0.9"}}})
        self.assertIsNone(maya.saved_calibration(CFG))
        got = maya.saved_calibration(CFG, any_context=True)
        self.assertEqual((got["settings"], got["context"]), ({"STRATA_GLM_PCIE_SHARE": "0.30"}, 32768))
        self.assertEqual(self.warned, [])              # a usable tuning: the stale one on the other slot is not noted

    def test_another_model_or_pc_is_not_noted(self):
        self.store({maya.hardware_key(CFG).replace("maya-s-v2", "other"): {"settings": {}},
                    maya.hardware_key(CFG).replace("Gen5 x16", "Gen4 x4").replace("94GB", "64GB"): {"settings": {}}})
        self.assertIsNone(maya.saved_calibration(CFG, any_context=True))
        self.assertEqual(self.warned, [])

    def test_no_nvidia_smi_or_card_is_a_question_mark(self):
        with patch.object(maya.shutil, "which", return_value=None):
            self.assertEqual(maya.hardware_key(CFG).split("|")[-2:], ["?", "580.65.06"])
        self.gpus, self.links = [], {}
        self.assertEqual(maya.hardware_key(CFG).split("|")[-2:], ["?", "?"])
        self.assertEqual(maya.hardware_key({"gpu": [], "args": []}).split("|")[-2:], ["?", "?"])

    def test_calibrate_writes_the_new_key(self):
        cfg_path = Path(self.tmp.name) / "abyss-glm.json"
        cfg_path.write_text(json.dumps(dict(CFG, exe="x")))
        with patch.object(maya, "busy_gpus", return_value=[]), patch.object(maya, "say"), patch.object(maya, "ok"), \
                patch.object(maya.CAL, "run", return_value={"settings": {"STRATA_GLM_PCIE_SHARE": "0.20"},
                                                           "report": {"tok_s": 40.1}}):
            self.assertTrue(maya.calibrate_config(cfg_path))
        store = json.loads(maya.CAL_STORE.read_text())
        self.assertEqual(list(store), [maya.hardware_key(CFG)])
        self.assertEqual(maya.saved_calibration(json.loads(cfg_path.read_text()))["settings"],
                         {"STRATA_GLM_PCIE_SHARE": "0.20"})


class AmdLink(unittest.TestCase):
    def test_hip_config_takes_the_narrowest_maximum_on_the_way(self):
        cfg = {"backend": "hip", "gpu": [0], "gpu_pci": ["0000:c7:00.0"]}
        chain = [(1, 4, 5, 16), (4, 4, 4, 4)]           # the card (idle at Gen1, can do Gen5 x16) behind an x4 root port
        with patch.object(maya, "pcie_chain", return_value=chain), patch.object(maya, "WIN", False), \
                patch.object(maya.Path, "read_text", return_value="6.14.14\n"):
            self.assertEqual(maya.gpu_link_driver(cfg, {}), ("Gen4 x4", "6.14.14"))
        with patch.object(maya, "pcie_chain", return_value=[]), patch.object(maya, "WIN", False), \
                patch.object(maya.Path, "read_text", side_effect=OSError):
            self.assertEqual(maya.gpu_link_driver(cfg, {}), ("?", "?"))


if __name__ == "__main__":
    unittest.main()

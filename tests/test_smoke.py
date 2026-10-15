"""Smoke test: the modules import, every drafter checkpoint and baseline graph loads, and the decode
settings of SPOT and SPOT-gamma validate as distribution-preserving. No data, GPU or model download;
it runs on a CPU in a few seconds."""
import glob
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class Smoke(unittest.TestCase):
    def test_imports(self):
        import spot.decode, spot.drafters, spot.graphs, spot.models, spot.sampling, spot.tasks  # noqa: F401

    def test_drafters_load(self):
        from spot.drafters import SlotDrafter
        paths = sorted(glob.glob(os.path.join(ROOT, "drafters", "*.pt")))
        paths += sorted(glob.glob(os.path.join(ROOT, "drafters", "tuned", "*.pt")))
        self.assertEqual(len(paths), 10)
        for p in paths:
            with self.subTest(drafter=os.path.relpath(p, ROOT)):
                d = SlotDrafter(p, device="cpu")
                self.assertEqual(sum(q.numel() for q in d.model.parameters()), 621057)  # the 0.62M order model
                self.assertEqual(d.m_top, 32)  # reads the stale top-32 probabilities
                self.assertEqual(d.trained_block, 16 if "_b16" in os.path.basename(p) else 32)

    def test_graphs_load(self):
        from spot.graphs import DraftGraph
        paths = sorted(glob.glob(os.path.join(ROOT, "graphs", "*_D24.json")))
        self.assertEqual(len(paths), 6)
        for p in paths:
            with self.subTest(graph=os.path.basename(p)):
                self.assertGreater(len(DraftGraph.load(p).nodes), 0)

    def test_decode_settings_are_distribution_preserving(self):
        from spot.decode import DecodeConfig
        common = dict(cache="on", unmasking="single", draft="slot", slot_mode="ddtree_b", verify="match",
                      suffix="skip", draft_layout="concat", draft_budget=8, gen_length=256, block_length=32)
        for name, extra in [("SPOT, T=0", dict(pad_pieces=8, temperature=0.0)),
                            ("SPOT-gamma, T=1", dict(pad_pieces=0, slot_gamma=0.1, temperature=1.0))]:
            with self.subTest(setting=name):
                cfg = DecodeConfig(**common, **extra)
                cfg.validate()
                preserving, reasons = cfg.is_distribution_preserving(True)
                self.assertTrue(preserving, reasons)


if __name__ == "__main__":
    unittest.main()

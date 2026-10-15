"""Tests for the comm-volume model and the theoretical-optimum composer."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = os.path.join(ROOT, "skills", "ascend-compute-simulation", "scripts")
sys.path.insert(0, S)

import comm_volume as cv  # noqa: E402

CFG = {"hidden_size": 5120, "num_hidden_layers": 40,
       "n_routed_experts": 384, "num_experts_per_tok": 6,
       "moe_intermediate_size": 2304}


class TestCommVolume(unittest.TestCase):
    def test_expected_distinct_ranks(self):
        # 8 ranks, top-6: 8*(1-(7/8)^6) = 4.41
        self.assertAlmostEqual(cv.expected_distinct_ranks(8, 6), 4.41, places=2)

    def test_tp_ep_counts(self):
        m = cv.model_layouts(CFG, 8)
        self.assertEqual(m["tp-ep"].collectives_per_layer, 3)
        self.assertEqual(m["dp-ep"].collectives_per_layer, 2)
        self.assertEqual(m["tp"].collectives_per_layer, 2)

    def test_dp_ep_fewer_collectives(self):
        m = cv.model_layouts(CFG, 8)
        self.assertLess(m["dp-ep"].collectives_per_layer,
                        m["tp-ep"].collectives_per_layer)

    def test_latency_projection_consistency(self):
        # 120 collectives x 430us = 51.6ms; the measured anchor is 51.7ms
        m = cv.model_layouts(CFG, 8)["tp-ep"]
        pred = m.collectives_per_layer * 40 * 430 / 1000
        self.assertAlmostEqual(pred, 51.6, places=1)

    def test_cli(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(CFG, f); f.close()
        try:
            r = subprocess.run([sys.executable, os.path.join(S, "comm_volume.py"),
                                "--config", f.name, "--latency-us", "430"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("120", r.stdout)
            self.assertIn("-33%", r.stdout)
        finally:
            os.unlink(f.name)


class TestTheoreticalOptimum(unittest.TestCase):
    def _csv(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        f.write("Name,Duration(us),Input Shapes\n")
        f.write('aclnnGroupedMatmulX,3.0,"6,512;128,512"\n')
        f.write('aclnnGroupedMatmulX,5.0,"6,512;128,512"\n')
        f.write('SparseFlashMlaY,15.9,"6,8,512"\n')
        f.close(); return f.name

    def _cfg(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(CFG, f); f.close(); return f.name

    def test_runs_and_decomposes(self):
        c, k = self._cfg(), self._csv()
        try:
            r = subprocess.run([sys.executable, os.path.join(S, "theoretical_optimum.py"),
                                "--config", c, "--kernel-details-csv", k,
                                "--measured-step-ms", "62.32"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("51.6 ms", r.stdout)
            self.assertIn("10.7 ms", r.stdout)
            self.assertIn("best demonstrated GEMM", r.stdout)
        finally:
            os.unlink(c); os.unlink(k)

    def test_json_mode(self):
        import theoretical_optimum as to
        c, k = self._cfg(), self._csv()
        try:
            cap = to.best_op_capability(k)
            self.assertIn("gemm", cap)
            self.assertAlmostEqual(cap["gemm"]["best_p10_us"], 3.0)
            self.assertIn("attn", cap)
            mb = to.necessary_memory_bytes(CFG, 8, 6)
            self.assertGreater(mb, 1e9)   # GB-class
            self.assertLess(mb, 1e10)
        finally:
            os.unlink(c); os.unlink(k)


if __name__ == "__main__":
    unittest.main()

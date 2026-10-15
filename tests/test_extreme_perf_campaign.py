"""Tests for the campaign scripts (fusion_headroom, upstream_drift)."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = os.path.join(ROOT, "skills", "extreme-perf-campaign", "scripts")
sys.path.insert(0, S)

import fusion_headroom as fh   # noqa: E402


class TestFusionHeadroom(unittest.TestCase):
    def test_compute_math(self):
        # 120 KB saved/site @ 50 GB/s = 120*1024/50e9 s = 2.458 us/site
        r = fh.compute(sites=60, step_ms=33.0, bytes_saved_kb=120, eff_bw_gbs=50.0)
        self.assertAlmostEqual(r["in_graph_us_per_site"], 2.458, places=2)
        # 60 sites * 2.458us = 147.5us of a 33000us step = 0.45%
        self.assertAlmostEqual(r["end_to_end_pct"], 0.45, places=2)

    def test_worked_example_shape(self):
        # The campaign's real numbers: 60KB tensor, ~60 sites, 33ms step
        r = fh.compute(sites=60, step_ms=33.0, bytes_saved_kb=2 * 60, eff_bw_gbs=50.0)
        self.assertLess(r["end_to_end_pct"], 1.5)   # the honest sub-1.5% band
        self.assertGreater(r["end_to_end_pct"], 0.2)

    def test_higher_batch_pays(self):
        # Same pattern at batch 64: tensors scale, step grows slower -> % up
        small = fh.compute(60, 33.0, 120, 50.0)["end_to_end_pct"]
        big = fh.compute(60, 45.0, 1280, 50.0)["end_to_end_pct"]  # 64x tokens, 1.36x step
        self.assertGreater(big, small * 5)

    def test_cli_stop_verdict(self):
        r = subprocess.run(
            [sys.executable, os.path.join(S, "fusion_headroom.py"),
             "--sites-per-step", "60", "--tensor-kb", "60", "--step-ms", "33"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        self.assertIn("STOP", r.stdout)
        self.assertIn("artifact", r.stdout)
        self.assertIn("DEFAULT stand-in", r.stdout)  # states the bw is not measured

    def test_cli_wire_verdict(self):
        r = subprocess.run(
            [sys.executable, os.path.join(S, "fusion_headroom.py"),
             "--sites-per-step", "600", "--tensor-kb", "640", "--step-ms", "33",
             "--eager-us", "268", "--fused-us", "127"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        self.assertIn("WIRE", r.stdout)

    def test_eager_delta(self):
        self.assertEqual(fh.eager_delta(267.8, 126.8), 141.0)


class TestUpstreamDrift(unittest.TestCase):
    def _repo(self):
        """A tiny fake git repo with two commits after the base date."""
        import shutil
        d = tempfile.mkdtemp(prefix="drift-")
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        def g(*a):
            subprocess.run(["git", "-C", d, *a], capture_output=True, env=env)
        g("init", "-q")
        os.makedirs(os.path.join(d, "pkg"))
        open(os.path.join(d, "pkg", "a.py"), "w").write("def engram_host(): pass\n")
        g("add", "-A"); g("commit", "-qm", "old base")
        open(os.path.join(d, "pkg", "b.py"), "w").write("x = 1\n")
        g("add", "-A"); g("commit", "-qm", "[Performance] new fused kernel")
        # Make both commits "recent" so --since catches the second one only.
        subprocess.run(["git", "-C", d, "commit", "--amend", "--no-edit",
                        "--date=now", "--quiet"], capture_output=True, env=env)
        return d

    def test_clone_mode_counts_and_overlap(self):
        d = self._repo()
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(S, "upstream_drift.py"),
                 "--repo", d, "--base-date", "2000-01-01",
                 "--keywords", "engram_host,nonexistent_thing",
                 "--model-filter", "fused"],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("UPSTREAM HAS IT", r.stdout)   # engram_host hits a.py
            self.assertIn("still yours to carry", r.stdout)  # nonexistent misses
            self.assertIn("Price the rebase", r.stdout)
        finally:
            import shutil; shutil.rmtree(d, ignore_errors=True)

    def test_missing_args_exits(self):
        r = subprocess.run(
            [sys.executable, os.path.join(S, "upstream_drift.py"),
             "--base-date", "2026-09-10"],
            capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--repo", r.stderr)


if __name__ == "__main__":
    unittest.main()

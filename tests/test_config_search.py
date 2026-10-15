"""Tests for the config-search planner.

Calibration fixtures use the REAL DeepSeek-V4.1-Flash W4A8 numbers measured
on 8×910B3 (2026-10-15): hbm 60.96 GiB, non_kv 43.49 GiB, KV pool 13.60 GiB
= 2,942,071 tokens at GPU_UTIL 0.92. The naive linear model misses the
anchor by 7% (12.59 vs 13.60 GiB), which is why the offset exists — the
tests pin that behaviour.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "serving-config-search", "scripts")
sys.path.insert(0, SCRIPTS)

import plan_config_search as pcs  # noqa: E402

# Real measured values (see module docstring).
CAL = {
    "hbm_gib": 60.96, "non_kv_gib": 43.49,
    "base_util": 0.92, "kv_gib": 13.60, "kv_tokens": 2942071,
}
BASE = {"GPU_UTIL": 0.92, "BAT_TOKENS": 8192, "MAX_SEQS": 4, "PREFIX": 0}


def search(**over):
    d = {
        "base": BASE,
        "axes": {"GPU_UTIL": [0.88, 0.90, 0.92, 0.94]},
        "constraints": [],
        "calibration": CAL,
        "sla": {"request_len": 131072, "min_concurrent": 8},
        "restart_cost_min": 15,
    }
    d.update(over)
    return d


class TestCapacityModel(unittest.TestCase):
    def setUp(self):
        self.m = pcs.CapacityModel(CAL)

    def test_offset_is_calibrated(self):
        # naive = 60.96*0.92 - 43.49 = 12.59; measured 13.60 -> offset ~ +1.01
        self.assertAlmostEqual(self.m.offset, 1.01, places=1)

    def test_anchor_is_exact(self):
        # With the offset the model reproduces the anchor exactly by design.
        self.assertAlmostEqual(self.m.anchor_kv_gib, 13.60, places=2)
        p = self.m.predict(0.92)
        self.assertAlmostEqual(p["kv_gib"], 13.60, places=2)

    def test_known_extrapolations(self):
        # Validated by hand against the real numbers (±1 token from float order).
        for util, want_tokens in ((0.88, 2414575), (0.90, 2678323), (0.94, 3205819)):
            got = self.m.predict(util)["kv_tokens"]
            self.assertAlmostEqual(got, want_tokens, delta=2)

    def test_unfit_is_reported(self):
        # Valid calibration, but a util too low to hold weights: the anchor
        # (13.60 GiB @ 0.92) is fine, predict() at 0.65 goes negative.
        # (A calibration whose own anchor is negative is refused at
        # construction — that is garbage input, tested below.)
        m = pcs.CapacityModel(CAL)
        p = m.predict(0.65)
        self.assertFalse(p["fits"])
        self.assertEqual(p["kv_tokens"], 0)

    def test_garbage_calibration_refused(self):
        cal = {k: v for k, v in CAL.items() if k != "kv_gib"}
        cal["hbm_gib"] = 45.0  # naive anchor negative and nothing to offset it
        with self.assertRaises(SystemExit):
            pcs.CapacityModel(cal)

    def test_uncalibrated_flag(self):
        m = pcs.CapacityModel({k: v for k, v in CAL.items() if k != "kv_gib"})
        self.assertTrue(m.uncalibrated)
        self.assertEqual(m.offset, 0.0)


class TestPlanning(unittest.TestCase):
    def test_baseline_marked_not_run(self):
        p = pcs.plan(search())
        base = p["trials"][0]
        self.assertEqual(base["status"], "baseline")

    def test_sla_prune_with_arithmetic(self):
        # 1M-context SLA: at util 0.88, ~2.41M tokens / 131072 = 18 -> feasible
        p = pcs.plan(search(sla={"request_len": 131072, "min_concurrent": 8}))
        pruned = [t for t in p["trials"] if t["status"] == "pruned"]
        self.assertFalse(pruned)
        # ...but requiring 30 concurrent at 1M context prunes everything
        p = pcs.plan(search(sla={"request_len": 1048576, "min_concurrent": 30}))
        pruned = [t for t in p["trials"] if t["status"] == "pruned"]
        self.assertTrue(pruned)
        self.assertTrue(all("predicted concurrency" in t["prune_reason"]
                            for t in pruned))

    def test_forbidden_region(self):
        cons = [{"type": "forbidden", "when": {"GPU_UTIL": 0.94},
                 "reason": "activation headroom collapse (measured 6x prefill slowdown)"}]
        p = pcs.plan(search(constraints=cons))
        t94 = next(t for t in p["trials"] if t["config"]["GPU_UTIL"] == 0.94)
        self.assertEqual(t94["status"], "pruned")
        self.assertIn("forbidden region", t94["prune_reason"])
        self.assertIn("6x", t94["prune_reason"])

    def test_pair_constraint(self):
        s = search(axes={"GPU_UTIL": [0.90], "KV32_REPACK": [True]})
        s["base"]["KV32_REPACK"] = False
        s["constraints"] = [{"type": "pair", "knobs": ["KV32_REPACK", "BAT_TOKENS"]}]
        p = pcs.plan(s)
        moved_alone = [t for t in p["trials"]
                       if t["status"] == "pruned" and "pair violation" in t["prune_reason"]]
        self.assertTrue(moved_alone)

    def test_accuracy_coupled_marked_not_pruned(self):
        s = search(axes={"GPU_UTIL": [0.90], "BAT_TOKENS": [2048]})
        s["constraints"] = [{"type": "accuracy_coupled", "knobs": ["BAT_TOKENS"]}]
        p = pcs.plan(s)
        t = next(t for t in p["trials"] if t["config"].get("BAT_TOKENS") == 2048)
        self.assertEqual(t["status"], "run")
        self.assertTrue(t["accuracy_coupled"])

    def test_ordering_nearest_first(self):
        s = search(axes={"GPU_UTIL": [0.90], "MAX_SEQS": [32]})
        p = pcs.plan(s)
        runners = [t for t in p["trials"] if t["status"] == "run"]
        sizes = [len(t["knobs_changed"]) for t in runners]
        self.assertEqual(sizes, sorted(sizes))

    def test_budget_reported(self):
        p = pcs.plan(search())
        self.assertEqual(p["planned_trials"], 3)  # 0.88/0.90/0.94 (0.92 is baseline)
        self.assertAlmostEqual(p["budget_min"], 45.0)


class TestCli(unittest.TestCase):
    def _write(self, d):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(d, f)
        f.close()
        return f.name

    def test_runs_and_renders(self):
        import subprocess
        p = self._write(search())
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "plan_config_search.py"), p],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("capacity model", r.stdout)
            self.assertIn("216,328 tokens/GiB", r.stdout)  # real calibration
            self.assertIn("budget", r.stdout)
        finally:
            os.unlink(p)

    def test_out_file(self):
        import subprocess
        src = self._write(search())
        out = os.path.join(tempfile.mkdtemp(), "trials.json")
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "plan_config_search.py"),
                 src, "--out", out],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            d = json.load(open(out))
            self.assertIn("trials", d)
            self.assertIn("budget_min", d)
        finally:
            os.unlink(src)

    def test_missing_key_exits(self):
        p = self._write({"base": {}})
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "plan_config_search.py"), p],
                capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing required key", r.stderr)
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()

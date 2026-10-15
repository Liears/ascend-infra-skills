"""Tests for the accuracy gate verdict tool."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "llm-serving-accuracy-gate", "scripts")
sys.path.insert(0, SCRIPTS)

import check_accuracy_gate as cag  # noqa: E402


def write(obj):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


FULL_PROTOCOL = {
    "harness": "acc_eval.py @ abc123",
    "dataset": "gsm8k test, 200 problems, 8-shot",
    "sampling": "chat mode, conc 4, effort 75, max_tokens 512",
    "draft": "off",
}


class TestL1b(unittest.TestCase):
    def test_pass_on_small_drop(self):
        r = cag.gate_l1b({**FULL_PROTOCOL, "correct": 197, "total": 200}, 198, 200)
        self.assertEqual(r.verdict, "PASS")
        self.assertEqual(r.drop_vs_baseline, 1)

    def test_fail_on_three_drop(self):
        r = cag.gate_l1b({**FULL_PROTOCOL, "correct": 195, "total": 200}, 198, 200)
        self.assertEqual(r.verdict, "FAIL")
        self.assertEqual(r.drop_vs_baseline, 3)

    def test_noise_band_is_one_question(self):
        # 196 = drop 2 -> still PASS per the rule (>2 fails)
        r = cag.gate_l1b({**FULL_PROTOCOL, "correct": 196, "total": 200}, 198, 200)
        self.assertEqual(r.verdict, "PASS")

    def test_unrecorded_protocol_refuses_to_judge(self):
        r = cag.gate_l1b({"correct": 198, "total": 200}, 198, 200)
        self.assertEqual(r.verdict, "UNRECORDED")
        self.assertTrue(any("protocol not recorded" in x for x in r.reasons))

    def test_denominator_mismatch_refuses(self):
        r = cag.gate_l1b({**FULL_PROTOCOL, "correct": 99, "total": 100}, 198, 200)
        self.assertEqual(r.verdict, "UNRECORDED")

    def test_multi_run_judged_on_worst(self):
        d = {**FULL_PROTOCOL,
             "runs": [{"correct": 198, "total": 200},
                      {"correct": 195, "total": 200}]}
        r = cag.gate_l1b(d, 198, 200)
        self.assertEqual(r.verdict, "FAIL")
        self.assertEqual(r.correct, 195)


class TestL2(unittest.TestCase):
    def _runs(self, pcts, total=30):
        return {**FULL_PROTOCOL,
                "runs": [{"correct": round(p * total / 100), "total": total}
                         for p in pcts]}

    def test_k_below_4_fails(self):
        r = cag.gate_l2(self._runs([90, 93]), "")
        self.assertEqual(r.verdict, "FAIL")
        self.assertTrue(any("k=2" in x for x in r.reasons))

    def test_mean_and_spread_reported(self):
        # counts are quantized to whole problems: 90%->27, 93.3%->28, 86.7%->26
        r = cag.gate_l2(self._runs([90, 93.3, 86.7, 90]), "")
        self.assertEqual(r.verdict, "PASS")
        self.assertAlmostEqual(r.mean_pct, 90.0, places=1)
        self.assertAlmostEqual(r.spread_pct, 6.67, places=1)
        self.assertEqual(r.k, 4)

    def test_each_problem_worth_stated(self):
        r = cag.gate_l2(self._runs([90, 93.3, 86.7, 90]), "")
        self.assertTrue(any("3.3" in x for x in r.reasons))

    def test_wide_spread_flagged_but_not_failed(self):
        r = cag.gate_l2(self._runs([80, 93.3, 86.7, 100]), "")
        self.assertEqual(r.verdict, "PASS")
        self.assertTrue(any("spread" in x and "caution" in x for x in r.reasons))

    def test_unrecorded_protocol_refuses(self):
        d = {"runs": [{"correct": 27, "total": 30}] * 4}
        r = cag.gate_l2(d, "")
        self.assertEqual(r.verdict, "UNRECORDED")

    def test_mixed_totals_refuse(self):
        d = {**FULL_PROTOCOL,
             "runs": [{"correct": 27, "total": 30},
                      {"correct": 90, "total": 100}]}
        r = cag.gate_l2(d, "")
        self.assertEqual(r.verdict, "UNRECORDED")


class TestCli(unittest.TestCase):
    def test_l1b_pass_exit_zero(self):
        p = write({**FULL_PROTOCOL, "correct": 198, "total": 200})
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "check_accuracy_gate.py"),
                 p, "--gate", "L1b", "--baseline", "198/200"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("PASS", r.stdout)
            self.assertIn("harness", r.stdout)  # evidence block present
        finally:
            os.unlink(p)

    def test_l1b_fail_exit_one(self):
        p = write({**FULL_PROTOCOL, "correct": 190, "total": 200})
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "check_accuracy_gate.py"),
                 p, "--gate", "L1b", "--baseline", "198/200"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 1)
            self.assertIn("FAIL", r.stdout)
        finally:
            os.unlink(p)

    def test_unrecorded_exit_one(self):
        p = write({"correct": 198, "total": 200})
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "check_accuracy_gate.py"),
                 p, "--gate", "L1b", "--baseline", "198/200"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 1)
            self.assertIn("UNRECORDED", r.stdout)
            self.assertIn("refuses to judge", r.stdout)
        finally:
            os.unlink(p)

    def test_l2_needs_k(self):
        p = write({**FULL_PROTOCOL, "runs": [{"correct": 28, "total": 30}]})
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "check_accuracy_gate.py"),
                 p, "--gate", "L2", "--official", "94-100 @ effort 25-100"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 1)
            self.assertIn("k=1", r.stdout)
        finally:
            os.unlink(p)

    def test_bad_baseline(self):
        p = write({**FULL_PROTOCOL, "correct": 198, "total": 200})
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "check_accuracy_gate.py"),
                 p, "--gate", "L1b", "--baseline", "not-a-ratio"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 2)
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()

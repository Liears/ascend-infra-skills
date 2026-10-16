"""Tests for upstream-pr-contribution (check_touch)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = os.path.join(ROOT, "skills", "upstream-pr-contribution", "scripts")


class TestCheckTouch(unittest.TestCase):
    def _repo(self):
        d = tempfile.mkdtemp(prefix="ct-")
        (Path(d) / "vllm_ascend" / "spec_decode").mkdir(parents=True)
        (Path(d) / "vllm_ascend" / "spec_decode" / "dspark_proposer.py").write_text("x = 1\n")
        ut = Path(d) / "tests" / "ut" / "spec_decode"
        ut.mkdir(parents=True)
        (ut / "test_dspark_proposer.py").write_text("def test_x(): pass\n")
        (ut / "test_unrelated_vision.py").write_text("def test_y(): pass\n")
        return d

    def test_finds_contract_tests(self):
        d = self._repo()
        r = subprocess.run(
            [sys.executable, os.path.join(S, "check_touch.py"), d,
             "vllm_ascend/spec_decode/dspark_proposer.py"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("test_dspark_proposer.py", r.stdout)
        self.assertIn("contract:", r.stdout)
        self.assertIn("update the behavior tests", r.stdout)  # checklist adapts

    def test_no_tests_no_contract_line(self):
        d = self._repo()
        (Path(d) / "vllm_ascend" / "attention").mkdir(parents=True)
        (Path(d) / "vllm_ascend" / "attention" / "brand_new_thing.py").write_text("y = 2\n")
        r = subprocess.run(
            [sys.executable, os.path.join(S, "check_touch.py"), d,
             "vllm_ascend/attention/brand_new_thing.py"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no existing unit tests", r.stdout)
        self.assertNotIn("update the behavior tests", r.stdout)

    def test_checklist_mentions_dco_and_base(self):
        d = self._repo()
        r = subprocess.run(
            [sys.executable, os.path.join(S, "check_touch.py"), d,
             "vllm_ascend/spec_decode/dspark_proposer.py"],
            capture_output=True, text=True, timeout=60)
        for needle in ("today's upstream main", "--amend -s", "default-off",
                       "root causes", "verbatim"):
            self.assertIn(needle, r.stdout)


if __name__ == "__main__":
    unittest.main()

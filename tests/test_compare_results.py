"""Tests for the benchmark comparison guard."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "llm-serving-auto-benchmark", "scripts")
sys.path.insert(0, SCRIPTS)

import compare_results as cr  # noqa: E402


def row(**kw):
    base = dict(
        framework="vllm-ascend", model="M", chip="910B3", cards=8, devices=8,
        toolchain="CANN 8.1.RC1", in_len=1024, out_len=256, concurrency=1,
        prefix_cache=0, graph_mode="eager", hccl="HCCL",
        tok_s=100.0, tok_s_basis="gen_tokens/wall",
    )
    base.update(kw)
    return cr.Row(**base)


class TestValidate(unittest.TestCase):
    def test_clean_row_has_no_problems(self):
        problems, untrusted = cr.validate([row()])
        self.assertEqual(problems, [])
        self.assertEqual(untrusted, set())

    def test_missing_required_field(self):
        r = row()
        r.tok_s = None
        problems, untrusted = cr.validate([r])
        self.assertTrue(any("missing" in p for p in problems))
        self.assertIn(id(r), untrusted)

    def test_devices_less_than_cards(self):
        r = row(devices=4, cards=8)
        problems, untrusted = cr.validate([r])
        self.assertTrue(any("devices" in p and "cards" in p for p in problems))

    def test_cards_blank_is_flagged(self):
        r = row(cards=None)
        problems, _ = cr.validate([r])
        self.assertTrue(any("cards blank" in p for p in problems))

    def test_bad_tok_s_basis(self):
        r = row(tok_s_basis="")
        problems, untrusted = cr.validate([r])
        self.assertTrue(any("tok_s_basis" in p for p in problems))
        self.assertIn(id(r), untrusted)

    def test_spec_decode_without_accept_len(self):
        r = row(spec_decode=1, accept_len=None)
        problems, _ = cr.validate([r])
        self.assertTrue(any("accept_len" in p for p in problems))


class TestGrouping(unittest.TestCase):
    def test_same_key_groups(self):
        rows = [row(framework="a", graph_mode="eager"),
                row(framework="b", graph_mode="eager")]
        self.assertEqual(len(cr.group(rows)), 1)

    def test_different_graph_mode_splits(self):
        rows = [row(framework="a", graph_mode="eager"),
                row(framework="b", graph_mode="aclgraph")]
        self.assertEqual(len(cr.group(rows)), 2)

    def test_different_concurrency_splits(self):
        rows = [row(framework="a", concurrency=1), row(framework="b", concurrency=8)]
        self.assertEqual(len(cr.group(rows)), 2)


class TestRenderExcludesUntrusted(unittest.TestCase):
    def test_flagged_row_cannot_win(self):
        good = row(framework="good", tok_s=100.0)
        bad = row(framework="bad", tok_s=999.0, tok_s_basis="")
        problems, untrusted = cr.validate([good, bad])
        out = cr.render([good, bad], problems, untrusted)
        self.assertIn("Highest tok/s among trusted rows: **good**", out)
        self.assertNotIn("**bad**", out.split("Highest")[1])

    def test_no_trusted_row_is_stated(self):
        # Two flagged rows: they form a group, but with no trusted member the
        # renderer must say so instead of naming a winner.
        a = row(framework="a", tok_s=1.0, tok_s_basis="")
        b = row(framework="b", tok_s=2.0, tok_s_basis="")
        problems, untrusted = cr.validate([a, b])
        out = cr.render([a, b], problems, untrusted)
        self.assertIn("No trusted row", out)

    def test_lone_flagged_row_is_marked_untrusted(self):
        bad = row(framework="only", tok_s=1.0, tok_s_basis="")
        problems, untrusted = cr.validate([bad])
        out = cr.render([bad], problems, untrusted)
        self.assertIn("flagged, untrusted", out)


class TestCli(unittest.TestCase):
    def _write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(rows, f)
        f.close()
        return f.name

    def test_clean_exit_zero(self):
        p = self._write([{**cr.asdict(row(framework="a"))}, {**cr.asdict(row(framework="b"))}])
        try:
            r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "compare_results.py"), p],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
        finally:
            os.unlink(p)

    def test_dirty_exit_one(self):
        r0 = cr.asdict(row(framework="a"))
        r0["tok_s_basis"] = ""
        p = self._write([r0])
        try:
            r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "compare_results.py"), p],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 1)
        finally:
            os.unlink(p)

    def test_template_is_valid_json(self):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "compare_results.py"), "--template"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        json.loads(r.stdout)


if __name__ == "__main__":
    unittest.main()

"""Tests for the compute simulator, including the shared-expert sizing trap."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "ascend-compute-simulation", "scripts")
sys.path.insert(0, SCRIPTS)

import model_compute_sim as mcs  # noqa: E402


DENSE = {
    "model_type": "dense",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "vocab_size": 32000,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
}

MOE = {
    "model_type": "moe_only",
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "vocab_size": 129280,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "q_lora_rank": 1280,
    "kv_lora_rank": 512,
    "n_routed_experts": 384,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2304,
    "n_shared_experts": 1,
    # note: NO top-level intermediate_size — the trap
}

WRAPPED = {"model_type": "vlm", "text_config": DENSE, "vision_config": {}}


def write(cfg):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(cfg, f)
    f.close()
    return f.name


class TestNormalize(unittest.TestCase):
    def test_unwraps_text_config(self):
        got = mcs.normalize_config(WRAPPED)
        self.assertEqual(got["hidden_size"], 4096)

    def test_plain_passthrough(self):
        got = mcs.normalize_config(DENSE)
        self.assertEqual(got["hidden_size"], 4096)


class TestDense(unittest.TestCase):
    def setUp(self):
        self.ops, self.facts = mcs.build_ops(DENSE, 1, 1)

    def test_has_projection_ops(self):
        names = {o.name for o in self.ops}
        self.assertIn("attn.q_proj", names)
        self.assertIn("attn.o_proj", names)
        self.assertIn("ffn.gate_up", names)

    def test_gate_up_flops(self):
        op = next(o for o in self.ops if o.name == "ffn.gate_up")
        # 2 * T * H * 2I = 2 * 1 * 4096 * 22016
        self.assertEqual(op.flops, 2 * 1 * 4096 * 22016)

    def test_no_moe_ops(self):
        names = {o.name for o in self.ops}
        self.assertNotIn("moe.router", names)


class TestMoe(unittest.TestCase):
    def setUp(self):
        self.ops, self.facts = mcs.build_ops(MOE, 1, 1)

    def test_mla_ops_present(self):
        names = {o.name for o in self.ops}
        self.assertIn("attn.q_a_proj", names)
        self.assertIn("attn.kv_b_proj", names)

    def test_shared_experts_nonzero_without_top_level_intermediate(self):
        # The bug this guards: a config with only moe_intermediate_size must
        # still produce non-zero shared-expert FLOPs.
        shared = [o for o in self.ops if o.name.startswith("ffn.shared")]
        self.assertTrue(shared, "no shared-expert ops produced")
        self.assertTrue(all(o.flops > 0 for o in shared),
                        "shared experts have zero FLOPs — sizing fell back to 0")

    def test_topk_applied(self):
        gu = next(o for o in self.ops if o.name == "ffn.experts.gate_up")
        # 2 * T * H * 2*I * topk
        self.assertEqual(gu.flops, 2 * 1 * 5120 * 2 * 2304 * 6)


class TestSummarize(unittest.TestCase):
    def test_lm_head_counted_once(self):
        ops, facts = mcs.build_ops(DENSE, 1, 1)
        r = mcs.summarize(DENSE, ops, facts, 1, 1, None, None, "", 1, 1)
        lm = next(o for o in r.ops if o.name == "lm_head")
        # total = per_layer*layers + lm_head, and lm_head must appear once
        per_layer = sum(o.flops for o in ops)
        self.assertAlmostEqual(r.flops_total, per_layer * 32 + lm.flops, places=0)

    def test_mfu_computed_only_with_peak(self):
        ops, facts = mcs.build_ops(DENSE, 1, 1)
        r = mcs.summarize(DENSE, ops, facts, 1, 1, 10.0, 313.0, "vendor", 1, 1)
        self.assertIsNotNone(r.mfu_pct)

    def test_no_peak_no_mfu(self):
        ops, facts = mcs.build_ops(DENSE, 1, 1)
        r = mcs.summarize(DENSE, ops, facts, 1, 1, 10.0, None, "", 1, 1)
        self.assertIsNone(r.mfu_pct)
        self.assertTrue(any("no --peak-tflops" in w for w in r.warnings))

    def test_tp_does_not_divide(self):
        ops, facts = mcs.build_ops(DENSE, 1, 1)
        base = mcs.summarize(DENSE, ops, facts, 1, 1, None, None, "", 1, 1)
        tp = mcs.summarize(DENSE, ops, facts, 1, 1, None, None, "", 8, 8)
        self.assertEqual(base.flops_total, tp.flops_total)
        self.assertTrue(any("NOT applied" in w for w in tp.warnings))

    def test_moe_upper_bound_warning(self):
        ops, facts = mcs.build_ops(MOE, 1, 1)
        r = mcs.summarize(MOE, ops, facts, 1, 1, None, None, "", 1, 1)
        self.assertTrue(any("upper bound" in w for w in r.warnings))


class TestCli(unittest.TestCase):
    def test_runs_on_dense(self):
        import subprocess
        p = write(DENSE)
        try:
            r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "model_compute_sim.py"),
                                "--config", p, "--batch", "1", "--seq", "1"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("total FLOPs", r.stdout)
        finally:
            os.unlink(p)

    def test_json_mode(self):
        import subprocess
        p = write(MOE)
        try:
            r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "model_compute_sim.py"),
                                "--config", p, "--json"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            d = json.loads(r.stdout)
            self.assertIn("flops_total", d)
        finally:
            os.unlink(p)

    def test_bad_config(self):
        import subprocess
        p = write({"not": "a model"})
        try:
            r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "model_compute_sim.py"),
                                "--config", p],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 2)
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()

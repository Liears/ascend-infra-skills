"""Tests for the read-only repo audit — especially the size-cue and version guards."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "ascend-open-source-finder", "scripts")
sys.path.insert(0, SCRIPTS)

import audit_repo as ar  # noqa: E402


def make_repo(files: dict[str, str]) -> str:
    root = tempfile.mkdtemp(prefix="audit-test-")
    for rel, content in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
    return root


class TestDeclaredWeightSize(unittest.TestCase):
    def test_size_with_cue_is_taken(self):
        line = "模型权重（273 GB，见 §3.1）"
        self.assertEqual(ar._declared_weight_size(line), "273 GB")

    def test_english_cue(self):
        line = "Download the weights (~212.6 GB) with modelscope."
        self.assertEqual(ar._declared_weight_size(line), "212.6 GB")

    def test_size_without_cue_is_rejected(self):
        # A KV-cache or HBM number near a model reference must NOT be reported
        # as the weight size. This is the bug the same-line cue exists to stop.
        line = "The HBM is 64 GB and the model runs there."
        self.assertEqual(ar._declared_weight_size(line), "")

    def test_no_number(self):
        self.assertEqual(ar._declared_weight_size("weight page"), "")


class TestAuditWeights(unittest.TestCase):
    def test_same_line_grounding(self):
        root = make_repo({
            "README.md": (
                "Weights: modelscope download chiro2001/Model-w4a8-Ascend --local-dir x  # 273 GB\n"
                "Unrelated: HBM is 64 GB.\n"
            )
        })
        ws = ar.audit_weights(root)
        found = [w for w in ws if "Model-w4a8-Ascend" in w["id"]]
        self.assertTrue(found)
        self.assertEqual(found[0]["declared_size"], "273 GB")

    def test_unknown_size_is_blank_not_zero(self):
        root = make_repo({"README.md": "See https://huggingface.co/org/model-x\n"})
        ws = ar.audit_weights(root)
        self.assertTrue(ws)
        self.assertEqual(ws[0]["declared_size"], "")


class TestAuditPatches(unittest.TestCase):
    def test_classifies_by_upstream(self):
        patch = (
            "diff --git a/vllm_ascend/core/x.py b/vllm_ascend/core/x.py\n"
            "--- a/vllm_ascend/core/x.py\n+++ b/vllm_ascend/core/x.py\n"
            "@@ -1 +1 @@\n-a\n+b\n"
        )
        root = make_repo({"patches/vllm-ascend/0001-x.patch": patch})
        res = ar.audit_patches(root)
        self.assertEqual(res["total"], 1)
        self.assertIn("vllm_ascend", res["by_upstream"])

    def test_empty(self):
        root = make_repo({"README.md": "nothing"})
        self.assertEqual(ar.audit_patches(root)["total"], 0)


class TestAuditStack(unittest.TestCase):
    def test_multiple_versions_are_all_kept(self):
        root = make_repo({
            "a.md": "Built on CANN 9.1.0.\n",
            "b.md": "Also works with CANN 9.2.0.\n",
        })
        res = ar.audit_stack(root)
        self.assertIn("cann", res)
        versions = res["cann"]["versions"]
        self.assertIn("9.1.0", versions)
        self.assertIn("9.2.0", versions)

    def test_single_version(self):
        root = make_repo({"a.md": "torch_npu 2.10.0.post4\n"})
        res = ar.audit_stack(root)
        self.assertEqual(list(res["torch_npu"]["versions"]), ["2.10.0.post4"])


class TestClassify(unittest.TestCase):
    def test_no_patches_is_a_red_flag(self):
        root = make_repo({"README.md": "Supports Ascend."})
        a = ar.Audit(root=root)
        a.patches = ar.audit_patches(root)
        a.scripts = ar.audit_scripts(root)
        a.evidence_docs = ar.audit_evidence(root)
        a.weights = ar.audit_weights(root)
        a.stack_pins = ar.audit_stack(root)
        a.tests = ar.audit_tests(root)
        ar.classify(a)
        self.assertTrue(any("no patch files" in r for r in a.red_flags))

    def test_ambiguous_toolchain_is_flagged(self):
        root = make_repo({
            "a.md": "CANN 9.1.0\n",
            "b.md": "CANN 9.2.0\n",
            "c.md": "vllm 0.27.1\n",
            "p/x.patch": "diff --git a/vllm/x.py b/vllm/x.py\n",
            "EXPECTED_PERF.md": "2026-10-01 result\n",
        })
        a = ar.Audit(root=root)
        a.patches = ar.audit_patches(root)
        a.scripts = ar.audit_scripts(root)
        a.evidence_docs = ar.audit_evidence(root)
        a.weights = ar.audit_weights(root)
        a.stack_pins = ar.audit_stack(root)
        a.tests = ar.audit_tests(root)
        ar.classify(a)
        self.assertTrue(any("ambiguous toolchain" in r for r in a.red_flags))


class TestCliOnFixture(unittest.TestCase):
    def test_runs_and_reports(self):
        import subprocess
        root = make_repo({
            "README.md": "model weights 273 GB\n",
            "patches/vllm-ascend/0001.patch": "diff --git a/vllm_ascend/a.py b/vllm_ascend/a.py\n",
            "EXPECTED_PERF.md": "2026-10-01\n",
            "scripts/serve_a2.sh": "#!/bin/bash\necho hi\n",
        })
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "audit_repo.py"), root],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("## Patches", r.stdout)
        self.assertIn("## Verdict", r.stdout)


if __name__ == "__main__":
    unittest.main()

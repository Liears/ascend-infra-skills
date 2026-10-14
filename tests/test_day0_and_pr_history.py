"""Tests for the Day-0 plan/bundle tools and the PR-history miner."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DAY0 = os.path.join(ROOT, "skills", "model-day0-support", "scripts")
PRH = os.path.join(ROOT, "skills", "model-pr-history-knowledge", "scripts")
sys.path.insert(0, DAY0)
sys.path.insert(0, PRH)

import plan_template as pt          # noqa: E402
import check_day0_bundle as cdb     # noqa: E402
import mine_pr_history as mpr       # noqa: E402


class TestPlanTemplate(unittest.TestCase):
    def test_renders_all_tasks(self):
        out = pt.render("ModelX", "910B3", "vllm-ascend")
        for t in pt.TEMPLATE:
            self.assertIn(t["id"], out)
            self.assertIn(t["title"], out)

    def test_gates_present(self):
        out = pt.render("ModelX", "910B3", "vllm-ascend")
        for g in pt.GATES:
            self.assertIn(g["id"], out)

    def test_cli_json(self):
        r = subprocess.run([sys.executable, os.path.join(DAY0, "plan_template.py"),
                            "--model", "M", "--chip", "910B3", "--json"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data["model"], "M")
        self.assertIn("tasks", data)


class TestBundleCheck(unittest.TestCase):
    def _bundle(self, files):
        root = tempfile.mkdtemp(prefix="bundle-")
        for rel, content in files.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
        return root

    def test_empty_bundle_fails_everything(self):
        root = self._bundle({"README.md": "nothing useful"})
        rep = cdb.check(root)
        self.assertFalse(rep.ok)
        self.assertEqual(len(rep.passes), 0)

    def test_complete_bundle_passes(self):
        root = self._bundle({
            "serve.sh": "#!/bin/bash\nvllm serve /model --port 8000\n",
            "stack.md": "Built on CANN 9.1.0 with torch_npu 2.10.0.post4.\n",
            "acc.md": "GSM8K accuracy: 198/200 on the real path.\n",
            "prof/kernel_details.csv": "Name,Duration(us)\naclnnX,1\n",
            "mem.md": "Available KV cache memory: 27.43 GiB, GPU KV cache size: 4,010,585 tokens\n",
            "limits.md": "## Known Limitations\n- long context only on 8 cards\n",
            "weights.md": "Download: modelscope download org/Model-w4a8 --local-dir x\n",
        })
        rep = cdb.check(root)
        self.assertTrue(rep.ok, [r.key for r in rep.failures])

    def test_limitations_section_detected(self):
        root = self._bundle({"a.md": "## Known Issues\n- foo\n"})
        rep = cdb.check(root)
        self.assertTrue(any(r.key == "limitations" for r in rep.passes))

    def test_cli_exit_codes(self):
        good = self._bundle({
            "serve.sh": "vllm serve /m\n",
            "stack.md": "CANN 9.1.0 torch_npu 2.10.0\n",
            "acc.md": "accuracy 198/200\n",
            "kernel_details.csv": "Name\na\n",
            "mem.md": "kv cache size: 1000 tokens\n",
            "lim.md": "## Limitations\n- x\n",
            "w.md": "https://huggingface.co/o/m\n",
        })
        r = subprocess.run([sys.executable, os.path.join(DAY0, "check_day0_bundle.py"), good],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

        bad = self._bundle({"x.md": "hi"})
        r = subprocess.run([sys.executable, os.path.join(DAY0, "check_day0_bundle.py"), bad],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1)


class TestClassifyCommit(unittest.TestCase):
    def test_perf(self):
        self.assertEqual(mpr.classify("[Performance][DSpark] Stack projections (#17263)"), "perf")

    def test_bugfix(self):
        self.assertEqual(mpr.classify("[BugFix] Validate TP limits (#18418)"), "bugfix")

    def test_feat(self):
        self.assertEqual(mpr.classify("[Feature][MRV2] Support DSA (#14762)"), "feat")

    def test_doc(self):
        self.assertEqual(mpr.classify("[Doc][Misc] Update links (#15093)"), "doc")

    def test_misc_fallback(self):
        self.assertEqual(mpr.classify("random subject"), "misc")


class TestMinerOnFixtureRepo(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="gitrepo-")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        self.env = env
        self._git("init", "-q")
        self._commit("initial", {"README.md": "hi\n"})
        self._commit("[Performance] optimize deepseek attention (#123)",
                     {"vllm_ascend/attention/dsa_v1.py": "x = 1\n"})
        self._commit("[Doc] update everything (#124)",
                     {"docs/deepseek_v4.md": "doc\n"})

    def _git(self, *args):
        return subprocess.run(["git", "-C", self.root, *args],
                              capture_output=True, text=True, env=self.env, timeout=30).stdout

    def _commit(self, msg, files):
        for rel, content in files.items():
            p = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(p) or self.root, exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
        self._git("add", "-A")
        self._git("commit", "-q", "-m", msg)

    def test_mines_matching_commits(self):
        commits = mpr.mine(self.root, "deepseek", 50)
        subjects = [c.subject for c in commits]
        self.assertTrue(any("optimize deepseek" in s for s in subjects))
        self.assertTrue(any("update everything" in s for s in subjects))

    def test_pr_number_extracted(self):
        commits = mpr.mine(self.root, "deepseek", 50)
        prs = {c.pr for c in commits}
        self.assertIn("123", prs)

    def test_render_has_anchor(self):
        commits = mpr.mine(self.root, "deepseek", 50)
        out = mpr.render(commits, "deepseek", self.root)
        self.assertIn("repo HEAD", out)

    def test_cli_with_out_creates_dir(self):
        out = os.path.join(tempfile.mkdtemp(), "nested", "dir", "d.md")
        r = subprocess.run([sys.executable, os.path.join(PRH, "mine_pr_history.py"),
                            "--repo", self.root, "--model", "deepseek", "--out", out],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.isfile(out))


if __name__ == "__main__":
    unittest.main()

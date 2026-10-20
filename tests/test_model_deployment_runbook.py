"""Tests for model-deployment-runbook (check_deploy_doc)."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(ROOT, "skills", "model-deployment-runbook", "scripts", "check_deploy_doc.py")

GOOD = """# 部署书
## 权重获取
下载到 <WEIGHTS> 后 sha256 校验。
## 镜像
从官方基底 quay.io/ascend/vllm-ascend:v0.30.0rc2 构建 <IMAGE_TAG>。
## 起容器
docker run ... <NAME> ... -v $C/vllm:/root/.cache/vllm ...
## 起服
vllm serve --host 0.0.0.0 127.0.0.1 探针。
## 验收
门禁基线 100/100。
"""

BAD = """# 部署书
## 权重获取
权重在 /mnt/local/models/foo。TBD 检查。
## 镜像
## 起容器
## 起服
## 验收
"""


class TestCheckDeployDoc(unittest.TestCase):
    def _run(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            path = f.name
        try:
            r = subprocess.run([sys.executable, SCRIPT, path], capture_output=True, text=True, timeout=60)
            return r
        finally:
            os.unlink(path)

    def test_good_doc_passes(self):
        r = self._run(GOOD)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("PASS", r.stdout)

    def test_hardcoded_and_tbd_fail(self):
        r = self._run(BAD)
        self.assertEqual(r.returncode, 1)
        self.assertIn("/mnt/local", r.stdout)
        self.assertIn("TBD", r.stdout)

    def test_container_internal_mount_allowed(self):
        r = self._run(GOOD)
        self.assertNotIn("/root", r.stdout)

    def test_zero_host_allowed_but_real_ip_flagged(self):
        doc = GOOD + "\n连接 192.168.1.5 使用。\n"
        r = self._run(doc)
        self.assertEqual(r.returncode, 1)
        self.assertIn("192.168.1.5", r.stdout)

    def test_pr_without_action_fails(self):
        """'Use PR #1234' is not actionable when the PR is not merged."""
        doc = GOOD + "\n镜像上加 PR #1234 即可。\n"
        r = self._run(doc)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no way to obtain its code", r.stdout)

    def test_pr_with_cherry_pick_passes(self):
        doc = GOOD + "\n镜像上加 PR #1234：\ngit fetch https://example/repo.git pull/1234/head && git cherry-pick -n FETCH_HEAD\n"
        r = self._run(doc)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_pr_stated_merged_passes(self):
        doc = GOOD + "\nPR #1234 已合入上游 main，直接 pull 即可。\n"
        r = self._run(doc)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_untagged_image_fails(self):
        doc = GOOD + "\ndocker pull quay.io/ascend/vllm-ascend\n"
        r = self._run(doc)
        self.assertEqual(r.returncode, 1)
        self.assertIn("without a tag or digest", r.stdout)

    def test_floating_tag_fails(self):
        doc = GOOD + "\ndocker pull quay.io/ascend/vllm-ascend:latest\n"
        r = self._run(doc)
        self.assertEqual(r.returncode, 1)
        self.assertIn("floating image tag", r.stdout)

    def test_pinned_digest_passes(self):
        doc = GOOD + "\nquay.io/ascend/vllm-ascend@sha256:" + "a" * 64 + "\n"
        r = self._run(doc)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_worked_example_passes(self):
        ex = os.path.join(ROOT, "examples/deepseek-v4.1-flash-910b/06-onboard/DEPLOY-1017.md")
        if os.path.exists(ex):
            r = subprocess.run([sys.executable, SCRIPT, ex], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout)


if __name__ == "__main__":
    unittest.main()

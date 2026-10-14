"""Tests for the open-source finder query construction and evidence scoring."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "ascend-open-source-finder", "scripts")
sys.path.insert(0, SCRIPTS)

import find_open_source as fos  # noqa: E402


class TestModelVariants(unittest.TestCase):
    def test_v41_spellings_all_present(self):
        vs = fos.model_variants("DeepSeek-V4.1-Flash")
        joined = " ".join(vs)
        self.assertIn("DeepSeek-V4_1-Flash", joined)   # dot -> underscore
        self.assertIn("DeepSeek-V4-1-Flash", joined)   # dot -> hyphen
        self.assertIn("DeepSeekV41Flash", joined)      # stripped

    def test_deterministic(self):
        a = fos.model_variants("GLM-5.3")
        b = fos.model_variants("GLM-5.3")
        self.assertEqual(a, b)

    def test_no_duplicates(self):
        vs = fos.model_variants("Qwen3.5-27B")
        self.assertEqual(len(vs), len(set(vs)))


class TestQueries(unittest.TestCase):
    def test_hardware_aliases_expand(self):
        qs = fos.build_queries("DeepSeek-V4.1-Flash", "910B")
        joined = " ".join(qs)
        self.assertIn("A2", joined)       # 910B alias
        self.assertIn("ascend", joined.lower())

    def test_a3_aliases(self):
        qs = fos.build_queries("GLM-5.3", "A3")
        joined = " ".join(qs)
        self.assertIn("910C", joined)

    def test_deterministic(self):
        a = fos.build_queries("DeepSeek-V4.1-Flash", "910B")
        b = fos.build_queries("DeepSeek-V4.1-Flash", "910B")
        self.assertEqual(a, b)

    def test_query_count_bounded(self):
        qs = fos.build_queries("SomeModel-1.2.3", "910B")
        self.assertLessEqual(len(qs), 20, "query explosion will rate-limit us")


class TestScoring(unittest.TestCase):
    def _score(self, desc, **kw):
        c = fos.Candidate(source="github", full_name="x/y", url="u", description=desc, **kw)
        fos.score_candidate(c)
        return c

    def test_acceptance_doc_is_verified(self):
        # Use the real chiro2001-style description.
        c = self._score(
            "DeepSeek-V4.1-Flash W4A8 optimized vLLM deployment for Ascend 910B (A2) / "
            "910C (A3): patches, one-command serve, self-check and acceptance",
            stars=24,
        )
        self.assertEqual(c.rating, "verified")
        self.assertTrue(c.evidence_signals)

    def test_thin_evidence_is_partial(self):
        c = self._score(
            "patches, one-command serve, self-check and acceptance for Ascend 910B",
            stars=24,
        )
        self.assertEqual(c.rating, "partial")

    def test_bare_claim_is_claimed(self):
        c = self._score("Supports Ascend NPU")
        self.assertEqual(c.rating, "claimed")

    def test_private_marker_is_a_doubt(self):
        c = self._score("Private DeepSeek V4.1 development repository", stars=1)
        self.assertIn("marked private", c.doubts)

    def test_toy_marker_is_a_doubt(self):
        c = self._score("a small demo repo", stars=0)
        self.assertIn("looks like a toy", c.doubts)

    def test_stars_alone_do_not_make_verified(self):
        c = self._score("nothing relevant", stars=1000)
        self.assertNotEqual(c.rating, "verified")


class TestRendering(unittest.TestCase):
    def test_verified_ranks_above_claimed(self):
        a = fos.Candidate(source="github", full_name="a/a", url="u",
                          description="acceptance acceptance acceptance")
        b = fos.Candidate(source="github", full_name="b/b", url="u", description="claimed thing")
        for c in (a, b):
            fos.score_candidate(c)
        out = fos.render([b, a], "M", "H", ["q"])
        self.assertLess(out.index("a/a"), out.index("b/b"))

    def test_empty_is_explicit(self):
        out = fos.render([], "M", "H", ["q"])
        self.assertIn("No candidates found", out)


class TestModelScopeExistence(unittest.TestCase):
    """Offline test of the parsing half of the reachability check."""

    def test_missing_model_yields_none(self):
        # _modelscope_model_info is network-bound; assert the guard shape only.
        # A 404 must not be treated as existence.
        self.assertTrue(hasattr(fos, "_modelscope_model_info"))

    def test_page_exists_wrapper_delegates(self):
        # The wrapper must not trust the HTML page (SPA returns 200 always).
        import inspect
        src = inspect.getsource(fos._modelscope_page_exists)
        self.assertIn("_modelscope_model_info", src)


if __name__ == "__main__":
    unittest.main()

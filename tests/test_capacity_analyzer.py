"""Tests for the capacity analyzer. Uses the exact log strings from the source."""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "llm-serving-capacity-planner", "scripts")
sys.path.insert(0, SCRIPTS)

import capacity_analyzer as ca  # noqa: E402

# A log built from the exact format strings in vllm_ascend/worker/worker.py
# and vllm/v1/core/kv_cache_utils.py.
REAL_LOG = """INFO 10-14 15:52:01 [worker.py:472] Free memory on device (62.31/64.0 GiB) on startup. Desired GPU memory utilization is (0.92, 58.88 GiB). Actual usage: 24.15 GiB for weights, 1.82 GiB for peak activation, 3.41 GiB for non-torch memory, 2.07 GiB for NPU graph memory. Replace gpu_memory_utilization with `--kv-cache-memory=26518185984` (24.70 GiB) to fit into requested memory, or `--kv-cache-memory=31024021504` (28.89 GiB) to fully utilize NPU free memory. Current KV cache memory: 27.43 GiB.
INFO 10-14 15:52:01 [worker.py:594] Available KV cache memory: 27.43 GiB
INFO 10-14 15:52:03 [kv_cache_utils.py:1744] GPU KV cache size: 4,010,585 tokens
INFO 10-14 15:52:03 [kv_cache_utils.py:1746] Maximum concurrency for 1,048,576 tokens per request: 3.82x
"""


class TestParse(unittest.TestCase):
    def setUp(self):
        self.b = ca.parse_log(REAL_LOG)

    def test_free_and_total(self):
        self.assertEqual(self.b.free_on_startup, 62.31)
        self.assertEqual(self.b.total_on_startup, 64.0)

    def test_utilization_and_requested(self):
        self.assertEqual(self.b.gpu_memory_utilization, 0.92)
        self.assertEqual(self.b.requested_memory, 58.88)

    def test_four_component_usage(self):
        self.assertEqual(self.b.weights, 24.15)
        self.assertEqual(self.b.peak_activation, 1.82)
        self.assertEqual(self.b.non_torch_memory, 3.41)
        self.assertEqual(self.b.npugraph_memory, 2.07)

    def test_non_kv_subtotal(self):
        self.assertAlmostEqual(self.b.non_kv_total(), 31.45, places=2)

    def test_available_kv(self):
        self.assertEqual(self.b.available_kv_cache, 27.43)

    def test_kv_tokens(self):
        self.assertEqual(self.b.kv_cache_size_tokens, 4010585)

    def test_max_concurrency(self):
        self.assertEqual(self.b.max_concurrency, 3.82)
        self.assertEqual(self.b.max_model_len, 1048576)

    def test_suggestion_is_not_treated_as_setting(self):
        # The REAL_LOG contains only a vLLM *suggestion* ("Replace
        # gpu_memory_utilization with `--kv-cache-memory=N`"); the flag was never
        # passed. Reporting it as an effective setting produces a false
        # diagnosis, so the parser must ignore suggestions.
        self.assertIsNone(self.b.kv_cache_memory_bytes_arg)

    def test_explicit_kv_arg_is_taken(self):
        b = ca.parse_log("--kv-cache-memory 26518185984 --port 8100\n")
        self.assertEqual(b.kv_cache_memory_bytes_arg, 26518185984)

    def test_explicit_kv_arg_equals_form(self):
        b = ca.parse_log("serving with --kv-cache-memory=13351176806\n")
        self.assertEqual(b.kv_cache_memory_bytes_arg, 13351176806)


class TestEdgeCases(unittest.TestCase):
    def test_empty_log(self):
        b = ca.parse_log("")
        self.assertIsNone(b.weights)
        self.assertIsNone(b.available_kv_cache)

    def test_unrelated_text(self):
        b = ca.parse_log("hello\nworld\n")
        self.assertIsNone(b.total_on_startup)

    def test_short_form_available_kv(self):
        b = ca.parse_log("Available KV cache memory: 12.50 GiB\n")
        self.assertEqual(b.available_kv_cache, 12.5)

    def test_initial_free_form(self):
        # The explicit-budget path, where profiling is skipped.
        b = ca.parse_log(
            "Initial free memory 62.31 GiB, reserved 27.43 GiB for KV Cache "
            "as specified by kv_cache_memory_bytes, skipping memory profiling.\n"
        )
        self.assertEqual(b.available_kv_cache, 27.43)

    def test_startup_oom_line_still_parses_free_total(self):
        b = ca.parse_log(
            "Free memory on device (61.20/64.0 GiB) on startup is less than desired "
            "GPU memory utilization (0.98, 62.72 GiB).\n"
        )
        self.assertEqual(b.free_on_startup, 61.2)
        self.assertEqual(b.total_on_startup, 64.0)


class TestConcurrency(unittest.TestCase):
    def setUp(self):
        self.b = ca.parse_log(REAL_LOG)

    def test_at_reference_length_uses_reported(self):
        est = ca.estimate_concurrency(self.b, [1048576])[0]
        self.assertAlmostEqual(est.concurrent_requests, 3.82, places=2)
        self.assertIn("reported directly", est.basis)

    def test_short_request_is_bounded_by_pool(self):
        est = ca.estimate_concurrency(self.b, [8192])[0]
        # pool 4,010,585 / 8192 = 489.6 ; scaled 3.82 * 128 = 489.0 -> min 489.0
        self.assertLessEqual(est.concurrent_requests, 489.6)
        self.assertGreater(est.concurrent_requests, 480)

    def test_no_kv_tokens_gives_none(self):
        b = ca.parse_log("Free memory on device (62.31/64.0 GiB) on startup\n")
        est = ca.estimate_concurrency(b, [1024])
        self.assertIsNone(est[0].concurrent_requests)


class TestDiagnosis(unittest.TestCase):
    def test_flags_npugraph_memory(self):
        b = ca.parse_log(REAL_LOG)
        notes = ca.diagnose(b)
        self.assertTrue(any("NPU graph memory" in n for n in notes))

    def test_no_false_kv_budget_diagnosis_from_suggestion(self):
        # REAL_LOG only *suggests* the flag; the diagnosis must not claim it was set.
        notes = ca.diagnose(ca.parse_log(REAL_LOG))
        self.assertFalse(any("kv-cache-memory" in n for n in notes))

    def test_flags_explicit_kv_budget_when_really_set(self):
        # A real log has the memory lines AND the effective flag.
        b = ca.parse_log(
            "Free memory on device (62.31/64.0 GiB) on startup\n"
            "Available KV cache memory: 12.43 GiB\n"
            "--kv-cache-memory=13351176806\n"
        )
        notes = ca.diagnose(b)
        self.assertTrue(any("kv-cache-memory" in n for n in notes))

    def test_unparseable_log(self):
        notes = ca.diagnose(ca.parse_log("nothing here"))
        self.assertTrue(any("no memory lines matched" in n for n in notes))


class TestCli(unittest.TestCase):
    def test_cli_json(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(REAL_LOG)
            path = f.name
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "capacity_analyzer.py"), path, "--json"],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            import json
            data = json.loads(r.stdout)
            self.assertEqual(data["breakdown"]["kv_cache_size_tokens"], 4010585)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

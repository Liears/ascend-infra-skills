"""Tests for the Ascend trace analyzer, using a real-shape synthetic trace."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "skills", "npu-profiler-analysis", "scripts")
sys.path.insert(0, SCRIPTS)

import analyze_profile as ap  # noqa: E402

KERNEL_CSV_HEADER = (
    "Device_id,Model ID,Task ID,Stream ID,Name,Type,OP State,Accelerator Core,"
    "Start Time(us),Duration(us),Wait Time(us),Block Num\n"
)


def make_prof(tmp: str, with_trace=True) -> str:
    out = os.path.join(tmp, "ASCEND_PROFILER_OUTPUT")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "kernel_details.csv"), "w") as f:
        f.write(KERNEL_CSV_HEADER)
        f.write('0,0,1,47,aclnnAddmm_MatMulCommon_MatMulV2,MatMulV2,dynamic,AI_CORE,100,8.96,0,16\n')
        f.write('0,0,2,47,aclnnAddmm_MatMulCommon_MatMulV2,MatMulV2,dynamic,AI_CORE,200,9.0,0,16\n')
        f.write('0,0,3,47,aclnnReduceSum_ReduceSumOpAiCore_ReduceSum,ReduceSum,dynamic,MIX_AIV,300,40.0,0,1\n')
        f.write('0,0,4,47,aclnnAdd_AddAiCore_Add,Add,dynamic,AI_VECTOR_CORE,400,5.0,0,1\n')
    if with_trace:
        events = [
            {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "host"}},
            {"ph": "X", "name": "AscendCL@aclrtLaunchKernelWithHostArgs", "pid": 1, "tid": 100,
             "ts": 0, "dur": 30.0, "args": {}},
            {"ph": "X", "name": "AscendCL@aclnnAddmm", "pid": 1, "tid": 100, "ts": 1, "dur": 10.0, "args": {}},
            {"ph": "X", "name": "aclnnAddmm_MatMulCommon_MatMulV2", "pid": 99, "tid": 47,
             "ts": 0, "dur": 8.96, "args": {"level": "kernel"}},
            {"ph": "X", "name": "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum", "pid": 99, "tid": 47,
             "ts": 10, "dur": 40.0, "args": {"level": "kernel"}},
        ]
        with open(os.path.join(out, "trace_view.json"), "w") as f:
            json.dump(events, f)
    return tmp


class TestOpType(unittest.TestCase):
    def test_matmul(self):
        self.assertEqual(ap.op_type_of("aclnnAddmm_MatMulCommon_MatMulV2"), "MatMul")

    def test_reducesum(self):
        self.assertEqual(ap.op_type_of("aclnnReduceSum_ReduceSumOpAiCore_ReduceSum"), "ReduceSum")

    def test_add(self):
        self.assertEqual(ap.op_type_of("aclnnAdd_AddAiCore_Add"), "Add")

    def test_unknown_falls_back_to_longest_token(self):
        got = ap.op_type_of("Something_Weird_Entirely")
        self.assertTrue(got)


class TestFindOutputDir(unittest.TestCase):
    def test_finds_nested(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(os.path.join(t, "root", "host_123_456_ascend_pt"))
            found = ap.find_output_dir(t)
            self.assertIsNotNone(found)
            self.assertTrue(found.endswith("ASCEND_PROFILER_OUTPUT"))

    def test_finds_direct(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(t)
            self.assertIsNotNone(ap.find_output_dir(os.path.join(t, "ASCEND_PROFILER_OUTPUT")))

    def test_none_when_absent(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(ap.find_output_dir(t))


class TestKernelDetails(unittest.TestCase):
    def test_columns_detected_as_reduced(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(t)
            rows, meta = ap.read_kernel_details(os.path.join(t, "ASCEND_PROFILER_OUTPUT"))
            self.assertTrue(meta["present"])
            self.assertLess(meta["columns"], 40)
            self.assertIn("reduced", meta["level"])
            self.assertEqual(len(rows), 4)

    def test_attribution_ranks_by_time(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(t)
            rows, _ = ap.read_kernel_details(os.path.join(t, "ASCEND_PROFILER_OUTPUT"))
            top = ap.attribute_kernels(rows, 10)
            self.assertEqual(top[0]["op_type"], "ReduceSum")   # 40us > 17.96us
            self.assertEqual(top[0]["count"], 1)


class TestTrace(unittest.TestCase):
    def test_host_device_split(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(t)
            tf = ap.read_trace(os.path.join(t, "ASCEND_PROFILER_OUTPUT"))
            self.assertEqual(tf.host_calls, 2)          # 2 AscendCL@ host events
            self.assertEqual(tf.device_kernels, 2)      # 2 device kernels
            self.assertAlmostEqual(tf.device_us, 48.96, places=2)
            self.assertAlmostEqual(tf.host_api_us, 40.0, places=2)

    def test_missing_trace_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(t, with_trace=False)
            tf = ap.read_trace(os.path.join(t, "ASCEND_PROFILER_OUTPUT"))
            self.assertEqual(tf.total_events, 0)


class TestAnalyze(unittest.TestCase):
    def test_full_run(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(t)
            a = ap.analyze(t, top=5)
            self.assertTrue(a.top_kernels)
            self.assertTrue(a.core_split)
            self.assertTrue(a.findings)

    def test_reduced_columns_produces_a_finding(self):
        with tempfile.TemporaryDirectory() as t:
            make_prof(t)
            a = ap.analyze(t, top=5)
            self.assertTrue(any("pipe breakdown" in f for f in a.findings))

    def test_missing_path(self):
        with tempfile.TemporaryDirectory() as t:
            a = ap.analyze(t, top=5)
            self.assertTrue(any("no trace_view.json" in f for f in a.findings))


COMM_CSV_HEADER = (
    "Device_id,Model ID,Task ID,Stream ID,Name,Type,OP State,Accelerator Core,"
    "Start Time(us),Duration(us),Wait Time(us),Block Num\n"
)


class TestCommunicationDominance(unittest.TestCase):
    """Real V4.1-Flash traces are ~83% collective time; the analyzer must say so."""

    def _prof(self, tmp):
        out = os.path.join(tmp, "ASCEND_PROFILER_OUTPUT")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "kernel_details.csv"), "w") as f:
            f.write(COMM_CSV_HEADER)
            # communication-heavy: 90% collective
            f.write('0,0,1,47,allReduce_x,Reduce,dynamic,COMMUNICATION,1,900.0,0,1\n')
            f.write('0,0,2,47,aclnnMatMul,MatMul,dynamic,AI_CORE,2,100.0,0,1\n')
        return out

    def test_comm_dominance_finding(self):
        with tempfile.TemporaryDirectory() as t:
            self._prof(t)
            a = ap.analyze(t, top=10)
            self.assertTrue(any("communication dominates" in f for f in a.findings))
            self.assertTrue(any("hottest *compute* op" in f for f in a.findings))
            self.assertGreaterEqual(a.core_split.get("COMMUNICATION", 0), 80)

    def test_compute_op_excludes_collectives(self):
        with tempfile.TemporaryDirectory() as t:
            self._prof(t)
            a = ap.analyze(t, top=10)
            line = next(f for f in a.findings if "hottest *compute* op" in f)
            self.assertIn("MatMul", line)
            self.assertNotIn("allReduce", line)


class TestHostDeviceRatioText(unittest.TestCase):
    def _prof_with_events(self, tmp, host_n, dev_n):
        out = os.path.join(tmp, "ASCEND_PROFILER_OUTPUT")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "kernel_details.csv"), "w") as f:
            f.write(COMM_CSV_HEADER)
            f.write('0,0,1,47,aclnnMatMul,MatMul,dynamic,AI_CORE,1,10.0,0,1\n')
        events = ([{"ph": "X", "name": "AscendCL@x", "pid": 1, "tid": 2, "ts": i, "dur": 1.0, "args": {}}
                   for i in range(host_n)]
                  + [{"ph": "X", "name": "aclnnK", "pid": 9, "tid": 47, "ts": i, "dur": 1.0, "args": {}}
                     for i in range(dev_n)])
        with open(os.path.join(out, "trace_view.json"), "w") as f:
            json.dump(events, f)
        return out

    def test_ratio_text_matches_the_number(self):
        with tempfile.TemporaryDirectory() as t:
            self._prof_with_events(t, host_n=4, dev_n=8)
            a = ap.analyze(t, top=5)
            line = next(f for f in a.findings if "host-API events vs" in f)
            self.assertIn("2.00:1", line)
            self.assertNotIn("ratio near 1", line)

    def test_low_ratio_says_overlap_poor(self):
        with tempfile.TemporaryDirectory() as t:
            self._prof_with_events(t, host_n=10, dev_n=10)
            a = ap.analyze(t, top=5)
            line = next(f for f in a.findings if "host-API events vs" in f)
            self.assertIn("1.00:1", line)
            self.assertIn("near 1:1", line)

    def test_high_ratio_describes_batching(self):
        with tempfile.TemporaryDirectory() as t:
            self._prof_with_events(t, host_n=2, dev_n=20)
            a = ap.analyze(t, top=5)
            line = next(f for f in a.findings if "host-API events vs" in f)
            self.assertIn("10.00:1", line)
            self.assertIn("graph mode", line)


if __name__ == "__main__":
    unittest.main()

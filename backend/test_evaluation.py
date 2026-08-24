import os
import tempfile
import unittest

from backend.evaluation import compare_reports, evaluate_events, write_report


class EvaluationTests(unittest.TestCase):
    def test_reports_false_merge_and_handoff_trace(self):
        truth = [
            {"event_id": "a1", "person_id": "A", "camera": "cam1"},
            {"event_id": "a2", "person_id": "A", "camera": "cam2", "handoff": True, "expected_gid": 1},
            {"event_id": "b1", "person_id": "B", "camera": "cam2"},
        ]
        report = evaluate_events(truth, [
            {"event_id": "a1", "global_id": 1}, {"event_id": "a2", "global_id": 1},
            {"event_id": "b1", "global_id": 1},
        ], {"model": "synthetic", "checkpoint": "none"})
        self.assertEqual(1, report["counts"]["false_merges"])
        self.assertEqual(1.0, report["metrics"]["handoff_accuracy"])
        self.assertFalse(compare_reports(evaluate_events(truth, []), report)["passed"])

    def test_report_is_machine_readable(self):
        report = evaluate_events([], [], {"config_version": "test"})
        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            write_report(report, path)
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

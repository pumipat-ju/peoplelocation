import os
import tempfile
import unittest

from backend.evaluation import (
    compare_reports,
    evaluate_events,
    format_summary,
    write_report,
)


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
        self.assertEqual(
            1.0,
            report["metrics"]["project_specific"]["handoff_accuracy"],
        )
        self.assertEqual({}, report["metrics"]["standard"])
        self.assertNotIn("idf1", report["metrics"]["project_specific"])
        self.assertFalse(compare_reports(evaluate_events(truth, []), report)["passed"])

    def test_temporal_switches_are_counted_from_event_order(self):
        truth = [
            {"event_id": "a1", "person_id": "A", "camera": "cam1", "event_time": 1.0},
            {"event_id": "a2", "person_id": "A", "camera": "cam1", "event_time": 2.0},
            {"event_id": "a3", "person_id": "A", "camera": "cam2", "event_time": 3.0, "handoff": True},
        ]
        report = evaluate_events(truth, [
            {"event_id": "a1", "global_id": 10},
            {"event_id": "a2", "global_id": 20},
            {"event_id": "a3", "global_id": 10},
        ])

        self.assertEqual(1, report["counts"]["false_splits"])
        self.assertEqual(2, report["counts"]["temporal_id_switches"])
        self.assertEqual(2, report["counts"]["temporal_prediction_transitions"])
        self.assertEqual(
            1.0,
            report["metrics"]["project_specific"]["temporal_id_switch_rate"],
        )
        self.assertEqual(0.0, report["metrics"]["project_specific"]["handoff_accuracy"])

    def test_handoff_can_use_previous_gid_without_numeric_gid_assumption(self):
        truth = [
            {"event_id": "a1", "person_id": "A", "camera": "cam1", "event_time": 1.0},
            {"event_id": "a2", "person_id": "A", "camera": "cam2", "event_time": 2.0, "handoff": True},
        ]
        report = evaluate_events(truth, [
            {"event_id": "a1", "global_id": "person-a"},
            {"event_id": "a2", "global_id": "person-a"},
        ], {"scenario": "string-gid-handoff"})

        self.assertEqual(1, report["counts"]["correct_handoffs"])
        self.assertIn("standard IDF1/HOTA are not reported", format_summary(report))

    def test_report_is_machine_readable(self):
        report = evaluate_events([], [], {"config_version": "test"})
        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            write_report(report, path)
            self.assertGreater(os.path.getsize(path), 0)
            with open(path, "r", encoding="utf-8") as saved:
                payload = saved.read()
            self.assertIn("peoplelocation-event-evaluation-v2", payload)
            self.assertNotIn('"idf1"', payload)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

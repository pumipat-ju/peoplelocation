"""Safe deadline regression runner for production Re-ID behavior."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import unittest

from backend.evaluation import evaluate_events, format_summary, write_report


WORKFLOW_VERSION = "peoplelocation-deadline-regression-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = (
    REPOSITORY_ROOT / "backend" / "fixtures" / "deadline_event_sequences.json"
)
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "deadline_regression_report.json"

SAFE_TEST_TARGETS = (
    "backend.test_deadline_regression_guards",
    "backend.test_global_presence_handoff",
    "backend.test_global_multicamera_batch_matching",
    "backend.test_production_global_assignment",
    "backend.test_identity_state_machine",
    "backend.test_identity_store",
    "backend.test_identity_persistence_config",
    "backend.test_tracklet_quality_gallery",
    "backend.test_timestamp_offsets",
    "backend.test_topology_travel_time",
    "backend.test_threshold_safety",
    "backend.test_evaluation",
    "backend.test_live_camera",
    "backend.test_per_camera_tracker",
)

PROTECTED_RUNTIME_PATHS = (
    REPOSITORY_ROOT / "static",
    REPOSITORY_ROOT / "backend" / "camera_topology.json",
    REPOSITORY_ROOT / "backend" / "identity_memory.sqlite3",
    REPOSITORY_ROOT / "labeled_data",
)

SCENARIO_TEST_COVERAGE = {
    "multi_camera_one_to_one": (
        "backend.test_global_multicamera_batch_matching."
        "GlobalMultiCameraBatchMatchingTests."
        "test_global_hungarian_prevents_per_camera_greedy_identity_conflict"
    ),
    "safe_unmatched_identity": (
        "backend.test_production_global_assignment."
        "TrustedEvidenceGlobalAssignmentTests."
        "test_two_rows_cannot_both_claim_the_only_existing_gid"
    ),
    "allowed_handoff": (
        "backend.test_global_presence_handoff."
        "GlobalPresenceHandoffTests."
        "test_non_overlap_handoff_deactivates_source_and_preserves_identity"
    ),
    "return_handoff": (
        "backend.test_global_presence_handoff."
        "GlobalPresenceHandoffTests."
        "test_return_handoff_recovers_same_gid_with_new_local_id"
    ),
    "dormant_recovery_and_expiry": (
        "backend.test_identity_state_machine.IdentityStateMachineTests"
    ),
    "restart_persistence": "backend.test_identity_store.IdentityStoreTests",
    "tracklet_gallery_safety": (
        "backend.test_tracklet_quality_gallery.TrackletQualityGalleryTests"
    ),
    "ambiguous_match": (
        "backend.test_global_multicamera_batch_matching."
        "GlobalMultiCameraBatchMatchingTests."
        "test_near_equal_candidates_are_deferred_instead_of_forced"
    ),
    "topology_and_time": "backend.test_topology_travel_time",
    "uploaded_offset": (
        "backend.test_deadline_regression_guards."
        "DeadlineProductionPathTests."
        "test_uploaded_offset_reaches_backend_event_time_logic_with_fake_capture"
    ),
    "missing_camera_nonblocking": (
        "backend.test_production_global_assignment."
        "GlobalAssignmentCoordinatorTests."
        "test_missing_camera_does_not_block_present_camera"
    ),
    "mocked_live_camera_boundary": "backend.test_live_camera",
}


def _fingerprint(path):
    relative = str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    if not path.exists():
        return {
            "path": relative,
            "kind": "missing",
            "exists": False,
            "sha256": None,
            "file_count": 0,
            "size": 0,
        }
    if path.is_file():
        content = path.read_bytes()
        return {
            "path": relative,
            "kind": "file",
            "exists": True,
            "sha256": hashlib.sha256(content).hexdigest(),
            "file_count": 1,
            "size": len(content),
        }

    manifest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        child_path = str(child.relative_to(path)).replace("\\", "/")
        manifest.update(child_path.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(str(stat.st_size).encode("ascii"))
        manifest.update(b"\0")
        manifest.update(str(stat.st_mtime_ns).encode("ascii"))
        manifest.update(b"\n")
        file_count += 1
        total_size += stat.st_size
    return {
        "path": relative,
        "kind": "directory",
        "exists": True,
        "sha256": manifest.hexdigest(),
        "file_count": file_count,
        "size": total_size,
    }


def _protected_state():
    return {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _fingerprint(path)
        for path in PROTECTED_RUNTIME_PATHS
    }


def _serialize_test_failure(test, traceback_text):
    return {
        "test": test.id(),
        "traceback": traceback_text,
    }


def _run_safe_tests():
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromNames(SAFE_TEST_TARGETS)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
    return {
        "targets": list(SAFE_TEST_TARGETS),
        "tests_run": result.testsRun,
        "failures": [
            _serialize_test_failure(test, traceback_text)
            for test, traceback_text in result.failures
        ],
        "errors": [
            _serialize_test_failure(test, traceback_text)
            for test, traceback_text in result.errors
        ],
        "skipped": [
            {"test": test.id(), "reason": reason}
            for test, reason in result.skipped
        ],
        "successful": result.wasSuccessful(),
        "runner_summary": output.getvalue().strip().splitlines()[-1],
    }


def _evaluate_fixture(fixture_path, generated_at):
    with fixture_path.open("r", encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    reports = []
    for scenario in fixture.get("scenarios", []):
        reports.append(evaluate_events(
            scenario["ground_truth"],
            scenario["predictions"],
            {
                "scenario": scenario["name"],
                "coverage": scenario.get("coverage", []),
                "fixture_version": fixture.get("fixture_version"),
                "fixture_path": str(
                    fixture_path.relative_to(REPOSITORY_ROOT)
                ).replace("\\", "/"),
                "generated_at": generated_at,
            },
        ))
    return reports


def _runtime_metadata():
    from backend import main

    topology = dict(main.topology_config)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "identity_db_path": ":memory:",
        "reid_enabled_for_tests": False,
        "reid_threshold_safety_mode": main.REID_THRESHOLD_SAFETY_MODE,
        "global_assignment_window_sec": main.GLOBAL_ASSIGNMENT_WINDOW_SEC,
        "reid_max_idle_sec": main.REID_MAX_IDLE_SEC,
        "identity_dormant_ttl_sec": main.IDENTITY_DORMANT_TTL_SEC,
        "topology": {
            "schema_version": main.TOPOLOGY_SCHEMA_VERSION,
            "loaded_version": topology.get("version"),
            "enforce": topology.get("enforce"),
            "transition_count": len(topology.get("transitions", [])),
            "valid": not bool(topology.get("_validation_error")),
            "path": str(
                Path(main.TOPOLOGY_CONFIG_PATH).resolve().relative_to(
                    REPOSITORY_ROOT
                )
            ).replace("\\", "/"),
        },
    }


def build_deadline_report(fixture_path=DEFAULT_FIXTURE_PATH):
    os.environ["IDENTITY_DB_PATH"] = ":memory:"
    os.environ["REID_ENABLED"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    generated_at = datetime.now(timezone.utc).isoformat()
    before = _protected_state()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix="peoplelocation-deadline-regression-"
    ) as runtime_directory:
        try:
            os.chdir(runtime_directory)
            tests = _run_safe_tests()
            runtime = _runtime_metadata()
        finally:
            os.chdir(original_cwd)
    scenarios = _evaluate_fixture(Path(fixture_path), generated_at)
    after = _protected_state()
    protected_unchanged = before == after
    passed = tests["successful"] and protected_unchanged

    return {
        "workflow_version": WORKFLOW_VERSION,
        "generated_at": generated_at,
        "scenario": "deadline_safe_regression",
        "passed": passed,
        "runtime": runtime,
        "test_suite": tests,
        "scenario_test_coverage": SCENARIO_TEST_COVERAGE,
        "evaluation_scenarios": scenarios,
        "frozen_subsystem_guards": {
            "real_device_access": "forbidden_by_mocked-import test",
            "production_database": ":memory:",
            "test_runtime_directory": "isolated temporary directory",
            "protected_paths_before": before,
            "protected_paths_after": after,
            "protected_paths_unchanged": protected_unchanged,
        },
        "readiness": {
            "automated_suite_passed": passed,
            "controlled_manual_demo_ready": passed,
            "real_world_accuracy_measured": False,
        },
    }


def format_deadline_summary(report, output_path):
    tests = report["test_suite"]
    lines = [
        "Deadline regression: " + ("PASS" if report["passed"] else "FAIL"),
        (
            f"Tests: {tests['tests_run']} run, {len(tests['failures'])} failed, "
            f"{len(tests['errors'])} errors, {len(tests['skipped'])} skipped"
        ),
        (
            "Frozen data unchanged: "
            + str(
                report["frozen_subsystem_guards"][
                    "protected_paths_unchanged"
                ]
            )
        ),
    ]
    lines.extend(
        format_summary(scenario)
        for scenario in report["evaluation_scenarios"]
    )
    lines.append(f"JSON report: {output_path}")
    return "\n\n".join(lines)


def main_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Run safe production Re-ID deadline regressions."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    args = parser.parse_args(argv)

    report = build_deadline_report(args.fixture.resolve())
    write_report(report, args.output)
    print(format_deadline_summary(report, args.output))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main_cli())

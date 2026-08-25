"""Deterministic event-level evaluation for cross-camera Global ID association."""

from collections import Counter, defaultdict
import json
from pathlib import Path


EVALUATION_SCHEMA_VERSION = "peoplelocation-event-evaluation-v2"

STANDARD_METRIC_NOTICE = {
    "implemented": [],
    "note": (
        "This deadline evaluator does not claim standard IDF1, HOTA, "
        "Rank-1, or mAP."
    ),
}

PROJECT_METRIC_DEFINITIONS = {
    "handoff_accuracy": (
        "Correct labeled handoffs divided by labeled handoffs. A handoff is "
        "correct when its predicted GID matches expected_gid, or the person's "
        "previous predicted GID when expected_gid is omitted."
    ),
    "false_merge_rate_per_event": (
        "Extra distinct ground-truth people sharing a predicted GID, divided "
        "by the number of labeled events."
    ),
    "false_split_rate_per_event": (
        "Extra predicted GIDs used by each ground-truth person, divided by "
        "the number of labeled events."
    ),
    "temporal_id_switch_rate": (
        "Changes between consecutive non-missing predicted GIDs for the same "
        "person, divided by evaluated consecutive prediction transitions."
    ),
    "event_assignment_coverage": (
        "Labeled events with a predicted GID divided by labeled events."
    ),
    "identity_consistency": (
        "Events assigned to each person's most frequent predicted GID divided "
        "by all labeled events, including missing predictions."
    ),
    "project_association_f1": (
        "Project-specific harmonic score derived from event coverage, false "
        "merges, and false splits. This is not standard IDF1."
    ),
}


def _indexed_events(events, label):
    indexed = []
    seen = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"{label} event {index} must be an object")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"{label} event {index} needs event_id")
        if event_id in seen:
            raise ValueError(f"Duplicate {label} event_id: {event_id}")
        seen.add(event_id)
        indexed.append((index, event))
    return indexed


def _sequence_sort_key(item):
    index, event = item
    event_time = event.get("event_time")
    if isinstance(event_time, (int, float)) and not isinstance(event_time, bool):
        return float(event_time), index
    return float(index), index


def evaluate_events(ground_truth, predictions, metadata=None):
    """Evaluate aligned labeled events without claiming standard MOT metrics."""
    truth_indexed = _indexed_events(ground_truth, "ground-truth")
    predicted_indexed = _indexed_events(predictions, "prediction")
    truth = {event["event_id"]: event for _, event in truth_indexed}
    predicted = {event["event_id"]: event for _, event in predicted_indexed}

    person_gids = defaultdict(list)
    gid_people = defaultdict(set)
    predicted_gid_by_event = {}
    traces = []
    matched = 0

    for _, event in truth_indexed:
        event_id = event["event_id"]
        person_id = event.get("person_id")
        if person_id is None:
            raise ValueError(f"Ground-truth event {event_id} needs person_id")
        prediction = predicted.get(event_id)
        gid = prediction.get("global_id") if prediction else None
        predicted_gid_by_event[event_id] = gid
        if gid is None:
            traces.append({
                "event_id": event_id,
                "person_id": person_id,
                "reason": "missing_prediction",
            })
            continue
        matched += 1
        person_gids[person_id].append(gid)
        gid_people[gid].add(person_id)

    unexpected_predictions = sorted(set(predicted) - set(truth))
    traces.extend(
        {"event_id": event_id, "reason": "unexpected_prediction"}
        for event_id in unexpected_predictions
    )

    person_sequences = defaultdict(list)
    for item in truth_indexed:
        person_sequences[item[1]["person_id"]].append(item)

    handoff_total = 0
    handoff_correct = 0
    temporal_id_switches = 0
    temporal_prediction_transitions = 0
    for person_id, sequence in person_sequences.items():
        previous_gid = None
        for _, event in sorted(sequence, key=_sequence_sort_key):
            event_id = event["event_id"]
            gid = predicted_gid_by_event[event_id]
            if event.get("handoff"):
                handoff_total += 1
                expected_gid = event.get("expected_gid", previous_gid)
                if gid is not None and expected_gid is not None and gid == expected_gid:
                    handoff_correct += 1
                else:
                    traces.append({
                        "event_id": event_id,
                        "person_id": person_id,
                        "reason": "handoff_wrong_gid",
                        "gid": gid,
                        "expected_gid": expected_gid,
                    })
            if gid is None:
                continue
            if previous_gid is not None:
                temporal_prediction_transitions += 1
                if gid != previous_gid:
                    temporal_id_switches += 1
                    traces.append({
                        "event_id": event_id,
                        "person_id": person_id,
                        "reason": "temporal_id_switch",
                        "from_gid": previous_gid,
                        "to_gid": gid,
                    })
            previous_gid = gid

    false_merges = sum(
        max(0, len(people) - 1)
        for people in gid_people.values()
    )
    false_splits = sum(
        max(0, len(set(gids)) - 1)
        for gids in person_gids.values()
    )
    consistent_events = sum(
        max(Counter(gids).values())
        for gids in person_gids.values()
        if gids
    )
    event_count = len(truth)
    missing_predictions = event_count - matched
    association_precision = matched / max(1, matched + false_merges)
    association_recall = matched / max(1, event_count + false_splits)
    project_association_f1 = (
        2 * association_precision * association_recall
        / max(1e-12, association_precision + association_recall)
    )

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metadata": metadata or {},
        "metric_definitions": {
            "standard": STANDARD_METRIC_NOTICE,
            "project_specific": PROJECT_METRIC_DEFINITIONS,
        },
        "counts": {
            "events": event_count,
            "matched": matched,
            "missing_predictions": missing_predictions,
            "unexpected_predictions": len(unexpected_predictions),
            "handoffs": handoff_total,
            "correct_handoffs": handoff_correct,
            "false_merges": false_merges,
            "false_splits": false_splits,
            "temporal_id_switches": temporal_id_switches,
            "temporal_prediction_transitions": temporal_prediction_transitions,
        },
        "metrics": {
            "standard": {},
            "project_specific": {
                "handoff_accuracy": handoff_correct / max(1, handoff_total),
                "false_merge_rate_per_event": false_merges / max(1, event_count),
                "false_split_rate_per_event": false_splits / max(1, event_count),
                "temporal_id_switch_rate": (
                    temporal_id_switches
                    / max(1, temporal_prediction_transitions)
                ),
                "event_assignment_coverage": matched / max(1, event_count),
                "identity_consistency": consistent_events / max(1, event_count),
                "project_association_f1": project_association_f1,
            },
        },
        "traces": traces,
    }


def format_summary(report):
    """Return a short human-readable summary for a machine report."""
    counts = report["counts"]
    metrics = report["metrics"]["project_specific"]
    scenario = report.get("metadata", {}).get("scenario", "unnamed")
    return "\n".join([
        f"Scenario: {scenario}",
        (
            "Events: {events} | assigned: {matched} | handoffs: "
            "{correct}/{total}"
        ).format(
            events=counts["events"],
            matched=counts["matched"],
            correct=counts["correct_handoffs"],
            total=counts["handoffs"],
        ),
        (
            "False merges: {merges} | false splits: {splits} | temporal "
            "ID switches: {switches}"
        ).format(
            merges=counts["false_merges"],
            splits=counts["false_splits"],
            switches=counts["temporal_id_switches"],
        ),
        (
            "Handoff accuracy: {handoff:.3f} | identity consistency: "
            "{consistency:.3f} | coverage: {coverage:.3f}"
        ).format(
            handoff=metrics["handoff_accuracy"],
            consistency=metrics["identity_consistency"],
            coverage=metrics["event_assignment_coverage"],
        ),
        "Metric family: project-specific; standard IDF1/HOTA are not reported.",
    ])


def write_report(report, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)


def compare_reports(baseline, current, max_false_merge_increase=0.0):
    """Conservative regression gate: false merges may not exceed tolerance."""
    baseline_rate = baseline["metrics"]["project_specific"][
        "false_merge_rate_per_event"
    ]
    current_rate = current["metrics"]["project_specific"][
        "false_merge_rate_per_event"
    ]
    delta = current_rate - baseline_rate
    return {
        "passed": delta <= max_false_merge_increase,
        "false_merge_rate_per_event_delta": delta,
    }

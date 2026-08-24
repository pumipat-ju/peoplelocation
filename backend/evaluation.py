"""Deterministic event-level evaluation for cross-camera Global ID association."""

from collections import Counter, defaultdict
import json


def evaluate_events(ground_truth, predictions, metadata=None):
    """Evaluate aligned events with ``event_id``, ``person_id``, ``camera`` and GID.

    Ground-truth uses ``person_id``; predictions use ``global_id``. Missing
    predictions count as false splits, while one predicted GID serving multiple
    people is a false merge.
    """
    truth = {event["event_id"]: event for event in ground_truth}
    predicted = {event["event_id"]: event for event in predictions}
    person_gids, gid_people = defaultdict(set), defaultdict(set)
    matched = 0
    handoff_total = handoff_correct = 0
    traces = []

    for event_id, event in truth.items():
        prediction = predicted.get(event_id)
        gid = prediction.get("global_id") if prediction else None
        if gid is None:
            traces.append({"event_id": event_id, "reason": "missing_prediction"})
            continue
        matched += 1
        person_gids[event["person_id"]].add(gid)
        gid_people[gid].add(event["person_id"])
        if event.get("handoff"):
            handoff_total += 1
            if event.get("expected_gid") == gid:
                handoff_correct += 1
            else:
                traces.append({"event_id": event_id, "reason": "handoff_wrong_gid", "gid": gid})

    false_merges = sum(max(0, len(people) - 1) for people in gid_people.values())
    false_splits = sum(max(0, len(gids) - 1) for gids in person_gids.values())
    id_switches = false_splits
    precision = matched / max(1, matched + false_merges)
    recall = matched / max(1, len(truth) + false_splits)
    idf1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "metadata": metadata or {},
        "counts": {"events": len(truth), "matched": matched, "false_merges": false_merges,
                   "false_splits": false_splits, "id_switches": id_switches},
        "metrics": {"handoff_accuracy": handoff_correct / max(1, handoff_total),
                    "false_merge_rate": false_merges / max(1, len(truth)),
                    "false_split_rate": false_splits / max(1, len(truth)),
                    "idf1": idf1, "association_accuracy": matched / max(1, len(truth))},
        "traces": traces,
    }


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)


def compare_reports(baseline, current, max_false_merge_increase=0.0):
    """Conservative regression gate: merges may never worsen beyond tolerance."""
    delta = current["metrics"]["false_merge_rate"] - baseline["metrics"]["false_merge_rate"]
    return {"passed": delta <= max_false_merge_increase, "false_merge_rate_delta": delta}

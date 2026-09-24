"""Global identity manager extracted from the production orchestration layer.

The manager receives the main runtime dependency namespace through
configure_identity_dependencies so policy and behavior remain unchanged.
"""

from .state_machine import GlobalIDStateMachine

_dependency_namespace = None

def configure_identity_dependencies(namespace):
    global _dependency_namespace
    _dependency_namespace = namespace
    globals().update(namespace)

def _sync_identity_dependencies():
    if _dependency_namespace is not None:
        globals().update(_dependency_namespace)

class GlobalIdentityManager:

    def __getattribute__(self, name):
        if name not in {"__dict__", "__class__", "_identity_dependency_sync"}:
            _sync_identity_dependencies()
        return object.__getattribute__(self, name)

    def __init__(self, identity_store=None):

        # Explicit stores make tests and short-lived model instances isolated.
        # The application singleton below opts into the durable SQLite store.
        self.identity_store = identity_store
        self.state_machine = GlobalIDStateMachine()

        # GID -> identity information
        self.identities = self.identity_store.load_identities() if self.identity_store else {}

        for gid, identity in self.identities.items():
            self._normalize_identity_presence(gid, identity)

        restored_next_gid = max(self.identities, default=0) + 1
        self.next_global_id = (
            max(restored_next_gid, self.identity_store.next_global_id())
            if self.identity_store
            else restored_next_gid
        )

        # (camera, local_track_id) -> GID
        self.local_to_global = {}

        # (camera, local_track_id) -> owner/generation-scoped tracklet record.
        # This is intentionally camera-local evidence, never camera runtime state.
        self.tracklets = {}

        # Occlusion
        self.occlusion_hold = {}

        # Recent same camera
        self.recent_same_cam = []

        # Recent cross camera
        self.recent_cross_cam = []

        # Re-entrant so the downstream coordinator can atomically validate a
        # camera generation and invoke assign_global_batch under one lock.
        self.lock = threading.RLock()

        # Diagnostics for the most recent synchronized multi-camera decision.
        self.last_global_batch_diagnostics = None

        # Lightweight same-camera pairwise swap-correction diagnostics.
        self.pairwise_swap_checks = 0
        self.pairwise_swap_corrections = 0
        self.pairwise_swap_corrected_rows = 0
        self.last_pairwise_swap_diagnostic = None
        self.pairwise_swap_candidate_count = 0
        self.pairwise_swap_rejected_count = 0
        self.pairwise_swap_max_avg_gain = float('-inf')
        self.pairwise_swap_max_row_gain = float('-inf')
        self.pairwise_swap_max_motion_gain = float('-inf')
        self.pairwise_swap_gain_counts = {
            float(level): 0
            for level in PAIRWISE_DIAGNOSTIC_GAIN_LEVELS
        }
        self.pairwise_swap_top_rejected = []
        self.pairwise_snapshot_groups = 0
        self.pairwise_snapshot_missing = 0
        self.pairwise_snapshot_last = None
        self.pairwise_matrix_diagnostics = []
        self.transition_recovery_checks = 0
        self.transition_recovery_candidates = 0
        self.transition_recovery_assignments = 0
        self.transition_recovery_rejections = 0
        self.tracklet_prototype_calls = 0
        self.tracklet_prototype_fallbacks = 0
        self.tracklet_prototype_low_consensus = 0
        self.tracklet_prototype_used = 0
        self.tracklet_prototype_total_samples_seen = 0
        self.last_tracklet_prototype_diagnostic = None
        self.persistent_tracklet_buffers = {}
        self.persistent_tracklet_buffer_updates = 0
        self.persistent_tracklet_buffer_resets = 0
        self.persistent_tracklet_buffer_pruned = 0
        self.last_persistent_tracklet_buffer = None
        self.soft_continuity_rows = 0
        self.soft_continuity_bonus_applied = 0
        self.soft_continuity_switches_blocked = 0
        self.soft_continuity_switches_allowed = 0
        self.soft_continuity_expired = 0
        self.last_soft_continuity = None
        self.soft_continuity_snapshot_hits = 0
        self.soft_continuity_live_hits = 0
        self.soft_continuity_presence_rejects = 0
        self.soft_continuity_generation_rejects = 0
        self.soft_continuity_identity_rejects = 0
        self.soft_continuity_candidates_preserved = 0
        self.soft_continuity_preserve_skips = 0
        self.soft_continuity_preserved_pair_fallbacks = 0
        self.last_preserved_candidate = None
        self.identity_prototype_calls = 0
        self.identity_prototype_fallbacks = 0
        self.merge_guard_checks = 0
        self.merge_guard_rejections = 0
        self.last_merge_guard_diagnostic = None
        self.last_transition_recovery_diagnostic = None
        self.last_global_map_diagnostic = None
        self.recent_lost_local_tracks = {}
        self.transition_recovery_lost_registered = 0
        self.transition_recovery_stale_mapping_checks = 0
        self.transition_recovery_reappeared_checks = 0
        self.delayed_transition_hypotheses = {}
        self.delayed_transition_strong_accepts = 0
        self.delayed_transition_vote_accepts = 0
        self.delayed_transition_vote_pending = 0
        self.delayed_transition_vote_resets = 0
        self.delayed_transition_rejects = 0
        self.last_delayed_transition = None
        self.wrong_gid_escape_state = {}
        self.wrong_gid_escape_checks = 0
        self.wrong_gid_escape_owner_kept = 0
        self.wrong_gid_escape_pending = 0
        self.wrong_gid_escape_confirmed = 0
        self.wrong_gid_escape_strong_immediate = 0
        self.wrong_gid_escape_resets = 0
        self.last_wrong_gid_escape = None


        # (camera, local track, coordinator generation, tracker generation) ->
        # bounded quality-approved evidence.  This is deliberately transient:
        # it owns no GID, gallery, presence, or capture-worker state.
        self.unresolved_cross_camera_handoffs = {}
        self.unresolved_handoff_capacity_drop_count = 0
        self.unresolved_handoff_forensic_trace = []

    @staticmethod
    def _unresolved_handoff_key(cam_name, detection):
        """Return a generation-scoped key for transient handoff evidence."""
        coordinator_generation = detection.get("coordinator_generation")
        tracker_generation = detection.get("camera_generation")
        try:
            coordinator_generation = (
                None
                if coordinator_generation is None
                else int(coordinator_generation)
            )
        except (TypeError, ValueError, OverflowError):
            coordinator_generation = None
        try:
            tracker_generation = (
                None
                if tracker_generation is None
                else int(tracker_generation)
            )
        except (TypeError, ValueError, OverflowError):
            tracker_generation = None
        return (
            str(cam_name),
            int(detection["tid"]),
            coordinator_generation,
            tracker_generation,
        )

    @staticmethod
    def _unresolved_handoff_age(record, event_time):
        first_event_time = record.get("first_event_time")
        if not isinstance(first_event_time, (int, float)):
            return float("inf")
        return max(0.0, float(event_time) - float(first_event_time))

    @staticmethod
    def _unresolved_record_is_valid(key, record):
        if not isinstance(record, dict):
            return False
        samples = record.get("samples")
        return (
            record.get("key") == key
            and isinstance(record.get("first_event_time"), (int, float))
            and isinstance(record.get("last_event_time"), (int, float))
            and isinstance(record.get("sample_count"), int)
            and 0 <= record.get("sample_count", -1)
            <= AMBIGUOUS_HANDOFF_MAX_SAMPLES
            and isinstance(samples, list)
            and len(samples) <= AMBIGUOUS_HANDOFF_MAX_SAMPLES
        )

    def _append_unresolved_handoff_sample(
        self,
        record,
        detection,
        event_time,
    ):
        """Append one quality-approved embedding without touching a gallery."""
        event_time = float(event_time)
        previous_event_time = float(record.get("last_event_time", event_time))
        if event_time < previous_event_time:
            record["last_quality_rejection_reason"] = "event_time_regression"
            return False
        record["last_event_time"] = event_time
        if not detection.get("local_track_confirmed", True):
            record["last_quality_rejection_reason"] = (
                "unconfirmed_local_track"
            )
            return False

        quality_reason = self._gallery_quality_reason(detection)
        if quality_reason is not None:
            record["last_quality_rejection_reason"] = quality_reason
            return False

        embedding, embedding_reason = self._validated_gallery_embedding(
            {"embedding": None, "gallery": []},
            detection.get("emb"),
        )
        if embedding_reason is not None:
            record["last_quality_rejection_reason"] = embedding_reason
            return False

        samples = record["samples"]
        if samples and samples[-1].get("event_time") == event_time:
            return False
        if samples and samples[-1]["emb"].size != embedding.size:
            record["last_quality_rejection_reason"] = (
                "embedding_dimension_mismatch"
            )
            return False

        samples.append({
            "emb": embedding.copy(),
            "event_time": event_time,
            "quality": self._tracklet_quality_score(detection),
        })
        if len(samples) > AMBIGUOUS_HANDOFF_MAX_SAMPLES:
            del samples[:-AMBIGUOUS_HANDOFF_MAX_SAMPLES]
        record["sample_count"] = min(
            record.get("sample_count", 0) + 1,
            AMBIGUOUS_HANDOFF_MAX_SAMPLES,
        )
        record["last_quality_rejection_reason"] = None
        return True

    @staticmethod
    def _unresolved_handoff_prototype(record):
        samples = record.get("samples", [])
        if not samples:
            return None
        try:
            aggregate = np.mean(
                [item["emb"] for item in samples],
                axis=0,
            )
        except (TypeError, ValueError, KeyError):
            return None
        norm = float(np.linalg.norm(aggregate))
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        return np.asarray(aggregate / norm, dtype=np.float32)

    def _unresolved_scoring_detection(self, record, detection):
        prototype = self._unresolved_handoff_prototype(record)
        if prototype is None:
            return detection
        aggregate_detection = dict(detection)
        aggregate_detection["emb"] = prototype
        aggregate_detection["unresolved_sample_count"] = record.get(
            "sample_count",
            0,
        )
        return aggregate_detection

    @staticmethod
    def _unresolved_candidate_diagnostics(candidate_details):
        diagnostics = []
        for item in candidate_details:
            topology = item.get("topology")
            diagnostics.append({
                "gid": item.get("gid"),
                "appearance": item.get("appearance"),
                "score": item.get("score"),
                "motion": item.get("motion"),
                "hard_gate_passed": item.get("hard_gate_passed"),
                "hard_gate_reason": item.get("hard_gate_reason"),
                "score_failure_reason": item.get("score_failure_reason"),
                "topology": (
                    {
                        "source_camera": topology.get("source_camera"),
                        "destination_camera": topology.get(
                            "destination_camera"
                        ),
                        "event_time_delta_sec": topology.get(
                            "event_time_delta_sec"
                        ),
                        "passed": topology.get("passed"),
                        "reason": topology.get("reason"),
                    }
                    if isinstance(topology, dict)
                    else None
                ),
            })
        return diagnostics

    @staticmethod
    def _unresolved_top1_candidate(candidate_details):
        viable = []
        for item in candidate_details:
            if not item.get("hard_gate_passed") or item.get("score") is None:
                continue
            try:
                score = float(item["score"])
                gid = int(item["gid"])
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(score):
                viable.append((score, gid))
        if not viable:
            return None
        viable.sort(key=lambda item: (-item[0], item[1]))
        return viable[0][1]

    def _record_unresolved_top1_observation(
        self,
        record,
        candidate_details,
        batch_id,
    ):
        """Accumulate bounded-window rank consistency without changing scores."""
        batch_id = str(batch_id)
        if record.get("last_top1_batch_id") == batch_id:
            return record.get("last_top1_gid")
        top1_gid = self._unresolved_top1_candidate(candidate_details)
        if top1_gid is None:
            return None

        counts = record.setdefault("top1_counts", {})
        previous_top1 = record.get("last_top1_gid")
        record["top1_observation_count"] = int(
            record.get("top1_observation_count", 0)
        ) + 1
        counts[top1_gid] = int(counts.get(top1_gid, 0)) + 1
        if previous_top1 is not None and int(previous_top1) != top1_gid:
            record["top1_switch_count"] = int(
                record.get("top1_switch_count", 0)
            ) + 1
        record["last_top1_gid"] = top1_gid
        record["last_top1_batch_id"] = batch_id
        return top1_gid

    @staticmethod
    def _unresolved_temporal_consensus(record):
        observation_count = int(record.get("top1_observation_count", 0))
        qualified_samples = int(record.get("sample_count", 0))
        raw_counts = record.get("top1_counts", {})
        counts = {}
        if isinstance(raw_counts, dict):
            for raw_gid, raw_count in raw_counts.items():
                try:
                    gid = int(raw_gid)
                    count = int(raw_count)
                except (TypeError, ValueError, OverflowError):
                    continue
                if count > 0:
                    counts[gid] = count

        ordered = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        dominant_gid = ordered[0][0] if ordered else None
        dominant_count = ordered[0][1] if ordered else 0
        runner_up_count = ordered[1][1] if len(ordered) > 1 else 0
        dominance_ratio = (
            float(dominant_count) / float(observation_count)
            if observation_count > 0
            else 0.0
        )
        last_top1_gid = record.get("last_top1_gid")
        try:
            last_top1_gid = (
                None if last_top1_gid is None else int(last_top1_gid)
            )
        except (TypeError, ValueError, OverflowError):
            last_top1_gid = None

        reason = None
        if qualified_samples < AMBIGUOUS_HANDOFF_MIN_SAMPLES:
            reason = "insufficient_qualified_samples"
        elif observation_count < AMBIGUOUS_HANDOFF_MIN_SAMPLES:
            reason = "insufficient_top1_observations"
        elif dominant_gid is None:
            reason = "no_temporal_top1_candidate"
        elif dominance_ratio < AMBIGUOUS_HANDOFF_TEMPORAL_TOP1_MIN_RATIO:
            reason = "top1_temporal_consistency_below_threshold"
        elif last_top1_gid != dominant_gid:
            reason = "latest_top1_differs_from_dominant"

        return {
            "eligible": reason is None,
            "reason": reason,
            "dominant_gid": dominant_gid,
            "dominant_count": dominant_count,
            "runner_up_count": runner_up_count,
            "observation_count": observation_count,
            "qualified_sample_count": qualified_samples,
            "dominance_ratio": dominance_ratio,
            "required_dominance_ratio": (
                AMBIGUOUS_HANDOFF_TEMPORAL_TOP1_MIN_RATIO
            ),
            "last_top1_gid": last_top1_gid,
            "top1_switch_count": int(record.get("top1_switch_count", 0)),
            "top1_counts": dict(sorted(counts.items())),
        }

    def _defer_ambiguous_cross_camera_handoff(
        self,
        cam_name,
        detection,
        event_time,
        candidate_details,
        margin,
        batch_id,
        sample_already_added=False,
    ):
        key = self._unresolved_handoff_key(cam_name, detection)
        record = self.unresolved_cross_camera_handoffs.get(key)
        if not self._unresolved_record_is_valid(key, record):
            self.unresolved_cross_camera_handoffs.pop(key, None)
            if (
                len(self.unresolved_cross_camera_handoffs)
                >= AMBIGUOUS_HANDOFF_MAX_RECORDS
            ):
                oldest_key = min(
                    self.unresolved_cross_camera_handoffs,
                    key=lambda item: self.unresolved_cross_camera_handoffs[
                        item
                    ].get("last_event_time", float("-inf")),
                )
                self.unresolved_cross_camera_handoffs.pop(oldest_key, None)
                self.unresolved_handoff_capacity_drop_count += 1
            record = {
                "key": key,
                "camera": cam_name,
                "local_track_id": int(detection["tid"]),
                "generation": key[2],
                "tracker_generation": key[3],
                "first_event_time": float(event_time),
                "last_event_time": float(event_time),
                "sample_count": 0,
                "samples": [],
                "candidate_gids": [],
                "candidates": [],
                "top1_top2_margin": None,
                "pending_reason": "ambiguous_top1_top2",
                "event_order_result": "waiting_for_competing_arrivals",
                "last_batch_id": str(batch_id),
                "last_quality_rejection_reason": None,
                "top1_observation_count": 0,
                "top1_counts": {},
                "top1_switch_count": 0,
                "last_top1_gid": None,
                "last_top1_batch_id": None,
            }
            self.unresolved_cross_camera_handoffs[key] = record

        if not sample_already_added:
            self._append_unresolved_handoff_sample(
                record,
                detection,
                event_time,
            )
        record["last_event_time"] = max(
            float(record["last_event_time"]),
            float(event_time),
        )
        record["candidate_gids"] = [
            item["gid"]
            for item in candidate_details
            if item.get("hard_gate_passed") and item.get("score") is not None
        ]
        record["candidates"] = self._unresolved_candidate_diagnostics(
            candidate_details
        )
        record["top1_top2_margin"] = margin
        record["last_batch_id"] = str(batch_id)
        self._record_unresolved_top1_observation(
            record,
            candidate_details,
            batch_id,
        )
        return key, record

    @staticmethod
    def _unresolved_handoff_status_item(record, event_time=None):
        reference_time = (
            record.get("last_event_time", 0.0)
            if event_time is None
            else float(event_time)
        )
        return {
            "camera": record.get("camera"),
            "local_track_id": record.get("local_track_id"),
            "generation": record.get("generation"),
            "tracker_generation": record.get("tracker_generation"),
            "first_event_time": record.get("first_event_time"),
            "last_event_time": record.get("last_event_time"),
            "event_time_age_sec": max(
                0.0,
                reference_time - float(record.get("first_event_time", 0.0)),
            ),
            "sample_count": record.get("sample_count", 0),
            "candidate_gids": list(record.get("candidate_gids", [])),
            "candidates": copy.deepcopy(record.get("candidates", [])),
            "top1_top2_margin": record.get("top1_top2_margin"),
            "pending_reason": record.get("pending_reason"),
            "resolution_reason": record.get("resolution_reason"),
            "event_order_result": record.get("event_order_result"),
            "last_quality_rejection_reason": record.get(
                "last_quality_rejection_reason"
            ),
            "last_batch_id": record.get("last_batch_id"),
            "temporal_consistency": (
                GlobalIdentityManager._unresolved_temporal_consensus(record)
            ),
        }

    def _clear_unresolved_handoff(self, cam_name, detection):
        return self.unresolved_cross_camera_handoffs.pop(
            self._unresolved_handoff_key(cam_name, detection),
            None,
        )

    def discard_unresolved_handoffs(self, cam_name=None):
        """Discard invalidated transient evidence without touching identities."""
        with self.lock:
            keys = [
                key
                for key in self.unresolved_cross_camera_handoffs
                if cam_name is None or key[0] == cam_name
            ]
            for key in keys:
                self.unresolved_cross_camera_handoffs.pop(key, None)
            return len(keys)

    @staticmethod
    def _presence_event_time(value, fallback):
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
        ):
            return float(value)
        try:
            fallback_value = float(fallback)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return fallback_value if np.isfinite(fallback_value) else 0.0

    def _normalize_identity_presence(self, gid, identity):
        """Normalize persisted camera presence without creating another store."""
        fallback_time = self._presence_event_time(
            identity.get("last_event_time", identity.get("last_seen", 0.0)),
            0.0,
        )
        raw_presence = identity.get("camera_presence")
        normalized = {}
        if isinstance(raw_presence, dict):
            for raw_camera, raw_record in raw_presence.items():
                if not isinstance(raw_camera, str) or not raw_camera:
                    continue
                if not isinstance(raw_record, dict):
                    continue
                first_seen = self._presence_event_time(
                    raw_record.get("first_seen_event_time"),
                    fallback_time,
                )
                last_seen = self._presence_event_time(
                    raw_record.get("last_seen_event_time"),
                    first_seen,
                )
                normalized[raw_camera] = {
                    "gid": int(gid),
                    "camera": raw_camera,
                    "local_track_id": raw_record.get("local_track_id"),
                    "first_seen_event_time": first_seen,
                    "last_seen_event_time": last_seen,
                    "active": bool(raw_record.get("active", False)),
                    "generation": raw_record.get("generation"),
                    "assignment_source": raw_record.get(
                        "assignment_source",
                        "restored",
                    ),
                    "inactive_reason": raw_record.get("inactive_reason"),
                    "deactivated_event_time": raw_record.get(
                        "deactivated_event_time"
                    ),
                }

        if identity.get("state") in {IDENTITY_DORMANT, IDENTITY_EXPIRED}:
            for record in normalized.values():
                record["active"] = False
                record["inactive_reason"] = (
                    record.get("inactive_reason")
                    or "identity_not_active"
                )

        last_camera = identity.get("last_cam")
        if (
            isinstance(last_camera, str)
            and last_camera
            and last_camera not in normalized
        ):
            normalized[last_camera] = {
                "gid": int(gid),
                "camera": last_camera,
                "local_track_id": None,
                "first_seen_event_time": fallback_time,
                "last_seen_event_time": fallback_time,
                "active": identity.get("state", IDENTITY_ACTIVE) in {
                    IDENTITY_PROVISIONAL,
                    IDENTITY_ACTIVE,
                },
                "generation": None,
                "assignment_source": "restored-legacy",
                "inactive_reason": None,
                "deactivated_event_time": None,
            }
        identity["camera_presence"] = normalized

        history = identity.get("handoff_history")
        if not isinstance(history, list):
            history = []
        identity["handoff_history"] = [
            dict(item)
            for item in history
            if isinstance(item, dict)
        ][-IDENTITY_HANDOFF_HISTORY_SIZE:]
        identity["presence_schema_version"] = 1
        return normalized

    def _presence_allows_local_claim(
        self,
        gid,
        cam_name,
        local_id,
        detection=None,
    ):
        """Reject stale camera-local evidence after a confirmed transfer."""
        identity = self.identities.get(gid)
        if not isinstance(identity, dict):
            return False
        presence = identity.get("camera_presence")
        if not isinstance(presence, dict):
            return identity.get("last_cam") == cam_name
        record = presence.get(cam_name)
        if not isinstance(record, dict) or not record.get("active", False):
            return False
        owner_local_id = record.get("local_track_id")
        if owner_local_id is not None:
            try:
                if int(owner_local_id) != int(local_id):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
        detection = detection or {}
        # Presence ownership is committed with the coordinator epoch.  The
        # per-camera tracker generation is a separate namespace and must not
        # be compared with that epoch.
        observed_generation = detection.get(
            "coordinator_generation"
        )
        owner_generation = record.get("generation")
        if (
            owner_generation is not None
            and observed_generation is not None
            and owner_generation != observed_generation
        ):
            return False
        return True

    @staticmethod
    def _confirmed_handoff_source(source):
        return source in {
            "cross-camera",
            "global-cross-camera",
            "local-transition-recovery-v2",
        }

    @staticmethod
    def _explicit_overlap_allowed(source_camera, destination_camera):
        with topology_lock:
            config = copy.deepcopy(topology_config)
        if config.get("_validation_error"):
            return False
        return any(
            rule.get("from_camera") == source_camera
            and rule.get("to_camera") == destination_camera
            and rule.get("overlap_allowed") is True
            for rule in config.get("transitions", [])
            if isinstance(rule, dict)
        )

    def _reconcile_local_track_presence(
        self,
        cam_name,
        local_id,
        generation,
        committed_gid,
        event_time,
    ):
        """Keep one active GID per camera/local-track/generation key.

        This is deliberately local to one camera/local-track namespace.  It
        must not deactivate a presence for the same GID on another camera,
        including an explicitly overlapping topology edge.
        """
        local_id = int(local_id)
        changed = []
        for other_gid, other_identity in self.identities.items():
            if int(other_gid) == int(committed_gid):
                continue
            presence = self._normalize_identity_presence(
                int(other_gid),
                other_identity,
            )
            record = presence.get(cam_name)
            if not isinstance(record, dict) or not record.get("active", False):
                continue
            if record.get("local_track_id") is None:
                continue
            try:
                same_local = int(record["local_track_id"]) == local_id
            except (TypeError, ValueError, OverflowError):
                same_local = False
            if not same_local:
                continue
            if record.get("generation") != generation:
                continue

            record["active"] = False
            record["inactive_reason"] = "local_track_reassigned"
            record["deactivated_event_time"] = float(event_time)
            changed.append(int(other_gid))

        if changed:
            self.last_global_map_diagnostic = {
                "event": "duplicate_local_track_ownership",
                "camera": cam_name,
                "local_track_id": local_id,
                "generation": generation,
                "active_gids": sorted({int(committed_gid), *changed}),
                "committed_gid": int(committed_gid),
                "repaired_gids": list(changed),
                "event_time": float(event_time),
            }

        return changed

    def _enforce_local_track_presence_invariant(
        self,
        cam_name,
        local_id,
        generation,
        committed_gid,
        event_time,
    ):
        """Final defensive repair/report for duplicate local-track owners."""
        local_id = int(local_id)
        active = []
        for gid, identity in self.identities.items():
            presence = self._normalize_identity_presence(int(gid), identity)
            record = presence.get(cam_name)
            if not isinstance(record, dict) or not record.get("active", False):
                continue
            if record.get("local_track_id") != local_id:
                continue
            if record.get("generation") != generation:
                continue
            active.append(int(gid))

        if len(active) <= 1:
            return []

        repaired = []
        for gid in active:
            if gid == int(committed_gid):
                continue
            record = self.identities[gid]["camera_presence"][cam_name]
            record["active"] = False
            record["inactive_reason"] = "local_track_reassigned"
            record["deactivated_event_time"] = float(event_time)
            repaired.append(gid)

        self.last_global_map_diagnostic = {
            "event": "duplicate_local_track_ownership",
            "camera": cam_name,
            "local_track_id": local_id,
            "generation": generation,
            "active_gids": active,
            "committed_gid": int(committed_gid),
            "repaired_gids": repaired,
            "event_time": float(event_time),
        }
        return repaired

    def _record_camera_presence(
        self,
        gid,
        identity,
        cam_name,
        local_id,
        now_ts,
        source,
        score,
        emb,
        map_pos,
        box_wh,
        generation=None,
    ):
        """Commit presence after assignment gates and one-to-one selection."""
        presence = self._normalize_identity_presence(gid, identity)
        previous_camera = identity.get("last_cam")
        previous_record = copy.deepcopy(presence.get(previous_camera))
        cross_camera = bool(previous_camera and previous_camera != cam_name)

        gate_detection = {
            "emb": emb,
            "map_pos": map_pos,
            "box_wh": box_wh,
        }
        topology = (
            self._topology_gate_details(
                identity,
                cam_name,
                gate_detection,
                now_ts,
            )
            if cross_camera
            else {
                "source_camera": previous_camera,
                "destination_camera": cam_name,
                "event_time_delta_sec": 0.0,
                "topology_rule": None,
                "overlap_allowed": None,
                "passed": True,
                "reason": "same_camera",
            }
        )
        hard_gate = (
            self._hard_gate_diagnostics(
                identity,
                cam_name,
                gate_detection,
                now_ts,
            )
            if cross_camera
            else {
                "passed": True,
                "reason": None,
                "topology": topology,
            }
        )
        if (
            cross_camera
            and self._confirmed_handoff_source(source)
            and not hard_gate["passed"]
        ):
            raise ValueError(
                "confirmed cross-camera assignment failed its hard-gate "
                f"recheck: {hard_gate.get('reason')}"
            )
        confirmed_handoff = (
            cross_camera
            and self._confirmed_handoff_source(source)
            and hard_gate["passed"]
        )

        destination = presence.get(cam_name)
        if not isinstance(destination, dict) or not destination.get("active", False):
            first_seen = float(now_ts)
        else:
            first_seen = self._presence_event_time(
                destination.get("first_seen_event_time"),
                now_ts,
            )
        destination = {
            "gid": int(gid),
            "camera": cam_name,
            "local_track_id": int(local_id),
            "first_seen_event_time": first_seen,
            "last_seen_event_time": float(now_ts),
            "active": True,
            "generation": generation,
            "assignment_source": source,
            "inactive_reason": None,
            "deactivated_event_time": None,
        }
        presence[cam_name] = destination

        handoff = None
        if confirmed_handoff:
            overlap_allowed = bool(
                topology.get("passed")
                and topology.get("overlap_allowed") is True
            )
            if not overlap_allowed:
                for camera, record in presence.items():
                    if camera == cam_name or not record.get("active", False):
                        continue
                    if self._explicit_overlap_allowed(camera, cam_name):
                        continue
                    record["active"] = False
                    record["inactive_reason"] = "confirmed_non_overlap_handoff"
                    record["deactivated_event_time"] = float(now_ts)

            exit_event_time = (
                previous_record.get("last_seen_event_time")
                if isinstance(previous_record, dict)
                else identity.get("last_event_time", identity.get("last_seen"))
            )
            exit_event_time = self._presence_event_time(exit_event_time, now_ts)
            appearance = self._gallery_similarity(emb, identity)
            handoff = {
                "gid": int(gid),
                "from_camera": previous_camera,
                "to_camera": cam_name,
                "exit_event_time": exit_event_time,
                "entry_event_time": float(now_ts),
                "event_time_delta_sec": float(now_ts) - exit_event_time,
                "appearance_score": float(appearance),
                "final_score": float(score),
                "assignment_source": source,
                "topology_result": copy.deepcopy(topology),
                "hard_gate_result": {
                    "passed": bool(hard_gate["passed"]),
                    "reason": hard_gate.get("reason"),
                },
                "previous_presence": previous_record,
                "new_presence": copy.deepcopy(destination),
                "reason": (
                    "confirmed_overlap_handoff"
                    if overlap_allowed
                    else "confirmed_non_overlap_handoff"
                ),
                "committed": True,
            }
            history = identity.setdefault("handoff_history", [])
            history.append(handoff)
            del history[:-IDENTITY_HANDOFF_HISTORY_SIZE]

        return {
            "previous_presence": previous_record,
            "new_presence": copy.deepcopy(destination),
            "handoff": handoff,
        }

    # ==================

    def reset_camera_local_state(
        self,
        cam_name
    ):
        """Remove camera-local evidence without deleting global identities."""
        with self.lock:
            local_keys = [
                key
                for key in self.local_to_global
                if key[0] == cam_name
            ]
            hold_keys = [
                key
                for key in self.occlusion_hold
                if key[0] == cam_name
            ]
            tracklet_keys = [
                key
                for key in self.tracklets
                if key[0] == cam_name
            ]
            unresolved_keys = [
                key
                for key in self.unresolved_cross_camera_handoffs
                if key[0] == cam_name
            ]
            recent_same_cam_count = len(self.recent_same_cam)
            recent_cross_cam_count = len(self.recent_cross_cam)

            for key in local_keys:
                self.local_to_global.pop(
                    key,
                    None
                )

            for key in hold_keys:
                self.occlusion_hold.pop(
                    key,
                    None
                )

            for key in tracklet_keys:
                self.tracklets.pop(key, None)

            for key in unresolved_keys:
                self.unresolved_cross_camera_handoffs.pop(key, None)

            self.recent_same_cam = [
                item
                for item in self.recent_same_cam
                if item.get("cam_name") != cam_name
            ]
            self.recent_cross_cam = [
                item
                for item in self.recent_cross_cam
                if item.get("cam_name") != cam_name
            ]

            return {
                "local_mappings_removed": len(
                    local_keys
                ),
                "occlusion_holds_removed": len(
                    hold_keys
                ),
                "tracklets_removed": len(tracklet_keys),
                "unresolved_handoffs_removed": len(unresolved_keys),
                "recent_same_cam_removed": (
                    recent_same_cam_count - len(self.recent_same_cam)
                ),
                "recent_cross_cam_removed": (
                    recent_cross_cam_count - len(self.recent_cross_cam)
                ),
            }

    # ==================

    def _assignable_identity(self, gid):
        identity = self.identities.get(gid)
        if identity is None:
            return None
        state = identity.get("state", IDENTITY_ACTIVE)
        return identity if state in IDENTITY_ASSIGNABLE_STATES else None

    def _globally_matchable_identity(self, gid):
        """Return durable identity evidence, never a provisional bootstrap."""
        identity = self._assignable_identity(gid)
        if identity is None:
            return None
        state = identity.get("state", IDENTITY_ACTIVE)
        if (
            state == IDENTITY_PROVISIONAL
            or identity.get("gallery_mature") is False
        ):
            return None
        # Missing maturity metadata is a schema-v1 compatibility case;
        # explicit False is authoritative for every new record.
        return identity

    def _matchable_identity_for_camera(self, gid, cam_name):
        """Allow short same-camera bootstrap continuity, not cross-camera reuse."""
        identity = self._globally_matchable_identity(gid)
        if identity is not None:
            return identity
        identity = self._assignable_identity(gid)
        if (
            identity is not None
            and identity.get("state", IDENTITY_ACTIVE) == IDENTITY_PROVISIONAL
            and identity.get("last_cam") == cam_name
        ):
            return identity
        return None

    @staticmethod
    def _identity_match_deadline(identity):
        last_seen = identity.get("last_seen")
        if not isinstance(last_seen, (int, float)):
            return None
        if identity.get("state", IDENTITY_ACTIVE) == IDENTITY_DORMANT:
            dormant_since = identity.get("state_updated_at")
            if not isinstance(dormant_since, (int, float)):
                dormant_since = float(last_seen)
            return float(dormant_since) + IDENTITY_DORMANT_TTL_SEC
        return float(last_seen) + REID_MAX_IDLE_SEC

    # ==================

    def _verify_local_track(
        self,
        gid,
        det
    ):
        identity = self._assignable_identity(gid)

        if identity is None:
            return False, -1.0

        appearance = self._gallery_similarity(
            det["emb"],
            identity
        )

        if appearance >= LOCAL_TRACK_STRONG_THRESHOLD:
            return True, appearance

        if appearance >= LOCAL_TRACK_VERIFY_THRESHOLD:
            return True, appearance

        return False, appearance

    # ========================================================
    # GALLERY SIMILARITY
    # ========================================================

    @staticmethod
    def _normalize_embedding_candidate(value, expected_size=None):
        return _reid_normalize_embedding_candidate(value, expected_size)

    def _identity_prototype_candidates(self, identity, expected_size):
        return _reid_identity_prototype_candidates(
            identity,
            expected_size,
            diversity_threshold=REID_GALLERY_DIVERSITY_THRESHOLD,
        )

    def _robust_identity_prototype(self, candidates):
        return _reid_robust_identity_prototype(
            candidates,
            min_samples=IDENTITY_PROTOTYPE_MIN_SAMPLES,
            min_consensus=IDENTITY_PROTOTYPE_MIN_CONSENSUS,
            max_samples=IDENTITY_PROTOTYPE_MAX_SAMPLES,
        )

    def _gallery_similarity(self, emb, identity):
        score = _reid_gallery_similarity(
            emb,
            identity,
            diversity_threshold=REID_GALLERY_DIVERSITY_THRESHOLD,
            prototype_enabled=IDENTITY_PROTOTYPE_ENABLED,
            prototype_min_samples=IDENTITY_PROTOTYPE_MIN_SAMPLES,
            prototype_min_consensus=IDENTITY_PROTOTYPE_MIN_CONSENSUS,
            prototype_weight=IDENTITY_PROTOTYPE_WEIGHT,
            support_weight=IDENTITY_PROTOTYPE_SUPPORT_WEIGHT,
        )
        if score >= -1.0:
            self.identity_prototype_calls += 1
        return score


    def _persistent_tracklet_key(
        self,
        cam_name,
        local_id,
        det,
    ):
        return (
            str(cam_name),
            int(local_id),
            self._tracklet_generation(det),
        )

    def add_fresh_tracklet_embedding(
        self,
        cam_name,
        local_id,
        emb,
        event_time,
        generation=None,
    ):
        """Register exactly one fresh OSNet observation for a local track."""
        self._update_persistent_tracklet_buffer_direct(
            cam_name,
            local_id,
            emb,
            event_time,
            generation=generation,
        )

    def _update_persistent_tracklet_buffer_direct(
        self,
        cam_name,
        local_id,
        emb,
        event_time,
        generation=None,
    ):
        if not PERSISTENT_TRACKLET_BUFFER_ENABLED:
            return

        normalized = self._normalize_embedding_candidate(emb)
        if normalized is None:
            return

        key = (
            str(cam_name),
            int(local_id),
            generation,
        )
        now_ts = float(event_time)

        record = self.persistent_tracklet_buffers.get(key)
        if record is None:
            record = {
                "samples": [],
                "last_event_time": now_ts,
            }
            self.persistent_tracklet_buffers[key] = record
        else:
            last_ts = float(record.get("last_event_time", now_ts))
            if (
                np.isfinite(last_ts)
                and now_ts - last_ts
                > PERSISTENT_TRACKLET_BUFFER_MAX_AGE_SEC
            ):
                record["samples"] = []
                self.persistent_tracklet_buffer_resets += 1

        record["last_event_time"] = now_ts

        samples = record["samples"]
        samples.append({
            "emb": normalized.copy(),
            "event_time": now_ts,
        })

        if len(samples) > PERSISTENT_TRACKLET_BUFFER_MAX_SAMPLES:
            drop_count = (
                len(samples)
                - PERSISTENT_TRACKLET_BUFFER_MAX_SAMPLES
            )
            del samples[:drop_count]
            self.persistent_tracklet_buffer_pruned += drop_count

        self.persistent_tracklet_buffer_updates += 1
        self.last_persistent_tracklet_buffer = {
            "camera": str(cam_name),
            "local_id": int(local_id),
            "sample_count": len(samples),
            "event_time": now_ts,
        }

    def _update_persistent_tracklet_buffer(
        self,
        cam_name,
        local_id,
        det,
        event_time,
    ):
        if not PERSISTENT_TRACKLET_BUFFER_ENABLED:
            return

        # Only a real OSNet refresh adds evidence. Cached embeddings are reused
        # for matching but must not be counted again as new temporal samples.
        if not bool(det.get("reid_fresh", False)):
            return

        emb = self._normalize_embedding_candidate(
            det.get("emb")
        )
        if emb is None:
            return

        key = self._persistent_tracklet_key(
            cam_name,
            local_id,
            det,
        )
        now_ts = float(event_time)

        record = self.persistent_tracklet_buffers.get(
            key
        )
        if record is None:
            record = {
                "samples": [],
                "last_event_time": now_ts,
            }
            self.persistent_tracklet_buffers[key] = record
        else:
            last_ts = float(
                record.get("last_event_time", now_ts)
            )
            if (
                np.isfinite(last_ts)
                and now_ts - last_ts
                > PERSISTENT_TRACKLET_BUFFER_MAX_AGE_SEC
            ):
                record["samples"] = []
                self.persistent_tracklet_buffer_resets += 1

        record["last_event_time"] = now_ts

        samples = record["samples"]
        samples.append({
            "emb": emb.copy(),
            "event_time": now_ts,
        })

        if len(samples) > PERSISTENT_TRACKLET_BUFFER_MAX_SAMPLES:
            drop_count = (
                len(samples)
                - PERSISTENT_TRACKLET_BUFFER_MAX_SAMPLES
            )
            del samples[:drop_count]
            self.persistent_tracklet_buffer_pruned += drop_count

        self.persistent_tracklet_buffer_updates += 1
        self.last_persistent_tracklet_buffer = {
            "camera": str(cam_name),
            "local_id": int(local_id),
            "sample_count": len(samples),
            "event_time": now_ts,
        }

    def _persistent_tracklet_samples(
        self,
        cam_name,
        local_id,
        det,
        event_time,
    ):
        key = (
            str(cam_name),
            int(local_id),
            self._tracklet_generation(det),
        )
        record = self.persistent_tracklet_buffers.get(
            key
        )
        if not isinstance(record, dict):
            return []

        last_ts = record.get("last_event_time")
        if not isinstance(last_ts, (int, float)):
            return []

        if (
            float(event_time) - float(last_ts)
            > PERSISTENT_TRACKLET_BUFFER_MAX_AGE_SEC
        ):
            return []

        return list(record.get("samples", []))


    def _tracklet_prototype_score(
        self,
        query_emb,
        samples,
    ):
        """Return robust multi-frame similarity from existing tracklet samples."""
        query = self._normalize_embedding_candidate(query_emb)
        if query is None:
            return None

        candidates = []
        for sample in samples:
            emb = (
                sample.get("emb")
                if isinstance(sample, dict)
                else sample
            )
            candidate = self._normalize_embedding_candidate(
                emb,
                expected_size=query.size,
            )
            if candidate is None:
                continue

            # V12: for one local track, high similarity across frames is
            # expected and should NOT be treated as a duplicate error.
            # Keep recent bounded samples so the prototype can actually form.
            if TRACKLET_PROTOTYPE_REJECT_NEAR_DUPLICATES:
                if any(
                    float(np.dot(candidate, existing))
                    >= REID_GALLERY_DIVERSITY_THRESHOLD
                    for existing in candidates
                ):
                    continue

            candidates.append(candidate)

        if not candidates:
            return None

        self.tracklet_prototype_calls += 1
        self.tracklet_prototype_total_samples_seen += len(
            candidates
        )

        # Recent bounded evidence only: keeps compute tiny and more relevant to
        # the current track session.
        if TRACKLET_PROTOTYPE_KEEP_RECENT_SAMPLES:
            candidates = candidates[
                -max(1, int(TRACKLET_PROTOTYPE_MAX_SAMPLES)):
            ]
        else:
            candidates = candidates[
                :max(1, int(TRACKLET_PROTOTYPE_MAX_SAMPLES))
            ]

        raw_scores = sorted(
            [
                float(np.dot(query, candidate))
                for candidate in candidates
            ],
            reverse=True,
        )
        topk = raw_scores[
            :max(
                1,
                min(
                    int(TRACKLET_SUPPORT_TOPK),
                    len(raw_scores),
                ),
            )
        ]
        support_score = float(np.median(topk))

        if (
            not TRACKLET_PROTOTYPE_ENABLED
            or len(candidates) < TRACKLET_PROTOTYPE_MIN_SAMPLES
        ):
            self.tracklet_prototype_fallbacks += 1
            self.last_tracklet_prototype_diagnostic = {
                "sample_count": len(candidates),
                "prototype_used": False,
                "support_score": support_score,
                "prototype_score": None,
                "consensus": None,
                "final_score": support_score,
            }
            return support_score

        matrix = np.stack(candidates, axis=0)
        pairwise = np.matmul(matrix, matrix.T)
        median_support = np.median(pairwise, axis=1)
        medoid_index = int(np.argmax(median_support))
        medoid = matrix[medoid_index]

        medoid_scores = np.matmul(matrix, medoid)
        selected = []
        for idx in np.argsort(-medoid_scores):
            score = float(medoid_scores[int(idx)])
            if (
                len(selected) >= TRACKLET_PROTOTYPE_MIN_SAMPLES
                and score < TRACKLET_PROTOTYPE_MIN_CONSENSUS
            ):
                continue
            selected.append(matrix[int(idx)])
            if len(selected) >= TRACKLET_PROTOTYPE_MAX_SAMPLES:
                break

        if len(selected) < TRACKLET_PROTOTYPE_MIN_SAMPLES:
            self.tracklet_prototype_fallbacks += 1
            self.tracklet_prototype_low_consensus += 1
            self.last_tracklet_prototype_diagnostic = {
                "sample_count": len(candidates),
                "prototype_used": False,
                "support_score": support_score,
                "prototype_score": None,
                "consensus": None,
                "final_score": support_score,
            }
            return support_score

        prototype = np.mean(
            np.stack(selected, axis=0),
            axis=0,
        )
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm < 1e-8:
            self.tracklet_prototype_fallbacks += 1
            return support_score
        prototype = prototype / norm

        prototype_score = float(np.dot(query, prototype))
        consensus_scores = [
            float(np.dot(prototype, item))
            for item in selected
        ]
        consensus = float(np.median(consensus_scores))

        if consensus < TRACKLET_PROTOTYPE_MIN_CONSENSUS:
            self.tracklet_prototype_low_consensus += 1
            # Low-consensus tracklet memory should not be allowed to create an
            # artificially optimistic match.
            final_score = min(
                prototype_score,
                support_score,
            )
        else:
            final_score = (
                TRACKLET_PROTOTYPE_WEIGHT
                * prototype_score
                + TRACKLET_SUPPORT_WEIGHT
                * support_score
            )

        final_score = float(
            max(-1.0, min(1.0, final_score))
        )

        self.tracklet_prototype_used += 1
        self.last_tracklet_prototype_diagnostic = {
            "sample_count": len(candidates),
            "selected_count": len(selected),
            "prototype_used": True,
            "support_score": support_score,
            "prototype_score": prototype_score,
            "consensus": consensus,
            "final_score": final_score,
        }

        return final_score

    def _local_track_similarity(self, gid, cam_name, local_id, det):
        """Use identity prototype plus robust multi-frame tracklet evidence."""
        identity = self.identities.get(gid)
        if identity is None:
            return -1.0

        identity_score = self._gallery_similarity(
            det.get("emb"),
            identity,
        )

        local_key = (
            cam_name,
            int(local_id),
        )
        record = self.tracklets.get(local_key)
        generation = self._tracklet_generation(det)

        if (
            record is None
            or not self._tracklet_record_is_valid(
                record,
                gid,
                generation,
                identity,
            )
        ):
            return float(identity_score)

        persistent_samples = self._persistent_tracklet_samples(
            cam_name,
            local_id,
            det,
            det.get("event_time", time.time()),
        )

        prototype_samples = (
            persistent_samples
            if persistent_samples
            else record.get("samples", [])
        )

        tracklet_score = self._tracklet_prototype_score(
            det.get("emb"),
            prototype_samples,
        )
        if tracklet_score is None:
            return float(identity_score)

        # Conservative fusion: strong agreement is rewarded, but a single
        # optimistic source cannot dominate the final local-continuity score.
        if identity_score < 0.0:
            return float(tracklet_score)

        combined = (
            0.55 * float(identity_score)
            + 0.45 * float(tracklet_score)
        )
        return float(
            max(-1.0, min(1.0, combined))
        )

    def _can_continue_provisional_local_track(
        self,
        gid,
        cam_name,
        local_id,
        det,
        now_ts,
        appearance,
    ):
        """Keep a confirmed local bootstrap alive without weakening global Re-ID."""
        identity = self.identities.get(gid)
        if (
            identity is None
            or identity.get("state", IDENTITY_ACTIVE) != IDENTITY_PROVISIONAL
            or identity.get("gallery_mature") is not False
            or identity.get("last_cam") != cam_name
            or not det.get("local_track_confirmed", True)
        ):
            return False

        local_key = (cam_name, int(local_id))
        mapping = self.local_to_global.get(local_key)
        if not isinstance(mapping, dict) or mapping.get("gid") != gid:
            return False
        mapping_last_seen = mapping.get("last_seen")
        if not isinstance(mapping_last_seen, (int, float)):
            return False
        local_gap = float(now_ts) - float(mapping_last_seen)
        if (
            not np.isfinite(local_gap)
            or local_gap < 0.0
            or local_gap > REID_PROVISIONAL_LOCAL_CONTINUITY_SEC
        ):
            return False
        provisional_since = identity.get("state_updated_at")
        if not isinstance(provisional_since, (int, float)):
            return False
        provisional_age = float(now_ts) - float(provisional_since)
        if (
            not np.isfinite(provisional_age)
            or provisional_age < 0.0
            or provisional_age > REID_PROVISIONAL_LOCAL_CONTINUITY_SEC
        ):
            return False
        if self._hard_gate_reason(identity, cam_name, det, now_ts) is not None:
            return False

        _, embedding_reason = self._validated_gallery_embedding(
            identity,
            det.get("emb"),
        )
        if embedding_reason is not None:
            return False

        if self._provisional_has_stronger_same_camera_competitor(
            gid,
            cam_name,
            det,
            now_ts,
            appearance,
        ):
            return False

        record = self.tracklets.get(local_key)
        if record is None:
            # The bootstrap frame may have been unsuitable for gallery
            # admission. The confirmed local tracker may anchor the first
            # usable sample, but the identity remains PROVISIONAL.
            return True
        if not self._tracklet_record_is_valid(
            record,
            gid,
            self._tracklet_generation(det),
            identity,
        ):
            return False

        # Poor crops are assignment-neutral: keep local continuity, then let
        # the gallery quality gate reject them. A usable crop must still be
        # consistent with existing accepted local evidence.
        if self._gallery_quality_reason(det) is not None:
            return True
        return appearance >= REID_RECENT_SAME_CAM_THRESHOLD

    def _provisional_has_stronger_same_camera_competitor(
        self,
        gid,
        cam_name,
        det,
        now_ts,
        appearance,
    ):
        """Let stronger same-camera evidence override provisional local ownership."""
        identity = self.identities.get(gid)
        if (
            identity is None
            or identity.get("state", IDENTITY_ACTIVE) != IDENTITY_PROVISIONAL
            or identity.get("gallery_mature") is not False
            or identity.get("last_cam") != cam_name
        ):
            return False

        # A confirmed local tracker is useful bootstrap evidence, but it must
        # yield when appearance points clearly to another eligible identity
        # from the same camera. This lets the batch matcher repair local-ID
        # swaps instead of cementing them into provisional identities.
        for competing_gid, competing_identity in self.identities.items():
            if competing_gid == gid:
                continue
            competing_identity = self._matchable_identity_for_camera(
                competing_gid,
                cam_name,
            )
            if (
                competing_identity is None
                or competing_identity.get("last_cam") != cam_name
                or self._hard_gate_reason(
                    competing_identity,
                    cam_name,
                    det,
                    now_ts,
                ) is not None
            ):
                continue
            competing_appearance = self._gallery_similarity(
                det.get("emb"),
                competing_identity,
            )
            if (
                np.isfinite(competing_appearance)
                and competing_appearance >= LOCAL_TRACK_VERIFY_THRESHOLD
                and competing_appearance - appearance
                >= ASSIGN_SAME_CAM_MIN_MARGIN
            ):
                return True
        return False


    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self, reference_time=None):

        # Re-ID state is timestamped on the observation timeline.  Wall-clock
        # processing delay must not age a person by more than its source gap.
        now = (
            time.time()
            if reference_time is None
            else float(reference_time)
        )

        # ----------------------------------------------------
        # Global identities
        # ----------------------------------------------------

        for gid, info in self.identities.items():
            presence_before = copy.deepcopy(info.get("camera_presence"))
            presence = self._normalize_identity_presence(gid, info)
            presence_changes = []
            for camera, record in presence.items():
                presence_last_seen = record.get("last_seen_event_time")
                if (
                    record.get("active", False)
                    and isinstance(presence_last_seen, (int, float))
                    and now >= float(presence_last_seen)
                    and now - float(presence_last_seen) > REID_MAX_IDLE_SEC
                ):
                    record["active"] = False
                    record["inactive_reason"] = "presence_timeout"
                    record["deactivated_event_time"] = float(now)
                    presence_changes.append({
                        "camera": camera,
                        "last_seen_event_time": float(presence_last_seen),
                        "deactivated_event_time": float(now),
                        "reason": "presence_timeout",
                    })

            state = info.get("state", IDENTITY_ACTIVE)
            last_seen = info.get("last_seen")
            transitioned = False
            try:
                if not isinstance(last_seen, (int, float)):
                    transitioned = self._transition_identity(
                        gid,
                        info,
                        IDENTITY_EXPIRED,
                        now,
                        "invalid_persisted_identity",
                    )
                else:
                    idle_sec = now - last_seen
                    if (
                        state in (IDENTITY_PROVISIONAL, IDENTITY_ACTIVE)
                        and idle_sec > REID_MAX_IDLE_SEC
                    ):
                        transitioned = self._transition_identity(
                            gid,
                            info,
                            IDENTITY_DORMANT,
                            now,
                            "idle_timeout",
                        )
                    elif state == IDENTITY_DORMANT:
                        dormant_deadline = self._identity_match_deadline(info)
                        if dormant_deadline is None or now > dormant_deadline:
                            transitioned = self._transition_identity(
                                gid,
                                info,
                                IDENTITY_EXPIRED,
                                now,
                                "dormant_ttl_expired",
                            )

                if presence_changes and self.identity_store and not transitioned:
                    self.identity_store.save_identity(
                        gid,
                        info,
                        "presence_update",
                        "presence_timeout",
                        now,
                        preceding_events=[{
                            "event_type": "presence_inactive",
                            "reason": "presence_timeout",
                            "timestamp": float(now),
                            "payload": {
                                "presence_changes": presence_changes,
                            },
                        }],
                    )
            except Exception:
                if presence_before is None:
                    info.pop("camera_presence", None)
                else:
                    info["camera_presence"] = presence_before
                raise

        if self.identity_store:
            self.identity_store.purge_expired_snapshots(
                now - IDENTITY_EXPIRED_SNAPSHOT_RETENTION_SEC
            )


        # ----------------------------------------------------
        # Local -> global
        # ----------------------------------------------------

        stale_local_keys = []

        for key, data in self.local_to_global.items():
            last_seen = data.get("last_seen") if isinstance(data, dict) else None
            if (
                not isinstance(last_seen, (int, float))
                or not np.isfinite(float(last_seen))
                or (
                    now >= last_seen
                    and now - last_seen > REID_MAX_IDLE_SEC
                )
            ):
                stale_local_keys.append(
                    key
                )

        for key in stale_local_keys:

            self.local_to_global.pop(
                key,
                None
            )
            self.tracklets.pop(
                key,
                None
            )

        # Tracklets are transient local evidence.  Remove malformed or stale
        # orphan records even if their mapping disappeared through another
        # safe path.  A backwards event-time jump must not age valid records.
        stale_tracklet_keys = []
        for key, record in self.tracklets.items():
            if not isinstance(record, dict):
                stale_tracklet_keys.append(key)
                continue
            owner_identity = self.identities.get(record.get("gid"))
            if (
                owner_identity is None
                or not self._tracklet_record_is_valid(
                    record,
                    record.get("gid"),
                    record.get("generation"),
                    owner_identity,
                )
            ):
                stale_tracklet_keys.append(key)
                continue
            last_seen = record["last_seen"]
            mapping = self.local_to_global.get(key)
            if mapping is not None:
                if (
                    isinstance(mapping, dict)
                    and record.get("gid") == mapping.get("gid")
                ):
                    continue
                stale_tracklet_keys.append(key)
                continue
            if now >= last_seen and now - last_seen > REID_MAX_IDLE_SEC:
                stale_tracklet_keys.append(key)

        for key in stale_tracklet_keys:
            self.tracklets.pop(key, None)


        # ----------------------------------------------------
        # Occlusion
        # ----------------------------------------------------

        stale_hold_keys = []

        for key, hold in self.occlusion_hold.items():

            if (
                now >
                hold.get(
                    "until_ts",
                    0
                )
            ):

                stale_hold_keys.append(
                    key
                )

        for key in stale_hold_keys:

            self.occlusion_hold.pop(
                key,
                None
            )


        # ----------------------------------------------------
        # Same camera cache
        # ----------------------------------------------------

        self.recent_same_cam = [

            item

            for item in self.recent_same_cam

            if (
                now - item.get(
                    "ts",
                    0
                )
                <=
                REID_RECENT_SAME_CAM_SEC
            )

            and
            item.get("gid")
            in
            self.identities

        ][-300:]


        # ----------------------------------------------------
        # Cross camera cache
        # ----------------------------------------------------

        self.recent_cross_cam = [

            item

            for item in self.recent_cross_cam

            if (
                now - item.get(
                    "ts",
                    0
                )
                <=
                REID_RECENT_CROSS_CAM_SEC
            )

            and
            item.get("gid")
            in
            self.identities

        ][-300:]


    # ========================================================
    # SIZE CHECK
    # ========================================================

    def _size_ratio_ok(
        self,
        box_wh,
        ref_wh
    ):

        if (
            box_wh is None
            or
            ref_wh is None
        ):
            return True

        bw, bh = box_wh
        rw, rh = ref_wh

        if min(
            bw,
            bh,
            rw,
            rh
        ) <= 0:

            return True

        wr = (
            min(bw, rw)
            /
            max(bw, rw)
        )

        hr = (
            min(bh, rh)
            /
            max(bh, rh)
        )

        return (
            wr >= REID_SIZE_GATE_RATIO
            and
            hr >= REID_SIZE_GATE_RATIO
        )


    # ========================================================
    # MAP DISTANCE
    # ========================================================

    def _map_distance(
        self,
        p1,
        p2
    ):

        if (
            p1 is None
            or
            p2 is None
        ):
            return None

        return float(
            np.linalg.norm(
                np.array(
                    p1,
                    dtype=np.float32
                )
                -
                np.array(
                    p2,
                    dtype=np.float32
                )
            )
        )


    # ========================================================
    # MAP SCORE
    # ========================================================

    def _map_score(
        self,
        identity,
        map_pos,
        cross_camera=False
    ):

        prev_pos = identity.get(
            "last_map_pos"
        )

        if (
            map_pos is None
            or
            prev_pos is None
        ):

            return 0.50

        dist = self._map_distance(
            map_pos,
            prev_pos
        )

        if dist is None:
            return 0.50

        if cross_camera:

            gate = (
                REID_MAP_GATE_CROSS_CAM_PX
            )

        else:

            gate = (
                REID_MAP_GATE_SAME_CAM_PX
            )

        score = (
            1.0
            -
            min(
                dist / max(gate, 1.0),
                1.0
            )
        )

        return float(
            max(
                0.0,
                min(1.0, score)
            )
        )


    # ========================================================
    # TIME SCORE
    # ========================================================

    def _time_score(
        self,
        identity,
        now_ts
    ):

        dt = max(
            0.0,
            now_ts
            -
            identity["last_seen"]
        )

        return float(
            max(
                0.0,
                1.0
                -
                min(
                    dt
                    /
                    max(
                        REID_MAX_IDLE_SEC,
                        1e-3
                    ),
                    1.0
                )
            )
        )


    # ========================================================
    # MOTION
    # ========================================================

    def _recent_history_for_gid(
        self,
        cam_name,
        gid,
        prev_assignments,
        limit=3
    ):

        if not prev_assignments:
            return []

        hist = []

        for item in reversed(
            prev_assignments
        ):

            if (
                item.get("cam_name")
                ==
                cam_name
                and
                item.get("gid")
                ==
                gid
            ):

                hist.append(item)

                if len(hist) >= limit:
                    break

        return list(
            reversed(hist)
        )


    def _predict_center(
        self,
        history,
        event_time=None
    ):

        if not history:
            return None

        if len(history) == 1:
            return history[-1].get(
                "center"
            )

        p1 = history[-2].get(
            "center"
        )

        p2 = history[-1].get(
            "center"
        )

        t1 = history[-2].get(
            "ts",
            0.0
        )

        t2 = history[-1].get(
            "ts",
            0.0
        )

        if (
            p1 is None
            or
            p2 is None
        ):

            return p2 or p1

        dt = max(
            t2 - t1,
            1e-3
        )

        vx = (
            p2[0] - p1[0]
        ) / dt

        vy = (
            p2[1] - p1[1]
        ) / dt

        prediction_time = (
            time.time()
            if event_time is None
            else float(event_time)
        )
        horizon = min(
            max(
                prediction_time - t2,
                0.0
            ),
            0.25
        )

        return (
            p2[0] + vx * horizon,
            p2[1] + vy * horizon
        )


    def _motion_score(
        self,
        cam_name,
        gid,
        det_box,
        prev_assignments,
        event_time=None
    ):

        history = (
            self._recent_history_for_gid(
                cam_name,
                gid,
                prev_assignments,
                limit=3
            )
        )

        if not history:
            return 0.50

        pred_center = (
            self._predict_center(
                history,
                event_time=event_time
            )
        )

        if pred_center is None:
            return 0.50

        det_center = bbox_center(
            det_box
        )

        dist = float(
            np.hypot(
                det_center[0]
                -
                pred_center[0],
                det_center[1]
                -
                pred_center[1]
            )
        )

        last_box = history[-1].get(
            "box",
            det_box
        )

        lx1, ly1, lx2, ly2 = (
            last_box
        )

        scale = max(
            40.0,
            np.hypot(
                lx2 - lx1,
                ly2 - ly1
            ) * 0.75
        )

        score = (
            1.0
            -
            min(
                dist / scale,
                1.6
            )
        )

        return float(
            max(
                -0.6,
                min(1.0, score)
            )
        )


    # ========================================================
    # CAN MATCH
    # ========================================================

    def _topology_gate_details(self, identity, cam_name, det, now_ts):
        source_camera = identity.get("last_cam")
        destination_camera = cam_name
        previous_event_time = identity.get(
            "last_event_time",
            identity.get("last_seen"),
        )
        details = {
            "source_camera": source_camera,
            "destination_camera": destination_camera,
            "event_time_delta_sec": None,
            "topology_rule": None,
            "overlap_allowed": None,
            "passed": False,
            "reason": None,
        }

        if source_camera == destination_camera:
            details.update({
                "passed": True,
                "reason": "same_camera",
            })
            return details

        with topology_lock:
            config = copy.deepcopy(topology_config)
        if config.get("_validation_error"):
            details["reason"] = "topology_config_invalid"
            return details

        if (
            isinstance(now_ts, bool)
            or not isinstance(now_ts, (int, float))
            or not np.isfinite(float(now_ts))
            or isinstance(previous_event_time, bool)
            or not isinstance(previous_event_time, (int, float))
            or not np.isfinite(float(previous_event_time))
        ):
            details["reason"] = "identity_event_time_invalid"
            return details

        event_time_delta = float(now_ts) - float(previous_event_time)
        details["event_time_delta_sec"] = event_time_delta

        if not config.get("enforce", False):
            details.update({
                "passed": True,
                "reason": "topology_disabled",
            })
            return details

        matching_rule = next((
            rule
            for rule in config.get("transitions", [])
            if (
                rule.get("from_camera") == source_camera
                and rule.get("to_camera") == destination_camera
            )
        ), None)
        if matching_rule is None:
            details["reason"] = "topology_transition_not_allowed"
            return details

        min_time = float(matching_rule.get(
            "min_travel_time_sec",
            matching_rule.get("min_travel_sec", 0.0),
        ))
        max_value = matching_rule.get(
            "max_travel_time_sec",
            matching_rule.get("max_travel_sec"),
        )
        max_time = None if max_value is None else float(max_value)
        overlap_allowed = matching_rule.get("overlap_allowed", False)
        normalized_rule = {
            "from_camera": source_camera,
            "to_camera": destination_camera,
            "min_travel_time_sec": min_time,
            "max_travel_time_sec": max_time,
            "overlap_allowed": bool(overlap_allowed),
        }
        details.update({
            "topology_rule": normalized_rule,
            "overlap_allowed": bool(overlap_allowed),
        })

        if event_time_delta < 0.0:
            details["reason"] = "event_time_before_previous_observation"
            return details
        if event_time_delta == 0.0:
            if overlap_allowed:
                details.update({
                    "passed": True,
                    "reason": "topology_overlap_allowed",
                })
            else:
                details["reason"] = "topology_overlap_not_allowed"
            return details
        if event_time_delta < min_time:
            details["reason"] = "topology_travel_too_fast"
            return details
        if max_time is not None and event_time_delta > max_time:
            details["reason"] = "topology_travel_too_slow"
            return details

        details.update({
            "passed": True,
            "reason": "topology_allowed",
        })
        return details

    def _topology_gate(self, identity, cam_name, det, now_ts):
        details = self._topology_gate_details(
            identity,
            cam_name,
            det,
            now_ts,
        )
        return details["passed"], details["reason"]

    def _hard_gate_diagnostics(self, identity, cam_name, det, now_ts):
        """Return hard-gate outcome plus topology audit metadata."""
        topology = self._topology_gate_details(
            identity,
            cam_name,
            det,
            now_ts,
        )
        state = identity.get("state", IDENTITY_ACTIVE)
        if state == IDENTITY_EXPIRED:
            reason = "identity_expired"
        elif state not in IDENTITY_ASSIGNABLE_STATES:
            reason = "identity_state_invalid"
        elif (
            identity.get("last_cam") != cam_name
            and not det.get("local_track_confirmed", True)
        ):
            reason = "unconfirmed_cross_camera_observation"
        elif not topology["passed"]:
            reason = topology["reason"]
        else:
            deadline = self._identity_match_deadline(identity)
            if deadline is None:
                reason = "identity_time_invalid"
            elif now_ts > deadline:
                reason = (
                    "identity_dormant_ttl_expired"
                    if state == IDENTITY_DORMANT
                    else "identity_idle_expired"
                )
            elif not self._size_ratio_ok(
                det.get("box_wh"),
                identity.get("box_wh"),
            ):
                reason = "incompatible_box_size"
            else:
                previous = identity.get("last_map_pos")
                current = det.get("map_pos")
                gate = (
                    REID_MAP_GATE_SAME_CAM_PX
                    if identity.get("last_cam") == cam_name
                    else REID_MAP_GATE_CROSS_CAM_PX
                )
                reason = (
                    "incompatible_location"
                    if (
                        previous is not None
                        and current is not None
                        and self._map_distance(previous, current) > gate
                    )
                    else None
                )
        return {
            "passed": reason is None,
            "reason": reason,
            "topology": topology,
        }

    def _can_match(
        self,
        identity,
        cam_name,
        now_ts,
        map_pos,
        box_wh
    ):

        state = identity.get("state", IDENTITY_ACTIVE)
        if state not in IDENTITY_ASSIGNABLE_STATES:
            return False
        topology_ok, _ = self._topology_gate(
            identity,
            cam_name,
            {"map_pos": map_pos},
            now_ts,
        )
        if not topology_ok:
            return False

        deadline = self._identity_match_deadline(identity)
        if deadline is None or now_ts > deadline:
            return False

        if not self._size_ratio_ok(
            box_wh,
            identity.get("box_wh")
        ):
            return False

        prev_pos = identity.get(
            "last_map_pos"
        )

        if (
            map_pos is not None
            and
            prev_pos is not None
        ):

            dist = self._map_distance(
                map_pos,
                prev_pos
            )

            if dist is not None:

                if (
                    identity.get(
                        "last_cam"
                    )
                    ==
                    cam_name
                ):

                    if (
                        dist
                        >
                        REID_MAP_GATE_SAME_CAM_PX
                    ):

                        return False

                else:

                    if (
                        dist
                        >
                        REID_MAP_GATE_CROSS_CAM_PX
                    ):

                        return False

        return True

    def _hard_gate_reason(self, identity, cam_name, det, now_ts):
        """Return a stable audit reason instead of only a boolean gate."""
        return self._hard_gate_diagnostics(
            identity,
            cam_name,
            det,
            now_ts,
        )["reason"]

    def _tracklet_quality_score(self, det):
        """Normalize available crop-quality evidence to [0, 1]."""
        try:
            confidence = float(
                det.get("detector_confidence", det.get("conf", 1.0))
            )
            crop_w, crop_h = det.get(
                "crop_size",
                det.get("box_wh", (0, 0)),
            )
            crop_w = float(crop_w)
            crop_h = float(crop_h)
            blur = float(det.get("blur_variance", REID_MIN_BLUR_VARIANCE))
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not all(np.isfinite(value) for value in (
            confidence,
            crop_w,
            crop_h,
            blur,
        )):
            return 0.0
        confidence_quality = max(0.0, min(1.0, confidence))
        crop_quality = max(
            0.0,
            min(1.0, min(crop_w, crop_h) / max(REID_MIN_CROP_SIZE, 1)),
        )
        blur_quality = max(
            0.0,
            min(1.0, blur / max(REID_MIN_BLUR_VARIANCE, 1e-6)),
        )
        occlusion_quality = 0.5 if det.get("overlap", False) else 1.0
        score = (
            confidence_quality
            * crop_quality
            * blur_quality
            * occlusion_quality
        )
        if not np.isfinite(score):
            return 0.0
        return float(max(0.0, min(1.0, score)))

    def _ambiguity_reason(
        self, pair_cache, idx, candidate_gids, cross_camera
    ):
        """Return (reason, margin) with an adaptive false-merge guard."""
        viable = []
        for gid in candidate_gids:
            pair = pair_cache.get((idx, gid), {})
            if "score" not in pair:
                continue
            try:
                score = float(pair["score"])
                appearance = float(pair.get("appearance", -1.0))
            except (TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite(score):
                continue
            viable.append((score, appearance, int(gid)))

        viable.sort(key=lambda item: item[0], reverse=True)
        if len(viable) < 2:
            return None, None

        self.merge_guard_checks += 1
        top_score, top_app, top_gid = viable[0]
        second_score, second_app, second_gid = viable[1]
        margin = float(top_score - second_score)

        base_required = (
            ASSIGN_CROSS_CAM_MIN_MARGIN
            if cross_camera
            else ASSIGN_SAME_CAM_MIN_MARGIN
        )
        required = float(base_required)

        appearance_margin = None
        if np.isfinite(top_app) and np.isfinite(second_app):
            appearance_margin = float(top_app - second_app)

        if (
            MERGE_GUARD_ENABLED
            and appearance_margin is not None
            and appearance_margin < MERGE_GUARD_APPEARANCE_MARGIN
        ):
            required = max(
                required,
                MERGE_GUARD_CROSS_CAM_MIN_MARGIN
                if cross_camera
                else MERGE_GUARD_SAME_CAM_MIN_MARGIN,
            )

        rejected = margin < required
        self.last_merge_guard_diagnostic = {
            "row": int(idx),
            "cross_camera": bool(cross_camera),
            "top_gid": int(top_gid),
            "second_gid": int(second_gid),
            "top_score": float(top_score),
            "second_score": float(second_score),
            "score_margin": float(margin),
            "top_appearance": float(top_app),
            "second_appearance": float(second_app),
            "appearance_margin": appearance_margin,
            "required_margin": float(required),
            "rejected": bool(rejected),
        }

        if rejected:
            self.merge_guard_rejections += 1
            return "ambiguous_top1_top2_merge_guard", margin
        return None, margin

    @staticmethod
    def _finite_pair(pair):
        if not isinstance(pair, dict) or "score" not in pair:
            return None
        try:
            score = float(pair["score"])
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(score):
            return None
        pair["score"] = score
        return pair


    # ========================================================
    # PAIR SCORE
    # ========================================================

    def _pair_score(
        self,
        gid,
        identity,
        cam_name,
        det,
        now_ts,
        prev_assignments
    ):

        appearance = (
            self._gallery_similarity(
                det["emb"],
                identity
            )
        )

        cross_camera = (
            identity.get("last_cam")
            !=
            cam_name
        )

        map_s = self._map_score(
            identity,
            det.get("map_pos"),
            cross_camera=cross_camera
        )

        time_s = self._time_score(
            identity,
            now_ts
        )
        # Motion is camera-coordinate evidence and is intentionally excluded
        # from cross-camera scoring. Keep an explicit diagnostic value so the
        # pair result is complete in both branches.
        motion = 0.0
        quality = self._tracklet_quality_score(det)
        quality_adjusted_appearance = max(0.0, appearance) * quality

        # ====================================================
        # CROSS CAMERA
        # ====================================================

        if cross_camera:

            appearance_score = quality_adjusted_appearance

            map_score = max(
                0.0,
                map_s
            )

            time_score = max(
                0.0,
                time_s
            )

            total = (
                ASSIGN_CROSS_CAM_APPEARANCE_WEIGHT * appearance_score
                + ASSIGN_CROSS_CAM_MAP_WEIGHT * map_score
                + ASSIGN_CROSS_CAM_TIME_WEIGHT * time_score
            )

            source_type = "cross-camera"

        # ====================================================
        # SAME CAMERA
        # ====================================================

        else:

            motion = (
                self._motion_score(
                    cam_name,
                    gid,
                    det["box"],
                    prev_assignments,
                    event_time=now_ts
                )
            )

            app_w = (
                ASSIGN_SAME_CAM_APPEARANCE_WEIGHT
            )

            motion_w = (
                ASSIGN_SAME_CAM_MOTION_WEIGHT
            )

            map_w = (
                ASSIGN_SAME_CAM_MAP_WEIGHT
            )

            time_w = (
                ASSIGN_SAME_CAM_TIME_WEIGHT
            )

            total = (
                app_w * quality_adjusted_appearance
                +
                motion_w * motion
                +
                map_w * map_s
                +
                time_w * time_s
            )

            if (
                identity.get(
                    "last_cam"
                )
                ==
                cam_name
            ):

                total += (
                    ASSIGN_SAME_CAM_BONUS
                )

            source_type = "same-camera"


        # ====================================================
        # STRONG APPEARANCE BONUS
        # ====================================================

        if (
            cross_camera
            and
            appearance
            >=
            REID_CROSS_CAM_STRONG_THRESHOLD
        ):

            total += 0.08


        # ====================================================
        # OCCLUSION
        # ====================================================

        if det.get(
            "overlap",
            False
        ):

            if (
                det.get(
                    "forced_gid"
                )
                ==
                gid
            ):

                total += (
                    ASSIGN_OVERLAP_FREEZE_BONUS
                )


        return {

            "gid": gid,

            "score": float(total),

            "appearance": float(
                appearance
            ),

            "quality_adjusted_appearance": float(quality_adjusted_appearance),

            "tracklet_quality": float(quality),

            "motion": float(
                motion
            ),

            "map": float(
                map_s
            ),

            "time": float(
                time_s
            ),

            "cross_camera": bool(
                cross_camera
            ),

            "source_type": source_type

        }


    # ========================================================
    # ACCEPT MATCH
    # ========================================================

    def _accept_match(
        self,
        pair,
        identity,
        cam_name,
        det
    ):

        if pair is None:
            return False

        appearance = (
            pair["appearance"]
        )

        score = pair["score"]

        cross_camera = (
            pair["cross_camera"]
        )


        # ====================================================
        # CROSS CAMERA
        # ====================================================

        if cross_camera:

            # ReID เธชเธนเธเธกเธฒเธ
            if (
                REID_THRESHOLD_SAFETY_MODE
                == "validated"
                and
                appearance
                >=
                REID_CROSS_CAM_STRONG_THRESHOLD
            ):

                return True

            # เธ•เนเธญเธเธเนเธฒเธเธ—เธฑเนเธ appearance + score
            if (
                appearance
                >=
                REID_CROSS_CAM_THRESHOLD
                and
                score
                >=
                ASSIGN_CROSS_CAM_SCORE_THRESHOLD
            ):

                return True

            return False


        # ====================================================
        # SAME CAMERA
        # ====================================================

        if (
            REID_THRESHOLD_SAFETY_MODE
            == "validated"
            and
            appearance
            >=
            REID_SAME_CAM_STRONG_THRESHOLD
        ):

            return True

        if (
            identity.get(
                "last_cam"
            )
            ==
            cam_name
            and
            appearance
            >=
            REID_SAME_CAM_THRESHOLD
            and
            score
            >=
            ASSIGN_SAME_CAM_SCORE_THRESHOLD
        ):

            return True

        return (
            appearance
            >=
            REID_SAME_CAM_THRESHOLD
            and
            score
            >=
            ASSIGN_SAME_CAM_SCORE_THRESHOLD
        )


    # ========================================================
    # UPDATE IDENTITY
    # ========================================================

    def _transition_identity(
        self,
        gid,
        identity,
        new_state,
        now_ts,
        reason,
        persist=True,
    ):
        old_state = identity.get("state", IDENTITY_PROVISIONAL)
        if old_state not in IDENTITY_ALLOWED_TRANSITIONS:
            raise ValueError(f"unknown identity state: {old_state!r}")
        if new_state not in IDENTITY_ALLOWED_TRANSITIONS:
            raise ValueError(f"unknown identity state: {new_state!r}")
        if old_state == new_state:
            return False
        if not reason:
            raise ValueError("identity transitions require a reason")

        normal_transition = new_state in IDENTITY_ALLOWED_TRANSITIONS.get(
            old_state,
            frozenset(),
        )
        # Corrupt legacy/runtime records with no usable timestamp are retired
        # explicitly instead of being made eligible for matching again.
        invalid_record_expiry = (
            reason == "invalid_persisted_identity"
            and new_state == IDENTITY_EXPIRED
            and old_state in {IDENTITY_PROVISIONAL, IDENTITY_ACTIVE}
        )
        if not normal_transition and not invalid_record_expiry:
            raise ValueError(
                f"invalid identity transition: {old_state!r} -> {new_state!r}"
            )

        transition_ts = float(now_ts)
        missing = object()
        previous = {
            "state": identity.get("state", missing),
            "state_updated_at": identity.get("state_updated_at", missing),
            "state_reason": identity.get("state_reason", missing),
            "state_transitions": missing,
        }
        existing_history = identity.get("state_transitions", missing)
        if existing_history is not missing:
            if not isinstance(existing_history, list):
                raise ValueError("identity transition history must be a list")
            previous["state_transitions"] = list(existing_history)

        identity["state"] = new_state
        identity["state_updated_at"] = transition_ts
        identity["state_reason"] = reason
        history = identity.setdefault("state_transitions", [])
        history.append({
            "from": old_state,
            "to": new_state,
            "ts": transition_ts,
            "reason": reason,
        })
        del history[:-IDENTITY_TRANSITION_HISTORY_SIZE]

        try:
            if persist and self.identity_store:
                self.identity_store.save_identity(
                    gid,
                    identity,
                    "state_transition",
                    reason,
                    transition_ts,
                )
        except Exception:
            for key, value in previous.items():
                if value is missing:
                    identity.pop(key, None)
                else:
                    identity[key] = value
            raise
        return True

    def identity_state_diagnostics(self):
        with self.lock:
            counts = {state: 0 for state in (
                IDENTITY_PROVISIONAL,
                IDENTITY_ACTIVE,
                IDENTITY_DORMANT,
                IDENTITY_EXPIRED,
            )}
            transitions = []
            presence = []
            handoffs = []
            for gid, identity in self.identities.items():
                state = identity.get("state", IDENTITY_ACTIVE)
                counts[state] = counts.get(state, 0) + 1
                transitions.extend(
                    {"gid": gid, **item}
                    for item in identity.get("state_transitions", [])
                    if isinstance(item, dict)
                    and isinstance(item.get("ts"), (int, float))
                )
                presence.extend(
                    {"gid": gid, **copy.deepcopy(record)}
                    for record in identity.get("camera_presence", {}).values()
                    if isinstance(record, dict)
                )
                handoffs.extend(
                    copy.deepcopy(item)
                    for item in identity.get("handoff_history", [])
                    if isinstance(item, dict)
                )
            persistence = (
                self.identity_store.status()
                if self.identity_store
                else {
                    "path": None,
                    "connected": False,
                    "schema_version": None,
                    "state_counts": {},
                    "recent_transitions": [],
                }
            )
            return {
                "state_counts": counts,
                "recent_transitions": sorted(
                    transitions,
                    key=lambda item: item["ts"],
                    reverse=True,
                )[:IDENTITY_TRANSITION_HISTORY_SIZE],
                "next_global_id": self.next_global_id,
                "camera_presence": sorted(
                    presence,
                    key=lambda item: (
                        item.get("gid", 0),
                        item.get("camera", ""),
                    ),
                ),
                "recent_handoffs": sorted(
                    handoffs,
                    key=lambda item: item.get("entry_event_time", 0.0),
                    reverse=True,
                )[:IDENTITY_HANDOFF_HISTORY_SIZE],
                "unresolved_cross_camera_handoffs": [
                    self._unresolved_handoff_status_item(record)
                    for _, record in sorted(
                        self.unresolved_cross_camera_handoffs.items(),
                        key=lambda item: (
                            item[1].get("first_event_time", 0.0),
                            repr(item[0]),
                        ),
                    )
                ],
                "unresolved_handoff_policy": {
                    "min_samples": AMBIGUOUS_HANDOFF_MIN_SAMPLES,
                    "max_samples": AMBIGUOUS_HANDOFF_MAX_SAMPLES,
                    "min_solo_event_sec": (
                        AMBIGUOUS_HANDOFF_MIN_SOLO_EVENT_SEC
                    ),
                    "max_event_sec": AMBIGUOUS_HANDOFF_MAX_EVENT_SEC,
                    "temporal_top1_min_ratio": (
                        AMBIGUOUS_HANDOFF_TEMPORAL_TOP1_MIN_RATIO
                    ),
                    "max_records": AMBIGUOUS_HANDOFF_MAX_RECORDS,
                    "capacity_drop_count": (
                        self.unresolved_handoff_capacity_drop_count
                    ),
                },
                "unresolved_handoff_forensic_trace": copy.deepcopy(
                    self.unresolved_handoff_forensic_trace
                ),
                "persistence": persistence,
            }
    def close(self):
        with self.lock:
            if not self.identity_store:
                return False
            return self.identity_store.close()

    def _gallery_quality_reason(self, det):
        confidence = det.get("detector_confidence", det.get("conf"))
        if confidence is None:
            return "low_detector_confidence"
        try:
            confidence = float(confidence)
            crop_w, crop_h = det.get(
                "crop_size",
                det.get("box_wh", (0, 0)),
            )
            crop_w = float(crop_w)
            crop_h = float(crop_h)
            blur_variance = float(det.get("blur_variance", 0.0))
            border_clip_ratio = float(det.get("border_clip_ratio", 0.0))
        except (TypeError, ValueError, OverflowError):
            return "invalid_quality_metadata"
        if not all(np.isfinite(value) for value in (
            confidence,
            crop_w,
            crop_h,
            blur_variance,
            border_clip_ratio,
        )):
            return "invalid_quality_metadata"
        if confidence < REID_MIN_DETECTION_CONFIDENCE:
            return "low_detector_confidence"
        if min(crop_w, crop_h) < REID_MIN_CROP_SIZE:
            return "crop_too_small"
        if blur_variance < REID_MIN_BLUR_VARIANCE:
            return "blurred_crop"
        if det.get("overlap", False) and REID_MAX_OVERLAP_FOR_GALLERY <= 0.0:
            return "overlap_or_occlusion"
        if border_clip_ratio > REID_MAX_BORDER_CLIP_RATIO:
            return "border_clipped"
        return None

    @staticmethod
    def _tracklet_generation(det):
        generation = det.get("camera_generation")
        if generation is None:
            generation = det.get("coordinator_generation")
        if generation is None:
            return None
        try:
            return int(generation)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _validated_gallery_embedding(identity, value):
        try:
            embedding = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return None, "invalid_embedding"
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            return None, "invalid_embedding"
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm):
            return None, "invalid_embedding"
        if norm < 1e-8:
            return None, "zero_embedding"

        expected_dimension = None
        candidates = [identity.get("embedding")]
        candidates.extend(identity.get("gallery", []))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                candidate_array = np.asarray(
                    candidate,
                    dtype=np.float32,
                ).reshape(-1)
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                candidate_array.size > 0
                and np.all(np.isfinite(candidate_array))
            ):
                expected_dimension = int(candidate_array.size)
                break
        if (
            expected_dimension is not None
            and embedding.size != expected_dimension
        ):
            return None, "embedding_dimension_mismatch"
        return embedding / norm, None

    @staticmethod
    def _tracklet_record_is_valid(record, gid, generation, identity):
        structurally_valid = (
            isinstance(record, dict)
            and record.get("gid") == gid
            and record.get("generation") == generation
            and isinstance(record.get("sample_count"), int)
            and record.get("sample_count", -1) >= 0
            and record.get("sample_count", 0) <= REID_TRACKLET_MAX_SAMPLES
            and isinstance(record.get("samples"), list)
            and len(record.get("samples", [])) <= REID_TRACKLET_MAX_SAMPLES
            and record.get("sample_count", 0) >= len(record.get("samples", []))
            and isinstance(record.get("gallery_committed"), bool)
            and isinstance(record.get("last_seen"), (int, float))
            and np.isfinite(float(record.get("last_seen", np.nan)))
        )
        if not structurally_valid:
            return False

        sample_dimension = None
        for item in record["samples"]:
            if not isinstance(item, dict):
                return False
            timestamp = item.get("ts")
            if (
                not isinstance(timestamp, (int, float))
                or not np.isfinite(float(timestamp))
            ):
                return False
            embedding, reason = (
                GlobalIdentityManager._validated_gallery_embedding(
                    identity,
                    item.get("emb"),
                )
            )
            if reason is not None:
                return False
            if sample_dimension is None:
                sample_dimension = int(embedding.size)
            elif embedding.size != sample_dimension:
                return False
        return True

    def _record_tracklet_sample(self, gid, cam_name, local_id, det, now_ts):
        """Update gallery only from diverse, quality-approved tracklet evidence."""
        local_key = (cam_name, int(local_id))
        identity = self.identities[gid]
        diagnostics = identity.get("gallery_diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            identity["gallery_diagnostics"] = diagnostics
        for key, default in (
            ("accepted_updates", 0),
            ("rejected_updates", 0),
            ("last_rejection_reason", None),
            ("tracklet_sample_count", 0),
            ("prototype_quality", 0.0),
        ):
            diagnostics.setdefault(key, default)

        generation = self._tracklet_generation(det)
        record = self.tracklets.get(local_key)
        if record is not None and not self._tracklet_record_is_valid(
            record,
            gid,
            generation,
            identity,
        ):
            self.tracklets.pop(local_key, None)
            record = None
            diagnostics["tracklet_sample_count"] = 0

        if not det.get("local_track_confirmed", True):
            self.tracklets.pop(local_key, None)
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = "unconfirmed_local_track"
            diagnostics["tracklet_sample_count"] = 0
            return False, "unconfirmed_local_track"

        reason = self._gallery_quality_reason(det)
        if reason is not None:
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = reason
            return False, reason

        embedding, reason = self._validated_gallery_embedding(
            identity,
            det.get("emb"),
        )
        if reason is not None:
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = reason
            return False, reason

        if record is not None and record["gallery_committed"]:
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = "tracklet_already_committed"
            diagnostics["tracklet_sample_count"] = record["sample_count"]
            return False, "tracklet_already_committed"

        next_sample_count = min(
            (record["sample_count"] if record is not None else 0) + 1,
            REID_TRACKLET_MAX_SAMPLES,
        )
        identity_snapshot = None
        tracklet_existed = False
        tracklet_snapshot = None
        if next_sample_count >= REID_TRACKLET_MIN_SAMPLES:
            identity_snapshot = copy.deepcopy(identity)
            tracklet_existed = local_key in self.tracklets
            tracklet_snapshot = copy.deepcopy(self.tracklets.get(local_key))

        if record is None:
            record = {
                "gid": gid,
                "generation": generation,
                "last_seen": float(now_ts),
                "last_event_time": float(now_ts),
                "sample_count": 0,
                "samples": [],
                "gallery_committed": False,
            }
            self.tracklets[local_key] = record

        record["last_seen"] = float(now_ts)
        record["last_event_time"] = float(now_ts)
        record["sample_count"] = next_sample_count
        samples = record["samples"]
        if not any(
            cosine_similarity(embedding, item["emb"])
            >= REID_GALLERY_DIVERSITY_THRESHOLD
            for item in samples
        ):
            samples.append({
                "emb": embedding.copy(),
                "ts": float(now_ts),
                "event_time": float(now_ts),
            })
            if len(samples) > REID_TRACKLET_MAX_SAMPLES:
                del samples[:-REID_TRACKLET_MAX_SAMPLES]

        prototype_raw = np.mean(
            [item["emb"] for item in samples],
            axis=0,
        )
        prototype_norm = float(np.linalg.norm(prototype_raw))
        diagnostics["tracklet_sample_count"] = record["sample_count"]
        if not np.isfinite(prototype_norm) or prototype_norm < 1e-8:
            if record["sample_count"] < REID_TRACKLET_MIN_SAMPLES:
                diagnostics["prototype_quality"] = 0.0
                diagnostics["last_rejection_reason"] = None
                return False, "tracklet_not_mature"
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = "invalid_tracklet_prototype"
            record["gallery_committed"] = True
            return False, "invalid_tracklet_prototype"
        prototype = np.asarray(
            prototype_raw / prototype_norm,
            dtype=np.float32,
        )
        diagnostics["prototype_quality"] = float(
            np.mean([cosine_similarity(item["emb"], prototype) for item in samples])
        )
        diagnostics["last_rejection_reason"] = None

        if record["sample_count"] < REID_TRACKLET_MIN_SAMPLES:
            return False, "tracklet_not_mature"

        gallery = identity.setdefault("gallery", [])
        normalized_gallery = []
        for item in gallery:
            try:
                gallery_item = np.asarray(item, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError, OverflowError):
                continue
            item_norm = float(np.linalg.norm(gallery_item))
            if (
                gallery_item.size != prototype.size
                or not np.all(np.isfinite(gallery_item))
                or not np.isfinite(item_norm)
                or item_norm < 1e-8
            ):
                continue
            normalized_gallery.append(gallery_item / item_norm)

        if any(
            cosine_similarity(prototype, item)
            >= REID_GALLERY_DIVERSITY_THRESHOLD
            for item in normalized_gallery
        ):
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = "prototype_near_duplicate"
            record["gallery_committed"] = True
            return False, "prototype_near_duplicate"

        candidate_gallery = (
            normalized_gallery + [prototype]
        )[-REID_GALLERY_SIZE:]
        aggregate_raw = np.mean(candidate_gallery, axis=0)
        aggregate_norm = float(np.linalg.norm(aggregate_raw))
        if not np.isfinite(aggregate_norm) or aggregate_norm < 1e-8:
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = "invalid_gallery_aggregate"
            record["gallery_committed"] = True
            return False, "invalid_gallery_aggregate"
        aggregate = np.asarray(
            aggregate_raw / aggregate_norm,
            dtype=np.float32,
        )

        try:
            transitioned = False
            if identity.get("state", IDENTITY_PROVISIONAL) == IDENTITY_PROVISIONAL:
                transitioned = self._transition_identity(
                    gid,
                    identity,
                    IDENTITY_ACTIVE,
                    now_ts,
                    "mature_tracklet",
                    persist=False,
                )
            identity["gallery"] = candidate_gallery
            identity["embedding"] = aggregate
            identity["gallery_mature"] = True
            diagnostics["accepted_updates"] += 1
            diagnostics["last_rejection_reason"] = None
            record["gallery_committed"] = True
            if self.identity_store:
                self.identity_store.save_identity(
                    gid,
                    identity,
                    "state_transition" if transitioned else "gallery_update",
                    "mature_tracklet" if transitioned else "quality_approved",
                    now_ts,
                )
        except Exception:
            if identity_snapshot is None:
                raise
            identity.clear()
            identity.update(identity_snapshot)
            if tracklet_existed:
                self.tracklets[local_key] = tracklet_snapshot
            else:
                self.tracklets.pop(local_key, None)
            raise
        return True, None

    def _gallery_assignment_diagnostics(self, gid, accepted, reason):
        identity = self.identities.get(gid, {})
        gallery = identity.get("gallery", [])
        if not isinstance(gallery, (list, tuple)):
            gallery = []
        gallery_diagnostics = identity.get("gallery_diagnostics", {})
        if not isinstance(gallery_diagnostics, dict):
            gallery_diagnostics = {}
        sample_count = gallery_diagnostics.get("tracklet_sample_count", 0)
        if not isinstance(sample_count, int) or sample_count < 0:
            sample_count = 0
        return {
            "gallery_update_accepted": bool(accepted),
            "gallery_rejection_reason": reason,
            "gallery_mature": identity.get("gallery_mature"),
            "tracklet_sample_count": sample_count,
            "gallery_size": len(gallery),
        }

    def _update_identity(
        self,
        gid,
        cam_name,
        emb,
        map_pos,
        box_wh,
        now_ts,
        last_score=None
    ):

        identity = (
            self.identities[gid]
        )

        transition_event = None
        if identity.get("state") == IDENTITY_DORMANT:
            reason = "cross_camera_recovery" if identity.get("last_cam") != cam_name else "same_camera_recovery"
            self._transition_identity(
                gid,
                identity,
                IDENTITY_ACTIVE,
                now_ts,
                reason,
                persist=False,
            )
            transition_event = {
                "event_type": "state_transition",
                "reason": reason,
                "timestamp": float(now_ts),
            }

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        identity["last_cam"] = cam_name

        identity["last_seen"] = now_ts
        identity["last_event_time"] = now_ts

        identity["last_map_pos"] = map_pos

        identity["box_wh"] = box_wh

        if last_score is not None:

            identity["last_score"] = float(
                last_score
            )

        return transition_event


    # ========================================================
    # REMEMBER SAME CAMERA
    # ========================================================

    def _remember_recent_same_cam(
        self,
        gid,
        cam_name,
        emb,
        map_pos,
        box_wh,
        now_ts
    ):

        self.recent_same_cam.append({

            "gid": gid,

            "cam_name": cam_name,

            "embedding": l2_normalize(
                emb
            ),

            "map_pos": map_pos,

            "box_wh": box_wh,

            "ts": now_ts

        })

        if len(
            self.recent_same_cam
        ) > 300:

            self.recent_same_cam = (
                self.recent_same_cam[
                    -300:
                ]
            )


    # ========================================================
    # REMEMBER CROSS CAMERA
    # ========================================================

    def _remember_cross_cam(
        self,
        gid,
        cam_name,
        emb,
        map_pos,
        now_ts
    ):

        self.recent_cross_cam.append({

            "gid": gid,

            "cam_name": cam_name,

            "embedding": l2_normalize(
                emb
            ),

            "map_pos": map_pos,

            "ts": now_ts

        })

        if len(
            self.recent_cross_cam
        ) > 300:

            self.recent_cross_cam = (
                self.recent_cross_cam[
                    -300:
                ]
            )


    # ========================================================
    # COMMIT ASSIGNMENT
    # ========================================================

    def _commit_assignment(
        self,
        gid,
        cam_name,
        local_id,
        emb,
        map_pos,
        box_wh,
        now_ts,
        score,
        source,
        generation=None,
    ):
        """Commit RAM and persistence as one recoverable assignment step."""
        state_record = self.state_machine.begin(
            cam_name,
            local_id,
            generation,
        )
        self.state_machine.authorize_commit(state_record, source)
        local_key = (cam_name, int(local_id))
        identity_existed = gid in self.identities
        identity_snapshot = (
            copy.deepcopy(self.identities[gid])
            if identity_existed
            else None
        )
        mapping_existed = local_key in self.local_to_global
        mapping_snapshot = copy.deepcopy(
            self.local_to_global.get(local_key)
        )
        tracklet_existed = local_key in self.tracklets
        tracklet_snapshot = copy.deepcopy(
            self.tracklets.get(local_key)
        )
        hold_existed = local_key in self.occlusion_hold
        hold_snapshot = copy.deepcopy(
            self.occlusion_hold.get(local_key)
        )
        recent_same_snapshot = self.recent_same_cam
        recent_same_length = len(recent_same_snapshot)
        recent_cross_snapshot = self.recent_cross_cam
        recent_cross_length = len(recent_cross_snapshot)

        try:
            result = self._commit_assignment_mutating(
                gid,
                cam_name,
                local_id,
                emb,
                map_pos,
                box_wh,
                now_ts,
                score,
                source,
                generation=generation,
            )
            self.state_machine.record_commit(
                cam_name,
                local_id,
                generation,
                gid,
                source,
                result.get("assignment_reason") if isinstance(result, dict) else None,
            )
            return result
        except Exception:
            if identity_existed:
                current_identity = self.identities.get(gid)
                if (
                    isinstance(current_identity, dict)
                    and isinstance(identity_snapshot, dict)
                ):
                    current_identity.clear()
                    current_identity.update(identity_snapshot)
                else:
                    self.identities[gid] = identity_snapshot
            else:
                self.identities.pop(gid, None)

            for container, existed, snapshot in (
                (self.local_to_global, mapping_existed, mapping_snapshot),
                (self.tracklets, tracklet_existed, tracklet_snapshot),
                (self.occlusion_hold, hold_existed, hold_snapshot),
            ):
                if existed:
                    container[local_key] = snapshot
                else:
                    container.pop(local_key, None)

            del recent_same_snapshot[recent_same_length:]
            self.recent_same_cam = recent_same_snapshot
            del recent_cross_snapshot[recent_cross_length:]
            self.recent_cross_cam = recent_cross_snapshot
            raise

    def _commit_assignment_mutating(
        self,
        gid,
        cam_name,
        local_id,
        emb,
        map_pos,
        box_wh,
        now_ts,
        score,
        source,
        generation=None,
    ):

        local_key = (
            cam_name,
            int(local_id)
        )

        if gid in self.identities and self._assignable_identity(gid) is None:
            raise ValueError(
                f"identity {gid} is not eligible for assignment"
            )

        # Reconcile the camera-local ownership namespace before mutating the
        # destination identity.  All assignment sources (fixed claims,
        # occlusion, transition recovery, and Hungarian results) converge here
        # under the manager lock, so no stale GID can remain active for the
        # same (camera, local track, generation) key.
        reassigned_gids = self._reconcile_local_track_presence(
            cam_name,
            local_id,
            generation,
            gid,
            now_ts,
        )

        previous_mapping = self.local_to_global.get(local_key)
        existing_tracklet = self.tracklets.get(local_key)
        mapping_owner_changed = (
            previous_mapping is not None
            and (
                not isinstance(previous_mapping, dict)
                or previous_mapping.get("gid") != gid
            )
        )
        tracklet_owner_changed = (
            existing_tracklet is not None
            and (
                not isinstance(existing_tracklet, dict)
                or existing_tracklet.get("gid") != gid
            )
        )
        if mapping_owner_changed or tracklet_owner_changed:
            self.tracklets.pop(local_key, None)

        existing_hold = self.occlusion_hold.get(local_key)
        if (
            existing_hold is not None
            and (
                not isinstance(existing_hold, dict)
                or existing_hold.get("gid") != gid
            )
        ):
            self.occlusion_hold.pop(local_key, None)

        transition_event = None
        presence_update = None
        if gid in self.identities:
            presence_update = self._record_camera_presence(
                gid,
                self.identities[gid],
                cam_name,
                local_id,
                now_ts,
                source,
                score,
                emb,
                map_pos,
                box_wh,
                generation=generation,
            )
            transition_event = self._update_identity(
                gid,
                cam_name,
                emb,
                map_pos,
                box_wh,
                now_ts,
                last_score=score
            )

        else:

            self.identities[gid] = {

                "embedding":
                    l2_normalize(emb),

                # The first frame is a temporary bootstrap for matching.
                # It does not become gallery evidence until its tracklet
                # reaches the quality-gated prototype stage.
                "gallery": [],

                "gallery_mature": False,

                "gallery_diagnostics": {
                    "accepted_updates": 0,
                    "rejected_updates": 0,
                    "last_rejection_reason": None,
                    "tracklet_sample_count": 0,
                    "prototype_quality": 0.0,
                },

                "last_cam":
                    cam_name,

                "last_seen":
                    now_ts,

                "last_event_time":
                    now_ts,

                "last_map_pos":
                    map_pos,

                "box_wh":
                    box_wh,

                "last_score":
                    float(score),

                "state": IDENTITY_PROVISIONAL,

                "state_updated_at": float(now_ts),

                "state_reason": "new_tracklet",

                "state_transitions": [{
                    "from": None, "to": IDENTITY_PROVISIONAL,
                    "ts": float(now_ts), "reason": "new_tracklet",
                }]

            }

            presence_update = self._record_camera_presence(
                gid,
                self.identities[gid],
                cam_name,
                local_id,
                now_ts,
                source,
                score,
                emb,
                map_pos,
                box_wh,
                generation=generation,
            )


        self.local_to_global[
            local_key
        ] = {

            "gid": gid,

            "last_seen": now_ts,

            "generation": generation,

        }

        self._enforce_local_track_presence_invariant(
            cam_name,
            local_id,
            generation,
            gid,
            now_ts,
        )


        self._remember_recent_same_cam(
            gid,
            cam_name,
            emb,
            map_pos,
            box_wh,
            now_ts
        )


        if source in ("cross-camera", "global-cross-camera"):

            self._remember_cross_cam(
                gid,
                cam_name,
                emb,
                map_pos,
                now_ts
            )

        if self.identity_store:
            for reassigned_gid in reassigned_gids:
                if reassigned_gid in self.identities:
                    self.identity_store.save_identity(
                        reassigned_gid,
                        self.identities[reassigned_gid],
                        "presence_reassigned",
                        "local_track_reassigned",
                        now_ts,
                    )
            preceding_events = []
            if transition_event is not None:
                preceding_events.append(transition_event)
            handoff = presence_update.get("handoff")
            if handoff is not None:
                preceding_events.append({
                    "event_type": "handoff",
                    "reason": handoff["reason"],
                    "timestamp": handoff["entry_event_time"],
                    "payload": {"handoff": handoff},
                })
            persistence_options = (
                {"preceding_events": preceding_events}
                if preceding_events
                else {}
            )
            self.identity_store.save_identity(
                gid,
                self.identities[gid],
                "assignment",
                source,
                now_ts,
                **persistence_options,
            )

        return {

            "gid": gid,

            "score": float(score),

            "source": source,

            "presence": presence_update["new_presence"],

            "handoff_committed": presence_update["handoff"] is not None,

            "handoff": presence_update["handoff"],

        }


    # ========================================================
    # NEW GLOBAL ID
    # ========================================================

    def _new_identity(
        self,
        cam_name,
        local_id,
        emb,
        map_pos,
        box_wh,
        now_ts,
        generation=None,
    ):

        gid = self.next_global_id

        self.next_global_id += 1

        return self._commit_assignment(

            gid,

            cam_name,

            local_id,

            emb,

            map_pos,

            box_wh,

            now_ts,

            1.0,

            "new",

            generation=generation,

        )


    # ========================================================
    # RECENT SAME CAMERA MATCH
    # ========================================================

    def _find_recent_same_cam_match(
        self,
        cam_name,
        emb,
        map_pos,
        box_wh,
        now_ts,
        used_gids=None,
        eligible_gids=None,
    ):

        best_gid = None

        best_score = -999.0

        for item in reversed(
            self.recent_same_cam
        ):

            if (
                item.get("cam_name")
                !=
                cam_name
            ):

                continue

            item_ts = item.get("ts")
            if not isinstance(item_ts, (int, float)):
                continue
            item_ts = float(item_ts)
            if not np.isfinite(item_ts):
                continue
            dt = now_ts - item_ts

            if (
                dt < 0.0
                or
                dt
                >
                REID_RECENT_SAME_CAM_SEC
            ):

                continue

            gid = item.get(
                "gid"
            )

            identity = self._matchable_identity_for_camera(gid, cam_name)
            if identity is None:
                continue

            if (
                used_gids is not None
                and
                gid in used_gids
            ):

                continue

            if eligible_gids is not None and gid not in eligible_gids:
                continue

            if self._hard_gate_reason(
                identity,
                cam_name,
                {
                    "map_pos": map_pos,
                    "box_wh": box_wh,
                },
                now_ts,
            ) is not None:
                continue

            if not self._size_ratio_ok(
                box_wh,
                item.get("box_wh")
            ):

                continue

            score = self._gallery_similarity(
                emb,
                identity,
            )
            if not np.isfinite(score):
                continue

            if (
                map_pos is not None
                and
                item.get("map_pos") is not None
            ):

                dist = self._map_distance(
                    map_pos,
                    item.get("map_pos")
                )

                if dist is not None:

                    score -= (
                        min(
                            dist
                            /
                            max(
                                REID_MAP_GATE_SAME_CAM_PX,
                                1.0
                            ),
                            1.0
                        )
                        *
                        0.10
                    )


            score -= (
                min(
                    dt
                    /
                    max(
                        REID_RECENT_SAME_CAM_SEC,
                        1e-6
                    ),
                    1.0
                )
                *
                0.08
            )


            if (
                score
                >
                best_score
                and
                score
                >=
                REID_RECENT_SAME_CAM_THRESHOLD
            ):

                best_gid = gid

                best_score = score


        return (
            best_gid,
            float(best_score)
        )


    # ========================================================
    # BATCH ASSIGNMENT
    # ========================================================

    def assign_batch(
        self,
        cam_name,
        detections,
        prev_assignments=None,
        event_time=None
    ):

        _sync_identity_dependencies()

        now_ts = (
            time.time()
            if event_time is None
            else float(event_time)
        )

        prev_assignments = (
            prev_assignments
            or
            []
        )

        results = [
            None
            for _ in detections
        ]

        for detection in detections:
            record = self.state_machine.begin(
                cam_name,
                detection["tid"],
                detection.get("coordinator_generation", detection.get("camera_generation")),
            )
            for state in ("candidates_built", "gated", "topology_classified", "joint_assignment"):
                self.state_machine.advance(record, state)


        with self.lock:

            self.cleanup(
                reference_time=now_ts
            )

            used_gids = set()

            pending_indices = []


            # =================================================
            # 1. LOCAL TRACK / OCCLUSION
            # =================================================

            for idx, det in enumerate(
                detections
            ):

                local_key = (
                    cam_name,
                    int(det["tid"])
                )


                # -------------------------------------------------
                # Occlusion hold
                # -------------------------------------------------

                hold = (
                    self.occlusion_hold.get(
                        local_key
                    )
                )

                if hold is not None:

                    if (
                        now_ts
                        <=
                        hold.get(
                            "until_ts",
                            0
                        )
                    ):

                        gid = hold.get(
                            "gid"
                        )

                        if (
                            self._assignable_identity(gid)
                            is not None
                            and self._presence_allows_local_claim(
                                gid,
                                cam_name,
                                det["tid"],
                                det,
                            )
                            and
                            gid
                            not in
                            used_gids
                        ):

                            results[idx] = (
                                self._commit_assignment(
                                    gid,
                                    cam_name,
                                    det["tid"],
                                    det["emb"],
                                    det.get(
                                        "map_pos"
                                    ),
                                    det.get(
                                        "box_wh"
                                    ),
                                    now_ts,
                                    hold.get(
                                        "score",
                                        1.0
                                    ),
                                    "occlusion-hold"
                                )
                            )

                            used_gids.add(
                                gid
                            )

                            continue


                # -------------------------------------------------
                # Existing local track
                # -------------------------------------------------

                existing = self.local_to_global.get(local_key)

                if existing is not None:

                    gid = existing.get("gid")

                    if (
                        self._assignable_identity(gid) is not None
                        and self._presence_allows_local_claim(
                            gid,
                            cam_name,
                            det["tid"],
                            det,
                        )
                        and
                        gid not in used_gids
                    ):

                        identity = self.identities[gid]

                        appearance = self._local_track_similarity(
                            gid,
                            cam_name,
                            det["tid"],
                            det,
                        )

                        # -----------------------------------------------
                        # Local ID + Appearance เธ•เธฃเธเธเธฑเธ
                        # -----------------------------------------------

                        if (
                            appearance >= LOCAL_TRACK_VERIFY_THRESHOLD
                            and not self._provisional_has_stronger_same_camera_competitor(
                                gid,
                                cam_name,
                                det,
                                now_ts,
                                appearance,
                            )
                        ):

                            results[idx] = (
                                self._commit_assignment(
                                    gid,
                                    cam_name,
                                    det["tid"],
                                    det["emb"],
                                    det.get("map_pos"),
                                    det.get("box_wh"),
                                    now_ts,
                                    appearance,
                                    "local-track-verified"
                                )
                            )

                            used_gids.add(gid)

                            continue

                        if self._can_continue_provisional_local_track(
                            gid,
                            cam_name,
                            det["tid"],
                            det,
                            now_ts,
                            appearance,
                        ):
                            results[idx] = self._commit_assignment(
                                gid,
                                cam_name,
                                det["tid"],
                                det["emb"],
                                det.get("map_pos"),
                                det.get("box_wh"),
                                now_ts,
                                appearance,
                                "provisional-local-continuity",
                            )
                            used_gids.add(gid)
                            continue

                        # -----------------------------------------------
                        # Local ID เน€เธ”เธดเธก เนเธ•เน Appearance เนเธกเนเธ•เธฃเธ
                        #
                        # เธญเธขเนเธฒเน€เธเธทเนเธญ Local ID
                        # เธเธฅเนเธญเธขเธฅเธ global matching
                        # -----------------------------------------------

                        if REID_DEBUG:

                            logger.warning(
                                f"[REID] Local ID conflict | "
                                f"CAM={cam_name} "
                                f"LID={det['tid']} "
                                f"GID={gid} "
                                f"appearance={appearance:.3f}"
                            )

                    self.local_to_global.pop(
                        local_key,
                        None
                    )
                    self.tracklets.pop(
                        local_key,
                        None
                    )


                # -------------------------------------------------
                # Forced GID
                # -------------------------------------------------

                if (
                    det.get(
                        "overlap",
                        False
                    )
                    and
                    det.get(
                        "forced_gid"
                    )
                    in
                    self.identities
                    and
                    self._assignable_identity(
                        det.get("forced_gid")
                    ) is not None
                    and
                    self.identities[det.get("forced_gid")].get("last_cam")
                    == cam_name
                    and
                    det.get(
                        "forced_gid"
                    )
                    not in
                    used_gids
                ):

                    gid = det.get(
                        "forced_gid"
                    )

                    results[idx] = (
                        self._commit_assignment(
                            gid,
                            cam_name,
                            det["tid"],
                            det["emb"],
                            det.get(
                                "map_pos"
                            ),
                            det.get(
                                "box_wh"
                            ),
                            now_ts,
                            1.0,
                            "occlusion-forced"
                        )
                    )

                    used_gids.add(
                        gid
                    )

                    continue


                pending_indices.append(
                    idx
                )


            # =================================================
            # 2. GLOBAL MATCHING
            # =================================================

            candidate_gids = []

            for gid, identity in (
                self.identities.items()
            ):

                if gid in used_gids:
                    continue

                identity = self._matchable_identity_for_camera(
                    gid,
                    cam_name,
                )
                if identity is None:
                    continue

                reusable = False

                for idx in pending_indices:

                    det = detections[idx]

                    if self._can_match(
                        identity,
                        cam_name,
                        now_ts,
                        det.get(
                            "map_pos"
                        ),
                        det.get(
                            "box_wh"
                        )
                    ):

                        reusable = True

                        break

                if reusable:

                    candidate_gids.append(
                        gid
                    )


            pair_cache = {}
            cache_blocked_indices = set()


            if (
                pending_indices
                and
                candidate_gids
            ):

                score_matrix = np.full(

                    (
                        len(
                            pending_indices
                        ),
                        len(
                            candidate_gids
                        )
                    ),

                    -1e6,

                    dtype=np.float32

                )


                for r, idx in enumerate(
                    pending_indices
                ):

                    det = detections[idx]

                    for c, gid in enumerate(
                        candidate_gids
                    ):

                        identity = self._matchable_identity_for_camera(
                            gid,
                            cam_name,
                        )
                        if identity is None:
                            pair_cache[(idx, gid)] = {
                                "gate_failure": "identity_not_globally_matchable"
                            }
                            continue

                        gate_reason = self._hard_gate_reason(
                            identity, cam_name, det, now_ts
                        )
                        if gate_reason is not None:
                            pair_cache[(idx, gid)] = {"gate_failure": gate_reason}
                            continue
                        pair = self._finite_pair(
                            self._pair_score(
                                gid,
                                identity,
                                cam_name,
                                det,
                                now_ts,
                                prev_assignments
                            )
                        )

                        if pair is None:
                            pair_cache[(idx, gid)] = {
                                "score_failure": "invalid_pair_score"
                            }
                            continue

                        pair_cache[
                            (idx, gid)
                        ] = pair

                        score_matrix[
                            r,
                            c
                        ] = pair[
                            "score"
                        ]


                # Resolve ambiguity for every row before Hungarian. Rows can
                # be omitted when there are fewer candidate GIDs than
                # detections, but omitted rows must not bypass the margin gate
                # through the recent same-camera cache.
                eligible_assignment_rows = []
                for matrix_row, idx in enumerate(pending_indices):
                    viable_pairs = [
                        pair_cache[(idx, gid)]
                        for gid in candidate_gids
                        if "score" in pair_cache.get((idx, gid), {})
                    ]
                    if not viable_pairs:
                        continue
                    top_pair = max(
                        viable_pairs,
                        key=lambda item: float(item["score"]),
                    )
                    ambiguity_reason, _ = self._ambiguity_reason(
                        pair_cache,
                        idx,
                        candidate_gids,
                        top_pair["cross_camera"],
                    )
                    if ambiguity_reason is not None:
                        cache_blocked_indices.add(idx)
                        continue
                    eligible_assignment_rows.append(matrix_row)

                if eligible_assignment_rows:
                    row_ind, col_ind = linear_sum_assignment(
                        -score_matrix[eligible_assignment_rows, :]
                    )
                else:
                    row_ind = np.asarray([], dtype=int)
                    col_ind = np.asarray([], dtype=int)

                matched_rows = set()


                for reduced_row, c in zip(
                    row_ind.tolist(),
                    col_ind.tolist()
                ):

                    r = eligible_assignment_rows[reduced_row]

                    idx = (
                        pending_indices[r]
                    )

                    gid = (
                        candidate_gids[c]
                    )

                    pair = pair_cache.get(
                        (idx, gid)
                    )

                    if pair is None or "score" not in pair:
                        continue

                    det = detections[
                        idx
                    ]

                    identity = (
                        self.identities.get(
                            gid
                        )
                    )

                    if identity is None:
                        continue

                    if not self._accept_match(
                        pair,
                        identity,
                        cam_name,
                        det
                    ):
                        continue


                    if gid in used_gids:
                        continue


                    # With unvalidated thresholds, do not let Hungarian
                    # tie-breaking turn multiple acceptable cross-camera
                    # candidates into an identity merge. A measured margin
                    # can replace this conservative rejection only after the
                    # threshold report is validation-backed.
                    if (
                        REID_THRESHOLD_SAFETY_MODE
                        == "conservative"
                        and pair["cross_camera"]
                    ):
                        acceptable_cross_camera = 0

                        for candidate_gid in candidate_gids:
                            if candidate_gid in used_gids:
                                continue

                            candidate_identity = (
                                self.identities.get(
                                    candidate_gid
                                )
                            )
                            candidate_pair = pair_cache.get(
                                (idx, candidate_gid)
                            )

                            if (
                                candidate_identity is not None
                                and candidate_pair is not None
                                and "score" in candidate_pair
                                and candidate_pair["cross_camera"]
                                and self._accept_match(
                                    candidate_pair,
                                    candidate_identity,
                                    cam_name,
                                    det
                                )
                            ):
                                acceptable_cross_camera += 1

                        if acceptable_cross_camera > 1:
                            cache_blocked_indices.add(idx)
                            continue
                    # ---------------------------------------------
                    # Source
                    # ---------------------------------------------

                    if pair[
                        "cross_camera"
                    ]:

                        source = (
                            "cross-camera"
                        )

                    else:

                        source = (
                            "batch-match"
                        )


                    results[idx] = (
                        self._commit_assignment(
                            gid,
                            cam_name,
                            det["tid"],
                            det["emb"],
                            det.get(
                                "map_pos"
                            ),
                            det.get(
                                "box_wh"
                            ),
                            now_ts,
                            pair["score"],
                            source
                        )
                    )


                    used_gids.add(
                        gid
                    )

                    matched_rows.add(
                        idx
                    )


                pending_indices = [

                    idx

                    for idx in pending_indices

                    if idx not in matched_rows

                ]


            # =================================================
            # 3. SAME CAMERA CACHE
            # =================================================

            still_pending = []


            for idx in pending_indices:

                det = detections[idx]

                if idx in cache_blocked_indices:
                    still_pending.append(idx)
                    continue

                eligible_cache_gids = {
                    gid
                    for gid in candidate_gids
                    if (
                        "score" in pair_cache.get((idx, gid), {})
                        and not pair_cache[(idx, gid)].get(
                            "cross_camera",
                            True,
                        )
                    )
                }

                gid, recent_score = (
                    self._find_recent_same_cam_match(
                        cam_name,
                        det["emb"],
                        det.get(
                            "map_pos"
                        ),
                        det.get(
                            "box_wh"
                        ),
                        now_ts,
                        used_gids=used_gids,
                        eligible_gids=eligible_cache_gids,
                    )
                )


                if (
                    gid is not None
                    and
                    self._matchable_identity_for_camera(gid, cam_name) is not None
                ):

                    results[idx] = (
                        self._commit_assignment(
                            gid,
                            cam_name,
                            det["tid"],
                            det["emb"],
                            det.get(
                                "map_pos"
                            ),
                            det.get(
                                "box_wh"
                            ),
                            now_ts,
                            recent_score,
                            "same-cam-cache"
                        )
                    )

                    used_gids.add(
                        gid
                    )

                else:

                    still_pending.append(
                        idx
                    )


            # =================================================
            # 4. CREATE NEW ID
            # =================================================

            for idx in still_pending:

                det = detections[idx]

                results[idx] = (
                    self._new_identity(
                        cam_name,
                        det["tid"],
                        det["emb"],
                        det.get(
                            "map_pos"
                        ),
                        det.get(
                            "box_wh"
                        ),
                        now_ts
                    )
                )


            # =================================================
            # 5. OCCLUSION HOLD
            # =================================================

            for idx, det in enumerate(
                detections
            ):

                if results[idx] is None:
                    continue

                if det.get(
                    "overlap",
                    False
                ) and det.get(
                    "local_track_confirmed",
                    True
                ):

                    local_key = (
                        cam_name,
                        int(det["tid"])
                    )

                    self.occlusion_hold[
                        local_key
                    ] = {

                        "gid":
                            results[idx]["gid"],

                        "until_ts":
                            now_ts
                            +
                            OCCLUSION_HOLD_SEC,

                        "score":
                            float(
                                results[idx][
                                    "score"
                                ]
                            )

                    }


            # Detector rows without a confirmed BoT-SORT ID receive a
            # frame-unique ephemeral key. They may use Re-ID evidence for the
            # current frame, but must not become local-ID history.
            for det in detections:
                if det.get(
                    "local_track_confirmed",
                    True
                ):
                    continue

                ephemeral_key = (
                    cam_name,
                    int(det["tid"])
                )
                self.local_to_global.pop(
                    ephemeral_key,
                    None
                )
                self.occlusion_hold.pop(
                    ephemeral_key,
                    None
                )
                self.tracklets.pop(
                    ephemeral_key,
                    None
                )

            # A successful identity assignment is deliberately separate from
            # gallery admission.  Thus every frame can retain normal tracking
            # behaviour while only a mature, quality-approved tracklet alters
            # the long-lived appearance memory.
            for idx, result in enumerate(results):
                if result is None:
                    continue
                accepted, reason = self._record_tracklet_sample(
                    result["gid"],
                    cam_name,
                    detections[idx]["tid"],
                    detections[idx],
                    now_ts,
                )
                result.update(
                    self._gallery_assignment_diagnostics(
                        result["gid"],
                        accepted,
                        reason,
                    )
                )


        return results



    def _soft_local_continuity_context(
        self,
        cam_name,
        detection,
        event_time,
        existing_override=None,
    ):
        """Return existing-GID continuity context without creating a claim."""
        if not SOFT_LOCAL_CONTINUITY_ENABLED:
            return None

        if (
            SOFT_LOCAL_CONTINUITY_REQUIRE_CONFIRMED
            and not detection.get("local_track_confirmed", True)
        ):
            return None

        local_key = (cam_name, int(detection["tid"]))
        existing = self.local_to_global.get(local_key)
        mapping_source = "live"

        if not isinstance(existing, dict):
            existing = existing_override
            mapping_source = "snapshot"

        if not isinstance(existing, dict):
            return None

        if mapping_source == "snapshot":
            self.soft_continuity_snapshot_hits += 1
        else:
            self.soft_continuity_live_hits += 1

        gid = existing.get("gid")
        if self._assignable_identity(gid) is None:
            self.soft_continuity_identity_rejects += 1
            return None

        last_seen = existing.get("last_seen")
        if not isinstance(last_seen, (int, float)):
            return None

        gap = float(event_time) - float(last_seen)
        if (
            not np.isfinite(gap)
            or gap < 0.0
            or gap > SOFT_LOCAL_CONTINUITY_MAX_GAP_SEC
        ):
            self.soft_continuity_expired += 1
            return None

        if SOFT_LOCAL_CONTINUITY_REQUIRE_SAME_GENERATION:
            previous_generation = existing.get("generation")
            current_generation = detection.get(
                "coordinator_generation",
                detection.get("camera_generation"),
            )
            if (
                previous_generation is not None
                and current_generation is not None
                and previous_generation != current_generation
            ):
                self.soft_continuity_generation_rejects += 1
                return None

        if mapping_source == "live":
            if not self._presence_allows_local_claim(
                gid,
                cam_name,
                detection["tid"],
                detection,
            ):
                self.soft_continuity_presence_rejects += 1
                return None

        # Snapshot mappings are deliberately allowed as SOFT evidence only.
        # They never create a final claim and cannot force an assignment.
        # The candidate must still exist in the global score matrix and must
        # survive the +min-gain switch rule below.
        return {
            "gid": int(gid),
            "gap_sec": float(gap),
            "generation": existing.get("generation"),
            "mapping_source": mapping_source,
            "snapshot_soft_only": bool(mapping_source == "snapshot"),
        }


    def _trusted_assignment_claim(
        self,
        cam_name,
        detection,
        event_time,
        drop_conflicting_local=False,
    ):
        """Return existing local/occlusion evidence without global scoring."""
        local_key = (cam_name, int(detection["tid"]))
        hold = self.occlusion_hold.get(local_key)
        if (
            hold is not None
            and event_time <= hold.get("until_ts", 0)
            and self._assignable_identity(hold.get("gid")) is not None
            and self._presence_allows_local_claim(
                hold.get("gid"),
                cam_name,
                detection["tid"],
                detection,
            )
        ):
            return {
                "gid": hold["gid"],
                "score": float(hold.get("score", 1.0)),
                "source": "occlusion-hold",
                "priority": 3,
            }

        existing = self.local_to_global.get(local_key)
        if existing is not None:
            gid = existing.get("gid")
            if (
                self._assignable_identity(gid) is not None
                and self._presence_allows_local_claim(
                    gid,
                    cam_name,
                    detection["tid"],
                    detection,
                )
            ):
                appearance = self._local_track_similarity(
                    gid,
                    cam_name,
                    detection["tid"],
                    detection,
                )
                if (
                    appearance >= LOCAL_TRACK_VERIFY_THRESHOLD
                    and not self._provisional_has_stronger_same_camera_competitor(
                        gid,
                        cam_name,
                        detection,
                        event_time,
                        appearance,
                    )
                ):
                    return {
                        "gid": gid,
                        "score": float(appearance),
                        "source": "local-track-verified",
                        "priority": 2,
                    }
                if self._can_continue_provisional_local_track(
                    gid,
                    cam_name,
                    detection["tid"],
                    detection,
                    event_time,
                    appearance,
                ):
                    return {
                        "gid": gid,
                        "score": float(appearance),
                        "source": "provisional-local-continuity",
                        "priority": 2,
                    }
                if REID_DEBUG:
                    logger.warning(
                        "[REID] Local ID conflict | CAM=%s LID=%s GID=%s appearance=%.3f",
                        cam_name,
                        detection["tid"],
                        gid,
                        appearance,
                    )
            if drop_conflicting_local:
                self.local_to_global.pop(local_key, None)
                self.tracklets.pop(local_key, None)

        forced_gid = detection.get("forced_gid")
        if (
            detection.get("overlap", False)
            and self._assignable_identity(forced_gid) is not None
            and self.identities[forced_gid].get("last_cam") == cam_name
        ):
            return {
                "gid": forced_gid,
                "score": 1.0,
                "source": "occlusion-forced",
                "priority": 1,
            }

        return None

    def preview_trusted_assignments(
        self,
        cam_name,
        detections,
        event_time=None,
        blocking=True,
    ):
        """Expose already-established evidence while global work is pending.

        This read-only preview keeps normal labels/map updates visible after a
        track has been globally established.  It never creates, matches, or
        mutates an identity; the coordinator's global batch remains the sole
        production assignment decision.
        """
        row_event_time = (
            time.time()
            if event_time is None
            else float(event_time)
        )
        results = [None for _ in detections]
        used_gids = set()

        acquired = self.lock.acquire(
            blocking=bool(blocking)
        )
        if not acquired:
            return results

        try:
            for index, detection in enumerate(detections):
                claim = self._trusted_assignment_claim(
                    cam_name,
                    detection,
                    float(detection.get("event_time", row_event_time)),
                    drop_conflicting_local=False,
                )
                if claim is None or claim["gid"] in used_gids:
                    continue
                results[index] = {
                    "gid": claim["gid"],
                    "score": claim["score"],
                    "source": claim["source"],
                    "gallery_update_accepted": False,
                    "gallery_rejection_reason": "pending_global_batch",
                }
                used_gids.add(claim["gid"])
        finally:
            self.lock.release()

        return results



    # ========================================================
    # REAL-TIME LOCAL TRACK TRANSITION RECOVERY
    # ========================================================

    @staticmethod
    def _transition_local_history(
        prev_assignments,
        local_id,
        gid,
        limit=3,
    ):
        history = [
            item
            for item in prev_assignments
            if (
                int(item.get("tid", -999999)) == int(local_id)
                and int(item.get("gid", -999999)) == int(gid)
                and item.get("box") is not None
                and isinstance(item.get("ts"), (int, float))
            )
        ]
        history.sort(
            key=lambda item: float(item["ts"])
        )
        return history[-max(2, int(limit)):]

    @staticmethod
    def _transition_predict_center(
        history,
        event_time,
    ):
        if not history:
            return None

        last = history[-1]
        last_center = bbox_center(last["box"])

        if len(history) < 2:
            return (
                float(last_center[0]),
                float(last_center[1]),
            )

        prev = history[-2]
        prev_center = bbox_center(prev["box"])

        t1 = float(prev["ts"])
        t2 = float(last["ts"])
        dt = t2 - t1

        if not np.isfinite(dt) or dt <= 1e-6:
            return (
                float(last_center[0]),
                float(last_center[1]),
            )

        vx = (
            float(last_center[0])
            - float(prev_center[0])
        ) / dt
        vy = (
            float(last_center[1])
            - float(prev_center[1])
        ) / dt

        future_dt = (
            float(event_time)
            - float(last["ts"])
        )
        future_dt = max(
            0.0,
            min(
                float(future_dt),
                float(LOCAL_TRANSITION_MAX_GAP_SEC),
            ),
        )

        return (
            float(last_center[0]) + vx * future_dt,
            float(last_center[1]) + vy * future_dt,
        )

    @staticmethod
    def _transition_motion_score(
        predicted_center,
        current_box,
        reference_box,
    ):
        if (
            predicted_center is None
            or current_box is None
            or reference_box is None
        ):
            return 0.0, float("inf")

        cx, cy = bbox_center(current_box)
        px, py = predicted_center

        distance = float(
            np.hypot(
                float(cx) - float(px),
                float(cy) - float(py),
            )
        )

        x1, y1, x2, y2 = [
            float(v)
            for v in reference_box
        ]
        diag = float(
            np.hypot(
                x2 - x1,
                y2 - y1,
            )
        )
        scale = max(
            45.0,
            diag * 1.35,
        )

        motion = 1.0 - min(
            distance / scale,
            1.0,
        )

        return (
            float(max(0.0, min(1.0, motion))),
            distance,
        )

    def _update_recent_lost_local_tracks(
        self,
        rows,
        previous,
        event_time,
    ):
        """Persist short-lived disappeared local-track evidence across frames.

        V1 only looked at ``prev_assignments`` from the immediate caller window.
        V2 keeps a bounded registry so a local track can disappear for several
        frames and still be considered for a real-time continuity recovery.
        """
        now_ts = float(event_time)
        active_by_camera = {}
        for cam_name, _, detection, _ in rows:
            if not detection.get("local_track_confirmed", True):
                continue
            active_by_camera.setdefault(cam_name, set()).add(int(detection["tid"]))

        # Expire old registry entries first.
        for key, record in list(self.recent_lost_local_tracks.items()):
            lost_ts = record.get("lost_ts")
            if (
                not isinstance(lost_ts, (int, float))
                or now_ts - float(lost_ts) > LOCAL_TRANSITION_LOST_TTL_SEC
            ):
                self.recent_lost_local_tracks.pop(key, None)

        # Register tracks that were present in previous assignment history but
        # are no longer active in the current camera batch.
        for cam_name, prev_items in previous.items():
            active = active_by_camera.get(cam_name, set())
            latest = {}
            for item in prev_items:
                tid = item.get("tid")
                gid = item.get("gid")
                ts = item.get("ts")
                box = item.get("box")
                if (
                    tid is None or gid is None or box is None
                    or not isinstance(ts, (int, float))
                ):
                    continue
                tid = int(tid)
                gid = int(gid)
                if tid in active:
                    continue
                key = (cam_name, tid)
                old = latest.get(key)
                if old is None or float(ts) > float(old["ts"]):
                    latest[key] = {
                        "cam_name": cam_name,
                        "tid": tid,
                        "gid": gid,
                        "ts": float(ts),
                        "lost_ts": now_ts,
                        "box": tuple(float(v) for v in box),
                        "box_wh": item.get("box_wh"),
                        "emb": item.get("emb"),
                    }

            for key, record in latest.items():
                previous_record = self.recent_lost_local_tracks.get(key)
                if previous_record is None or float(record["ts"]) >= float(previous_record.get("ts", -1.0)):
                    self.recent_lost_local_tracks[key] = record
                    self.transition_recovery_lost_registered += 1

        # Bound memory per camera.
        cameras = {key[0] for key in self.recent_lost_local_tracks}
        for cam_name in cameras:
            entries = [
                (key, value)
                for key, value in self.recent_lost_local_tracks.items()
                if key[0] == cam_name
            ]
            entries.sort(key=lambda item: float(item[1].get("lost_ts", 0.0)), reverse=True)
            for key, _ in entries[LOCAL_TRANSITION_MAX_LOST_PER_CAMERA:]:
                self.recent_lost_local_tracks.pop(key, None)

    def _transition_lost_candidates(
        self,
        cam_name,
        current_tid,
        current_active_tids,
        event_time,
    ):
        now_ts = float(event_time)
        candidates = []
        for (lost_cam, lost_tid), record in self.recent_lost_local_tracks.items():
            if lost_cam != cam_name:
                continue
            if lost_tid == int(current_tid):
                # same-ID reappearance is allowed, but only after a real gap
                pass
            elif lost_tid in current_active_tids:
                continue
            age = now_ts - float(record.get("ts", now_ts))
            lost_age = now_ts - float(record.get("lost_ts", now_ts))
            if (
                not np.isfinite(age)
                or not np.isfinite(lost_age)
                or age < LOCAL_TRANSITION_REAPPEAR_MIN_GAP_SEC
                or lost_age < 0.0
                or lost_age > LOCAL_TRANSITION_LOST_TTL_SEC
            ):
                continue
            candidates.append(record)
        return candidates


    def _delayed_transition_decision(
        self,
        cam_name,
        current_tid,
        best,
        margin,
        event_time,
    ):
        """Evidence-gated transition decision for a newly/reappeared local ID.

        Very strong transitions may be accepted immediately. Borderline but
        plausible transitions must repeat consistently before they can replace
        the normal V20 assignment path. This avoids making a long-lived wrong
        identity from one ambiguous observation.
        """
        if not DELAYED_TRANSITION_ENABLED:
            return True, "disabled"

        score = float(best.get("score", -1.0))
        motion = float(best.get("motion", -1.0))
        gid = int(best["gid"])
        old_tid = int(best["old_tid"])
        now_ts = float(event_time)

        strong = bool(
            score >= DELAYED_TRANSITION_STRONG_SCORE
            and motion >= DELAYED_TRANSITION_STRONG_MOTION
            and margin >= DELAYED_TRANSITION_STRONG_MARGIN
        )

        key = (str(cam_name), int(current_tid))
        if strong:
            self.delayed_transition_strong_accepts += 1
            self.delayed_transition_hypotheses.pop(key, None)
            self.last_delayed_transition = {
                "camera": str(cam_name),
                "current_tid": int(current_tid),
                "gid": gid,
                "old_tid": old_tid,
                "votes": 1,
                "accepted": True,
                "reason": "strong-immediate",
                "score": score,
                "motion": motion,
                "margin": float(margin),
            }
            return True, "strong-immediate"

        plausible = bool(
            score >= DELAYED_TRANSITION_NORMAL_SCORE
            and motion >= DELAYED_TRANSITION_NORMAL_MOTION
            and margin >= DELAYED_TRANSITION_NORMAL_MARGIN
        )

        if not plausible:
            self.delayed_transition_rejects += 1
            self.delayed_transition_hypotheses.pop(key, None)
            self.last_delayed_transition = {
                "camera": str(cam_name),
                "current_tid": int(current_tid),
                "gid": gid,
                "old_tid": old_tid,
                "votes": 0,
                "accepted": False,
                "reason": "below-normal-gates",
                "score": score,
                "motion": motion,
                "margin": float(margin),
            }
            return False, "below-normal-gates"

        state = self.delayed_transition_hypotheses.get(key)
        if (
            not isinstance(state, dict)
            or int(state.get("gid", -1)) != gid
            or int(state.get("old_tid", -1)) != old_tid
            or now_ts - float(state.get("last_ts", now_ts))
            > DELAYED_TRANSITION_MAX_VOTE_GAP_SEC
        ):
            if isinstance(state, dict):
                self.delayed_transition_vote_resets += 1
            state = {
                "gid": gid,
                "old_tid": old_tid,
                "votes": 0,
                "last_ts": now_ts,
            }

        state["votes"] = int(state.get("votes", 0)) + 1
        state["last_ts"] = now_ts
        self.delayed_transition_hypotheses[key] = state

        votes = int(state["votes"])
        accepted = votes >= DELAYED_TRANSITION_REQUIRED_VOTES

        if accepted:
            self.delayed_transition_vote_accepts += 1
            self.delayed_transition_hypotheses.pop(key, None)
            reason = "multi-observation-confirmed"
        else:
            self.delayed_transition_vote_pending += 1
            reason = "pending-more-evidence"

        self.last_delayed_transition = {
            "camera": str(cam_name),
            "current_tid": int(current_tid),
            "gid": gid,
            "old_tid": old_tid,
            "votes": votes,
            "accepted": bool(accepted),
            "reason": reason,
            "score": score,
            "motion": motion,
            "margin": float(margin),
        }
        return bool(accepted), reason


    def _transition_recovery_claims(
        self,
        rows,
        previous,
        fixed_claims,
    ):
        """V2: recover new/reappeared local tracks from a persistent lost registry.

        Safety rules for real time:
        - never steal a GID from another currently active local track;
        - no additional OSNet inference;
        - allow an existing local mapping only when it is stale/reappeared;
        - require motion, score and top-1/top-2 margin gates.
        """
        if not LOCAL_TRANSITION_RECOVERY_ENABLED:
            return []

        fixed_rows = {int(row) for row, _ in fixed_claims}
        current_local_ids = {}
        gid_active_owner = {}
        trusted_presence_cameras_by_gid = {}
        for row, (cam_name, _, detection, _) in enumerate(rows):
            if not detection.get("local_track_confirmed", True):
                continue
            tid = int(detection["tid"])
            current_local_ids.setdefault(cam_name, set()).add(tid)
            mapping = self.local_to_global.get((cam_name, tid))
            if isinstance(mapping, dict) and mapping.get("gid") is not None:
                mapped_gid = int(mapping["gid"])
                gid_active_owner[(cam_name, mapped_gid)] = tid
                if self._presence_allows_local_claim(
                    mapped_gid,
                    cam_name,
                    tid,
                    detection,
                ):
                    trusted_presence_cameras_by_gid.setdefault(
                        mapped_gid,
                        set(),
                    ).add(cam_name)

        # Occlusion/local fixed claims can be trustworthy even when their
        # ownership did not come from the live local mapping lookup above.
        for fixed_row, claim in fixed_claims:
            cam_name, _, detection, _ = rows[int(fixed_row)]
            gid = claim.get("gid")
            if (
                gid is not None
                and self._presence_allows_local_claim(
                    int(gid),
                    cam_name,
                    detection["tid"],
                    detection,
                )
            ):
                trusted_presence_cameras_by_gid.setdefault(
                    int(gid),
                    set(),
                ).add(cam_name)

        recovery_claims = []

        for row, (cam_name, _, detection, row_event_time) in enumerate(rows):
            if not detection.get("local_track_confirmed", True):
                continue

            current_tid = int(detection["tid"])
            local_key = (cam_name, current_tid)
            mapping = self.local_to_global.get(local_key)
            mapping_last_seen = mapping.get("last_seen") if isinstance(mapping, dict) else None
            mapping_gap = None
            if isinstance(mapping_last_seen, (int, float)):
                mapping_gap = float(row_event_time) - float(mapping_last_seen)

            mapped_gid = (
                mapping.get("gid")
                if isinstance(mapping, dict)
                else None
            )
            previous_generation = (
                mapping.get("generation")
                if isinstance(mapping, dict)
                else None
            )
            current_generation = detection.get(
                "coordinator_generation",
                detection.get("camera_generation"),
            )
            same_generation = bool(
                previous_generation is None
                or current_generation is None
                or previous_generation == current_generation
            )
            valid_mapping_continuity = bool(
                isinstance(mapping, dict)
                and mapped_gid is not None
                and mapping_gap is not None
                and np.isfinite(mapping_gap)
                and 0.0 <= mapping_gap <= SOFT_LOCAL_CONTINUITY_MAX_GAP_SEC
                and same_generation
                and self._assignable_identity(mapped_gid) is not None
                and self._presence_allows_local_claim(
                    mapped_gid,
                    cam_name,
                    current_tid,
                    detection,
                )
            )
            is_stale_mapping = bool(
                LOCAL_TRANSITION_ALLOW_STALE_MAPPING
                and isinstance(mapping, dict)
                and not valid_mapping_continuity
                and mapping_gap is not None
                and np.isfinite(mapping_gap)
                and mapping_gap >= LOCAL_TRANSITION_REAPPEAR_MIN_GAP_SEC
            )

            # A normal trusted row stays on the fast path.  V2 only inspects
            # unmapped rows or a mapping that has genuinely reappeared after a gap.
            if row in fixed_rows and not is_stale_mapping:
                continue
            if mapping is not None and not is_stale_mapping:
                continue
            if is_stale_mapping:
                self.transition_recovery_stale_mapping_checks += 1

            lost_candidates = self._transition_lost_candidates(
                cam_name,
                current_tid,
                current_local_ids.get(cam_name, set()),
                row_event_time,
            )
            if not lost_candidates:
                continue

            self.transition_recovery_checks += 1
            self.transition_recovery_reappeared_checks += 1
            scored = []

            for lost in lost_candidates:
                gid = int(lost["gid"])
                owner_tid = gid_active_owner.get((cam_name, gid))
                if owner_tid is not None and owner_tid != current_tid:
                    # Never steal an identity from an active track.
                    continue
                identity = self._matchable_identity_for_camera(gid, cam_name)
                if identity is None:
                    continue

                cross_camera = bool(
                    identity.get("last_cam")
                    and identity.get("last_cam") != cam_name
                )
                if cross_camera:
                    trusted_other_cameras = {
                        camera
                        for camera in trusted_presence_cameras_by_gid.get(
                            gid,
                            set(),
                        )
                        if camera != cam_name
                    }
                    if trusted_other_cameras:
                        # Transition recovery is camera-local continuity
                        # evidence. It must not reclaim a GID that still has a
                        # trustworthy live owner in another camera.
                        continue
                    if self._hard_gate_reason(
                        identity,
                        cam_name,
                        detection,
                        row_event_time,
                    ) is not None:
                        continue

                # Use the latest previous history for velocity when available.
                prev_assignments = previous.get(cam_name, [])
                history = self._transition_local_history(
                    prev_assignments,
                    int(lost["tid"]),
                    gid,
                    limit=LOCAL_TRANSITION_HISTORY,
                )
                if history:
                    predicted_center = self._transition_predict_center(
                        history,
                        row_event_time,
                    )
                else:
                    predicted_center = bbox_center(lost["box"])

                motion, distance = self._transition_motion_score(
                    predicted_center,
                    detection.get("box"),
                    lost.get("box"),
                )
                appearance = self._gallery_similarity(
                    detection.get("emb"),
                    identity,
                )
                if not np.isfinite(float(appearance)):
                    appearance = -1.0
                appearance01 = max(0.0, min(1.0, (float(appearance) + 1.0) * 0.5))
                size = self._pairwise_size_score(
                    detection.get("box_wh"),
                    lost.get("box_wh"),
                )
                score = (
                    LOCAL_TRANSITION_MOTION_WEIGHT * motion
                    + LOCAL_TRANSITION_APPEARANCE_WEIGHT * appearance01
                    + LOCAL_TRANSITION_SIZE_WEIGHT * size
                )
                scored.append({
                    "gid": gid,
                    "old_tid": int(lost["tid"]),
                    "score": float(score),
                    "motion": float(motion),
                    "appearance": float(appearance),
                    "size": float(size),
                    "distance": float(distance),
                    "predicted_center": predicted_center,
                    "age_sec": float(row_event_time) - float(lost["ts"]),
                })

            if not scored:
                continue

            self.transition_recovery_candidates += 1
            scored.sort(key=lambda item: item["score"], reverse=True)
            best = scored[0]
            second = scored[1] if len(scored) > 1 else None
            margin = float(best["score"] - second["score"]) if second is not None else 1.0
            base_accepted = bool(
                best["score"] >= LOCAL_TRANSITION_MIN_SCORE
                and best["motion"] >= LOCAL_TRANSITION_MIN_MOTION
                and margin >= LOCAL_TRANSITION_MIN_MARGIN
            )

            if base_accepted:
                accepted, delayed_reason = self._delayed_transition_decision(
                    cam_name,
                    current_tid,
                    best,
                    margin,
                    row_event_time,
                )
            else:
                accepted = False
                delayed_reason = "base-transition-gates-failed"

            self.last_transition_recovery_diagnostic = {
                "camera": cam_name,
                "row": int(row),
                "current_tid": current_tid,
                "event_time": float(row_event_time),
                "accepted": accepted,
                "delayed_reason": delayed_reason,
                "stale_mapping": is_stale_mapping,
                "mapping_gap": mapping_gap,
                "best": best,
                "second": second,
                "margin": margin,
                "candidate_count": len(scored),
            }

            if not accepted:
                self.transition_recovery_rejections += 1
                continue

            # If this row previously had a fixed trusted claim, replace it with
            # the stronger transition-recovery claim only when the row was stale.
            if row in fixed_rows:
                fixed_claims[:] = [item for item in fixed_claims if int(item[0]) != row]
                fixed_rows.discard(row)

            self.transition_recovery_assignments += 1
            recovery_claims.append((
                row,
                {
                    "gid": int(best["gid"]),
                    "score": float(best["score"]),
                    "source": "local-transition-recovery-v2",
                    "priority": 3,
                },
            ))

            # Consume only the winning lost-track record.  This prevents a
            # second current track from inheriting the same historical event.
            self.recent_lost_local_tracks.pop((cam_name, int(best["old_tid"])), None)

        return recovery_claims

    # ========================================================
    # SAME-CAMERA PAIRWISE SWAP CORRECTION
    # ========================================================

    @staticmethod
    def _pairwise_size_score(box_wh, ref_wh):
        if box_wh is None or ref_wh is None:
            return 0.50
        try:
            bw, bh = [float(v) for v in box_wh]
            rw, rh = [float(v) for v in ref_wh]
        except (TypeError, ValueError, OverflowError):
            return 0.50
        if min(bw, bh, rw, rh) <= 0:
            return 0.50
        wr = min(bw, rw) / max(bw, rw)
        hr = min(bh, rh) / max(bh, rh)
        return float(max(0.0, min(1.0, 0.5 * (wr + hr))))

    def _build_pairwise_gid_snapshot(
        self,
        cam_name,
        candidate_gids,
        prev_assignments,
        event_time,
    ):
        """Build a previous-frame GID trajectory snapshot before current commits.

        Snapshot is keyed by GID, never Local ID.  It therefore remains usable
        when BoT-SORT swaps Local IDs between two nearby people.
        """
        snapshot = {}
        now_ts = float(event_time)

        for gid in candidate_gids:
            history = self._recent_history_for_gid(
                cam_name,
                gid,
                prev_assignments,
                limit=max(2, int(PAIRWISE_SNAPSHOT_HISTORY)),
            )
            if not history:
                continue

            last = history[-1]
            last_ts = last.get("ts")
            if not isinstance(last_ts, (int, float)):
                continue

            gap = now_ts - float(last_ts)
            if (
                not np.isfinite(gap)
                or gap < 0.0
                or gap > PAIRWISE_SNAPSHOT_MAX_GAP_SEC
            ):
                continue

            last_box = last.get("box")
            if last_box is None:
                continue

            last_center = bbox_center(last_box)
            predicted = self._predict_center(
                history,
                event_time=now_ts,
            )
            if predicted is None:
                predicted = last_center

            identity = self.identities.get(gid)
            if identity is None:
                continue

            snapshot[gid] = {
                "gid": gid,
                "history": history,
                "last_box": last_box,
                "last_center": (
                    float(last_center[0]),
                    float(last_center[1]),
                ),
                "predicted_center": (
                    float(predicted[0]),
                    float(predicted[1]),
                ),
                "box_wh": identity.get("box_wh"),
                "identity": identity,
                "last_ts": float(last_ts),
                "gap_sec": float(gap),
            }

        return snapshot

    @staticmethod
    def _snapshot_motion_score(snapshot_item, detection):
        pred_x, pred_y = snapshot_item["predicted_center"]
        cur_x, cur_y = bbox_center(detection["box"])

        dist = float(np.hypot(
            float(cur_x) - float(pred_x),
            float(cur_y) - float(pred_y),
        ))

        last_box = snapshot_item["last_box"]
        lx1, ly1, lx2, ly2 = [float(v) for v in last_box]
        diag = float(np.hypot(lx2 - lx1, ly2 - ly1))
        scale = max(
            35.0,
            diag * float(PAIRWISE_MOTION_DISTANCE_SCALE),
        )

        # 1.0 at the predicted point, smoothly dropping to 0.
        motion = 1.0 - min(dist / scale, 1.0)
        return float(max(0.0, min(1.0, motion))), dist, scale

    def _pairwise_snapshot_pair_score(
        self,
        snapshot_item,
        detection,
    ):
        motion, distance, motion_scale = self._snapshot_motion_score(
            snapshot_item,
            detection,
        )

        identity = snapshot_item["identity"]
        appearance = self._gallery_similarity(
            detection.get("emb"),
            identity,
        )
        if not np.isfinite(float(appearance)):
            appearance = -1.0

        appearance01 = max(
            0.0,
            min(1.0, (float(appearance) + 1.0) * 0.5),
        )

        size_score = self._pairwise_size_score(
            detection.get("box_wh"),
            snapshot_item.get("box_wh"),
        )

        score = (
            PAIRWISE_SWAP_MOTION_WEIGHT * motion
            + PAIRWISE_SWAP_APPEARANCE_WEIGHT * appearance01
            + PAIRWISE_SWAP_SIZE_WEIGHT * size_score
        )

        return {
            "score": float(score),
            "motion": float(motion),
            "appearance": float(appearance),
            "size": float(size_score),
            "distance": float(distance),
            "motion_scale": float(motion_scale),
            "predicted_center": snapshot_item["predicted_center"],
        }

    def _correct_pairwise_local_swaps(
        self,
        rows,
        fixed_claims,
        previous,
    ):
        if not PAIRWISE_SWAP_CORRECTION_ENABLED or len(fixed_claims) < 2:
            return fixed_claims

        claims_by_camera = {}
        for row, claim in fixed_claims:
            if claim.get("source") not in {
                "local-track-verified",
                "provisional-local-continuity",
            }:
                continue
            cam_name, _, detection, row_event_time = rows[row]
            if not detection.get("local_track_confirmed", True):
                continue
            claims_by_camera.setdefault(cam_name, []).append(
                (row, claim, detection, row_event_time)
            )

        replacements = {}

        for cam_name, items in claims_by_camera.items():
            if len(items) < 2 or len(items) > PAIRWISE_SWAP_MAX_GROUP:
                continue

            candidate_gids = [item[1].get("gid") for item in items]
            if None in candidate_gids or len(set(candidate_gids)) != len(candidate_gids):
                continue

            prev_assignments = previous.get(cam_name, [])
            n = len(items)
            score_matrix = np.full((n, n), -1e6, dtype=np.float32)
            details = {}

            # All rows in this camera batch share the same synchronized event
            # time in production/evaluation. Build the GID snapshot BEFORE any
            # current-frame local claim is committed.
            snapshot_event_time = max(
                float(item[3])
                for item in items
            )
            snapshot = self._build_pairwise_gid_snapshot(
                cam_name,
                candidate_gids,
                prev_assignments,
                snapshot_event_time,
            )

            if len(snapshot) != len(candidate_gids):
                self.pairwise_snapshot_missing += 1
                continue

            self.pairwise_snapshot_groups += 1
            self.pairwise_snapshot_last = {
                "camera": cam_name,
                "event_time": float(snapshot_event_time),
                "gids": {
                    int(gid): {
                        "predicted_center": tuple(
                            snapshot[gid]["predicted_center"]
                        ),
                        "gap_sec": float(snapshot[gid]["gap_sec"]),
                    }
                    for gid in candidate_gids
                },
            }

            usable = True
            for i, (row, claim, detection, row_event_time) in enumerate(items):
                for j, gid in enumerate(candidate_gids):
                    snap = snapshot.get(gid)
                    if snap is None:
                        usable = False
                        break
                    pair = self._pairwise_snapshot_pair_score(
                        snap,
                        detection,
                    )
                    score_matrix[i, j] = pair["score"]
                    details[(i, j)] = pair
                if not usable:
                    break

            if not usable:
                continue

            self.pairwise_swap_checks += 1

            baseline_cols = list(range(n))
            baseline_scores = [
                float(score_matrix[i, i]) for i in range(n)
            ]
            baseline_avg = float(np.mean(baseline_scores))

            if PAIRWISE_MATRIX_DIAGNOSTICS_ENABLED:
                matrix_record = {
                    "camera": cam_name,
                    "event_time": float(snapshot_event_time),
                    "rows": [
                        {
                            "row_index": int(items[i][0]),
                            "local_id": int(items[i][2]["tid"]),
                            "existing_gid": int(candidate_gids[i]),
                            "box": tuple(
                                float(v)
                                for v in items[i][2]["box"]
                            ),
                            "center": tuple(
                                float(v)
                                for v in bbox_center(items[i][2]["box"])
                            ),
                        }
                        for i in range(n)
                    ],
                    "candidate_gids": [
                        int(gid)
                        for gid in candidate_gids
                    ],
                    "snapshot": {
                        int(gid): {
                            "predicted_center": tuple(
                                float(v)
                                for v in snapshot[gid]["predicted_center"]
                            ),
                            "last_center": tuple(
                                float(v)
                                for v in snapshot[gid]["last_center"]
                            ),
                            "gap_sec": float(snapshot[gid]["gap_sec"]),
                        }
                        for gid in candidate_gids
                    },
                    "score_matrix": [
                        [
                            float(score_matrix[i, j])
                            for j in range(n)
                        ]
                        for i in range(n)
                    ],
                    "motion_matrix": [
                        [
                            float(details[(i, j)]["motion"])
                            for j in range(n)
                        ]
                        for i in range(n)
                    ],
                    "appearance_matrix": [
                        [
                            float(details[(i, j)]["appearance"])
                            for j in range(n)
                        ]
                        for i in range(n)
                    ],
                    "size_matrix": [
                        [
                            float(details[(i, j)]["size"])
                            for j in range(n)
                        ]
                        for i in range(n)
                    ],
                    "distance_matrix": [
                        [
                            float(details[(i, j)]["distance"])
                            for j in range(n)
                        ]
                        for i in range(n)
                    ],
                }
                self.pairwise_matrix_diagnostics.append(
                    matrix_record
                )
                del self.pairwise_matrix_diagnostics[
                    :-max(
                        1,
                        int(PAIRWISE_MATRIX_DIAGNOSTIC_MAX_RECORDS),
                    )
                ]

            row_ind, col_ind = linear_sum_assignment(-score_matrix)
            assigned_col = {
                int(i): int(j) for i, j in zip(row_ind, col_ind)
            }
            if (
                PAIRWISE_MATRIX_DIAGNOSTICS_ENABLED
                and self.pairwise_matrix_diagnostics
            ):
                matrix_record = self.pairwise_matrix_diagnostics[-1]
                if (
                    matrix_record.get("camera") == cam_name
                    and abs(
                        float(
                            matrix_record.get(
                                "event_time",
                                -1.0,
                            )
                        )
                        - float(snapshot_event_time)
                    ) < 1e-6
                ):
                    matrix_record[
                        "hungarian_assignment"
                    ] = {
                        int(row): int(col)
                        for row, col in assigned_col.items()
                    }
                    matrix_record[
                        "diagonal_assignment"
                    ] = {
                        int(i): int(i)
                        for i in range(n)
                    }
                    matrix_record[
                        "baseline_avg"
                    ] = float(baseline_avg)

            best_scores = [
                float(score_matrix[i, assigned_col[i]]) for i in range(n)
            ]
            best_avg = float(np.mean(best_scores))
            avg_gain = best_avg - baseline_avg

            changed_rows = [
                i for i in range(n)
                if assigned_col[i] != i
            ]
            if not changed_rows:
                continue

            self.pairwise_swap_candidate_count += 1
            self.pairwise_swap_max_avg_gain = max(
                float(self.pairwise_swap_max_avg_gain),
                float(avg_gain),
            )
            for level in PAIRWISE_DIAGNOSTIC_GAIN_LEVELS:
                if avg_gain >= float(level):
                    self.pairwise_swap_gain_counts[float(level)] = (
                        self.pairwise_swap_gain_counts.get(float(level), 0) + 1
                    )

            row_diags = []
            min_row_gain = float("inf")
            min_motion_gain = float("inf")
            max_row_gain = float("-inf")
            max_motion_gain = float("-inf")

            for i in changed_rows:
                old_pair = details[(i, i)]
                new_pair = details[(i, assigned_col[i])]
                row_gain = float(new_pair["score"] - old_pair["score"])
                motion_gain = float(new_pair["motion"] - old_pair["motion"])

                min_row_gain = min(min_row_gain, row_gain)
                min_motion_gain = min(min_motion_gain, motion_gain)
                max_row_gain = max(max_row_gain, row_gain)
                max_motion_gain = max(max_motion_gain, motion_gain)

                self.pairwise_swap_max_row_gain = max(
                    float(self.pairwise_swap_max_row_gain),
                    row_gain,
                )
                self.pairwise_swap_max_motion_gain = max(
                    float(self.pairwise_swap_max_motion_gain),
                    motion_gain,
                )

                row_diags.append({
                    "row": int(items[i][0]),
                    "old_gid": int(candidate_gids[i]),
                    "new_gid": int(candidate_gids[assigned_col[i]]),
                    "old_score": float(old_pair["score"]),
                    "new_score": float(new_pair["score"]),
                    "row_gain": row_gain,
                    "old_motion": float(old_pair["motion"]),
                    "new_motion": float(new_pair["motion"]),
                    "motion_gain": motion_gain,
                    "old_appearance": float(old_pair["appearance"]),
                    "new_appearance": float(new_pair["appearance"]),
                    "old_size": float(old_pair["size"]),
                    "new_size": float(new_pair["size"]),
                    "old_distance": float(old_pair.get("distance", 0.0)),
                    "new_distance": float(new_pair.get("distance", 0.0)),
                    "old_predicted_center": old_pair.get("predicted_center"),
                    "new_predicted_center": new_pair.get("predicted_center"),
                })

            reject_reasons = []
            if avg_gain < PAIRWISE_SWAP_MIN_AVG_GAIN:
                reject_reasons.append("avg_gain")
            if min_row_gain < PAIRWISE_SWAP_MIN_ROW_GAIN:
                reject_reasons.append("row_gain")
            if min_motion_gain < PAIRWISE_SWAP_MIN_MOTION_GAIN:
                reject_reasons.append("motion_gain")

            if reject_reasons:
                self.pairwise_swap_rejected_count += 1
                rejected = {
                    "camera": cam_name,
                    "avg_gain": float(avg_gain),
                    "baseline_avg": float(baseline_avg),
                    "candidate_avg": float(best_avg),
                    "min_row_gain": float(min_row_gain),
                    "min_motion_gain": float(min_motion_gain),
                    "max_row_gain": float(max_row_gain),
                    "max_motion_gain": float(max_motion_gain),
                    "reasons": list(reject_reasons),
                    "rows": row_diags,
                }
                self.pairwise_swap_top_rejected.append(rejected)
                self.pairwise_swap_top_rejected.sort(
                    key=lambda item: float(item.get("avg_gain", -1e9)),
                    reverse=True,
                )
                del self.pairwise_swap_top_rejected[
                    max(1, int(PAIRWISE_DIAGNOSTIC_TOP_K)):
                ]
                continue

            self.pairwise_swap_corrections += 1
            self.pairwise_swap_corrected_rows += len(changed_rows)
            self.last_pairwise_swap_diagnostic = {
                "camera": cam_name,
                "avg_gain": float(avg_gain),
                "baseline_avg": float(baseline_avg),
                "corrected_avg": float(best_avg),
                "min_row_gain": float(min_row_gain),
                "min_motion_gain": float(min_motion_gain),
                "thresholds": {
                    "avg_gain": float(PAIRWISE_SWAP_MIN_AVG_GAIN),
                    "row_gain": float(PAIRWISE_SWAP_MIN_ROW_GAIN),
                    "motion_gain": float(PAIRWISE_SWAP_MIN_MOTION_GAIN),
                },
                "rows": row_diags,
            }

            for i in changed_rows:
                row, claim, detection, row_event_time = items[i]
                new_gid = candidate_gids[assigned_col[i]]
                new_pair = details[(i, assigned_col[i])]
                replacements[row] = {
                    "gid": new_gid,
                    "score": float(new_pair["score"]),
                    "source": "pairwise-motion-corrected",
                    "priority": int(claim.get("priority", 2)) + 1,
                }

        return [
            (row, replacements.get(row, claim))
            for row, claim in fixed_claims
        ]


    # ========================================================
    # GLOBAL MULTI-CAMERA BATCH ASSIGNMENT
    # ========================================================


    def _wrong_gid_escape_decision(
        self,
        cam_name,
        detection,
        row_event_time,
        previous_mapping,
        challenger_gid,
        challenger_pair,
        owner_pair,
    ):
        """Gate same-local GID changes with temporal hysteresis.

        A challenger must beat the current owner repeatedly before a switch is
        committed. A very strong challenger can switch immediately. This
        protects accuracy from one-frame relabels while still allowing a
        stable wrong GID to be corrected when better evidence persists.
        """
        if not WRONG_GID_ESCAPE_ENABLED:
            return True, 'disabled', 0

        if not isinstance(previous_mapping, dict):
            return True, 'no-previous-mapping', 0

        previous_gid = previous_mapping.get('gid')
        if previous_gid is None or int(previous_gid) == int(challenger_gid):
            return True, 'same-or-no-owner', 0

        self.wrong_gid_escape_checks += 1

        challenger_score = float(challenger_pair.get('score', -1.0))
        owner_score = float(owner_pair.get('score', -1.0)) if isinstance(owner_pair, dict) else -1.0
        advantage = challenger_score - owner_score

        key = (str(cam_name), int(detection['tid']))
        now_ts = float(row_event_time)

        strong = bool(
            challenger_score >= WRONG_GID_ESCAPE_STRONG_SCORE
            and advantage >= WRONG_GID_ESCAPE_STRONG_ADVANTAGE
        )
        if strong:
            self.wrong_gid_escape_strong_immediate += 1
            self.wrong_gid_escape_state.pop(key, None)
            self.last_wrong_gid_escape = {
                'camera': str(cam_name),
                'local_id': int(detection['tid']),
                'owner_gid': int(previous_gid),
                'challenger_gid': int(challenger_gid),
                'owner_score': owner_score,
                'challenger_score': challenger_score,
                'advantage': advantage,
                'votes': 1,
                'accepted': True,
                'reason': 'strong-immediate',
            }
            return True, 'strong-immediate', 1

        plausible = bool(
            challenger_score >= WRONG_GID_ESCAPE_MIN_CHALLENGER_SCORE
            and advantage >= WRONG_GID_ESCAPE_MIN_SCORE_ADVANTAGE
        )

        if not plausible:
            self.wrong_gid_escape_state.pop(key, None)
            self.wrong_gid_escape_owner_kept += 1
            self.last_wrong_gid_escape = {
                'camera': str(cam_name),
                'local_id': int(detection['tid']),
                'owner_gid': int(previous_gid),
                'challenger_gid': int(challenger_gid),
                'owner_score': owner_score,
                'challenger_score': challenger_score,
                'advantage': advantage,
                'votes': 0,
                'accepted': False,
                'reason': 'challenger-not-better-enough',
            }
            return False, 'challenger-not-better-enough', 0

        state = self.wrong_gid_escape_state.get(key)
        if (
            not isinstance(state, dict)
            or int(state.get('challenger_gid', -1)) != int(challenger_gid)
            or int(state.get('owner_gid', -1)) != int(previous_gid)
            or now_ts - float(state.get('last_ts', now_ts)) > WRONG_GID_ESCAPE_MAX_VOTE_GAP_SEC
        ):
            if isinstance(state, dict):
                self.wrong_gid_escape_resets += 1
            state = {
                'owner_gid': int(previous_gid),
                'challenger_gid': int(challenger_gid),
                'votes': 0,
                'last_ts': now_ts,
            }

        state['votes'] = int(state.get('votes', 0)) + 1
        state['last_ts'] = now_ts
        self.wrong_gid_escape_state[key] = state

        votes = int(state['votes'])
        accepted = votes >= WRONG_GID_ESCAPE_REQUIRED_VOTES
        if accepted:
            self.wrong_gid_escape_confirmed += 1
            self.wrong_gid_escape_state.pop(key, None)
            reason = 'multi-observation-confirmed'
        else:
            self.wrong_gid_escape_pending += 1
            self.wrong_gid_escape_owner_kept += 1
            reason = 'pending-more-evidence'

        self.last_wrong_gid_escape = {
            'camera': str(cam_name),
            'local_id': int(detection['tid']),
            'owner_gid': int(previous_gid),
            'challenger_gid': int(challenger_gid),
            'owner_score': owner_score,
            'challenger_score': challenger_score,
            'advantage': advantage,
            'votes': votes,
            'accepted': bool(accepted),
            'reason': reason,
        }
        return bool(accepted), reason, votes


    def assign_global_batch(
        self,
        camera_detections,
        prev_assignments_by_camera=None,
        event_time=None,
        batch_id=None,
        assignment_window_sec=None,
    ):
        _sync_identity_dependencies()
        """Match simultaneous detections from all cameras in one assignment.

        ``camera_detections`` is a mapping of camera name to that camera's
        detection list.  This is deliberately model-only: callers decide how
        to form the bounded rendezvous window, while this method makes one
        atomic identity decision for the supplied window.
        """
        default_event_time = (
            time.time()
            if event_time is None
            else float(event_time)
        )
        previous = prev_assignments_by_camera or {}
        results = {
            cam_name: [None for _ in detections]
            for cam_name, detections in camera_detections.items()
        }
        rows = [
            (
                cam_name,
                index,
                detection,
                float(
                    detection.get(
                        "event_time",
                        default_event_time
                    )
                ),
            )
            for cam_name, detections in camera_detections.items()
            for index, detection in enumerate(detections)
        ]
        event_times = [row[3] for row in rows]
        batch_event_time = (
            max(event_times)
            if event_times
            else default_event_time
        )
        resolved_batch_id = (
            str(batch_id)
            if batch_id is not None
            else f"global-{batch_event_time:.6f}"
        )

        for cam_name, _index, detection, _row_event_time in rows:
            record = self.state_machine.begin(
                cam_name,
                detection["tid"],
                detection.get("coordinator_generation", detection.get("camera_generation")),
            )
            self.state_machine.advance(record, "candidates_built")
            self.state_machine.advance(record, "gated")
            self.state_machine.advance(record, "topology_classified")
            self.state_machine.advance(record, "joint_assignment")

        with self.lock:
            previous_local_mappings = {
                row: (
                    dict(mapping)
                    if mapping is not None
                    else None
                )
                for row, (cam_name, _, detection, _) in enumerate(rows)
                for mapping in [
                    self.local_to_global.get(
                        (cam_name, int(detection["tid"]))
                    )
                ]
            }
            self.cleanup(
                reference_time=batch_event_time
            )
            self._update_recent_lost_local_tracks(
                rows,
                previous,
                batch_event_time,
            )
            current_unresolved_keys = {
                self._unresolved_handoff_key(cam_name, detection)
                for cam_name, _, detection, _ in rows
            }
            for key, record in list(
                self.unresolved_cross_camera_handoffs.items()
            ):
                if not self._unresolved_record_is_valid(key, record):
                    self.unresolved_cross_camera_handoffs.pop(key, None)
                    continue
                last_event_time = float(record["last_event_time"])
                if (
                    key not in current_unresolved_keys
                    and batch_event_time >= last_event_time
                    and batch_event_time - last_event_time
                    > AMBIGUOUS_HANDOFF_MAX_EVENT_SEC
                ):
                    self.unresolved_cross_camera_handoffs.pop(key, None)

            unresolved_keys_by_row = {}
            unresolved_records_by_row = {}
            scoring_detections = {}
            sample_added_rows = set()
            for row, (cam_name, _, detection, row_event_time) in enumerate(rows):
                key = self._unresolved_handoff_key(cam_name, detection)
                record = self.unresolved_cross_camera_handoffs.get(key)
                if not self._unresolved_record_is_valid(key, record):
                    self.unresolved_cross_camera_handoffs.pop(key, None)
                    continue
                self._append_unresolved_handoff_sample(
                    record,
                    detection,
                    row_event_time,
                )
                unresolved_keys_by_row[row] = key
                unresolved_records_by_row[row] = record
                scoring_detections[row] = self._unresolved_scoring_detection(
                    record,
                    detection,
                )
                sample_added_rows.add(row)
            used_gids_by_camera = {}

            def used_gids_for_camera(cam_name):
                return used_gids_by_camera.setdefault(cam_name, set())

            assigned_rows = set()
            rejections = []
            ownership_blocked_rows = set()

            # Resolve trusted camera-local evidence before building the
            # Hungarian matrix. Its GID is reserved only against another row
            # from the same camera; a separately gated cross-camera row may
            # still represent a valid handoff or simultaneous visibility.
            fixed_claims = []
            for row, (cam_name, _, detection, row_event_time) in enumerate(rows):
                claim = self._trusted_assignment_claim(
                    cam_name,
                    detection,
                    row_event_time,
                    drop_conflicting_local=True,
                )

                if claim is not None:
                    fixed_claims.append((row, claim))

            transition_claims = (
                self._transition_recovery_claims(
                    rows,
                    previous,
                    fixed_claims,
                )
            )

            if transition_claims:
                fixed_rows = {
                    int(row)
                    for row, _ in fixed_claims
                }
                for row, claim in transition_claims:
                    if int(row) not in fixed_rows:
                        fixed_claims.append(
                            (row, claim)
                        )
                        fixed_rows.add(int(row))

            fixed_claims = self._correct_pairwise_local_swaps(
                rows,
                fixed_claims,
                previous,
            )

            # A trusted, currently observed local owner retains cross-camera
            # ownership until it actually disappears or its mapping/presence
            # becomes invalid.  Keep this snapshot before committing fixed
            # rows, because those commits update identity recency in place.
            trusted_active_source_cameras_by_gid = {}
            for fixed_row, claim in fixed_claims:
                source_camera, _, source_detection, _ = rows[int(fixed_row)]
                source_gid = int(claim["gid"])
                if not self._presence_allows_local_claim(
                    source_gid,
                    source_camera,
                    source_detection["tid"],
                    source_detection,
                ):
                    continue
                trusted_active_source_cameras_by_gid.setdefault(
                    source_gid,
                    set(),
                ).add(source_camera)

            def blocking_trusted_source(gid, destination_camera):
                for source_camera in sorted(
                    trusted_active_source_cameras_by_gid.get(int(gid), ())
                ):
                    if source_camera == destination_camera:
                        continue
                    if self._explicit_overlap_allowed(
                        source_camera,
                        destination_camera,
                    ):
                        continue
                    return source_camera
                return None

            fixed_claims.sort(
                key=lambda item: (
                    -item[1]["priority"],
                    -item[1]["score"],
                    item[0],
                )
            )

            for row, claim in fixed_claims:
                cam_name, index, detection, row_event_time = rows[row]
                gid = claim["gid"]
                camera_used_gids = used_gids_for_camera(cam_name)
                if gid in camera_used_gids:
                    rejections.append({
                        "row": row,
                        "camera": cam_name,
                        "track_id": detection["tid"],
                        "gid": gid,
                        "reason": "trusted_gid_conflict",
                    })
                    local_key = (cam_name, int(detection["tid"]))
                    mapping = self.local_to_global.get(local_key)
                    if (
                        isinstance(mapping, dict)
                        and mapping.get("gid") == gid
                    ):
                        self.local_to_global.pop(local_key, None)
                    tracklet = self.tracklets.get(local_key)
                    if (
                        tracklet is not None
                        and (
                            not isinstance(tracklet, dict)
                            or tracklet.get("gid") == gid
                        )
                    ):
                        self.tracklets.pop(local_key, None)
                    hold = self.occlusion_hold.get(local_key)
                    if (
                        isinstance(hold, dict)
                        and hold.get("gid") == gid
                    ):
                        self.occlusion_hold.pop(local_key, None)
                    continue

                results[cam_name][index] = self._commit_assignment(
                    gid,
                    cam_name,
                    detection["tid"],
                    detection["emb"],
                    detection.get("map_pos"),
                    detection.get("box_wh"),
                    row_event_time,
                    claim["score"],
                    claim["source"],
                    generation=detection.get(
                        "coordinator_generation",
                        detection.get("camera_generation"),
                    ),
                )
                self._clear_unresolved_handoff(cam_name, detection)
                camera_used_gids.add(gid)
                assigned_rows.add(row)

            pending_rows = [
                row
                for row in range(len(rows))
                if row not in assigned_rows
            ]
            candidate_gids = [
                gid
                for gid in self.identities
                if (
                    any(
                        (
                            gid not in used_gids_for_camera(rows[row][0])
                            and self._matchable_identity_for_camera(
                                gid,
                                rows[row][0],
                            ) is not None
                        )
                        for row in pending_rows
                    )
                )
            ]

            # V20: preserve the previous GID for each pending same-local row.
            # Preservation is row-scoped: the old GID is only given a special
            # fallback for the row that actually owned it previously.
            preserved_previous_gids_by_row = {}

            for row in pending_rows:
                previous_mapping = previous_local_mappings.get(row)
                if not isinstance(previous_mapping, dict):
                    continue

                previous_gid = previous_mapping.get("gid")
                if previous_gid is None:
                    continue

                previous_gid = int(previous_gid)

                if previous_gid in used_gids_for_camera(rows[row][0]):
                    self.soft_continuity_preserve_skips += 1
                    continue

                if self._assignable_identity(previous_gid) is None:
                    self.soft_continuity_preserve_skips += 1
                    continue

                preserved_previous_gids_by_row[row] = previous_gid

                if previous_gid not in candidate_gids:
                    candidate_gids.append(previous_gid)
                    self.soft_continuity_candidates_preserved += 1
                    self.last_preserved_candidate = {
                        "row": int(row),
                        "gid": int(previous_gid),
                        "reason": "previous-local-mapping",
                    }

            candidate_gids = sorted(set(candidate_gids))
            score_matrix = np.full(
                (len(pending_rows), len(candidate_gids)),
                -1e6,
                dtype=np.float32
            )
            pair_cache = {}

            for matrix_row, row in enumerate(pending_rows):
                cam_name, _, detection, row_event_time = rows[row]
                scoring_detection = scoring_detections.get(row, detection)
                for column, gid in enumerate(candidate_gids):
                    if gid in used_gids_for_camera(cam_name):
                        pair_cache[(row, gid)] = {
                            "gate_failure": "used_by_same_camera_claim"
                        }
                        continue
                    blocking_source = blocking_trusted_source(gid, cam_name)
                    if blocking_source is not None:
                        ownership_blocked_rows.add(row)
                        pair_cache[(row, gid)] = {
                            "gate_failure": (
                                "trusted_active_source_ownership"
                            ),
                            "source_camera": blocking_source,
                        }
                        rejections.append({
                            "row": row,
                            "camera": cam_name,
                            "track_id": detection["tid"],
                            "gid": gid,
                            "source_camera": blocking_source,
                            "reason": "trusted_active_source_ownership",
                        })
                        continue
                    identity = self._matchable_identity_for_camera(
                        gid,
                        cam_name,
                    )

                    preserved_previous_gid = (
                        preserved_previous_gids_by_row.get(row)
                    )
                    preserved_pair = (
                        preserved_previous_gid is not None
                        and int(gid) == int(preserved_previous_gid)
                    )

                    if identity is None and preserved_pair:
                        identity = self._assignable_identity(gid)
                        if identity is not None:
                            self.soft_continuity_preserved_pair_fallbacks += 1

                    if identity is None:
                        pair_cache[(row, gid)] = {
                            "gate_failure": "identity_not_globally_matchable"
                        }
                        continue
                    topology_details = self._topology_gate_details(
                        identity,
                        cam_name,
                        scoring_detection,
                        row_event_time
                    )
                    gate_reason = self._hard_gate_reason(
                        identity,
                        cam_name,
                        scoring_detection,
                        row_event_time,
                    )
                    if gate_reason is not None:
                        pair_cache[(row, gid)] = {
                            "gate_failure": gate_reason,
                            "topology": topology_details,
                        }
                        continue

                    pair = self._finite_pair(
                        self._pair_score(
                            gid,
                            identity,
                            cam_name,
                            scoring_detection,
                            row_event_time,
                            previous.get(cam_name, []),
                        )
                    )
                    if pair is None:
                        pair_cache[(row, gid)] = {
                            "score_failure": "invalid_pair_score",
                            "topology": topology_details,
                        }
                        continue
                    pair["topology"] = topology_details
                    pair_cache[(row, gid)] = pair
                    score_matrix[matrix_row, column] = pair["score"]

            # V17: apply continuity only as a score prior after every ReID
            # candidate has been evaluated. This preserves V15's full
            # prototype/global matching path while discouraging weak relabels.
            for matrix_row, row in enumerate(pending_rows):
                cam_name, _, detection, row_event_time = rows[row]
                continuity = self._soft_local_continuity_context(
                    cam_name,
                    detection,
                    row_event_time,
                    existing_override=previous_local_mappings.get(row),
                )
                if continuity is None:
                    continue

                existing_gid = int(continuity["gid"])
                if existing_gid not in candidate_gids:
                    continue

                existing_column = candidate_gids.index(existing_gid)
                existing_pair = pair_cache.get((row, existing_gid), {})
                if "score" not in existing_pair:
                    continue

                self.soft_continuity_rows += 1

                existing_raw = float(existing_pair["score"])
                existing_adjusted = min(
                    1.0,
                    existing_raw + SOFT_LOCAL_CONTINUITY_BONUS,
                )
                existing_pair["raw_score_before_continuity"] = existing_raw
                existing_pair["continuity_bonus"] = float(
                    SOFT_LOCAL_CONTINUITY_BONUS
                )
                existing_pair["score"] = float(existing_adjusted)
                score_matrix[matrix_row, existing_column] = float(
                    existing_adjusted
                )
                self.soft_continuity_bonus_applied += 1

                blocked = []
                allowed = []
                for column, candidate_gid in enumerate(candidate_gids):
                    if int(candidate_gid) == existing_gid:
                        continue
                    candidate_pair = pair_cache.get((row, candidate_gid), {})
                    if "score" not in candidate_pair:
                        continue

                    candidate_score = float(candidate_pair["score"])
                    required = existing_raw + SOFT_LOCAL_SWITCH_MIN_GAIN

                    if candidate_score < required:
                        candidate_pair["soft_continuity_blocked"] = True
                        candidate_pair["soft_continuity_required_score"] = float(
                            required
                        )
                        score_matrix[matrix_row, column] = -1e6
                        blocked.append(
                            (int(candidate_gid), candidate_score)
                        )
                        self.soft_continuity_switches_blocked += 1
                    else:
                        candidate_pair["soft_continuity_override_allowed"] = True
                        candidate_pair["soft_continuity_required_score"] = float(
                            required
                        )
                        allowed.append(
                            (int(candidate_gid), candidate_score)
                        )
                        self.soft_continuity_switches_allowed += 1

                self.last_soft_continuity = {
                    "camera": cam_name,
                    "local_id": int(detection["tid"]),
                    "existing_gid": existing_gid,
                    "gap_sec": float(continuity["gap_sec"]),
                    "existing_raw_score": float(existing_raw),
                    "existing_adjusted_score": float(existing_adjusted),
                    "mapping_source": continuity.get("mapping_source"),
                    "blocked": blocked,
                    "allowed": allowed,
                }

            candidate_details_by_row = {}
            top1_top2_margin_by_row = {}
            for row in pending_rows:
                viable_scores = sorted(
                    [
                        float(pair_cache[(row, gid)]["score"])
                        for gid in candidate_gids
                        if (
                            (row, gid) in pair_cache
                            and "score" in pair_cache[(row, gid)]
                        )
                    ],
                    reverse=True,
                )
                top1_top2_margin_by_row[row] = (
                    float(viable_scores[0] - viable_scores[1])
                    if len(viable_scores) >= 2
                    else None
                )
                candidate_details_by_row[row] = [
                    {
                        "gid": gid,
                        "hard_gate_passed": (
                            "gate_failure" not in pair_cache[(row, gid)]
                        ),
                        "hard_gate_reason": pair_cache[(row, gid)].get(
                            "gate_failure"
                        ),
                        "score_failure_reason": pair_cache[(row, gid)].get(
                            "score_failure"
                        ),
                        "appearance": pair_cache[(row, gid)].get(
                            "appearance"
                        ),
                        "score": pair_cache[(row, gid)].get("score"),
                        "motion": pair_cache[(row, gid)].get("motion"),
                        "topology": pair_cache[(row, gid)].get("topology"),
                    }
                    for gid in candidate_gids
                    if (row, gid) in pair_cache
                ]

                record = unresolved_records_by_row.get(row)
                if record is not None:
                    record["candidate_gids"] = [
                        item["gid"]
                        for item in candidate_details_by_row[row]
                        if (
                            item.get("hard_gate_passed")
                            and item.get("score") is not None
                        )
                    ]
                    record["candidates"] = (
                        self._unresolved_candidate_diagnostics(
                            candidate_details_by_row[row]
                        )
                    )
                    record["top1_top2_margin"] = (
                        top1_top2_margin_by_row[row]
                    )
                    record["last_batch_id"] = resolved_batch_id
                    self._record_unresolved_top1_observation(
                        record,
                        candidate_details_by_row[row],
                        resolved_batch_id,
                    )

            selected = []
            resolution_reasons_by_row = {}
            ambiguous_rows = set()
            deferred_cross_camera_rows = set()
            viable_rows = set()
            for matrix_row, row in enumerate(pending_rows):
                viable_candidates = [
                    (gid, pair_cache[(row, gid)])
                    for gid in candidate_gids
                    if "score" in pair_cache.get((row, gid), {})
                ]
                if not viable_candidates:
                    continue
                viable_rows.add(row)
                top_gid, top_pair = max(
                    viable_candidates,
                    key=lambda item: float(item[1]["score"]),
                )
                ambiguity_reason, margin = self._ambiguity_reason(
                    pair_cache,
                    row,
                    candidate_gids,
                    top_pair["cross_camera"],
                )
                if ambiguity_reason is None:
                    continue
                ambiguous_rows.add(row)
                cam_name, _, detection, _ = rows[row]
                all_cross_camera = all(
                    bool(pair.get("cross_camera"))
                    for _, pair in viable_candidates
                )
                if top_pair.get("cross_camera") and all_cross_camera:
                    key, record = self._defer_ambiguous_cross_camera_handoff(
                        cam_name,
                        detection,
                        rows[row][3],
                        candidate_details_by_row.get(row, []),
                        margin,
                        resolved_batch_id,
                        sample_already_added=(row in sample_added_rows),
                    )
                    unresolved_keys_by_row[row] = key
                    unresolved_records_by_row[row] = record
                    scoring_detections[row] = (
                        self._unresolved_scoring_detection(record, detection)
                    )
                    deferred_cross_camera_rows.add(row)
                rejections.append({
                    "row": row,
                    "camera": cam_name,
                    "track_id": detection["tid"],
                    "gid": top_gid,
                    "reason": ambiguity_reason,
                    "score": float(top_pair["score"]),
                    "top1_top2_margin": margin,
                })

            pending_hold_rows = set()
            accepted_cross_candidates_by_row = {}
            for row, record in unresolved_records_by_row.items():
                if row not in pending_rows:
                    continue
                row_age = self._unresolved_handoff_age(
                    record,
                    rows[row][3],
                )
                if record.get("sample_count", 0) < AMBIGUOUS_HANDOFF_MIN_SAMPLES:
                    pending_hold_rows.add(row)
                    record["event_order_result"] = "waiting_for_min_samples"
                elif row_age < AMBIGUOUS_HANDOFF_MIN_SOLO_EVENT_SEC:
                    pending_hold_rows.add(row)
                    record["event_order_result"] = (
                        "waiting_for_competing_arrivals"
                    )

                cam_name, _, detection, _ = rows[row]
                scoring_detection = scoring_detections.get(row, detection)
                accepted_candidates = []
                for gid in candidate_gids:
                    pair = pair_cache.get((row, gid))
                    identity = self.identities.get(gid)
                    if (
                        pair is None
                        or "score" not in pair
                        or not pair.get("cross_camera", False)
                        or identity is None
                        or not self._accept_match(
                            pair,
                            identity,
                            cam_name,
                            scoring_detection,
                        )
                    ):
                        continue
                    accepted_candidates.append(gid)
                if accepted_candidates:
                    accepted_cross_candidates_by_row[row] = tuple(
                        sorted(accepted_candidates)
                    )
                elif row not in ambiguous_rows:
                    # A pending cross-camera handoff cannot silently become a
                    # same-camera/global claim after another row moves a GID.
                    pending_hold_rows.add(row)
                    record["event_order_result"] = (
                        "waiting_for_cross_camera_candidate"
                    )

            grouped_unresolved_rows = {}
            for row, accepted_gids in accepted_cross_candidates_by_row.items():
                record = unresolved_records_by_row[row]
                if record.get("sample_count", 0) < AMBIGUOUS_HANDOFF_MIN_SAMPLES:
                    continue
                source_cameras = {
                    self.identities[gid].get("last_cam")
                    for gid in accepted_gids
                    if gid in self.identities
                }
                if len(source_cameras) != 1:
                    continue
                group_key = (
                    rows[row][0],
                    next(iter(source_cameras)),
                    accepted_gids,
                )
                grouped_unresolved_rows.setdefault(group_key, []).append(row)

            joint_constraint_by_row = {}
            for (_, _, accepted_gids), group_rows in grouped_unresolved_rows.items():
                if len(group_rows) < 2 or len(group_rows) != len(accepted_gids):
                    continue
                if any(
                    self._unresolved_handoff_age(
                        unresolved_records_by_row[row],
                        rows[row][3],
                    ) > AMBIGUOUS_HANDOFF_MAX_EVENT_SEC
                    for row in group_rows
                ):
                    continue

                # Do not reserve a candidate away from an unrelated row.  If
                # such competition exists, leave the whole group unresolved
                # for a later complete global decision.
                group_row_set = set(group_rows)
                outside_conflict = False
                for other_row in pending_rows:
                    if other_row in group_row_set:
                        continue
                    other_cam, _, other_detection, _ = rows[other_row]
                    other_scoring_detection = scoring_detections.get(
                        other_row,
                        other_detection,
                    )
                    for gid in accepted_gids:
                        pair = pair_cache.get((other_row, gid))
                        identity = self.identities.get(gid)
                        if (
                            pair is not None
                            and "score" in pair
                            and identity is not None
                            and self._accept_match(
                                pair,
                                identity,
                                other_cam,
                                other_scoring_detection,
                            )
                        ):
                            outside_conflict = True
                            break
                    if outside_conflict:
                        break
                if outside_conflict:
                    continue

                ordered_rows = sorted(
                    group_rows,
                    key=lambda row: (
                        unresolved_records_by_row[row]["first_event_time"],
                        rows[row][2]["tid"],
                    ),
                )
                ordered_gids = sorted(
                    accepted_gids,
                    key=lambda gid: (
                        self.identities[gid].get(
                            "last_event_time",
                            self.identities[gid].get("last_seen", 0.0),
                        ),
                        gid,
                    ),
                )
                arrival_times = [
                    unresolved_records_by_row[row]["first_event_time"]
                    for row in ordered_rows
                ]
                departure_times = [
                    self.identities[gid].get(
                        "last_event_time",
                        self.identities[gid].get("last_seen", 0.0),
                    )
                    for gid in ordered_gids
                ]
                if (
                    len(set(arrival_times)) != len(arrival_times)
                    or len(set(departure_times)) != len(departure_times)
                    or not all(
                        isinstance(value, (int, float))
                        and np.isfinite(float(value))
                        for value in departure_times
                    )
                ):
                    continue

                for row, gid in zip(ordered_rows, ordered_gids):
                    joint_constraint_by_row[row] = gid
                    unresolved_records_by_row[row]["event_order_result"] = (
                        "order_preserving_joint_assignment"
                    )

            eligible_matrix_rows = [
                matrix_row
                for matrix_row, row in enumerate(pending_rows)
                if (
                    row in joint_constraint_by_row
                    or (
                        row in viable_rows
                        and row not in ambiguous_rows
                        and row not in pending_hold_rows
                    )
                )
            ]
            if eligible_matrix_rows and candidate_gids:
                eligible_rows_by_camera = {}
                for matrix_row in eligible_matrix_rows:
                    row = pending_rows[matrix_row]
                    eligible_rows_by_camera.setdefault(
                        rows[row][0],
                        [],
                    ).append(matrix_row)

                # A physical identity is one-to-one among local tracks in a
                # camera, not across cameras. Each pair still has to pass the
                # existing hard gate and acceptance logic above and below.
                for camera_matrix_rows in eligible_rows_by_camera.values():
                    camera_score_matrix = score_matrix[
                        camera_matrix_rows,
                        :,
                    ].copy()
                    for camera_row, matrix_row in enumerate(
                        camera_matrix_rows
                    ):
                        row = pending_rows[matrix_row]
                        constrained_gid = joint_constraint_by_row.get(row)
                        if constrained_gid is None:
                            continue
                        camera_score_matrix[camera_row, :] = -1e6
                        constrained_column = candidate_gids.index(
                            constrained_gid
                        )
                        camera_score_matrix[
                            camera_row,
                            constrained_column,
                        ] = score_matrix[matrix_row, constrained_column]
                    row_ind, col_ind = linear_sum_assignment(
                        -camera_score_matrix
                    )
                    for camera_row, column in zip(
                        row_ind.tolist(),
                        col_ind.tolist()
                    ):
                        matrix_row = camera_matrix_rows[camera_row]
                        row = pending_rows[matrix_row]
                        gid = candidate_gids[column]
                        pair = pair_cache.get((row, gid))
                        cam_name, _, detection, _ = rows[row]
                        scoring_detection = scoring_detections.get(
                            row,
                            detection,
                        )
                        identity = self.identities.get(gid)
                        if (
                            pair is None
                            or "score" not in pair
                            or identity is None
                            or camera_score_matrix[camera_row, column] <= -1e5
                            or (
                                row in joint_constraint_by_row
                                and joint_constraint_by_row[row] != gid
                            )
                        ):
                            continue
                        if not self._accept_match(
                            pair,
                            identity,
                            cam_name,
                            scoring_detection,
                        ):
                            rejections.append({
                                "row": row,
                                "camera": cam_name,
                                "track_id": detection["tid"],
                                "gid": gid,
                                "reason": "acceptance_threshold",
                                "score": float(pair["score"]),
                            })
                            continue
                        selected.append((row, gid, pair))
                        resolution_reasons_by_row[row] = (
                            "event_order_joint_one_to_one"
                            if row in joint_constraint_by_row
                            else (
                                "aggregated_margin_sufficient"
                                if row in unresolved_records_by_row
                                else "global_hungarian"
                            )
                        )

            selected_by_gid = {}
            for selected_item in selected:
                selected_by_gid.setdefault(
                    int(selected_item[1]),
                    [],
                ).append(selected_item)

            filtered_selected = []
            for gid, gid_selections in selected_by_gid.items():
                selected_cameras = {
                    rows[row][0]
                    for row, _, _ in gid_selections
                }
                simultaneous_visibility_allowed = all(
                    self._explicit_overlap_allowed(left_camera, right_camera)
                    for left_camera in selected_cameras
                    for right_camera in selected_cameras
                    if left_camera != right_camera
                )
                if len(selected_cameras) <= 1 or simultaneous_visibility_allowed:
                    filtered_selected.extend(gid_selections)
                    continue

                # Multiple new destinations may independently pass the gate
                # from the same source camera, but without explicit overlap
                # they cannot all be a plausible simultaneous presence. Keep
                # the strongest deterministic handoff. A fixed claim in a
                # different camera is intentionally not part of this set.
                winner = min(
                    gid_selections,
                    key=lambda item: (
                        -float(item[2]["score"]),
                        rows[item[0]][0],
                        int(rows[item[0]][2]["tid"]),
                        int(item[0]),
                    ),
                )
                filtered_selected.append(winner)
                for rejected_row, _, rejected_pair in gid_selections:
                    if rejected_row == winner[0]:
                        continue
                    cam_name, _, detection, _ = rows[rejected_row]
                    rejections.append({
                        "row": rejected_row,
                        "camera": cam_name,
                        "track_id": detection["tid"],
                        "gid": gid,
                        "reason": "cross_camera_overlap_not_allowed",
                        "score": float(rejected_pair["score"]),
                    })
                    resolution_reasons_by_row.pop(rejected_row, None)
            selected = sorted(filtered_selected, key=lambda item: item[0])

            resolved_ambiguities = []
            selected_rows = {row for row, _, _ in selected}
            retained_rejections = []
            for rejection in rejections:
                if (
                    rejection.get("row") in selected_rows
                    and rejection.get("reason") == "ambiguous_top1_top2"
                ):
                    resolved_ambiguities.append({
                        **rejection,
                        "resolution_reason": resolution_reasons_by_row.get(
                            rejection.get("row")
                        ),
                    })
                else:
                    retained_rejections.append(rejection)
            rejections = retained_rejections

            # Reserve the complete per-camera Hungarian result before any
            # commit-time hysteresis runs. A row may retain an unreserved
            # previous GID, but it must never steal a GID selected for another
            # row in the same camera merely because that row has not committed.
            selected_gid_reservations = {
                (rows[selected_row][0], int(selected_gid)): int(selected_row)
                for selected_row, selected_gid, _ in selected
            }

            # Commit only after Hungarian has selected the complete global
            # one-to-one set. V26 adds hysteresis only when the SAME local ID
            # is about to change from its previous GID to a challenger GID.
            for row, gid, pair in selected:
                cam_name, index, detection, row_event_time = rows[row]
                commit_detection = scoring_detections.get(row, detection)
                source = "global-cross-camera" if pair["cross_camera"] else "global-batch"

                previous_mapping = previous_local_mappings.get(row)
                previous_gid = (
                    int(previous_mapping.get("gid"))
                    if isinstance(previous_mapping, dict)
                    and previous_mapping.get("gid") is not None
                    else None
                )

                commit_gid = int(gid)
                commit_pair = pair
                commit_source = source
                previous_gid_reserved_row = (
                    selected_gid_reservations.get((cam_name, previous_gid))
                    if previous_gid is not None
                    else None
                )

                if (
                    previous_gid is not None
                    and previous_gid != int(gid)
                    and previous_gid not in used_gids_for_camera(cam_name)
                    and (
                        previous_gid_reserved_row is None
                        or previous_gid_reserved_row == row
                    )
                    and self._assignable_identity(previous_gid) is not None
                ):
                    owner_pair = pair_cache.get((row, previous_gid))
                    if not isinstance(owner_pair, dict) or "score" not in owner_pair:
                        owner_identity = self._assignable_identity(previous_gid)
                        if owner_identity is not None:
                            owner_pair = self._finite_pair(
                                self._pair_score(
                                    previous_gid,
                                    owner_identity,
                                    cam_name,
                                    commit_detection,
                                    row_event_time,
                                    previous.get(cam_name, []),
                                )
                            )

                    owner_score_ok = bool(
                        isinstance(owner_pair, dict)
                        and "score" in owner_pair
                        and float(owner_pair["score"]) >= WRONG_GID_ESCAPE_MIN_OWNER_SCORE
                    )

                    if owner_score_ok:
                        switch_allowed, escape_reason, escape_votes = (
                            self._wrong_gid_escape_decision(
                                cam_name,
                                detection,
                                row_event_time,
                                previous_mapping,
                                int(gid),
                                pair,
                                owner_pair,
                            )
                        )
                        if not switch_allowed:
                            commit_gid = int(previous_gid)
                            commit_pair = owner_pair
                            commit_source = "same-local-hysteresis-hold"
                        else:
                            commit_source = (
                                "same-local-confirmed-escape"
                                if not pair.get("cross_camera", False)
                                else "global-cross-camera"
                            )

                # Final per-camera uniqueness guard. Reservation conflicts
                # fall back to this row's original Hungarian selection; the
                # same GID remains legal in a different camera after its own
                # hard-gate and acceptance checks.
                reserved_row = selected_gid_reservations.get(
                    (cam_name, commit_gid)
                )
                if reserved_row is not None and reserved_row != row:
                    commit_gid = int(gid)
                    commit_pair = pair
                    commit_source = source

                camera_used_gids = used_gids_for_camera(cam_name)
                if commit_gid in camera_used_gids:
                    rejections.append({
                        "row": row,
                        "camera": cam_name,
                        "track_id": detection["tid"],
                        "gid": commit_gid,
                        "reason": "global_gid_conflict_before_commit",
                    })
                    continue

                blocking_source = blocking_trusted_source(
                    commit_gid,
                    cam_name,
                )
                if blocking_source is not None:
                    rejections.append({
                        "row": row,
                        "camera": cam_name,
                        "track_id": detection["tid"],
                        "gid": commit_gid,
                        "source_camera": blocking_source,
                        "reason": "trusted_active_source_ownership",
                    })
                    continue

                results[cam_name][index] = self._commit_assignment(
                    commit_gid, cam_name, detection["tid"], commit_detection["emb"],
                    detection.get("map_pos"), detection.get("box_wh"),
                    row_event_time,
                    commit_pair["score"], commit_source,
                    generation=detection.get(
                        "coordinator_generation",
                        detection.get("camera_generation"),
                    ),
                )
                unresolved_record = unresolved_records_by_row.get(row)
                if unresolved_record is not None:
                    unresolved_record["pending_reason"] = None
                    unresolved_record["resolution_reason"] = (
                        resolution_reasons_by_row.get(row)
                    )
                self._clear_unresolved_handoff(cam_name, detection)
                camera_used_gids.add(commit_gid)
                assigned_rows.add(row)

            # Preserve the existing same-camera cache fallback, while keeping
            # every GID already used by this camera unavailable.
            cache_blocked_rows = set(ambiguous_rows)
            cache_blocked_rows.update(unresolved_records_by_row)
            cache_blocked_rows.update(ownership_blocked_rows)
            for row in pending_rows:
                if row in assigned_rows:
                    continue
                if row in cache_blocked_rows:
                    continue
                cam_name, index, detection, row_event_time = rows[row]
                eligible_cache_gids = {
                    gid
                    for gid in candidate_gids
                    if (
                        "score" in pair_cache.get((row, gid), {})
                        and not pair_cache[(row, gid)].get(
                            "cross_camera",
                            True,
                        )
                    )
                }
                gid, recent_score = self._find_recent_same_cam_match(
                    cam_name,
                    detection["emb"],
                    detection.get("map_pos"),
                    detection.get("box_wh"),
                    row_event_time,
                    used_gids=used_gids_for_camera(cam_name),
                    eligible_gids=eligible_cache_gids,
                )
                if (
                    gid is None
                    or self._matchable_identity_for_camera(gid, cam_name) is None
                ):
                    continue
                results[cam_name][index] = self._commit_assignment(
                    gid,
                    cam_name,
                    detection["tid"],
                    detection["emb"],
                    detection.get("map_pos"),
                    detection.get("box_wh"),
                    row_event_time,
                    recent_score,
                    "same-cam-cache",
                    generation=detection.get(
                        "coordinator_generation",
                        detection.get("camera_generation"),
                    ),
                )
                self._clear_unresolved_handoff(cam_name, detection)
                used_gids_for_camera(cam_name).add(gid)
                assigned_rows.add(row)

            new_identity_reasons = {}
            pending_reasons_by_row = {}
            temporal_consistency_decisions_by_row = {}
            for row, (cam_name, index, detection, row_event_time) in enumerate(rows):
                if row in assigned_rows:
                    continue
                if row in ownership_blocked_rows:
                    pending_reasons_by_row[row] = (
                        "trusted_active_source_ownership_pending"
                    )
                    continue
                unresolved_record = unresolved_records_by_row.get(row)
                if unresolved_record is not None:
                    unresolved_age = self._unresolved_handoff_age(
                        unresolved_record,
                        row_event_time,
                    )
                    if unresolved_age < AMBIGUOUS_HANDOFF_MAX_EVENT_SEC:
                        pending_reasons_by_row[row] = (
                            "ambiguous_cross_camera_handoff_pending"
                        )
                        continue
                    unresolved_record["event_order_result"] = (
                        "bounded_window_exhausted"
                    )

                    temporal_decision = self._unresolved_temporal_consensus(
                        unresolved_record
                    )
                    temporal_gid = temporal_decision.get("dominant_gid")
                    temporal_pair = (
                        pair_cache.get((row, temporal_gid))
                        if temporal_gid is not None
                        else None
                    )
                    temporal_identity = (
                        self._matchable_identity_for_camera(
                            temporal_gid,
                            cam_name,
                        )
                        if temporal_gid is not None
                        else None
                    )
                    scoring_detection = scoring_detections.get(
                        row,
                        detection,
                    )

                    if temporal_decision["eligible"]:
                        if (
                            not isinstance(temporal_pair, dict)
                        ):
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                "dominant_candidate_not_currently_viable"
                            )
                        elif temporal_pair.get("gate_failure") is not None:
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = temporal_pair.get(
                                "gate_failure"
                            )
                        elif "score" not in temporal_pair:
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                "dominant_candidate_not_currently_viable"
                            )
                        elif temporal_identity is None:
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                "dominant_identity_not_matchable"
                            )
                        elif not temporal_pair.get("cross_camera", False):
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                "dominant_candidate_not_cross_camera"
                            )
                        elif temporal_gid in used_gids_for_camera(cam_name):
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                "same_camera_gid_already_used"
                            )

                    if temporal_decision["eligible"]:
                        presence = temporal_identity.get(
                            "camera_presence",
                            {},
                        )
                        destination_presence = (
                            presence.get(cam_name)
                            if isinstance(presence, dict)
                            else None
                        )
                        owner_local_id = (
                            destination_presence.get("local_track_id")
                            if isinstance(destination_presence, dict)
                            and destination_presence.get("active", False)
                            else None
                        )
                        if (
                            owner_local_id is not None
                            and int(owner_local_id) != int(detection["tid"])
                        ):
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                "same_camera_presence_owned_by_other_local"
                            )

                    if temporal_decision["eligible"]:
                        topology = temporal_pair.get("topology")
                        if (
                            not isinstance(topology, dict)
                            or topology.get("passed") is not True
                        ):
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                "topology_time_gate_failed"
                            )

                    if temporal_decision["eligible"]:
                        hard_gate = self._hard_gate_diagnostics(
                            temporal_identity,
                            cam_name,
                            scoring_detection,
                            row_event_time,
                        )
                        temporal_decision["hard_gate"] = hard_gate
                        if not hard_gate.get("passed", False):
                            temporal_decision["eligible"] = False
                            temporal_decision["reason"] = (
                                hard_gate.get("reason")
                                or "hard_gate_failed"
                            )

                    if (
                        temporal_decision["eligible"]
                        and not self._accept_match(
                            temporal_pair,
                            temporal_identity,
                            cam_name,
                            scoring_detection,
                        )
                    ):
                        temporal_decision["eligible"] = False
                        temporal_decision["reason"] = (
                            "acceptance_threshold"
                        )

                    if temporal_decision["eligible"]:
                        temporal_decision["reason"] = (
                            "temporal_consistent_top1_at_window_end"
                        )
                        temporal_decision["committed_gid"] = int(
                            temporal_gid
                        )
                        unresolved_record["pending_reason"] = None
                        unresolved_record["resolution_reason"] = (
                            temporal_decision["reason"]
                        )
                        unresolved_record["event_order_result"] = (
                            "temporal_consistency_resolution"
                        )
                        unresolved_record[
                            "temporal_consistency_resolution"
                        ] = copy.deepcopy(temporal_decision)
                        temporal_consistency_decisions_by_row[row] = (
                            copy.deepcopy(temporal_decision)
                        )
                        resolution_reasons_by_row[row] = (
                            temporal_decision["reason"]
                        )
                        results[cam_name][index] = self._commit_assignment(
                            int(temporal_gid),
                            cam_name,
                            detection["tid"],
                            scoring_detection["emb"],
                            detection.get("map_pos"),
                            detection.get("box_wh"),
                            row_event_time,
                            temporal_pair["score"],
                            "global-cross-camera",
                            generation=detection.get(
                                "coordinator_generation",
                                detection.get("camera_generation"),
                            ),
                        )
                        results[cam_name][index]["assignment_reason"] = (
                            temporal_decision["reason"]
                        )
                        self._clear_unresolved_handoff(
                            cam_name,
                            detection,
                        )
                        used_gids_for_camera(cam_name).add(
                            int(temporal_gid)
                        )
                        assigned_rows.add(row)
                        continue

                    unresolved_record[
                        "temporal_consistency_resolution"
                    ] = copy.deepcopy(temporal_decision)
                    temporal_consistency_decisions_by_row[row] = (
                        copy.deepcopy(temporal_decision)
                    )
                    self._clear_unresolved_handoff(cam_name, detection)

                row_rejections = [
                    rejection["reason"]
                    for rejection in rejections
                    if rejection.get("row") == row
                ]
                row_pairs = [
                    pair_cache[(row, gid)]
                    for gid in candidate_gids
                    if (row, gid) in pair_cache
                ]
                if unresolved_record is not None:
                    new_reason = "ambiguous_handoff_window_exhausted"
                elif not candidate_gids:
                    new_reason = "no_eligible_candidate"
                elif row_rejections:
                    new_reason = row_rejections[-1]
                elif row_pairs and all(
                    "gate_failure" in pair
                    for pair in row_pairs
                ):
                    new_reason = "all_candidates_hard_gated"
                else:
                    new_reason = "unmatched_global_assignment"
                fallback_detection = (
                    scoring_detections.get(row, detection)
                    if unresolved_record is not None
                    else detection
                )

                print(
                    "[NEW_ID_DEBUG]",
                    {
                        "camera": cam_name,
                        "tid": detection["tid"],
                        "row": row,
                        "candidate_gids": candidate_gids,
                        "row_pairs": row_pairs,
                        "row_rejections": row_rejections,
                        "unresolved": unresolved_record is not None,
                        "new_reason": new_reason,
                        "existing_gids": list(self.identities.keys()),
                    },
                )
                                
                results[cam_name][index] = self._new_identity(
                    cam_name, detection["tid"], fallback_detection["emb"],
                    detection.get("map_pos"), detection.get("box_wh"),
                    row_event_time,
                    generation=detection.get(
                        "coordinator_generation",
                        detection.get("camera_generation"),
                    ),
                )
                results[cam_name][index]["assignment_reason"] = new_reason
                new_identity_reasons[row] = new_reason
                assigned_rows.add(row)

            for cam_name, index, detection, row_event_time in rows:
                result = results[cam_name][index]
                if result is None:
                    continue
                if (
                    detection.get("overlap", False)
                    and detection.get("local_track_confirmed", True)
                ):
                    self.occlusion_hold[(cam_name, int(detection["tid"]))] = {
                        "gid": result["gid"],
                        "until_ts": row_event_time + OCCLUSION_HOLD_SEC,
                        "score": float(result["score"]),
                    }

                if not detection.get("local_track_confirmed", True):
                    ephemeral_key = (cam_name, int(detection["tid"]))
                    self.local_to_global.pop(ephemeral_key, None)
                    self.occlusion_hold.pop(ephemeral_key, None)
                    self.tracklets.pop(ephemeral_key, None)

                accepted, reason = self._record_tracklet_sample(
                    result["gid"],
                    cam_name,
                    detection["tid"],
                    detection,
                    row_event_time,
                )
                result.update(
                    self._gallery_assignment_diagnostics(
                        result["gid"],
                        accepted,
                        reason,
                    )
                )

            gate_failures = [
                {
                    "camera": rows[row][0],
                    "track_id": rows[row][2]["tid"],
                    "gid": gid,
                    "reason": pair["gate_failure"],
                    "row": row,
                }
                for (row, gid), pair in pair_cache.items()
                if "gate_failure" in pair
            ]
            handoff_decisions = []
            handoff_decisions_by_row = {}
            for row, (cam_name, index, detection, row_event_time) in enumerate(rows):
                result = results[cam_name][index]
                handoff = result.get("handoff") if result is not None else None
                final_gid = result["gid"] if result is not None else None
                candidate_gid = final_gid
                pair = (
                    pair_cache.get((row, final_gid))
                    if final_gid is not None
                    else None
                )
                if pair is None:
                    row_candidates = [
                        (gid, candidate)
                        for (candidate_row, gid), candidate in pair_cache.items()
                        if candidate_row == row
                    ]
                    scored_candidates = [
                        item
                        for item in row_candidates
                        if "score" in item[1]
                    ]
                    if scored_candidates:
                        candidate_gid, pair = max(
                            scored_candidates,
                            key=lambda item: float(item[1]["score"]),
                        )
                    elif row_candidates:
                        candidate_gid, pair = row_candidates[0]
                    else:
                        candidate_gid, pair = None, {}
                committed = isinstance(handoff, dict)
                if committed:
                    rejection_reason = None
                elif result is None:
                    rejection_reason = pending_reasons_by_row.get(
                        row,
                        "ambiguous_cross_camera_handoff_pending",
                    )
                elif row in new_identity_reasons:
                    rejection_reason = new_identity_reasons[row]
                elif result.get("source") in {
                    "global-batch",
                    "same-cam-cache",
                    "local-track-verified",
                    "provisional-local-continuity",
                    "occlusion-hold",
                }:
                    rejection_reason = "not_cross_camera"
                else:
                    rejection_reason = "handoff_not_confirmed"

                topology_result = (
                    handoff.get("topology_result")
                    if committed
                    else pair.get("topology")
                )
                hard_gate_result = (
                    handoff.get("hard_gate_result")
                    if committed
                    else {
                        "passed": (
                            "gate_failure" not in pair if pair else None
                        ),
                        "reason": pair.get("gate_failure"),
                    }
                )
                decision = {
                    "row": row,
                    "gid": final_gid,
                    "candidate_gid": candidate_gid,
                    "from_camera": (
                        handoff.get("from_camera")
                        if committed
                        else (
                            topology_result.get("source_camera")
                            if isinstance(topology_result, dict)
                            else None
                        )
                    ),
                    "to_camera": cam_name,
                    "previous_presence": (
                        handoff.get("previous_presence") if committed else None
                    ),
                    "new_presence": (
                        result.get("presence") if result is not None else None
                    ),
                    "event_time": float(row_event_time),
                    "event_time_delta_sec": (
                        handoff.get("event_time_delta_sec")
                        if committed
                        else (
                            topology_result.get("event_time_delta_sec")
                            if isinstance(topology_result, dict)
                            else None
                        )
                    ),
                    "candidate_appearance": (
                        handoff.get("appearance_score")
                        if committed
                        else pair.get("appearance")
                    ),
                    "final_score": (
                        float(result["score"])
                        if result is not None
                        else (
                            float(pair["score"])
                            if pair and "score" in pair
                            else None
                        )
                    ),
                    "hard_gate_result": hard_gate_result,
                    "topology_result": topology_result,
                    "margin": top1_top2_margin_by_row.get(row),
                    "assignment_source": (
                        result["source"]
                        if result is not None
                        else "unresolved-cross-camera"
                    ),
                    "resolution_reason": resolution_reasons_by_row.get(row),
                    "handoff_committed": committed,
                    "handoff_rejection_reason": rejection_reason,
                }
                handoff_decisions.append(decision)
                handoff_decisions_by_row[row] = decision

            unresolved_handoff_forensic_events = []
            for row, record in sorted(unresolved_records_by_row.items()):
                cam_name, index, detection, row_event_time = rows[row]
                result = results[cam_name][index]
                previous_mapping = previous_local_mappings.get(row)
                previous_gid = (
                    int(previous_mapping["gid"])
                    if isinstance(previous_mapping, dict)
                    and previous_mapping.get("gid") is not None
                    else None
                )
                forensic_candidates = self._unresolved_candidate_diagnostics(
                    candidate_details_by_row.get(row, [])
                )
                if not forensic_candidates:
                    forensic_candidates = copy.deepcopy(
                        record.get("candidates", [])
                    )
                viable_candidate_gids = [
                    int(item["gid"])
                    for item in forensic_candidates
                    if (
                        item.get("gid") is not None
                        and item.get("hard_gate_passed")
                        and item.get("score") is not None
                    )
                ]
                forensic_candidate_gids = [
                    int(item["gid"])
                    for item in forensic_candidates
                    if item.get("gid") is not None
                ]
                expected_gid = previous_gid
                expected_gid_source = (
                    "previous_local_mapping"
                    if previous_gid is not None
                    else None
                )
                if expected_gid is None and len(viable_candidate_gids) == 1:
                    expected_gid = viable_candidate_gids[0]
                    expected_gid_source = "sole_viable_candidate"

                decision = handoff_decisions_by_row[row]
                row_rejection_reasons = [
                    item.get("reason")
                    for item in rejections
                    if item.get("row") == row and item.get("reason")
                ]
                sample_event = {
                    "event": "pending_sample",
                    "batch_id": resolved_batch_id,
                    "frame": detection.get(
                        "frame_index",
                        detection.get("sequence_index"),
                    ),
                    "camera": cam_name,
                    "local_id": int(detection["tid"]),
                    "event_time": float(row_event_time),
                    "expected_gid": expected_gid,
                    "expected_gid_source": expected_gid_source,
                    "previous_gid": previous_gid,
                    "candidate_gids": forensic_candidate_gids,
                    "viable_candidate_gids": viable_candidate_gids,
                    "candidates": forensic_candidates,
                    "unresolved_age_sec": self._unresolved_handoff_age(
                        record,
                        row_event_time,
                    ),
                    "sample_count": record.get("sample_count", 0),
                    "min_samples": AMBIGUOUS_HANDOFF_MIN_SAMPLES,
                    "min_solo_event_sec": (
                        AMBIGUOUS_HANDOFF_MIN_SOLO_EVENT_SEC
                    ),
                    "max_event_sec": AMBIGUOUS_HANDOFF_MAX_EVENT_SEC,
                    "pending_reason": record.get("pending_reason"),
                    "event_order_result": record.get("event_order_result"),
                    "quality_rejection_reason": record.get(
                        "last_quality_rejection_reason"
                    ),
                    "rejection_reason": (
                        row_rejection_reasons[-1]
                        if row_rejection_reasons
                        else decision.get("handoff_rejection_reason")
                    ),
                    "handoff_rejection_reason": decision.get(
                        "handoff_rejection_reason"
                    ),
                    "final_gid": (
                        int(result["gid"]) if result is not None else None
                    ),
                    "assignment_source": (
                        result.get("source")
                        if result is not None
                        else "unresolved-cross-camera"
                    ),
                }
                unresolved_handoff_forensic_events.append(sample_event)

                if result is not None and row not in new_identity_reasons:
                    unresolved_handoff_forensic_events.append({
                        **copy.deepcopy(sample_event),
                        "event": "pending_committed_existing_gid",
                        "committed_gid": int(result["gid"]),
                        "rejection_reason": None,
                    })
                elif (
                    result is not None
                    and new_identity_reasons.get(row)
                    == "ambiguous_handoff_window_exhausted"
                ):
                    unresolved_handoff_forensic_events.append({
                        **copy.deepcopy(sample_event),
                        "event": "pending_cancelled_new_gid",
                        "new_gid": int(result["gid"]),
                        "rejection_reason": (
                            "ambiguous_handoff_window_exhausted"
                        ),
                    })

            if unresolved_handoff_forensic_events:
                self.unresolved_handoff_forensic_trace.extend(
                    copy.deepcopy(unresolved_handoff_forensic_events)
                )
                if (
                    len(self.unresolved_handoff_forensic_trace)
                    > UNRESOLVED_HANDOFF_FORENSIC_MAX_EVENTS
                ):
                    del self.unresolved_handoff_forensic_trace[
                        :-UNRESOLVED_HANDOFF_FORENSIC_MAX_EVENTS
                    ]

            assignments = [
                {
                    "row": row,
                    "camera": cam_name,
                    "track_id": detection["tid"],
                    "gid": results[cam_name][index]["gid"],
                    "score": float(results[cam_name][index]["score"]),
                    "source": results[cam_name][index]["source"],
                    "identity_state": self.identities.get(
                        results[cam_name][index]["gid"],
                        {}
                    ).get("state", IDENTITY_PROVISIONAL),
                    "assignment_state": "committed",
                    "reason": results[cam_name][index].get(
                        "assignment_reason",
                        results[cam_name][index]["source"],
                    ),
                    "gallery_update_accepted": results[cam_name][index].get(
                        "gallery_update_accepted"
                    ),
                    "gallery_rejection_reason": results[cam_name][index].get(
                        "gallery_rejection_reason"
                    ),
                    "gallery_mature": results[cam_name][index].get(
                        "gallery_mature"
                    ),
                    "tracklet_sample_count": results[cam_name][index].get(
                        "tracklet_sample_count", 0
                    ),
                    "gallery_size": results[cam_name][index].get(
                        "gallery_size", 0
                    ),
                    "presence": results[cam_name][index].get("presence"),
                    "handoff_committed": handoff_decisions_by_row[row][
                        "handoff_committed"
                    ],
                    "handoff_rejection_reason": handoff_decisions_by_row[row][
                        "handoff_rejection_reason"
                    ],
                }
                for row, (cam_name, index, detection, _) in enumerate(rows)
                if results[cam_name][index] is not None
            ]

            for row, (cam_name, index, detection, _row_event_time) in enumerate(rows):
                if results[cam_name][index] is None:
                    self.state_machine.record_pending(
                        cam_name,
                        detection["tid"],
                        detection.get("coordinator_generation", detection.get("camera_generation")),
                        pending_reasons_by_row.get(row),
                    )
            self.last_global_batch_diagnostics = {
                "batch_id": resolved_batch_id,
                "event_time": batch_event_time,
                "window_start_event_time": (
                    min(event_times) if event_times else batch_event_time
                ),
                "window_end_event_time": (
                    max(event_times) if event_times else batch_event_time
                ),
                "assignment_window_sec": (
                    GLOBAL_ASSIGNMENT_WINDOW_SEC
                    if assignment_window_sec is None
                    else float(assignment_window_sec)
                ),
                "cameras": sorted(camera_detections),
                "observation_count": len(rows),
                "rows": [
                    {
                        "row": row,
                        "camera": cam_name,
                        "track_id": detection["tid"],
                        "event_time": row_event_time,
                        "sequence_index": detection.get(
                            "frame_index",
                            detection.get("sequence_index"),
                        ),
                        "previous_local_mapping": (
                            previous_local_mappings.get(row)
                        ),
                        "candidate_gids": [
                            item["gid"]
                            for item in candidate_details_by_row.get(row, [])
                            if (
                                item["hard_gate_passed"]
                                and item["score"] is not None
                            )
                        ],
                        "candidates": candidate_details_by_row.get(row, []),
                        "top1_top2_margin": top1_top2_margin_by_row.get(row),
                        "assignment_state": (
                            "committed"
                            if results[cam_name][index] is not None
                            else "pending"
                        ),
                        "batch_id": resolved_batch_id,
                        "generation": detection.get(
                            "coordinator_generation",
                            detection.get("camera_generation"),
                        ),
                        "final_gid": (
                            results[cam_name][index]["gid"]
                            if results[cam_name][index] is not None
                            else None
                        ),
                        "final_state": (
                            self.identities.get(
                                results[cam_name][index]["gid"],
                                {},
                            ).get("state", IDENTITY_PROVISIONAL)
                            if results[cam_name][index] is not None
                            else None
                        ),
                        "gallery_update_accepted": (
                            results[cam_name][index].get(
                                "gallery_update_accepted"
                            )
                            if results[cam_name][index] is not None
                            else None
                        ),
                        "gallery_rejection_reason": (
                            results[cam_name][index].get(
                                "gallery_rejection_reason"
                            )
                            if results[cam_name][index] is not None
                            else "pending_ambiguous_handoff"
                        ),
                        "gallery_mature": (
                            results[cam_name][index].get("gallery_mature")
                            if results[cam_name][index] is not None
                            else None
                        ),
                        "tracklet_sample_count": (
                            results[cam_name][index].get(
                                "tracklet_sample_count",
                                0,
                            )
                            if results[cam_name][index] is not None
                            else 0
                        ),
                        "gallery_size": (
                            results[cam_name][index].get("gallery_size", 0)
                            if results[cam_name][index] is not None
                            else 0
                        ),
                        "new_identity_reason": new_identity_reasons.get(row),
                        "pending_reason": pending_reasons_by_row.get(row),
                        "resolution_reason": resolution_reasons_by_row.get(row),
                        "temporal_consistency": (
                            temporal_consistency_decisions_by_row.get(row)
                        ),
                        "unresolved_handoff": (
                            self._unresolved_handoff_status_item(
                                unresolved_records_by_row[row],
                                row_event_time,
                            )
                            if row in unresolved_records_by_row
                            else None
                        ),
                        "handoff": handoff_decisions_by_row[row],
                    }
                    for row, (
                        cam_name,
                        index,
                        detection,
                        row_event_time,
                    ) in enumerate(rows)
                ],
                "candidate_gids": candidate_gids,
                "gate_failures": gate_failures,
                "topology_gate_decisions": [
                    {
                        "row": row,
                        "camera": rows[row][0],
                        "track_id": rows[row][2]["tid"],
                        "gid": gid,
                        "hard_gate_passed": "gate_failure" not in pair,
                        "hard_gate_reason": pair.get("gate_failure"),
                        **pair["topology"],
                    }
                    for (row, gid), pair in pair_cache.items()
                    if pair.get("topology") is not None
                ],
                "rejections": rejections,
                "resolved_ambiguities": resolved_ambiguities,
                "assignments": assignments,
                "handoff_decisions": handoff_decisions,
                "unresolved_handoffs": [
                    self._unresolved_handoff_status_item(
                        record,
                        batch_event_time,
                    )
                    for _, record in sorted(
                        self.unresolved_cross_camera_handoffs.items(),
                        key=lambda item: (
                            item[1].get("first_event_time", 0.0),
                            repr(item[0]),
                        ),
                    )
                ],
                "unresolved_handoff_forensic_events": copy.deepcopy(
                    unresolved_handoff_forensic_events
                ),
                "temporal_consistency_decisions": [
                    {
                        "row": row,
                        **copy.deepcopy(decision),
                    }
                    for row, decision in sorted(
                        temporal_consistency_decisions_by_row.items()
                    )
                ],
                "selected": [
                    {
                        "camera": rows[row][0],
                        "track_id": rows[row][2]["tid"],
                        "gid": gid,
                        "score": float(pair["score"]),
                    }
                    for row, gid, pair in selected
                ],
                "state_machine": self.state_machine.snapshot(),
            }

            # Transition-recovery is a fixed identity claim and therefore does
            # not enter the Hungarian candidate matrix. Preserve the existing
            # forensic contract by exposing that recovered GID as the sole
            # validated candidate in the row diagnostics.
            for row, (cam_name, index, _detection, _row_event_time) in enumerate(rows):
                trace = self.last_global_batch_diagnostics["rows"][row]
                result = results[cam_name][index]
                if (
                    not trace["candidate_gids"]
                    and isinstance(result, dict)
                    and result.get("source") == "local-transition-recovery-v2"
                ):
                    recovered_gid = int(result["gid"])
                    recovered_score = float(result.get("score", 1.0))
                    trace["candidate_gids"] = [recovered_gid]
                    trace["candidates"] = [{
                        "gid": recovered_gid,
                        "hard_gate_passed": True,
                        "hard_gate_reason": None,
                        "score_failure_reason": None,
                        "appearance": 1.0,
                        "score": recovered_score,
                        "motion": None,
                        "topology": None,
                    }]

            # --------------------------------------------------------
            # GID CHANGE DIAGNOSTICS (V1 behavior is unchanged)
            # --------------------------------------------------------
            # Record only cases where a confirmed local track had a previous
            # Global ID and the final batch assignment changed that GID.
            # This is diagnostic-only: it never changes scoring or assignment.
            gid_change_diagnostics = []
            for row, (cam_name, index, detection, row_event_time) in enumerate(rows):
                previous_mapping = previous_local_mappings.get(row)
                previous_gid = (
                    previous_mapping.get("gid")
                    if isinstance(previous_mapping, dict)
                    else None
                )
                final_result = results[cam_name][index]
                final_gid = (
                    final_result.get("gid")
                    if isinstance(final_result, dict)
                    else None
                )

                if (
                    previous_gid is None
                    or final_gid is None
                    or int(previous_gid) == int(final_gid)
                ):
                    continue

                candidates = candidate_details_by_row.get(row, [])
                scored_candidates = sorted(
                    [
                        item for item in candidates
                        if item.get("hard_gate_passed")
                        and item.get("score") is not None
                    ],
                    key=lambda item: float(item["score"]),
                    reverse=True,
                )
                best = scored_candidates[0] if scored_candidates else None
                second = scored_candidates[1] if len(scored_candidates) > 1 else None
                previous_candidate = next(
                    (
                        item for item in candidates
                        if item.get("gid") == previous_gid
                    ),
                    None,
                )
                final_candidate = next(
                    (
                        item for item in candidates
                        if item.get("gid") == final_gid
                    ),
                    None,
                )

                diagnostic = {
                    "camera": cam_name,
                    "track_id": int(detection["tid"]),
                    "frame_index": detection.get("frame_index"),
                    "event_time": float(row_event_time),
                    "previous_gid": int(previous_gid),
                    "final_gid": int(final_gid),
                    "assignment_source": final_result.get("source"),
                    "assignment_reason": final_result.get(
                        "assignment_reason", final_result.get("source")
                    ),
                    "previous_score": (
                        previous_candidate.get("score")
                        if isinstance(previous_candidate, dict)
                        else None
                    ),
                    "previous_appearance": (
                        previous_candidate.get("appearance")
                        if isinstance(previous_candidate, dict)
                        else None
                    ),
                    "previous_hard_gate_passed": (
                        previous_candidate.get("hard_gate_passed")
                        if isinstance(previous_candidate, dict)
                        else None
                    ),
                    "previous_hard_gate_reason": (
                        previous_candidate.get("hard_gate_reason")
                        if isinstance(previous_candidate, dict)
                        else None
                    ),
                    "final_candidate_score": (
                        final_candidate.get("score")
                        if isinstance(final_candidate, dict)
                        else None
                    ),
                    "final_candidate_appearance": (
                        final_candidate.get("appearance")
                        if isinstance(final_candidate, dict)
                        else None
                    ),
                    "best_gid": best.get("gid") if best else None,
                    "best_score": best.get("score") if best else None,
                    "best_appearance": best.get("appearance") if best else None,
                    "second_gid": second.get("gid") if second else None,
                    "second_score": second.get("score") if second else None,
                    "second_appearance": second.get("appearance") if second else None,
                    "top1_top2_margin": top1_top2_margin_by_row.get(row),
                    "new_identity_reason": new_identity_reasons.get(row),
                    "pending_reason": pending_reasons_by_row.get(row),
                    "resolution_reason": resolution_reasons_by_row.get(row),
                }
                gid_change_diagnostics.append(diagnostic)

                logger.warning(
                    "[REID][GID-CHANGE] CAM=%s frame=%s LID=%s G%s->G%s "
                    "source=%s prev_score=%s prev_app=%s best=G%s score=%s app=%s "
                    "second=G%s score=%s margin=%s gate=%s",
                    cam_name,
                    detection.get("frame_index"),
                    detection["tid"],
                    previous_gid,
                    final_gid,
                    final_result.get("source"),
                    diagnostic["previous_score"],
                    diagnostic["previous_appearance"],
                    diagnostic["best_gid"],
                    diagnostic["best_score"],
                    diagnostic["best_appearance"],
                    diagnostic["second_gid"],
                    diagnostic["second_score"],
                    diagnostic["top1_top2_margin"],
                    diagnostic["previous_hard_gate_reason"],
                )

            self.last_global_batch_diagnostics[
                "gid_change_diagnostics"
            ] = gid_change_diagnostics

            # --------------------------------------------------------
            # NEW LOCAL TRACK -> GLOBAL RE-ID DIAGNOSTICS
            # --------------------------------------------------------
            # Diagnostic-only. Record how a newly-seen confirmed BoT-SORT
            # local track is resolved by the existing V1 Global ID logic.
            # No thresholds, scores, assignments, cache, or gallery behavior
            # are changed by this block.
            new_local_track_diagnostics = []
            for row, (cam_name, index, detection, row_event_time) in enumerate(rows):
                if previous_local_mappings.get(row) is not None:
                    continue
                if not detection.get("local_track_confirmed", True):
                    continue

                final_result = results[cam_name][index]
                final_gid = (
                    final_result.get("gid")
                    if isinstance(final_result, dict)
                    else None
                )
                if final_gid is None:
                    continue

                candidates = candidate_details_by_row.get(row, [])
                scored_candidates = sorted(
                    [
                        item for item in candidates
                        if item.get("hard_gate_passed")
                        and item.get("score") is not None
                    ],
                    key=lambda item: float(item["score"]),
                    reverse=True,
                )
                best = scored_candidates[0] if scored_candidates else None
                second = (
                    scored_candidates[1]
                    if len(scored_candidates) > 1
                    else None
                )
                final_candidate = next(
                    (
                        item for item in candidates
                        if item.get("gid") == final_gid
                    ),
                    None,
                )

                diagnostic = {
                    "camera": cam_name,
                    "track_id": int(detection["tid"]),
                    "frame_index": detection.get("frame_index"),
                    "event_time": float(row_event_time),
                    "final_gid": int(final_gid),
                    "assignment_source": final_result.get("source"),
                    "assignment_reason": final_result.get(
                        "assignment_reason", final_result.get("source")
                    ),
                    "final_candidate_score": (
                        final_candidate.get("score")
                        if isinstance(final_candidate, dict)
                        else None
                    ),
                    "final_candidate_appearance": (
                        final_candidate.get("appearance")
                        if isinstance(final_candidate, dict)
                        else None
                    ),
                    "best_gid": best.get("gid") if best else None,
                    "best_score": best.get("score") if best else None,
                    "best_appearance": best.get("appearance") if best else None,
                    "second_gid": second.get("gid") if second else None,
                    "second_score": second.get("score") if second else None,
                    "second_appearance": (
                        second.get("appearance") if second else None
                    ),
                    "top1_top2_margin": top1_top2_margin_by_row.get(row),
                    "new_identity_reason": new_identity_reasons.get(row),
                    "pending_reason": pending_reasons_by_row.get(row),
                    "resolution_reason": resolution_reasons_by_row.get(row),
                }
                new_local_track_diagnostics.append(diagnostic)

                logger.info(
                    "[REID][NEW-LOCAL] CAM=%s frame=%s LID=%s -> G%s "
                    "source=%s best=G%s score=%s app=%s second=G%s "
                    "score=%s margin=%s reason=%s",
                    cam_name,
                    detection.get("frame_index"),
                    detection["tid"],
                    final_gid,
                    final_result.get("source"),
                    diagnostic["best_gid"],
                    diagnostic["best_score"],
                    diagnostic["best_appearance"],
                    diagnostic["second_gid"],
                    diagnostic["second_score"],
                    diagnostic["top1_top2_margin"],
                    diagnostic["assignment_reason"],
                )

            self.last_global_batch_diagnostics[
                "new_local_track_diagnostics"
            ] = new_local_track_diagnostics

            logger.info(
                "[REID][GLOBAL] batch=%s cameras=%s observations=%d assignments=%s rejections=%d",
                resolved_batch_id,
                ",".join(sorted(camera_detections)) or "none",
                len(rows),
                ",".join(
                    f"{item['camera']}:{item['track_id']}->G{item['gid']}({item['source']})"
                    for item in assignments
                ) or "none",
                len(gate_failures) + len(rejections),
            )

        return results


    # ========================================================
    # RESOLVE ID
    # ========================================================

    def resolve_identity(
        self,
        cam_name,
        local_id,
        emb,
        map_pos=None,
        box_wh=None,
        forbidden_gids=None,
        forced_gid=None,
        allow_new=True
    ):

        _sync_identity_dependencies()

        det = {

            "tid":
                local_id,

            "emb":
                emb,

            "map_pos":
                map_pos,

            "box_wh":
                box_wh,

            "box":
                (
                    0,
                    0,
                    box_wh[0]
                    if box_wh
                    else 1,
                    box_wh[1]
                    if box_wh
                    else 1
                ),

            "overlap":
                forced_gid is not None,

            "forced_gid":
                forced_gid

        }

        result = self.assign_batch(
            cam_name,
            [det],
            prev_assignments=[]
        )[0]

        return (
            result["gid"],
            result["score"],
            result["source"]
        )


# ============================================================
# DOWNSTREAM GLOBAL ASSIGNMENT COORDINATOR
# ============================================================

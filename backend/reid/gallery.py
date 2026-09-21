"""Gallery/prototype helpers shared by Re-ID consumers.

The identity manager remains responsible for lifecycle and admission policy;
these functions only implement the existing vector operations.
"""

import numpy as np

from .similarity import l2_normalize


def normalize_embedding_candidate(value, expected_size=None):
    try:
        candidate = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        candidate.size == 0
        or not np.all(np.isfinite(candidate))
        or (
            expected_size is not None
            and candidate.size != int(expected_size)
        )
    ):
        return None
    norm = float(np.linalg.norm(candidate))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return candidate / norm


def normalize_gallery_embedding(value, expected_size=None):
    try:
        embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        return None
    if expected_size is not None and embedding.size != int(expected_size):
        return None
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return l2_normalize(embedding)


def identity_prototype_candidates(identity, expected_size, diversity_threshold=0.985):
    raw = list(identity.get("gallery", []) or [])
    if identity.get("embedding") is not None:
        raw.append(identity["embedding"])
    candidates = []
    for value in raw:
        candidate = normalize_gallery_embedding(value, expected_size)
        if candidate is None:
            continue
        if any(float(np.dot(candidate, old)) >= diversity_threshold for old in candidates):
            continue
        candidates.append(candidate)
    return candidates


def robust_identity_prototype(candidates, min_samples=2, min_consensus=0.70, max_samples=7):
    if not candidates:
        return None, 0.0, []
    if len(candidates) == 1:
        return candidates[0], 1.0, [candidates[0]]
    matrix = np.stack(candidates, axis=0)
    pairwise = np.matmul(matrix, matrix.T)
    medoid_index = int(np.argmax(np.median(pairwise, axis=1)))
    medoid = matrix[medoid_index]
    ordered = np.argsort(-np.matmul(matrix, medoid))
    selected = []
    for index in ordered:
        score = float(np.matmul(matrix[int(index)], medoid))
        if len(selected) >= min_samples and score < min_consensus:
            continue
        selected.append(matrix[int(index)])
        if len(selected) >= max_samples:
            break
    if not selected:
        selected = [medoid]
    prototype = l2_normalize(np.mean(np.stack(selected, axis=0), axis=0))
    consensus = float(np.median([np.dot(prototype, item) for item in selected]))
    return prototype, consensus, selected


def gallery_similarity(
    emb,
    identity,
    diversity_threshold=0.985,
    prototype_enabled=True,
    prototype_min_samples=2,
    prototype_min_consensus=0.0,
    prototype_weight=0.5,
    support_weight=0.5,
):
    query = normalize_embedding_candidate(emb)
    if query is None:
        return -1.0
    candidates = []
    raw = list(identity.get("gallery", []) or [])
    if identity.get("embedding") is not None:
        raw.append(identity["embedding"])
    for value in raw:
        candidate = normalize_embedding_candidate(value, query.size)
        if candidate is None:
            continue
        if any(float(np.dot(candidate, old)) >= diversity_threshold for old in candidates):
            continue
        candidates.append(candidate)
    if not candidates:
        return -1.0
    raw_scores = sorted(
        max(-1.0, min(1.0, float(np.dot(query, candidate))))
        for candidate in candidates
    )
    support_score = float(np.median(raw_scores[:min(3, len(raw_scores))]))
    if not prototype_enabled or len(candidates) < prototype_min_samples:
        return support_score
    prototype, consensus, _ = robust_identity_prototype(
        candidates,
        min_samples=prototype_min_samples,
        min_consensus=prototype_min_consensus,
    )
    if prototype is None:
        return support_score
    prototype_score = max(-1.0, min(1.0, float(np.dot(query, prototype))))
    if consensus < prototype_min_consensus:
        return min(prototype_score, support_score)
    return float(max(
        -1.0,
        min(1.0, prototype_weight * prototype_score + support_weight * support_score),
    ))

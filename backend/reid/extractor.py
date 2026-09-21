"""OSNet extractor implementation for the extracted Re-ID package."""

import logging
import os
import warnings
from contextlib import nullcontext

import cv2
import numpy as np

from .config import (
    REID_DEVICE_CONFIG,
    REID_INPUT_H,
    REID_INPUT_W,
    REID_MODEL_NAME,
    REID_MODEL_PATH,
    REID_MIN_CROP_SIZE,
)
from .similarity import l2_normalize

try:
    from ..reid_config import (
        OSNET_EMBEDDING_DIMENSION,
        OSNET_PIXEL_MEAN,
        OSNET_PIXEL_STD,
        osnet_preprocessing_metadata,
        read_osnet_checkpoint_metadata,
        validate_osnet_checkpoint_metadata,
    )
except ImportError:  # pragma: no cover
    from reid_config import (
        OSNET_EMBEDDING_DIMENSION,
        OSNET_PIXEL_MEAN,
        OSNET_PIXEL_STD,
        osnet_preprocessing_metadata,
        read_osnet_checkpoint_metadata,
        validate_osnet_checkpoint_metadata,
    )

logger = logging.getLogger(__name__)
TORCH_IMPORT_ERROR = None
TORCHREID_IMPORT_ERROR = None
FEATURE_EXTRACTOR_IMPORT_ERROR = None
try:
    import torch
except Exception as error:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = str(error)
try:
    import torchreid
except Exception as error:  # pragma: no cover
    torchreid = None
    TORCHREID_IMPORT_ERROR = str(error)

FeatureExtractor = None
if torchreid is not None:
    try:
        from torchreid.utils import FeatureExtractor
    except Exception as primary_error:  # pragma: no cover
        try:
            from torchreid.reid.utils import FeatureExtractor
        except Exception as fallback_error:  # pragma: no cover
            FEATURE_EXTRACTOR_IMPORT_ERROR = (
                f"primary={primary_error}; fallback={fallback_error}"
            )


def resolve_reid_device():
    if REID_DEVICE_CONFIG == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if REID_DEVICE_CONFIG.startswith("cuda") and (
        torch is None or not torch.cuda.is_available()
    ):
        raise RuntimeError("REID_DEVICE requests CUDA but CUDA is unavailable")
    if REID_DEVICE_CONFIG != "cpu" and not REID_DEVICE_CONFIG.startswith("cuda"):
        raise ValueError("REID_DEVICE must be auto, cpu, cuda, or cuda:<index>")
    return REID_DEVICE_CONFIG


def load_validated_osnet_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("OSNet checkpoint must contain a state dictionary")
    metadata = read_osnet_checkpoint_metadata(checkpoint)
    declared = checkpoint.get("architecture")
    if declared is not None and declared != REID_MODEL_NAME:
        raise RuntimeError(
            f"OSNet checkpoint architecture mismatch: expected={REID_MODEL_NAME}, checkpoint={declared}"
        )
    try:
        validate_osnet_checkpoint_metadata(
            metadata,
            expected_architecture=REID_MODEL_NAME,
            expected_embedding_dimension=OSNET_EMBEDDING_DIMENSION,
            expected_preprocessing=osnet_preprocessing_metadata(),
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    state = checkpoint
    for key in ("model_state_dict", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    normalized = {}
    excluded = []
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("classifier."):
            excluded.append(key)
            continue
        normalized[key] = value
    if not normalized:
        raise RuntimeError("OSNet checkpoint contains no model tensors")
    model_state = model.state_dict()
    expected = {key for key in model_state if not key.startswith("classifier.")}
    missing = sorted(expected - set(normalized))
    unexpected = sorted(set(normalized) - expected)
    mismatches = sorted(
        key for key in expected & set(normalized)
        if tuple(normalized[key].shape) != tuple(model_state[key].shape)
    )
    if missing or unexpected or mismatches:
        raise RuntimeError(
            f"OSNet checkpoint is incompatible with {REID_MODEL_NAME}; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={mismatches[:5]}"
        )
    model.load_state_dict(normalized, strict=False)
    model.eval()
    return len(normalized), metadata, sorted(excluded)


class OSNetFeatureExtractor:
    """Behavior-compatible OSNet x1.0 feature extractor."""

    def __init__(self):
        if torch is None:
            raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")
        if torchreid is None:
            raise RuntimeError(f"torchreid import failed: {TORCHREID_IMPORT_ERROR}")
        if FeatureExtractor is None:
            raise RuntimeError(
                f"FeatureExtractor import failed: {FEATURE_EXTRACTOR_IMPORT_ERROR}"
            )
        if not os.path.isfile(REID_MODEL_PATH):
            kind = "directory" if os.path.isdir(REID_MODEL_PATH) else "missing"
            raise FileNotFoundError(f"OSNet checkpoint is not a file ({kind}): {REID_MODEL_PATH}")
        self.name = os.path.basename(REID_MODEL_PATH)
        self.device = resolve_reid_device()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The pretrained weights .* cannot be loaded.*")
            version = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
            safe_globals = getattr(getattr(torch, "serialization", None), "safe_globals", None)
            context = safe_globals([version]) if safe_globals and version else nullcontext()
            with context:
                self.extractor = FeatureExtractor(
                    model_name=REID_MODEL_NAME,
                    model_path=REID_MODEL_PATH,
                    image_size=(REID_INPUT_H, REID_INPUT_W),
                    pixel_mean=list(OSNET_PIXEL_MEAN),
                    pixel_std=list(OSNET_PIXEL_STD),
                    device=self.device,
                    verbose=False,
                )
        self.loaded_tensor_count, self.checkpoint_metadata, self.excluded_classifier_keys = (
            load_validated_osnet_checkpoint(self.extractor.model, REID_MODEL_PATH)
        )
        if self.extractor.model.training:
            raise RuntimeError("OSNet feature extractor must be in eval mode")
        smoke = self.extract(np.full((REID_INPUT_H, REID_INPUT_W, 3), 127, dtype=np.uint8))
        if smoke is None or smoke.size == 0 or not np.all(np.isfinite(smoke)) or not np.isclose(np.linalg.norm(smoke), 1.0, atol=1e-5):
            raise RuntimeError("OSNet checkpoint loaded but embedding smoke test failed")
        self.embedding_dimension = int(smoke.size)
        if self.embedding_dimension != OSNET_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"OSNet embedding dimension differs from shared contract: expected={OSNET_EMBEDDING_DIMENSION}, inference={self.embedding_dimension}"
            )

    def extract(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return None
        height, width = person_crop.shape[:2]
        if height < REID_MIN_CROP_SIZE or width < REID_MIN_CROP_SIZE:
            return None
        try:
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            features = self.extractor([rgb])
            if features is None:
                return None
            feature = features[0].detach().cpu().numpy() if hasattr(features, "detach") else np.asarray(features[0])
            return l2_normalize(feature.reshape(-1).astype(np.float32))
        except Exception as error:  # pragma: no cover
            logger.warning("OSNet feature extraction failed: %s", error)
            return None

    def extract_batch(self, person_crops):
        if not person_crops:
            return []
        results = [None] * len(person_crops)
        valid, indices = [], []
        for index, crop in enumerate(person_crops):
            if crop is None or crop.size == 0:
                continue
            height, width = crop.shape[:2]
            if height < REID_MIN_CROP_SIZE or width < REID_MIN_CROP_SIZE:
                continue
            try:
                valid.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                indices.append(index)
            except Exception:
                continue
        if not valid:
            return results
        try:
            context = torch.inference_mode() if torch is not None else nullcontext()
            with context:
                features = self.extractor(valid)
            features = features.detach().cpu().numpy() if hasattr(features, "detach") else np.asarray(features)
            for feature_index, original_index in enumerate(indices):
                results[original_index] = l2_normalize(np.asarray(features[feature_index], dtype=np.float32).reshape(-1))
        except Exception as error:  # pragma: no cover
            logger.warning("OSNet batch feature extraction failed: %s", error)
        return results


_runtime_dependencies = None

def configure_extractor_dependencies(namespace):
    global _runtime_dependencies
    _runtime_dependencies = namespace
    globals().update(namespace)

def _sync_extractor_dependencies():
    if _runtime_dependencies is not None:
        globals().update(_runtime_dependencies)

def _publish_runtime_status():
    if _runtime_dependencies is not None:
        _runtime_dependencies["REID_RUNTIME_STATUS"] = REID_RUNTIME_STATUS

class LightweightAppearanceFeatureExtractor:

    def __init__(self):
        self.name = "lightweight"
        self.device = "cuda"
        self.embedding_dimension = (
            (2 * 12 * 4 * 4)
            +
            (16 * 32)
            +
            1
        )

    def _hsv_hist(
        self,
        img_bgr,
        h_bins=12,
        s_bins=4,
        v_bins=4
    ):

        hsv = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2HSV
        )

        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [h_bins, s_bins, v_bins],
            [0, 180, 0, 256, 0, 256]
        )

        hist = cv2.normalize(
            hist,
            hist
        ).flatten().astype(np.float32)

        return hist

    def _region_hist(self, img_bgr):

        h, w = img_bgr.shape[:2]

        upper = img_bgr[
            :max(1, int(h * 0.45)),
            :
        ]

        lower = img_bgr[
            max(0, int(h * 0.45)):,
            :
        ]

        upper_hist = self._hsv_hist(upper)

        lower_hist = self._hsv_hist(lower)

        return np.concatenate(
            [
                upper_hist,
                lower_hist
            ]
        ).astype(np.float32)

    def _shape_feature(self, img_bgr):

        h, w = img_bgr.shape[:2]

        aspect = np.array(
            [w / max(h, 1)],
            dtype=np.float32
        )

        gray = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (16, 32)
        )

        gray = gray.astype(
            np.float32
        ) / 255.0

        coarse = gray.flatten()

        return np.concatenate(
            [
                aspect,
                coarse
            ]
        ).astype(np.float32)

    def extract(self, person_crop):

        if person_crop is None:
            return None

        if person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]

        if (
            h < REID_MIN_CROP_SIZE
            or
            w < REID_MIN_CROP_SIZE
        ):
            return None

        crop = cv2.resize(
            person_crop,
            (64, 128)
        )

        hist_feat = self._region_hist(crop)

        shape_feat = self._shape_feature(crop)

        feat = np.concatenate(
            [
                hist_feat,
                shape_feat
            ]
        ).astype(np.float32)

        return l2_normalize(feat)


# ============================================================
# OSNET MARKET1501

        # เนเธเน FeatureExtractor เนเธ”เธขเธ•เธฃเธ
        # เนเธฅเธฐเธฃเธฐเธเธธ Market1501 weight
        # ----------------------------------------------------








# BUILD REID EXTRACTOR
# ============================================================

def build_feature_extractor():
    global REID_RUNTIME_STATUS

    _sync_extractor_dependencies()

    initialization_error = None

    if USE_OSNET:

        try:

            extractor = OSNetFeatureExtractor()

            REID_RUNTIME_STATUS = {
                "enabled": True,
                "model_architecture": REID_MODEL_NAME,
                "checkpoint_path": REID_MODEL_PATH,
                "checkpoint_name": os.path.basename(
                    REID_MODEL_PATH
                ),
                "checkpoint_loaded": True,
                "device": extractor.device,
                "fallback_active": False,
                "embedding_dimension": (
                    extractor.embedding_dimension
                ),
                "expected_embedding_dimension": (
                    OSNET_EMBEDDING_DIMENSION
                ),
                "preprocessing": (
                    osnet_preprocessing_metadata()
                ),
                "checkpoint_metadata": (
                    extractor.checkpoint_metadata
                ),
                "crop_mode": REID_CROP_MODE,
                "production_crop_source": "detector_tracker_bbox",
                "model_eval_mode": (
                    not extractor.extractor.model.training
                ),
                "excluded_classifier_keys": (
                    extractor.excluded_classifier_keys
                ),
                "offline_v5_validation_threshold_reference": (
                    V5_OFFLINE_VALIDATION_THRESHOLD_REFERENCE
                ),
                "threshold_safety_mode": (
                    REID_THRESHOLD_SAFETY_MODE
                ),
                "similarity_only_shortcut_enabled": (
                    REID_THRESHOLD_SAFETY_MODE
                    == "validated"
                ),
                "active_extractor": extractor.name,
                "error": None
            }

            logger.info(
                "[ReID] Runtime | enabled=%s | "
                "architecture=%s | checkpoint=%s | "
                "loaded=%s | device=%s | fallback=%s | "
                "embedding_dim=%s | crop_mode=%s | model_eval=%s | "
                "excluded_classifier_keys=%s | threshold_safety=%s",
                REID_RUNTIME_STATUS["enabled"],
                REID_RUNTIME_STATUS["model_architecture"],
                REID_RUNTIME_STATUS["checkpoint_path"],
                REID_RUNTIME_STATUS["checkpoint_loaded"],
                REID_RUNTIME_STATUS["device"],
                REID_RUNTIME_STATUS["fallback_active"],
                REID_RUNTIME_STATUS["embedding_dimension"],
                REID_RUNTIME_STATUS["crop_mode"],
                REID_RUNTIME_STATUS["model_eval_mode"],
                REID_RUNTIME_STATUS["excluded_classifier_keys"],
                REID_RUNTIME_STATUS["threshold_safety_mode"]
            )

            _publish_runtime_status()
            return extractor

        except Exception as e:
            initialization_error = (
                f"{type(e).__name__}: {e}"
            )

            logger.error(
                "[ReID] OSNet initialization failed: %s",
                initialization_error
            )

    extractor = (
        LightweightAppearanceFeatureExtractor()
    )

    REID_RUNTIME_STATUS = {
        "enabled": bool(USE_OSNET),
        "model_architecture": REID_MODEL_NAME,
        "checkpoint_path": REID_MODEL_PATH,
        "checkpoint_name": os.path.basename(
            REID_MODEL_PATH
        ),
        "checkpoint_loaded": False,
        "device": extractor.device,
        "fallback_active": True,
        "embedding_dimension": (
            extractor.embedding_dimension
        ),
        "expected_embedding_dimension": (
            OSNET_EMBEDDING_DIMENSION
        ),
        "preprocessing": (
            osnet_preprocessing_metadata()
        ),
        "checkpoint_metadata": None,
        "crop_mode": REID_CROP_MODE,
        "production_crop_source": "detector_tracker_bbox",
        "model_eval_mode": None,
        "excluded_classifier_keys": [],
        "offline_v5_validation_threshold_reference": (
            V5_OFFLINE_VALIDATION_THRESHOLD_REFERENCE
        ),
        "threshold_safety_mode": (
            REID_THRESHOLD_SAFETY_MODE
        ),
        "similarity_only_shortcut_enabled": (
            REID_THRESHOLD_SAFETY_MODE
            == "validated"
        ),
        "active_extractor": extractor.name,
        "error": initialization_error
    }

    logger.info(
        "[ReID] Runtime | enabled=%s | "
        "architecture=%s | checkpoint=%s | "
        "loaded=%s | device=%s | fallback=%s | "
        "embedding_dim=%s | crop_mode=%s | model_eval=%s | "
        "excluded_classifier_keys=%s | threshold_safety=%s | error=%s",
        REID_RUNTIME_STATUS["enabled"],
        REID_RUNTIME_STATUS["model_architecture"],
        REID_RUNTIME_STATUS["checkpoint_path"],
        REID_RUNTIME_STATUS["checkpoint_loaded"],
        REID_RUNTIME_STATUS["device"],
        REID_RUNTIME_STATUS["fallback_active"],
        REID_RUNTIME_STATUS["embedding_dimension"],
        REID_RUNTIME_STATUS["crop_mode"],
        REID_RUNTIME_STATUS["model_eval_mode"],
        REID_RUNTIME_STATUS["excluded_classifier_keys"],
        REID_RUNTIME_STATUS["threshold_safety_mode"],
        REID_RUNTIME_STATUS["error"]
    )

    _publish_runtime_status()
    return extractor


# ============================================================
# CAMERA PROCESSOR
# ============================================================

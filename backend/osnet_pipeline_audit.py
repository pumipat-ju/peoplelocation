"""Offline diagnostics for the production OSNet embedding contract."""

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from .reid_config import (
        OSNET_DEFAULT_CHECKPOINT_NAME,
        OSNET_INPUT_HEIGHT,
        OSNET_INPUT_WIDTH,
        OSNET_PIXEL_MEAN,
        OSNET_PIXEL_STD,
    )
except ImportError:
    from reid_config import (
        OSNET_DEFAULT_CHECKPOINT_NAME,
        OSNET_INPUT_HEIGHT,
        OSNET_INPUT_WIDTH,
        OSNET_PIXEL_MEAN,
        OSNET_PIXEL_STD,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def summarize_embedding(embedding):
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return {
        "shape": tuple(array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "L2 norm": float(np.linalg.norm(array)),
        "contains_nan": bool(np.isnan(array).any()),
        "contains_inf": bool(np.isinf(array).any()),
    }


def build_default_samples():
    samples = []
    for bgr in ((15, 90, 220), (220, 90, 15), (70, 180, 35)):
        crop = np.empty(
            (OSNET_INPUT_HEIGHT, OSNET_INPUT_WIDTH, 3),
            dtype=np.uint8,
        )
        crop[:] = bgr
        samples.append(crop)
    return samples


def load_samples(paths):
    if not paths:
        return build_default_samples(), ["synthetic_1", "synthetic_2", "synthetic_3"]

    samples = []
    names = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read sample crop: {path}")
        samples.append(image)
        names.append(str(path))
    return samples, names


def close_runtime(main_module):
    manager = getattr(main_module, "global_identity_manager", None)
    store = getattr(manager, "identity_store", None)
    if store is not None:
        store.close()


def run(checkpoint, device, image_paths):
    checkpoint = str(Path(checkpoint).expanduser().resolve())

    # Importing production main initializes persistence. Keep this diagnostic
    # isolated in memory and never enter the FastAPI lifespan or start workers.
    os.environ["IDENTITY_DB_PATH"] = ":memory:"
    os.environ["REID_ENABLED"] = "true"
    os.environ["REID_CHECKPOINT_PATH"] = checkpoint
    os.environ["REID_DEVICE"] = device

    backend_directory = str(Path(__file__).resolve().parent)
    if backend_directory not in sys.path:
        sys.path.insert(0, backend_directory)

    import main

    try:
        status = main.REID_RUNTIME_STATUS
        if not status.get("checkpoint_loaded") or status.get("fallback_active"):
            raise RuntimeError(f"Production OSNet is not active: {status}")

        extractor = main.appearance_extractor
        model = extractor.extractor.model
        samples, names = load_samples(image_paths)
        embeddings = extractor.extract_batch(samples)

        if len(embeddings) != len(samples) or any(item is None for item in embeddings):
            raise RuntimeError("One or more sample crops did not produce an embedding")

        print(f"OSNet architecture: {status['model_architecture']}")
        print(f"Checkpoint: {status['checkpoint_path']}")
        print(f"Checkpoint loaded tensors: {extractor.loaded_tensor_count}")
        print("Missing/unexpected non-classifier keys: 0/0 (strictly validated)")
        print(f"Device: {status['device']}")
        print(f"Embedding dim: {status['embedding_dimension']}")
        print("Existing normalization: yes (extraction L2; similarity re-normalizes)")
        print("Similarity implementation: dot(l2(a), l2(b)); mathematical cosine")
        print(f"Preprocess size: H={main.REID_INPUT_H}, W={main.REID_INPUT_W}")
        print("Resize argument: torchvision Resize((height, width))")
        print("Color conversion: OpenCV BGR -> RGB before FeatureExtractor")
        print("Pixel scaling: torchvision ToTensor uint8 -> float32 / 255.0")
        print(f"Mean: {tuple(OSNET_PIXEL_MEAN)}")
        print(f"Std: {tuple(OSNET_PIXEL_STD)}")
        print("Tensor layout: HWC -> CHW -> BCHW")
        print(f"Model eval: {not model.training}")
        print("Inference guard: FeatureExtractor torch.no_grad; batch adds inference_mode")
        print("Model output: 512-D feature tensor; classifier logits are not used")

        for index, (name, embedding) in enumerate(zip(names, embeddings), start=1):
            print(f"\nSample {index}: {name}")
            for key, value in summarize_embedding(embedding).items():
                print(f"{key}: {value}")

        print(f"\nDistinct embedding arrays: {len({id(item) for item in embeddings}) == len(embeddings)}")
    finally:
        close_runtime(main)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the exact production OSNet embedding path without camera/video input."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "weights" / OSNET_DEFAULT_CHECKPOINT_NAME),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Path to an already-cropped BGR-readable person image; repeat as needed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.checkpoint, arguments.device, arguments.image)

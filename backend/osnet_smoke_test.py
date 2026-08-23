import argparse
import asyncio
import json
import os
import sys
import tempfile

import cv2
import numpy as np

from reid_config import (
    OSNET_ARCHITECTURE,
    OSNET_DEFAULT_CHECKPOINT_NAME,
    build_osnet_checkpoint_metadata
)


def configure_synthetic_checkpoint():
    import torch
    import torchreid

    temporary_directory = (
        tempfile.TemporaryDirectory()
    )

    checkpoint_path = os.path.join(
        temporary_directory.name,
        OSNET_DEFAULT_CHECKPOINT_NAME
    )

    model = torchreid.models.build_model(
        name=OSNET_ARCHITECTURE,
        num_classes=1,
        loss="triplet",
        pretrained=False,
        use_gpu=False
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": build_osnet_checkpoint_metadata(
                training_dataset="synthetic_smoke",
                epoch=1,
                best_metric=0.0
            )
        },
        checkpoint_path
    )

    os.environ["REID_ENABLED"] = "true"
    os.environ["REID_CHECKPOINT_PATH"] = checkpoint_path
    os.environ["REID_DEVICE"] = "cpu"

    return temporary_directory


def run_smoke_test():
    import main

    status = dict(
        main.REID_RUNTIME_STATUS
    )

    if (
        not status.get("enabled")
        or not status.get("checkpoint_loaded")
        or status.get("fallback_active")
    ):
        raise RuntimeError(
            "Production OSNet is not active: "
            + json.dumps(
                status,
                ensure_ascii=False
            )
        )

    crop = np.zeros(
        (
            main.REID_INPUT_H,
            main.REID_INPUT_W,
            3
        ),
        dtype=np.uint8
    )
    crop[:, :, 0] = 15
    crop[:, :, 1] = 90
    crop[:, :, 2] = 220

    embedding = (
        main.appearance_extractor
        .extract(crop)
    )

    if (
        embedding is None
        or embedding.size == 0
        or not np.all(
            np.isfinite(embedding)
        )
        or np.linalg.norm(embedding) < 1e-8
    ):
        raise RuntimeError(
            "OSNet did not produce a valid embedding"
        )

    if (
        status.get("embedding_dimension")
        != int(embedding.size)
    ):
        raise RuntimeError(
            "OSNet embedding dimension differs from runtime status"
        )

    rgb_crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )
    direct_features = (
        main.appearance_extractor
        .extractor([rgb_crop])
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    direct_features = main.l2_normalize(
        direct_features
    )

    if not np.allclose(
        embedding,
        direct_features,
        rtol=1e-5,
        atol=1e-6
    ):
        raise RuntimeError(
            "Production BGR-to-RGB preprocessing differs from the shared contract"
        )

    status_response = asyncio.run(
        main.get_status()
    )
    api_status = json.loads(
        status_response.body
    )["reid"]

    if api_status != status:
        raise RuntimeError(
            "/api/status Re-ID diagnostics differ from runtime state"
        )

    print(
        json.dumps(
            {
                "success": True,
                "reid": status,
                "api_status_verified": True,
                "preprocessing_verified": True,
                "embedding_norm": float(
                    np.linalg.norm(embedding)
                )
            },
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Validate production OSNet checkpoint loading and embedding inference"
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Generate a temporary architecture-compatible checkpoint "
            "to test the production loader"
        )
    )
    arguments = parser.parse_args()
    temporary_directory = None

    try:
        if arguments.self_test:
            temporary_directory = (
                configure_synthetic_checkpoint()
            )

        run_smoke_test()
    except Exception as error:
        print(
            f"OSNet smoke test failed: {error}",
            file=sys.stderr
        )
        raise SystemExit(1)
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

OSNET_ARCHITECTURE = "osnet_x1_0"
OSNET_EMBEDDING_DIMENSION = 512

OSNET_INPUT_HEIGHT = 256
OSNET_INPUT_WIDTH = 128

OSNET_COLOR_SPACE = "RGB"
OSNET_PIXEL_MEAN = (
    0.485,
    0.456,
    0.406
)
OSNET_PIXEL_STD = (
    0.229,
    0.224,
    0.225
)

OSNET_PREPROCESSING_VERSION = "imagenet_rgb_v1"
OSNET_CHECKPOINT_FORMAT_VERSION = 1
OSNET_DEFAULT_CHECKPOINT_NAME = (
    f"{OSNET_ARCHITECTURE}_market1501.pth"
)


def osnet_preprocessing_metadata():
    return {
        "version": OSNET_PREPROCESSING_VERSION,
        "color_space": OSNET_COLOR_SPACE,
        "image_height": OSNET_INPUT_HEIGHT,
        "image_width": OSNET_INPUT_WIDTH,
        "pixel_mean": list(OSNET_PIXEL_MEAN),
        "pixel_std": list(OSNET_PIXEL_STD)
    }


def build_osnet_checkpoint_metadata(
    training_dataset,
    epoch,
    best_metric,
    training_dataset_version=None
):
    return {
        "format_version": (
            OSNET_CHECKPOINT_FORMAT_VERSION
        ),
        "architecture": OSNET_ARCHITECTURE,
        "embedding_dimension": (
            OSNET_EMBEDDING_DIMENSION
        ),
        "training_dataset": training_dataset,
        "training_dataset_version": (
            training_dataset_version
        ),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "preprocessing": (
            osnet_preprocessing_metadata()
        )
    }


def read_osnet_checkpoint_metadata(
    checkpoint
):
    metadata = checkpoint.get(
        "metadata",
        {}
    )

    metadata = (
        dict(metadata)
        if isinstance(metadata, dict)
        else {}
    )

    config = checkpoint.get(
        "config",
        {}
    )

    if isinstance(config, dict):
        metadata.setdefault(
            "architecture",
            config.get("model_name")
        )
        metadata.setdefault(
            "training_dataset",
            config.get("dataset")
        )

        if "preprocessing" not in metadata:
            height = config.get(
                "image_height"
            )
            width = config.get(
                "image_width"
            )

            if height is not None and width is not None:
                preprocessing = (
                    osnet_preprocessing_metadata()
                )
                preprocessing[
                    "image_height"
                ] = int(height)
                preprocessing[
                    "image_width"
                ] = int(width)
                metadata[
                    "preprocessing"
                ] = preprocessing

    metadata.setdefault(
        "epoch",
        checkpoint.get("epoch")
    )
    metadata.setdefault(
        "best_metric",
        checkpoint.get("best_score")
    )

    return metadata


def validate_osnet_checkpoint_metadata(
    metadata,
    expected_architecture=OSNET_ARCHITECTURE,
    expected_embedding_dimension=(
        OSNET_EMBEDDING_DIMENSION
    ),
    expected_preprocessing=None
):
    architecture = metadata.get(
        "architecture"
    )

    if (
        architecture is not None
        and architecture != expected_architecture
    ):
        raise ValueError(
            "OSNet checkpoint architecture mismatch: "
            f"expected={expected_architecture}, "
            f"checkpoint={architecture}"
        )

    embedding_dimension = metadata.get(
        "embedding_dimension"
    )

    if (
        embedding_dimension is not None
        and int(embedding_dimension)
        != int(expected_embedding_dimension)
    ):
        raise ValueError(
            "OSNet checkpoint embedding dimension mismatch: "
            f"expected={expected_embedding_dimension}, "
            f"checkpoint={embedding_dimension}"
        )

    checkpoint_preprocessing = metadata.get(
        "preprocessing"
    )

    if checkpoint_preprocessing is not None:
        expected = (
            expected_preprocessing
            or osnet_preprocessing_metadata()
        )

        for key in (
            "version",
            "color_space",
            "image_height",
            "image_width",
            "pixel_mean",
            "pixel_std"
        ):
            if (
                checkpoint_preprocessing.get(key)
                != expected.get(key)
            ):
                raise ValueError(
                    "OSNet checkpoint preprocessing mismatch: "
                    f"field={key}, expected={expected.get(key)}, "
                    "checkpoint="
                    f"{checkpoint_preprocessing.get(key)}"
                )

    return metadata

"""Offline manifest-driven OSNet x1.0 fine-tuning.

Only train/manifest.jsonl and val/manifest.jsonl are accepted. The held-out
test manifest is intentionally neither an argument nor opened by this module.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision import transforms
import torchvision
import torchreid


EXPERIMENT_VERSION = "osnet-peoplelocation-finetune-v1"
SEED = 20260911
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, required_split):
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != required_split:
                raise ValueError(
                    f"Unexpected split at {path}:{line_number}: "
                    f"{record.get('split')!r}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    sample_ids = [record["sample_id"] for record in records]
    duplicates = [key for key, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample IDs in {path}: {duplicates[:5]}")
    return records


def validate_sources(train_records, val_records):
    train_sources = Counter(record["dataset_source"] for record in train_records)
    val_sources = Counter(record["dataset_source"] for record in val_records)
    if set(train_sources) != {"peoplelocation", "market1501", "msmt17"}:
        raise ValueError(f"Unexpected training sources: {dict(train_sources)}")
    if set(val_sources) != {"peoplelocation"}:
        raise ValueError(f"Validation must be PeopleLocation-only: {dict(val_sources)}")
    train_ids = {record["identity_key"] for record in train_records}
    val_ids = {record["identity_key"] for record in val_records}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"Train/validation identity leakage: {overlap[:5]}")
    forbidden = []
    for record in train_records + val_records:
        serialized = json.dumps(record).replace("\\", "/")
        if any(token in serialized for token in (
            "list_val.txt", "list_query.txt", "list_gallery.txt",
            "/MSMT17_V1/test/", "bounding_box_test/",
        )):
            forbidden.append(record["sample_id"])
    if forbidden:
        raise ValueError(f"Forbidden external test/validation sources: {forbidden[:5]}")
    return {
        "train_source_images": dict(sorted(train_sources.items())),
        "validation_source_images": dict(sorted(val_sources.items())),
        "train_identities": len(train_ids),
        "validation_identities": len(val_ids),
        "identity_overlap": 0,
        "forbidden_source_records": 0,
    }


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def training_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Pad(10),
        transforms.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random"
        ),
    ])


def validation_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestReidDataset(Dataset):
    def __init__(self, records, project_root, transform, label_map=None):
        self.records = records
        self.project_root = project_root.resolve()
        self.transform = transform
        identities = sorted({record["identity_key"] for record in records})
        self.label_map = label_map or {
            identity: index for index, identity in enumerate(identities)
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = (self.project_root / record["image_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            bbox = record.get("bbox_xyxy")
            if bbox is not None:
                image = image.crop(tuple(int(value) for value in bbox))
            image = self.transform(image)
        identity = record["identity_key"]
        label = self.label_map.get(identity, -1)
        return {
            "image": image,
            "label": label,
            "dataset_source": record["dataset_source"],
            "identity_key": identity,
            "camera_id": record["camera_id"],
            "sample_id": record["sample_id"],
        }


class IdentityBalancedBatchSampler(BatchSampler):
    """One deterministic P x K group per identity per epoch."""

    def __init__(self, records, identities_per_batch, instances_per_identity, seed):
        self.identities_per_batch = identities_per_batch
        self.instances_per_identity = instances_per_identity
        self.seed = seed
        self.epoch = 0
        grouped = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record["identity_key"]].append(index)
        self.grouped = dict(grouped)
        self.identities = sorted(grouped)
        if identities_per_batch < 2 or instances_per_identity < 2:
            raise ValueError("P and K must both be at least two for batch-hard triplet")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (
            len(self.identities) + self.identities_per_batch - 1
        ) // self.identities_per_batch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        identities = list(self.identities)
        rng.shuffle(identities)
        remainder = len(identities) % self.identities_per_batch
        if remainder:
            needed = self.identities_per_batch - remainder
            final_ids = set(identities[-remainder:])
            candidates = [item for item in identities if item not in final_ids]
            identities.extend(rng.sample(candidates, needed))
        for offset in range(0, len(identities), self.identities_per_batch):
            batch = []
            for identity in identities[offset:offset + self.identities_per_batch]:
                indices = self.grouped[identity]
                if len(indices) >= self.instances_per_identity:
                    selected = rng.sample(indices, self.instances_per_identity)
                else:
                    selected = [rng.choice(indices) for _ in range(
                        self.instances_per_identity
                    )]
                batch.extend(selected)
            yield batch


def batch_hard_triplet_loss(features, labels, margin=0.3):
    distances = torch.cdist(features, features, p=2)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    if not bool(same.any(dim=1).all()) or not bool(different.any(dim=1).all()):
        raise ValueError("Every anchor needs a positive and a negative in the batch")
    hardest_positive = distances.masked_fill(~same, float("-inf")).max(1).values
    hardest_negative = distances.masked_fill(~different, float("inf")).min(1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def load_pretrained_model(checkpoint_path, num_classes, device):
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=num_classes,
        loss="triplet", pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if not key.startswith("classifier."):
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
        raise ValueError(
            f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    nn.init.normal_(model.classifier.weight, std=0.001)
    nn.init.zeros_(model.classifier.bias)
    return model.to(device), {
        "loaded_backbone_keys": len(cleaned),
        "reinitialized_keys": sorted(missing),
        "unexpected_keys": list(unexpected),
    }


def compute_roc_eer(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if not positives or not negatives or not np.isfinite(scores).all():
        raise ValueError("ROC-AUC/EER requires finite positive and negative scores")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tpr = np.r_[0.0, tp[distinct_ends] / positives]
    fpr = np.r_[0.0, fp[distinct_ends] / negatives]
    thresholds = np.r_[np.inf, sorted_scores[distinct_ends]]
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    roc_auc = float(trapezoid(tpr, fpr))
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossing = np.flatnonzero(difference >= 0)
    if crossing.size and crossing[0] > 0:
        right = int(crossing[0])
        left = right - 1
        fraction = -difference[left] / (difference[right] - difference[left])
        eer = fpr[left] + fraction * (fpr[right] - fpr[left])
        threshold = thresholds[right]
    else:
        index = int(np.argmin(np.abs(difference)))
        eer = (fpr[index] + fnr[index]) / 2.0
        threshold = thresholds[index]
    return roc_auc, float(eer), float(threshold)


def validation_metrics(embeddings, identities, cameras):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    similarities = embeddings @ embeddings.T
    identities = np.asarray(identities)
    cameras = np.asarray(cameras)
    ranks = []
    average_precisions = []
    for query in range(len(identities)):
        gallery = np.flatnonzero(cameras != cameras[query])
        relevant_mask = identities[gallery] == identities[query]
        if not relevant_mask.any():
            continue
        order = np.lexsort((
            gallery,
            -similarities[query, gallery],
        ))
        ranked_relevant = relevant_mask[order]
        hit_positions = np.flatnonzero(ranked_relevant) + 1
        ranks.append(int(hit_positions[0]))
        average_precisions.append(float(np.mean(
            np.arange(1, len(hit_positions) + 1) / hit_positions
        )))
    cross_mask = np.triu(cameras[:, None] != cameras[None, :], k=1)
    pair_scores = similarities[cross_mask]
    pair_labels = (identities[:, None] == identities[None, :])[cross_mask]
    roc_auc, eer, eer_threshold = compute_roc_eer(pair_scores, pair_labels)
    same_scores = pair_scores[pair_labels]
    different_scores = pair_scores[~pair_labels]
    ranks = np.asarray(ranks)
    return {
        "rank_1": float(np.mean(ranks <= 1)),
        "rank_5": float(np.mean(ranks <= 5)),
        "mAP": float(np.mean(average_precisions)),
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "same_id_mean_similarity": float(same_scores.mean()),
        "different_id_mean_similarity": float(different_scores.mean()),
        "similarity_gap": float(same_scores.mean() - different_scores.mean()),
        "valid_queries": int(len(ranks)),
        "verification_pairs": int(len(pair_scores)),
        "same_id_pairs": int(pair_labels.sum()),
        "different_id_pairs": int((~pair_labels).sum()),
        "protocol": "all validation samples as queries; other-camera gallery",
    }


def evaluate(model, loader, device):
    model.eval()
    features = []
    identities = []
    cameras = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device, non_blocking=True))
            output = F.normalize(output, p=2, dim=1)
            features.append(output.cpu().numpy())
            identities.extend(batch["identity_key"])
            cameras.extend(batch["camera_id"])
    return validation_metrics(np.concatenate(features), identities, cameras)


def dry_run(model, train_loader, classification_loss, device):
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    if labels.min().item() < 0 or labels.max().item() >= model.classifier.out_features:
        raise ValueError("Dry-run labels are outside the classifier range")
    model.train()
    logits, features = model(images)
    ce = classification_loss(logits, labels)
    triplet = batch_hard_triplet_loss(features, labels)
    total = ce + triplet
    total.backward()
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        embeddings = model(images)
    if embeddings.shape[1] != 512:
        raise ValueError(f"Expected 512-D embeddings, got {tuple(embeddings.shape)}")
    if not all(torch.isfinite(value).all() for value in (
        logits, features, embeddings, ce, triplet, total
    )):
        raise ValueError("Dry-run produced NaN or Inf")
    return {
        "batch_shape": list(images.shape),
        "label_min": int(labels.min()),
        "label_max": int(labels.max()),
        "unique_labels": int(labels.unique().numel()),
        "logits_shape": list(logits.shape),
        "feature_shape": list(features.shape),
        "embedding_shape": list(embeddings.shape),
        "classification_loss": float(ce.detach().cpu()),
        "triplet_loss": float(triplet.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "finite": True,
        "passed": True,
    }


def checkpoint_payload(model, optimizer, scheduler, epoch, metrics, config):
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "architecture": "osnet_x1_0",
        "num_classes": model.classifier.out_features,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation_metrics": metrics,
        "label_map": config["label_map"],
        "config": config,
    }


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(output_root, history):
    (output_root / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "epoch", "learning_rate", "classification_loss", "triplet_loss",
        "total_loss", "rank_1", "rank_5", "mAP", "roc_auc", "eer",
        "same_id_mean_similarity", "different_id_mean_similarity",
        "similarity_gap", "epoch_seconds",
    ]
    with (output_root / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in history)


def render_report(config, history, best_epoch, runtime_seconds, smoke_test):
    best = history[best_epoch - 1]
    possible_overfitting = (
        best_epoch < len(history)
        and history[-1]["total_loss"] < best["total_loss"]
        and max(row["mAP"] for row in history[best_epoch:]) < best["mAP"]
    )
    overfitting_text = (
        "Possible overfitting/validation instability: training loss continued to "
        "decrease after the best epoch while validation mAP did not recover."
        if possible_overfitting else
        "No clear post-best training-loss/validation-mAP divergence was observed."
    )
    return "\n".join([
        "# OSNet x1.0 Fine-Tuning Report",
        "",
        f"- Device: `{config['environment']['selected_device']}`",
        f"- Torch: `{config['environment']['torch_version']}`",
        f"- Initial checkpoint: `{config['initial_checkpoint']['path']}`",
        f"- Training identities/images: `{config['dataset']['train_identities']}` / "
        f"`{config['dataset']['train_images']}`",
        f"- Validation identities/images: `{config['dataset']['validation_identities']}` / "
        f"`{config['dataset']['validation_images']}`",
        f"- Epochs completed: `{len(history)}`",
        f"- Best epoch: `{best_epoch}` selected by validation mAP, then Rank-1/AUC",
        f"- Best validation mAP: `{best['mAP']:.6f}`",
        f"- Rank-1 / Rank-5: `{best['rank_1']:.6f}` / `{best['rank_5']:.6f}`",
        f"- ROC-AUC / EER: `{best['roc_auc']:.6f}` / `{best['eer']:.6f}`",
        f"- Same/different mean similarity: "
        f"`{best['same_id_mean_similarity']:.6f}` / "
        f"`{best['different_id_mean_similarity']:.6f}`",
        f"- Similarity gap: `{best['similarity_gap']:.6f}`",
        f"- Runtime seconds: `{runtime_seconds:.3f}`",
        f"- Best checkpoint smoke test: `{'PASS' if smoke_test['passed'] else 'FAIL'}`",
        f"- Overfitting assessment: {overfitting_text}",
        "",
        "Validation used only PeopleLocation `val/manifest.jsonl`. The held-out test "
        "manifest was not opened or evaluated. No production integration was performed.",
        "",
    ])


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def require_cuda_device(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA_REQUIRED: CUDA is unavailable or the selected device is not CUDA; "
            "stopping before dry-run/training without CPU fallback"
        )


def run(args):
    project_root = args.project_root.resolve()
    train_manifest = args.train_manifest.resolve()
    val_manifest = args.val_manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    for path in (train_manifest, val_manifest, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch_version={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"selected_device={device}", flush=True)
    print(f"gpu_name={gpu_name}", flush=True)
    require_cuda_device(device)
    seed_everything(args.seed)

    train_records = read_manifest(train_manifest, "train")
    val_records = read_manifest(val_manifest, "val")
    source_audit = validate_sources(train_records, val_records)
    train_identities = sorted({record["identity_key"] for record in train_records})
    label_map = {identity: index for index, identity in enumerate(train_identities)}
    print(
        f"train_images={len(train_records)} train_identities={len(label_map)} "
        f"val_images={len(val_records)} "
        f"val_identities={len({r['identity_key'] for r in val_records})}",
        flush=True,
    )

    train_dataset = ManifestReidDataset(
        train_records, project_root, training_transform(), label_map
    )
    val_dataset = ManifestReidDataset(
        val_records, project_root, validation_transform()
    )
    sampler = IdentityBalancedBatchSampler(
        train_records, args.identities_per_batch,
        args.instances_per_identity, args.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.validation_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    model, load_audit = load_pretrained_model(
        checkpoint_path, len(label_map), device
    )
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    dry_run_result = dry_run(model, train_loader, classification_loss, device)
    print("dry_run=" + json.dumps(dry_run_result, sort_keys=True), flush=True)
    if args.dry_run_only:
        return {"dry_run": dry_run_result, "source_audit": source_audit}

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    environment = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torchreid_version": package_version("torchreid"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "gpu_name": gpu_name,
    }
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "architecture": "osnet_x1_0",
        "feature_dimension": 512,
        "num_classes": len(label_map),
        "label_map": label_map,
        "random_seed": args.seed,
        "deterministic_algorithms_warn_only": True,
        "dataset": {
            "train_manifest": train_manifest.relative_to(project_root).as_posix(),
            "train_manifest_sha256": sha256_file(train_manifest),
            "train_images": len(train_records),
            "train_identities": len(label_map),
            "train_source_images": source_audit["train_source_images"],
            "validation_manifest": val_manifest.relative_to(project_root).as_posix(),
            "validation_manifest_sha256": sha256_file(val_manifest),
            "validation_images": len(val_records),
            "validation_identities": source_audit["validation_identities"],
            "validation_source_images": source_audit["validation_source_images"],
            "held_out_test_manifest_used": False,
        },
        "initial_checkpoint": {
            "path": checkpoint_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(checkpoint_path),
            **load_audit,
        },
        "preprocessing": {
            "color_space": "RGB", "height": IMAGE_HEIGHT, "width": IMAGE_WIDTH,
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        },
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "random_crop_padding_pixels": 10,
            "random_erasing_probability": 0.5,
            "aggressive_identity_distortion": False,
        },
        "training": {
            "epochs": args.epochs,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "classification_loss": "cross_entropy_label_smoothing_0.1",
            "metric_loss": "batch_hard_triplet_margin_0.3",
            "identities_per_batch": args.identities_per_batch,
            "instances_per_identity": args.instances_per_identity,
            "batch_size": args.identities_per_batch * args.instances_per_identity,
            "batches_per_epoch": len(sampler),
            "sampling": "one deterministic P×K group per identity per epoch",
        },
        "validation": {
            "selection": "maximum mAP; tie-break by Rank-1 then ROC-AUC",
            "peoplelocation_only": True,
            "held_out_test_used": False,
        },
        "environment": environment,
        "dry_run": dry_run_result,
        "source_audit": source_audit,
    }
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_key = None
    best_epoch = 0
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        totals = {"classification": 0.0, "triplet": 0.0, "total": 0.0}
        samples = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits, features = model(images)
                ce = classification_loss(logits, labels)
                triplet = batch_hard_triplet_loss(features, labels)
                loss = ce + triplet
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = images.shape[0]
            samples += count
            totals["classification"] += float(ce.detach().cpu()) * count
            totals["triplet"] += float(triplet.detach().cpu()) * count
            totals["total"] += float(loss.detach().cpu()) * count
            if batch_index % 20 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_index}/"
                    f"{len(train_loader)} loss={totals['total']/samples:.6f}",
                    flush=True,
                )
        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "classification_loss": totals["classification"] / samples,
            "triplet_loss": totals["triplet"] / samples,
            "total_loss": totals["total"] / samples,
            **metrics,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        candidate_key = (metrics["mAP"], metrics["rank_1"], metrics["roc_auc"])
        payload = checkpoint_payload(
            model, optimizer, scheduler, epoch, metrics, config
        )
        save_checkpoint(output_root / "latest_checkpoint.pth", payload)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_epoch = epoch
            save_checkpoint(output_root / "best_checkpoint.pth", payload)
        write_history(output_root, history)
        print(
            f"epoch={epoch} mAP={metrics['mAP']:.6f} "
            f"rank1={metrics['rank_1']:.6f} rank5={metrics['rank_5']:.6f} "
            f"auc={metrics['roc_auc']:.6f} eer={metrics['eer']:.6f} "
            f"gap={metrics['similarity_gap']:.6f}",
            flush=True,
        )
        scheduler.step()

    runtime_seconds = time.perf_counter() - training_start
    latest = torch.load(
        output_root / "latest_checkpoint.pth", map_location="cpu", weights_only=False
    )
    save_checkpoint(output_root / "final_checkpoint.pth", latest)
    best_checkpoint = torch.load(
        output_root / "best_checkpoint.pth", map_location="cpu", weights_only=False
    )
    smoke_model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=len(label_map),
        loss="triplet", pretrained=False,
    ).to(device)
    smoke_model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    smoke_model.eval()
    sample = next(iter(val_loader))["image"][:1].to(device)
    with torch.inference_mode():
        smoke_embedding = smoke_model(sample)
    smoke_test = {
        "shape": list(smoke_embedding.shape),
        "finite": bool(torch.isfinite(smoke_embedding).all()),
        "passed": (
            smoke_embedding.shape == (1, 512)
            and bool(torch.isfinite(smoke_embedding).all())
        ),
    }
    if not smoke_test["passed"]:
        raise RuntimeError("Best checkpoint smoke test failed")
    summary = {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_metrics": history[best_epoch - 1],
        "training_runtime_seconds": runtime_seconds,
        "checkpoint_paths": {
            "latest": str(output_root / "latest_checkpoint.pth"),
            "best": str(output_root / "best_checkpoint.pth"),
            "final": str(output_root / "final_checkpoint.pth"),
        },
        "smoke_test": smoke_test,
        "overfitting_assessment": (
            f"Possible overfitting/validation instability after epoch {best_epoch}: "
            "training loss continued decreasing while validation mAP stayed below "
            "its best."
            if (
                best_epoch < len(history)
                and history[-1]["total_loss"]
                < history[best_epoch - 1]["total_loss"]
                and max(row["mAP"] for row in history[best_epoch:])
                < history[best_epoch - 1]["mAP"]
            ) else
            "No clear post-best training-loss/validation-mAP divergence was observed."
        ),
        "held_out_test_evaluated": False,
        "production_integrated": False,
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "FINETUNING_REPORT.md").write_text(
        render_report(config, history, best_epoch, runtime_seconds, smoke_test),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest", type=Path,
        default=Path("datasets/peoplelocation_reid_v1/train/manifest.jsonl"),
    )
    parser.add_argument(
        "--val-manifest", type=Path,
        default=Path("datasets/peoplelocation_reid_v1/val/manifest.jsonl"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("weights/osnet_x1_0_market1501.pth"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("backend/reid_experiments/finetune_osnet_v1"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--instances-per-identity", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

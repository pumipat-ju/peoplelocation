"""Offline domain-balanced OSNet x1.0 fine-tuning experiment."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader

from .finetune_osnet import (
    ManifestReidDataset,
    batch_hard_triplet_loss,
    dry_run,
    evaluate,
    load_pretrained_model,
    package_version,
    read_manifest,
    require_cuda_device,
    save_checkpoint,
    seed_everything,
    sha256_file,
    training_transform,
    validate_sources,
    validation_transform,
)


EXPERIMENT_VERSION = "osnet-peoplelocation-domain-balanced-v1"
SOURCE_SLOTS = {"peoplelocation": 6, "market1501": 6, "msmt17": 8}


class DomainBalancedBatchSampler(BatchSampler):
    """Sample fixed source-level identity quotas without duplicating files."""

    def __init__(self, records, source_slots, instances_per_identity,
                 batches_per_epoch, seed):
        self.records = records
        self.source_slots = dict(source_slots)
        self.instances_per_identity = instances_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        grouped = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(records):
            grouped[record["dataset_source"]][record["identity_key"]].append(index)
        self.grouped = {
            source: dict(identities) for source, identities in grouped.items()
        }
        if set(self.grouped) != set(self.source_slots):
            raise ValueError(
                f"Source mismatch: records={sorted(self.grouped)}, "
                f"quotas={sorted(self.source_slots)}"
            )
        if instances_per_identity < 2:
            raise ValueError("Triplet sampling needs at least two instances per identity")
        if any(value <= 0 for value in self.source_slots.values()):
            raise ValueError("Every source quota must be positive")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches_per_epoch

    def expected_source_counts(self):
        return {
            source: slots * self.instances_per_identity * self.batches_per_epoch
            for source, slots in self.source_slots.items()
        }

    @staticmethod
    def _draw_identity_slots(rng, identities, count, decks, positions, source):
        selected = []
        while len(selected) < count:
            if positions[source] >= len(decks[source]):
                decks[source] = list(identities)
                rng.shuffle(decks[source])
                positions[source] = 0
            take = min(count - len(selected), len(decks[source]) - positions[source])
            selected.extend(
                decks[source][positions[source]:positions[source] + take]
            )
            positions[source] += take
        return selected

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        identities_by_source = {
            source: sorted(groups) for source, groups in self.grouped.items()
        }
        decks = {source: list(ids) for source, ids in identities_by_source.items()}
        for deck in decks.values():
            rng.shuffle(deck)
        positions = {source: 0 for source in decks}
        for _ in range(self.batches_per_epoch):
            identity_slots = []
            for source in sorted(self.source_slots):
                chosen = self._draw_identity_slots(
                    rng, identities_by_source[source], self.source_slots[source],
                    decks, positions, source,
                )
                identity_slots.extend((source, identity) for identity in chosen)
            rng.shuffle(identity_slots)
            slot_counts = Counter(identity_slots)
            batch = []
            for (source, identity), occurrences in slot_counts.items():
                candidates = self.grouped[source][identity]
                needed = occurrences * self.instances_per_identity
                if len(candidates) >= needed:
                    selected = rng.sample(candidates, needed)
                else:
                    selected = list(candidates)
                    selected.extend(
                        rng.choice(candidates) for _ in range(needed - len(candidates))
                    )
                    rng.shuffle(selected)
                batch.extend(selected)
            rng.shuffle(batch)
            yield batch


def sampler_source_counts(records, batches):
    counts = Counter()
    for batch in batches:
        counts.update(records[index]["dataset_source"] for index in batch)
    return dict(sorted(counts.items()))


def source_ratios(counts):
    total = sum(counts.values())
    return {source: count / total for source, count in sorted(counts.items())}


def checkpoint_payload(model, optimizer, scheduler, epoch, metrics, config,
                       source_counts):
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "architecture": "osnet_x1_0",
        "num_classes": model.classifier.out_features,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation_metrics": metrics,
        "effective_source_sampling_counts": source_counts,
        "label_map": config["label_map"],
        "config": config,
    }


def write_history(output_root, history):
    (output_root / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "epoch", "backbone_learning_rate", "classifier_learning_rate",
        "classification_loss", "triplet_loss", "total_loss", "rank_1",
        "rank_5", "mAP", "roc_auc", "eer", "same_id_mean_similarity",
        "different_id_mean_similarity", "similarity_gap",
        "peoplelocation_samples", "market1501_samples", "msmt17_samples",
        "peoplelocation_ratio", "market1501_ratio", "msmt17_ratio",
        "mAP_improved", "epochs_without_map_improvement", "epoch_seconds",
    ]
    with (output_root / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in history)


def report_text(config, history, best_epoch, runtime, stopped_early):
    best = next(row for row in history if row["epoch"] == best_epoch)
    last = history[-1]
    return "\n".join([
        "# Domain-Balanced OSNet x1.0 Fine-Tuning", "",
        f"- Device: {config['environment']['selected_device']}",
        f"- GPU: {config['environment']['gpu_name']}",
        f"- Base checkpoint: {config['initial_checkpoint']['path']}",
        f"- Backbone LR: {config['training']['backbone_learning_rate']}",
        f"- Classifier LR: {config['training']['classifier_learning_rate']}",
        f"- Epochs completed: {len(history)} / {config['training']['max_epochs']}",
        f"- Early stopping triggered: {stopped_early}",
        f"- Best epoch: {best_epoch}",
        f"- Best mAP: {best['mAP']:.6f}",
        f"- Rank-1 / Rank-5: {best['rank_1']:.6f} / {best['rank_5']:.6f}",
        f"- ROC-AUC / EER: {best['roc_auc']:.6f} / {best['eer']:.6f}",
        f"- Same-ID / different-ID mean similarity: "
        f"{best['same_id_mean_similarity']:.6f} / "
        f"{best['different_id_mean_similarity']:.6f}",
        f"- Similarity gap: {best['similarity_gap']:.6f}",
        f"- Effective counts per epoch: PeopleLocation "
        f"{last['peoplelocation_samples']}, Market-1501 "
        f"{last['market1501_samples']}, MSMT17 {last['msmt17_samples']}",
        f"- Last-epoch source ratios: PeopleLocation "
        f"{last['peoplelocation_ratio']:.3f}, Market-1501 "
        f"{last['market1501_ratio']:.3f}, MSMT17 {last['msmt17_ratio']:.3f}",
        f"- Runtime seconds: {runtime:.3f}", "",
        "Best checkpoint selection used validation mAP, with Rank-1 and ROC-AUC "
        "only as exact-mAP tie-breakers. Early stopping used mAP improvement only.",
        "Validation used only PeopleLocation val/manifest.jsonl. The held-out test "
        "manifest was not opened or evaluated. No production integration occurred.", "",
    ])


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
    identities = sorted({record["identity_key"] for record in train_records})
    label_map = {identity: index for index, identity in enumerate(identities)}
    if len(label_map) != 1795:
        raise ValueError(f"Expected 1795 train identities, found {len(label_map)}")
    sampler = DomainBalancedBatchSampler(
        train_records, SOURCE_SLOTS, args.instances_per_identity,
        args.batches_per_epoch, args.seed,
    )
    dry_batches = list(iter(sampler))
    dry_source_counts = sampler_source_counts(train_records, dry_batches)
    if dry_source_counts != sampler.expected_source_counts():
        raise ValueError(
            f"Source balancing dry-run failed: {dry_source_counts}"
        )
    dry_source_ratios = source_ratios(dry_source_counts)
    expected_ratios = {"peoplelocation": 0.3, "market1501": 0.3, "msmt17": 0.4}
    if any(abs(dry_source_ratios[key] - value) > 1e-12
           for key, value in expected_ratios.items()):
        raise ValueError(f"Unexpected source ratios: {dry_source_ratios}")

    train_dataset = ManifestReidDataset(
        train_records, project_root, training_transform(), label_map
    )
    val_dataset = ManifestReidDataset(
        val_records, project_root, validation_transform()
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.validation_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    model, load_audit = load_pretrained_model(
        checkpoint_path, len(label_map), device
    )
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    dry_result = dry_run(model, train_loader, classification_loss, device)
    dry_result.update({
        "cuda_active": device.type == "cuda",
        "source_counts_per_epoch": dry_source_counts,
        "source_ratios_per_epoch": dry_source_ratios,
        "labels_valid": True,
        "manifests_loaded": True,
    })
    print("dry_run=" + json.dumps(dry_result, sort_keys=True), flush=True)
    if args.dry_run_only:
        return dry_result

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    model, load_audit = load_pretrained_model(
        checkpoint_path, len(label_map), device
    )
    classifier_parameters = list(model.classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in classifier_ids
    ]
    optimizer = torch.optim.Adam([
        {"params": backbone_parameters, "lr": args.backbone_learning_rate,
         "name": "backbone"},
        {"params": classifier_parameters, "lr": args.classifier_learning_rate,
         "name": "classifier"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    environment = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(), "torch_version": torch.__version__,
        "torchreid_version": package_version("torchreid"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device), "gpu_name": gpu_name,
    }
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "architecture": "osnet_x1_0", "feature_dimension": 512,
        "num_classes": len(label_map), "label_map": label_map,
        "random_seed": args.seed,
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
            "validation_peoplelocation_only": True,
            "held_out_test_manifest_used": False,
        },
        "initial_checkpoint": {
            "path": checkpoint_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "is_previous_finetuned_checkpoint": False,
            **load_audit,
        },
        "preprocessing": {
            "color_space": "RGB", "height": 256, "width": 128,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "random_crop_padding_pixels": 10,
            "random_erasing_probability": 0.5,
            "aggressive_identity_distortion": False,
        },
        "training": {
            "max_epochs": args.max_epochs, "early_stopping_patience": 2,
            "early_stopping_metric": "validation_mAP",
            "optimizer": "Adam",
            "backbone_learning_rate": args.backbone_learning_rate,
            "classifier_learning_rate": args.classifier_learning_rate,
            "previous_experiment_learning_rate": 0.0003,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "classification_loss": "cross_entropy_label_smoothing_0.1",
            "metric_loss": "batch_hard_triplet_margin_0.3",
            "instances_per_identity": args.instances_per_identity,
            "identity_slots_per_batch": SOURCE_SLOTS,
            "batch_size": sum(SOURCE_SLOTS.values()) * args.instances_per_identity,
            "batches_per_epoch": args.batches_per_epoch,
            "target_source_ratios": expected_ratios,
            "physical_file_duplication": False,
        },
        "selection": "maximum validation mAP; exact tie by Rank-1 then ROC-AUC",
        "dry_run": dry_result, "source_audit": source_audit,
        "environment": environment,
    }
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    history = []
    best_key = None
    best_map = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    training_start = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        totals = Counter()
        effective_sources = Counter()
        samples = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
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
            effective_sources.update(batch["dataset_source"])
            if batch_index % 20 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch}/{args.max_epochs} batch={batch_index}/"
                    f"{len(train_loader)} loss={totals['total']/samples:.6f}",
                    flush=True,
                )
        source_counts = dict(sorted(effective_sources.items()))
        if source_counts != sampler.expected_source_counts():
            raise ValueError(f"Effective sampling drift: {source_counts}")
        ratios = source_ratios(source_counts)
        metrics = evaluate(model, val_loader, device)
        map_improved = metrics["mAP"] > best_map + 1e-12
        if map_improved:
            best_map = metrics["mAP"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "classifier_learning_rate": optimizer.param_groups[1]["lr"],
            "classification_loss": totals["classification"] / samples,
            "triplet_loss": totals["triplet"] / samples,
            "total_loss": totals["total"] / samples,
            **metrics,
            "peoplelocation_samples": source_counts["peoplelocation"],
            "market1501_samples": source_counts["market1501"],
            "msmt17_samples": source_counts["msmt17"],
            "peoplelocation_ratio": ratios["peoplelocation"],
            "market1501_ratio": ratios["market1501"],
            "msmt17_ratio": ratios["msmt17"],
            "mAP_improved": map_improved,
            "epochs_without_map_improvement": epochs_without_improvement,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        candidate_key = (metrics["mAP"], metrics["rank_1"], metrics["roc_auc"])
        payload = checkpoint_payload(
            model, optimizer, scheduler, epoch, metrics, config, source_counts
        )
        save_checkpoint(output_root / "latest_checkpoint.pth", payload)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_epoch = epoch
            save_checkpoint(output_root / "best_checkpoint.pth", payload)
        write_history(output_root, history)
        print(
            f"epoch={epoch} ratios={ratios} mAP={metrics['mAP']:.6f} "
            f"rank1={metrics['rank_1']:.6f} auc={metrics['roc_auc']:.6f} "
            f"eer={metrics['eer']:.6f} gap={metrics['similarity_gap']:.6f}",
            flush=True,
        )
        scheduler.step()
        if epochs_without_improvement >= 2:
            stopped_early = True
            print(f"early_stopping_at_epoch={epoch}", flush=True)
            break

    runtime = time.perf_counter() - training_start
    summary = {
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_metrics": next(
            row for row in history if row["epoch"] == best_epoch
        ),
        "effective_source_sampling_counts_by_epoch": [{
            "epoch": row["epoch"],
            "peoplelocation": row["peoplelocation_samples"],
            "market1501": row["market1501_samples"],
            "msmt17": row["msmt17_samples"],
        } for row in history],
        "effective_source_sampling_ratios": source_ratios(
            sampler.expected_source_counts()
        ),
        "early_stopping_triggered": stopped_early,
        "training_runtime_seconds": runtime,
        "best_checkpoint": str(output_root / "best_checkpoint.pth"),
        "latest_checkpoint": str(output_root / "latest_checkpoint.pth"),
        "held_out_test_evaluated": False,
        "production_integrated": False,
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "FINETUNING_REPORT.md").write_text(
        report_text(config, history, best_epoch, runtime, stopped_early),
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
        default=Path("backend/reid_experiments/finetune_osnet_v2_balanced"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--instances-per-identity", type=int, default=2)
    parser.add_argument("--batches-per-epoch", type=int, default=113)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--classifier-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

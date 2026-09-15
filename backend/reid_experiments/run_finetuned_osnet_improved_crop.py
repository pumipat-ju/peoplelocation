"""Runner: Fine-tuned OSNet x1.0 with Raw-GT-bbox Improved Crop."""

from backend.reid_experiments.common import runner_main


if __name__ == "__main__":
    runner_main("04_finetuned_osnet_improved_crop")

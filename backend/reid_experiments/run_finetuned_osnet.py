"""Runner: Fine-tuned OSNet x1.0 with the original production crop."""

from backend.reid_experiments.common import runner_main


if __name__ == "__main__":
    runner_main("03_finetuned_osnet")

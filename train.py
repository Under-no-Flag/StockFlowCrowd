"""Train and evaluate the core Sim-SH-ENB configurations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from stockflowcrowd import Forecaster
from stockflowcrowd.model import RAW_KEYS

BACKBONES = ("megacrn", "stid", "staeformer")
VARIANTS = ("B", "S", "A", "SA")
GROUPS = {"D": slice(0, 5), "N": slice(5, 10), "C": slice(10, 34)}


def load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return dict(arrays)


def load_split(data_dir, split):
    arrays = load_npz(data_dir / "data" / f"{split}.npz")
    if split == "test":
        arrays.update(load_npz(data_dir / "test_labels.npz"))
    origins = len(arrays["history"])
    if arrays["history"].shape != (origins, 24, 34):
        raise ValueError(f"{split}: expected 24 history steps and 34 channels")
    if (
        arrays["target"].shape != (origins, 6, 34)
        or arrays["target_mask"].shape != arrays["target"].shape
    ):
        raise ValueError(f"{split}: expected six target horizons and 34 channels")
    return arrays


def history_batch(arrays, rows, device):
    return {
        name: torch.as_tensor(arrays[name][rows], device=device) for name in RAW_KEYS
    }


def training_loss(prediction, aux, target, mask, scales, support, population):
    valid_target = torch.where(mask, target, prediction.detach())
    cells = (((prediction - valid_target).abs() / scales) * mask).sum(0) / support
    loss = torch.stack([cells[:, group].mean() for group in GROUPS.values()]).mean()
    if "occupancy_prediction" in aux:
        n = GROUPS["N"]
        observation = aux["occupancy_prediction"]
        valid_occupancy = torch.where(
            mask[..., n], target[..., n], observation.detach()
        )
        errors = ((observation - valid_occupancy).abs() / scales[n]) * mask[..., n]
        auxiliary_cells = errors.sum(0) / support[:, n]
        loss = loss + 0.1 * auxiliary_cells.mean()
    return loss * (population / len(prediction)) + aux["memory_regularizer"]


@torch.no_grad()
def score(model, arrays, scales, device, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(arrays["history"]), batch_size):
        rows = slice(start, start + batch_size)
        predictions.append(model(history_batch(arrays, rows, device)).cpu().numpy())
    point = np.concatenate(predictions).astype(np.float64)
    mask = arrays["target_mask"]
    support = mask.sum(0)
    if (support == 0).any():
        raise ValueError("Every scored channel and horizon needs a valid label")
    if not np.isfinite(point).all() or not np.isfinite(arrays["target"][mask]).all():
        raise ValueError("Nonfinite prediction or valid label")
    error = np.where(mask, point - arrays["target"], 0.0)
    mae = np.abs(error).sum(0) / support
    rmse = np.sqrt(np.square(error).sum(0) / support)
    scaled = mae / scales
    metrics = {
        name: {
            "MAE": float(mae[:, group].mean()),
            "RMSE": float(rmse[:, group].mean()),
            "sMAE": float(scaled[:, group].mean()),
        }
        for name, group in GROUPS.items()
    }
    return {"J": sum(v["sMAE"] for v in metrics.values()) / len(GROUPS), **metrics}


def fit(args, stats, spatial_arrays, device):
    if args.checkpoint.exists():
        raise FileExistsError(f"Choose a new checkpoint path: {args.checkpoint}")

    train = load_split(args.data_dir, "train")
    validation = load_split(args.data_dir, "validation")
    model = Forecaster(
        args.backbone, args.variant, stats, spatial_arrays, seed=args.seed
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.RandomState(args.seed + 40000)
    population = len(train["history"])
    support = torch.as_tensor(train["target_mask"].sum(0), device=device)
    if (support == 0).any():
        raise ValueError("Every channel and horizon needs a valid training label")

    best = None
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(population)
        for start in range(0, population, args.batch_size):
            rows = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            point, aux = model(history_batch(train, rows, device), return_aux=True)
            target = torch.as_tensor(train["target"][rows], device=device)
            mask = torch.as_tensor(train["target_mask"][rows], device=device)
            loss = training_loss(
                point, aux, target, mask, model.scales, support, population
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()

        metrics = score(model, validation, stats["scales"], device, args.batch_size)
        print(f"Epoch {epoch:>3}  validation J={metrics['J']:.5f}", flush=True)
        if metrics["J"] < best_score:
            best_score = metrics["J"]
            best = {
                "backbone": args.backbone,
                "variant": args.variant,
                "seed": args.seed,
                "epoch": epoch,
                "validation": metrics,
                "model": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
            }

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("xb") as output:
        torch.save(best, output)
    print(f"Saved epoch {best['epoch']} to {args.checkpoint}")


def evaluate(args, stats, spatial_arrays, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Forecaster(
        checkpoint["backbone"], checkpoint["variant"], stats, spatial_arrays
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    test = load_split(args.data_dir, "test")
    print(
        json.dumps(
            score(model, test, stats["scales"], device, args.batch_size), indent=2
        )
    )


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Fit with validation-only checkpoint selection"
    )
    test = commands.add_parser(
        "evaluate", help="Score a saved checkpoint on the three test events"
    )
    for command in (train, test):
        command.add_argument(
            "--data-dir",
            type=Path,
            required=True,
            help="Extracted Sim-SH-ENB directory",
        )
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--batch-size", type=positive_int, default=64)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--backbone", choices=BACKBONES, default="megacrn")
    train.add_argument("--variant", choices=VARIANTS, default="SA")
    train.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    train.add_argument("--epochs", type=positive_int, default=80)
    train.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access a CUDA device")
    device = torch.device(args.device)

    stats = load_npz(args.data_dir / "data" / "stats.npz")
    spatial_arrays = json.loads((args.data_dir / "contract.json").read_text())["arrays"]
    if args.command == "train":
        fit(args, stats, spatial_arrays, device)
    else:
        evaluate(args, stats, spatial_arrays, device)


if __name__ == "__main__":
    main()

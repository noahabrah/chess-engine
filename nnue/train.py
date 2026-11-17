"""Train the NNUE network on the data from prepare_data.py.

Loads all data/train_*.npz into memory (or up to --max-positions), trains on the
Mac's GPU (MPS, otherwise CUDA or CPU) and validates on data/val.npz after every epoch.
Each run gets its own directory checkpoints/<timestamp>/ with
- best.pt (network with the lowest validation loss) and last.pt (network after the last epoch)
- log.txt (settings and all output of the run)

Usage:
    uv run --group train python -m nnue.train --epochs 20
    uv run --group train python -m nnue.train --max-positions 2000000 --epochs 3   # quick test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from engine.evaluate import evaluate_pesto

from .model import EVAL_SCALE, NNUE, loss_fn
from .prepare_data import decode_pieces


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_files(files: list[Path], max_positions: int | None) -> dict[str, np.ndarray]:
    """Load .npz files and concatenate them (at most max_positions positions)."""
    parts: dict[str, list[np.ndarray]] = {"pieces": [], "stm": [], "score": []}
    total = 0
    for path in files:
        if max_positions is not None and total >= max_positions:
            break
        with np.load(path) as data:
            for key in parts:  # noqa: PLC0206
                parts[key].append(data[key])
            total += len(data["score"])
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    if max_positions is not None:
        arrays = {key: value[:max_positions] for key, value in arrays.items()}
    return arrays


def to_device(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "pieces": torch.from_numpy(arrays["pieces"].astype(np.int32)).to(device),
        "stm": torch.from_numpy(arrays["stm"]).to(device),
        "score": torch.from_numpy(arrays["score"]).to(device),
    }


def pesto_baseline(val: dict[str, np.ndarray], count: int = 20_000) -> tuple[float, float]:
    """Loss of the handcrafted evaluation (PeSTO) on the same data, for comparison.

    PeSTO centipawns are on a slightly different scale than Stockfish's. To keep the
    comparison fair, PeSTO is scaled by the factor that fits best.
    """
    n = min(count, len(val["score"]))
    pesto = np.array([evaluate_pesto(decode_pieces(val["pieces"][i], val["stm"][i])) for i in range(n)])
    score = torch.tensor(val["score"][:n])
    best_factor = min(np.linspace(0.3, 2.0, 35), key=lambda k: loss_fn(torch.tensor(k * pesto / EVAL_SCALE), score))
    output = torch.tensor(best_factor * pesto / EVAL_SCALE)
    return loss_fn(output, score).item(), mean_abs_error_cp(output, score)


def mean_abs_error_cp(output: torch.Tensor, score: torch.Tensor) -> float:
    """Mean absolute error in centipawns, only for undecided positions (|score| < 1000)."""
    mask = score.abs() < 1000
    return (output[mask] * EVAL_SCALE - score[mask].float()).abs().mean().item()


@torch.no_grad()
def validate(model: NNUE, val: dict[str, torch.Tensor], batch_size: int) -> tuple[float, float]:
    model.eval()
    outputs = [
        model(val["pieces"][i:i + batch_size], val["stm"][i:i + batch_size])
        for i in range(0, len(val["score"]), batch_size)
    ]
    model.train()
    output = torch.cat(outputs)
    return loss_fn(output, val["score"]).item(), mean_abs_error_cp(output, val["score"])


class Logger:
    """Prints lines and appends them to a file at the same time."""

    def __init__(self, path: Path) -> None:
        self.file = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()       # write immediately: after an abort everything up to here is in the file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="directory with train_*.npz and val.npz")
    parser.add_argument("--max-positions", type=int, default=None, help="use only this many training positions")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out", default=None, help="checkpoint directory (default: checkpoints/<time>)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    data_dir = Path(args.data)
    out_dir = Path(args.out or f"checkpoints/{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    log(f"Start {time.strftime('%Y-%m-%d %H:%M:%S')}, settings: {vars(args)}")

    train_files = sorted(data_dir.glob("train_*.npz"))
    train_np = load_files(train_files, args.max_positions)
    val_np = load_files([data_dir / "val.npz"], None)
    n_train = len(train_np["score"])
    log(f"Device: {device}, training: {n_train:,} positions, validation: {len(val_np['score']):,}")

    pesto_loss, pesto_mae = pesto_baseline(val_np)
    zero_loss = loss_fn(torch.zeros(len(val_np["score"])), torch.tensor(val_np["score"])).item()
    log(f"Baselines on val: always 0 -> loss {zero_loss:.5f} | "
        f"PeSTO -> loss {pesto_loss:.5f}, error {pesto_mae:.0f} cp")

    # The whole data set fits into memory: index it on the GPU instead of using a DataLoader
    train = to_device(train_np, device)
    val = to_device(val_np, device)
    del train_np, val_np

    model = NNUE(args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = n_train // args.batch_size
    # The learning rate decreases smoothly to almost 0 over the whole run (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        permutation = torch.randperm(n_train, device=device)
        total = torch.zeros((), device=device)     # summed on the GPU: no sync after every step
        for step in range(steps_per_epoch):
            index = permutation[step * args.batch_size:(step + 1) * args.batch_size]
            loss = loss_fn(model(train["pieces"][index], train["stm"][index]), train["score"][index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.detach()

        train_loss = (total / steps_per_epoch).item()
        val_loss, val_mae = validate(model, val, args.batch_size)
        checkpoint = {"model": model.state_dict(), "hidden": args.hidden, "epoch": epoch, "val_loss": val_loss}
        torch.save(checkpoint, out_dir / "last.pt")
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")
            marker = "  *"
        log(f"Epoch {epoch:2d}: train {train_loss:.5f}  val {val_loss:.5f}  "
            f"error {val_mae:.0f} cp  ({time.perf_counter() - start:.0f} s){marker}")

    log(f"\nBest val loss {best_val:.5f} (PeSTO {pesto_loss:.5f}) -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

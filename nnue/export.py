"""Export a trained network for the engine (as plain NumPy arrays, no PyTorch needed to play).

Usage:
    uv run --group train python -m nnue.export checkpoints/<time>/best.pt
    -> engine/nnue.npz
"""

import argparse

import numpy as np
import torch

from .model import EVAL_SCALE, NNUE


def export(checkpoint_path: str, out_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = NNUE(checkpoint["hidden"])
    model.load_state_dict(checkpoint["model"])

    np.savez(
        out_path,
        # Row f = contribution of feature f to the accumulator -> the engine sums the rows of all pieces
        ft_weight=model.ft.weight.detach().numpy().T.astype(np.float32),   # [768, hidden]
        ft_bias=model.ft.bias.detach().numpy().astype(np.float32),          # [hidden]
        out_weight=model.out.weight.detach().numpy()[0].astype(np.float32),  # [2 * hidden]
        out_bias=np.float32(model.out.bias.item()),
        eval_scale=np.float32(EVAL_SCALE),
    )
    print(f"{checkpoint_path} (epoch {checkpoint['epoch']}, val {checkpoint['val_loss']:.5f}) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="engine/nnue.npz")
    args = parser.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()

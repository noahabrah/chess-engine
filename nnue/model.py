"""The NNUE network: 768 -> 256 x 2 -> 1.

Input (see prepare_data.py): the piece list of a position, one entry per piece:
    piece * 64 + square   (piece 0-5 = white pawn…king, 6-11 = black)
These are the indices of the ones in a 768-wide vector (12 piece planes x 64 squares).

Perspectives: every position is fed in twice,
- once from White's point of view (as stored),
- once from Black's point of view: colors swapped and board mirrored (square ^ 56).
Both go through the same first layer (feature transformer, "accumulator").
The side to move's half comes first, then the opponent's. So the network always
evaluates "from my point of view" and does not have to learn White and Black separately.

Output: one number; times EVAL_SCALE = evaluation in centipawns for the side to move.
"""

import torch
from torch import nn

NUM_FEATURES = 768
PADDING = NUM_FEATURES      # stand-in for empty slots (-1) in the piece list
EVAL_SCALE = 400            # network output 1.0 = 400 cp; sigmoid(cp / 400) ≈ expected score


def flip_features(pieces: torch.Tensor) -> torch.Tensor:
    """Piece list from Black's point of view: swap colors, mirror the board. -1 stays -1."""
    piece = torch.div(pieces, 64, rounding_mode="floor")
    square = pieces % 64
    flipped = ((piece + 6) % 12) * 64 + (square ^ 56)
    return torch.where(pieces >= 0, flipped, pieces)


def to_dense(pieces: torch.Tensor) -> torch.Tensor:
    """Piece list [B, 32] -> 768-wide vector of zeros and ones [B, 768].

    Padding (-1) is written to an extra column 768, which is cut off afterwards.
    """
    index = torch.where(pieces >= 0, pieces, PADDING).long()
    dense = torch.zeros(pieces.shape[0], NUM_FEATURES + 1, device=pieces.device)
    dense.scatter_(1, index, 1.0)
    return dense[:, :NUM_FEATURES]


class NNUE(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.hidden = hidden
        self.ft = nn.Linear(NUM_FEATURES, hidden)   # feature transformer (shared by both perspectives)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, pieces: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        """pieces [B, 32] (White's view), stm [B] (1 = White to move) -> output [B] (x EVAL_SCALE = cp)."""
        white = self.ft(to_dense(pieces))
        black = self.ft(to_dense(flip_features(pieces)))
        white_to_move = stm.bool().unsqueeze(1)
        us = torch.where(white_to_move, white, black)
        them = torch.where(white_to_move, black, white)
        hidden = torch.cat([us, them], dim=1).clamp(0.0, 1.0)     # clipped ReLU
        return self.out(hidden).squeeze(1)


def loss_fn(output: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Error in win-probability space: sigmoid(network) vs. sigmoid(Stockfish / 400).

    Small errors in clearly decided positions (+1500 instead of +1600) barely count,
    errors in balanced positions count a lot.
    """
    target = torch.sigmoid(score.float() / EVAL_SCALE)
    return torch.mean((torch.sigmoid(output) - target) ** 2)

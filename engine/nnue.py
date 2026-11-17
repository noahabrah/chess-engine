"""NNUE evaluation inside the engine (NumPy only, no PyTorch).

Computes the same as nnue/model.py, using the weights saved by nnue/export.py:
1. Features: every piece as  piece * 64 + square, once from White's point of view and
   once from Black's (colors swapped, board mirrored).
2. One accumulator per side: bias + sum of the weight rows of all pieces
   (this is the first layer; multiplying a vector of mostly zeros would be a waste).
3. Side to move first, then the opponent, clipped to 0..1 (clipped ReLU).
4. Output layer -> one number, times eval_scale = centipawns for the side to move.

The accumulators are recomputed for every position. Real engines update them
incrementally after each move, which would be the next speed-up.

The network is loaded from engine/nnue.npz at import time, if the file exists.
`network` is the active network (None = use PeSTO), switched with set_enabled().
"""

from pathlib import Path

import chess
import numpy as np

DEFAULT_PATH = Path(__file__).with_name("nnue.npz")
MAX_EVAL = 10_000   # safety limit: an evaluation must never look like a mate score

# Feature offset (plane 0-11 times 64) of a piece from White's and from Black's point of view, by [color][piece_type]
_WHITE_VIEW_PLANE = [[0] * 7, [0] * 7]
_BLACK_VIEW_PLANE = [[0] * 7, [0] * 7]
for _color in chess.COLORS:
    for _piece_type in chess.PIECE_TYPES:
        own = 0 if _color == chess.WHITE else 6
        _WHITE_VIEW_PLANE[_color][_piece_type] = (_piece_type - 1 + own) * 64
        _BLACK_VIEW_PLANE[_color][_piece_type] = (_piece_type - 1 + 6 - own) * 64


class Network:
    def __init__(
        self,
        ft_weight: np.ndarray,
        ft_bias: np.ndarray,
        out_weight: np.ndarray,
        out_bias: float,
        eval_scale: float,
    ) -> None:
        hidden = len(ft_bias)
        self.ft_weight = np.ascontiguousarray(ft_weight, dtype=np.float32)    # [768, hidden]
        self.ft_bias = np.asarray(ft_bias, dtype=np.float32)
        self.out_us = np.asarray(out_weight[:hidden], dtype=np.float32)     # side to move
        self.out_them = np.asarray(out_weight[hidden:], dtype=np.float32)   # opponent
        self.out_bias = float(out_bias)
        self.eval_scale = float(eval_scale)

    @classmethod
    def from_file(cls, path: str | Path) -> "Network":
        with np.load(path) as data:
            return cls(data["ft_weight"], data["ft_bias"], data["out_weight"],
                       data["out_bias"], data["eval_scale"])

    def evaluate(self, board: chess.Board) -> int:
        """Evaluation in centipawns from the side to move's point of view."""
        features: list[int] = []        # all features from White's point of view ...
        flipped: list[int] = []         # ... and the same ones from Black's point of view
        for color in chess.COLORS:
            white_view = _WHITE_VIEW_PLANE[color]
            black_view = _BLACK_VIEW_PLANE[color]
            for piece_type in chess.PIECE_TYPES:
                for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                    features.append(white_view[piece_type] + square)
                    flipped.append(black_view[piece_type] + (square ^ 56))

        # A single NumPy lookup for both sides: gather the rows, then sum each half
        count = len(features)
        rows = self.ft_weight[features + flipped]
        white = self.ft_bias + rows[:count].sum(axis=0)
        black = self.ft_bias + rows[count:].sum(axis=0)

        us, them = (white, black) if board.turn == chess.WHITE else (black, white)
        output = (
            np.minimum(np.maximum(us, 0.0), 1.0) @ self.out_us
            + np.minimum(np.maximum(them, 0.0), 1.0) @ self.out_them
            + self.out_bias
        )
        score = int(output * self.eval_scale)
        return max(-MAX_EVAL, min(MAX_EVAL, score))


loaded: Network | None = Network.from_file(DEFAULT_PATH) if DEFAULT_PATH.exists() else None
network: Network | None = loaded    # active network; None = PeSTO


def set_enabled(enabled: bool) -> None:
    """Switch NNUE on/off (UCI option UseNNUE). Without a network file it stays off."""
    global network
    network = loaded if enabled else None

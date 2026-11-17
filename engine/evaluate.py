"""Static evaluation of a position: NNUE (engine/nnue.py), otherwise PeSTO with tapered eval.

evaluate() uses the neural network if engine/nnue.npz exists and NNUE is switched on
(UCI option UseNNUE), and the handcrafted PeSTO evaluation otherwise:

- Every piece counts its base value + square bonus (pst.py), separately for middlegame and endgame.
- The game phase (24 = full board, 0 = only kings and pawns) blends the two values,
  e.g. the king should stay in the corner in the middlegame but walk to the center in the endgame.

The result is always in centipawns from the side to move's point of view (for negamax):
positive = good for the side that is about to move.
"""

import chess

from . import nnue
from .pst import EG_TABLE, MAX_PHASE, MG_TABLE, PHASE_WEIGHT


def game_phase(board: chess.Board) -> int:
    """24 = middlegame (all pieces on the board), 0 = endgame (no minor or major pieces left)."""
    phase = 0
    for piece_type, weight in PHASE_WEIGHT.items():
        if weight:
            phase += weight * board.pieces_mask(piece_type, chess.WHITE).bit_count()
            phase += weight * board.pieces_mask(piece_type, chess.BLACK).bit_count()
    return min(phase, MAX_PHASE)    # can exceed 24 after promotions


def evaluate(board: chess.Board) -> int:
    """Evaluation used by the search: the network if it is active, otherwise PeSTO."""
    network = nnue.network
    if network is not None:
        return network.evaluate(board)
    return evaluate_pesto(board)


def evaluate_pesto(board: chess.Board) -> int:
    """Handcrafted evaluation: PeSTO piece-square tables, blended by game phase."""
    mg = [0, 0]     # index = color: chess.BLACK = 0, chess.WHITE = 1
    eg = [0, 0]

    for color in chess.COLORS:
        mg_tables = MG_TABLE[color]
        eg_tables = EG_TABLE[color]
        for piece_type in chess.PIECE_TYPES:
            mg_table = mg_tables[piece_type]
            eg_table = eg_tables[piece_type]
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                mg[color] += mg_table[square]
                eg[color] += eg_table[square]

    us, them = board.turn, not board.turn
    phase = game_phase(board)
    mg_score = mg[us] - mg[them]
    eg_score = eg[us] - eg[them]
    # int(… / …) rounds towards 0 (// would round negative numbers down). That keeps
    # score for White == -(score for Black) exactly, which negamax relies on.
    return int((mg_score * phase + eg_score * (MAX_PHASE - phase)) / MAX_PHASE)

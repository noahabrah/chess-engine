import json

import chess

from nnue.prepare_data import MAX_SCORE, encode_pieces, process_line


def make_line(fen: str, line: str, depth: int = 30, **score) -> str:
    """One line in the format of the Lichess evaluation database (score: cp=… or mate=…)."""
    return json.dumps({"fen": fen, "evals": [{"depth": depth, "knodes": 1000, "pvs": [{**score, "line": line}]}]})


QUEEN_UP_WHITE = "4k3/8/8/8/8/8/8/3QK3 w - -"     # White is a queen up
QUEEN_UP_BLACK_TO_MOVE = "4k3/8/8/8/8/8/8/3QK3 b - -"


def test_encode_pieces():
    pieces = encode_pieces(chess.Board())
    assert len(pieces) == 32
    assert 0 * 64 + chess.E2 in pieces          # white pawn e2
    assert 5 * 64 + chess.E1 in pieces          # white king e1
    assert 6 * 64 + chess.E7 in pieces          # black pawn e7
    assert 11 * 64 + chess.E8 in pieces         # black king e8


def test_score_is_converted_to_side_to_move():
    pieces, white_to_move, score = process_line(make_line(QUEEN_UP_WHITE, "d1d7", cp=900))
    assert white_to_move and score == 900
    # Same evaluation from White's point of view, but Black to move -> negative
    _, white_to_move, score = process_line(make_line(QUEEN_UP_BLACK_TO_MOVE, "e8f7", cp=900))
    assert not white_to_move and score == -900
    assert len(pieces) == 3


def test_mate_and_large_scores_are_clipped():
    assert process_line(make_line(QUEEN_UP_WHITE, "d1d7", mate=5))[2] == MAX_SCORE
    assert process_line(make_line(QUEEN_UP_WHITE, "d1d7", mate=-3))[2] == -MAX_SCORE
    assert process_line(make_line(QUEEN_UP_WHITE, "d1d7", cp=5000))[2] == MAX_SCORE


def test_uses_deepest_eval():
    data = {"fen": QUEEN_UP_WHITE, "evals": [
        {"depth": 25, "pvs": [{"cp": 100, "line": "d1d7"}]},
        {"depth": 40, "pvs": [{"cp": 950, "line": "d1d7"}]},
        {"depth": 30, "pvs": [{"cp": 200, "line": "d1d7"}]},
    ]}
    assert process_line(json.dumps(data))[2] == 950


def test_rejects_unsuitable_positions():
    assert process_line(make_line(QUEEN_UP_WHITE, "d1d7", depth=12, cp=900)) == "depth too low"
    # Black is in check (queen on d8)
    assert process_line(make_line("3Qk3/8/8/8/8/8/8/4K3 b - -", "e8d8", cp=0)) == "in check"
    # the best move is a capture
    assert process_line(make_line("4k3/8/8/3p4/4P3/8/8/4K3 w - -", "e4d5", cp=100)) == "best move is a capture/promotion"
    # en passant also counts as a capture
    assert process_line(make_line("4k3/8/8/3pP3/8/8/8/4K3 w - d6", "e5d6", cp=100)) == "best move is a capture/promotion"
    # promotion
    assert process_line(make_line("4k3/P7/8/8/8/8/8/4K3 w - -", "a7a8q", cp=900)) == "best move is a capture/promotion"
    assert process_line(json.dumps({"fen": QUEEN_UP_WHITE, "evals": []})) == "no evaluation"
    # impossible: 33 pieces, or a pawn on the back rank
    assert process_line(make_line("QQQQkQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/8/8/8/4K3 w - -", "e1e2", cp=0)) == "invalid position"
    assert process_line(make_line("P3k3/8/8/8/8/8/8/4K3 w - -", "e1e2", cp=0)) == "invalid position"

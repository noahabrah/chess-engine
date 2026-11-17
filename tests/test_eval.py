import chess

from engine.evaluate import evaluate_pesto as evaluate, game_phase


def test_startpos_is_equal():
    assert evaluate(chess.Board()) == 0


def test_score_is_from_side_to_move():
    # White is a queen up
    white_to_move = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    black_to_move = chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
    assert evaluate(white_to_move) > 800
    assert evaluate(black_to_move) == -evaluate(white_to_move)


def test_mirrored_position_has_same_score():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    assert evaluate(board) == evaluate(board.mirror())


def test_game_phase():
    assert game_phase(chess.Board()) == 24
    assert game_phase(chess.Board("4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1")) == 0
    assert game_phase(chess.Board("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1")) == 8


def test_knight_prefers_center():
    center = chess.Board("4k3/8/8/8/4N3/8/8/4K3 b - - 0 1")
    corner = chess.Board("4k3/8/8/8/8/8/8/N3K3 b - - 0 1")
    assert evaluate(center) < evaluate(corner)      # from Black's point of view: Ne4 is worse


def test_king_belongs_to_center_in_endgame():
    # Pure pawn endgame: the central king is better
    central = chess.Board("4k3/4p3/8/8/4K3/8/4P3/8 w - - 0 1")
    corner = chess.Board("4k3/4p3/8/8/8/8/4P3/K7 w - - 0 1")
    assert evaluate(central) > evaluate(corner)


def test_king_prefers_safety_in_middlegame():
    castled = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1")
    center = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N1KN2/PPPP1PPP/R1BQ1R2 w - - 0 1")
    assert evaluate(castled) > evaluate(center)

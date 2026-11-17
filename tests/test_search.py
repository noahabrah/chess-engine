import threading
import time

import chess

import engine.search
from engine.evaluate import evaluate
from engine.search import (
    MATE_SCORE,
    Searcher,
    SearchLimits,
    has_non_pawn_material,
    lmr_reduction,
    mvv_lva,
    noisy_moves,
    order_moves,
    score_to_uci,
    search,
    time_limits,
)

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def run(fen: str, **limits) -> tuple[chess.Move | None, dict]:
    """Search fen and return (move, arguments of the last report)."""
    info: dict = {}
    move = search(chess.Board(fen), SearchLimits(**limits), threading.Event(), info.update)
    return move, info


# ---------------------------------------------------------------- mate / stalemate

def test_finds_mate_in_one():
    move, info = run("6k1/8/6K1/8/8/8/8/R7 w - - 0 1", depth=1)
    assert move == chess.Move.from_uci("a1a8")
    assert info["score_mate"] == 1


def test_finds_mate_in_two():
    move, info = run("k7/8/2K5/8/8/8/8/1R6 w - - 0 1", depth=3)
    assert info["score_mate"] == 2
    board = chess.Board("k7/8/2K5/8/8/8/8/1R6 w - - 0 1")
    for m in info["pv"]:
        board.push(m)
    assert board.is_checkmate()
    assert move == info["pv"][0]


def test_prefers_mate_over_stalemate():
    # Qg6 would be stalemate, Qg7 is mate
    move, info = run("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1", depth=2)
    assert move == chess.Move.from_uci("g1g7")
    assert info["score_mate"] == 1


def test_check_extension_finds_mate_beyond_depth():
    # Smothered mate: 1.Qg8+ Rxg8 2.Nf7#. Without the check extension this needs depth 3.
    move, info = run("5r1k/6pp/7N/3Q4/8/8/8/6K1 w - - 0 1", depth=2)
    assert move == chess.Move.from_uci("d5g8")
    assert info["score_mate"] == 2


def test_no_move_when_stalemated_or_mated():
    assert run("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", depth=3)[0] is None   # stalemate
    assert run("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", depth=3)[0] is None   # checkmate


def test_reports_negative_mate_when_getting_mated():
    # Black to move: the only move is Kb8, then Rh8#
    _, info = run("k7/8/1K6/8/8/8/8/7R b - - 0 1", depth=2)
    assert info["score_mate"] == -1


# ---------------------------------------------------------------- material

def test_captures_hanging_queen():
    move, info = run("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", depth=2)
    assert move == chess.Move.from_uci("e4d5")
    assert info["score_cp"] > 0


# ---------------------------------------------------------------- quiescence

def test_does_not_grab_defended_pawn_at_horizon():
    # Qxd5 seems to win a pawn, but exd5 takes the queen back.
    # Without quiescence search, depth 1 would not see the recapture.
    move, info = run("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", depth=1)
    assert move != chess.Move.from_uci("d1d5")
    assert info["score_cp"] > 500           # queen against two pawns, nothing lost


def test_quiescence_sees_promotion():
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    searcher = Searcher(board, SearchLimits(), threading.Event())
    assert searcher.quiescence(0, -MATE_SCORE, MATE_SCORE) > evaluate(board) + 500


def test_quiescence_stands_pat_in_quiet_position():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    searcher = Searcher(board, SearchLimits(), threading.Event())
    assert searcher.quiescence(0, -MATE_SCORE, MATE_SCORE) == evaluate(board)


def test_noisy_moves_are_sorted_by_mvv_lva():
    # White can take the queen with a pawn or a pawn with the rook
    board = chess.Board("4k3/8/8/2q5/1P6/8/2p5/2R1K3 w - - 0 1")
    moves = [m.uci() for m in noisy_moves(board)]
    assert moves == ["b4c5", "c1c2"]
    assert mvv_lva(board, chess.Move.from_uci("b4c5")) > mvv_lva(board, chess.Move.from_uci("c1c2"))


def test_order_moves():
    board = chess.Board("4k3/8/8/2q5/1P6/8/2p5/2R1K3 w - - 0 1")
    moves = list(board.legal_moves)
    first = chess.Move.from_uci("e1d2")

    order_moves(board, moves, first)
    assert [m.uci() for m in moves[:3]] == ["e1d2", "b4c5", "c1c2"]   # first_move, then MVV-LVA

    order_moves(board, moves)
    assert [m.uci() for m in moves[:2]] == ["b4c5", "c1c2"]
    assert not any(board.is_capture(m) for m in moves[2:])            # then only quiet moves


def test_order_moves_with_killers_and_history():
    board = chess.Board("4k3/8/8/2q5/1P6/8/2p5/2R1K3 w - - 0 1")
    moves = list(board.legal_moves)
    killer = chess.Move.from_uci("e1f1")
    good_history = chess.Move.from_uci("c1d1")
    history = [[[0] * 64 for _ in range(64)] for _ in chess.COLORS]
    history[chess.WHITE][good_history.from_square][good_history.to_square] = 500

    order_moves(board, moves, None, [killer, None], history)
    # captures, then the killer, then the quiet move with the best history
    assert [m.uci() for m in moves[:4]] == ["b4c5", "c1c2", "e1f1", "c1d1"]


def test_quiet_cutoff_updates_killers_and_history():
    board = chess.Board()
    searcher = Searcher(board, SearchLimits(), threading.Event())
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")

    searcher.update_quiet_stats(e4, depth=3, ply=2)
    searcher.update_quiet_stats(d4, depth=2, ply=2)
    searcher.update_quiet_stats(d4, depth=2, ply=2)     # same killer: don't store it twice

    assert searcher.killers[2] == [d4, e4]
    assert searcher.history[chess.WHITE][e4.from_square][e4.to_square] == 9
    assert searcher.history[chess.WHITE][d4.from_square][d4.to_square] == 8


# ---------------------------------------------------------------- null move pruning

ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


def test_has_non_pawn_material():
    assert has_non_pawn_material(chess.Board())
    assert not has_non_pawn_material(chess.Board("4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1"))
    # Only the side to move counts: White has a knight, Black only pawns
    assert has_non_pawn_material(chess.Board("4k3/pppp4/8/8/8/8/PPPP4/4KN2 w - - 0 1"))
    assert not has_non_pawn_material(chess.Board("4k3/pppp4/8/8/8/8/PPPP4/4KN2 b - - 0 1"))


def test_null_move_pruning_saves_nodes(monkeypatch):
    _, with_null = run(ITALIAN, depth=5)
    monkeypatch.setattr(engine.search, "NULL_MOVE_MIN_DEPTH", 99)   # disable
    _, without_null = run(ITALIAN, depth=5)
    assert with_null["nodes"] < without_null["nodes"] * 0.8


def test_no_null_move_in_pawn_endgame(monkeypatch):
    # Pawn endgame (risk of zugzwang): null move pruning must not change anything
    fen = "8/5k2/3p4/1p1Pp3/pP2Pp2/P4P2/8/6K1 w - - 0 1"
    move_with, with_null = run(fen, depth=6)
    monkeypatch.setattr(engine.search, "NULL_MOVE_MIN_DEPTH", 99)
    move_without, without_null = run(fen, depth=6)
    assert move_with == move_without
    assert with_null["nodes"] == without_null["nodes"]


# ---------------------------------------------------------------- PVS / LMR

def test_lmr_reduction():
    assert lmr_reduction(3, 1) == 0                     # early move: no reduction
    assert lmr_reduction(3, 40) == 1                    # at least depth 1 is always left
    assert lmr_reduction(8, 5) <= lmr_reduction(8, 20)  # later move -> more reduction
    assert lmr_reduction(4, 10) <= lmr_reduction(10, 10)  # deeper -> more reduction
    assert lmr_reduction(200, 1000) <= 198              # no error outside the table


def test_lmr_saves_nodes(monkeypatch):
    _, with_lmr = run(ITALIAN, depth=6)
    monkeypatch.setattr(engine.search, "LMR_MIN_DEPTH", 99)     # disable
    _, without_lmr = run(ITALIAN, depth=6)
    assert with_lmr["nodes"] < without_lmr["nodes"] * 0.8


def test_principal_variation_is_legal():
    # Even after null-window searches and re-searches the PV must be a legal move sequence
    for fen in (chess.STARTING_FEN, ITALIAN, KIWIPETE):
        _, info = run(fen, depth=5)
        board = chess.Board(fen)
        assert info["pv"]
        for move in info["pv"]:
            assert move in board.legal_moves
            board.push(move)


# ---------------------------------------------------------------- draws

def score_after(board: chess.Board, uci_move: str) -> int:
    """Negamax score after uci_move, from the point of view of the side then to move."""
    board.push(chess.Move.from_uci(uci_move))
    searcher = Searcher(board, SearchLimits(), threading.Event())
    return searcher.negamax(1, 1, -MATE_SCORE, MATE_SCORE, [])


def test_repetition_is_draw():
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    for m in ["a1a2", "h8g8", "a2a1", "g8h8"]:
        board.push_uci(m)
    # a1a2 repeats a position -> 0, although White is a rook up
    assert score_after(board, "a1a2") == 0


def test_fifty_move_rule_is_draw():
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 99 80")
    assert score_after(board, "a1a2") == 0


def test_insufficient_material_is_draw():
    board = chess.Board("7k/8/8/8/8/8/8/N6K w - - 0 1")
    assert score_after(board, "a1b3") == 0


# ---------------------------------------------------------------- stopping

def test_returns_legal_move_when_stopped_immediately():
    board = chess.Board()
    stop = threading.Event()
    stop.set()
    move = search(board, SearchLimits(depth=6), stop, lambda **_: None)
    assert move in board.legal_moves
    assert board.fen() == chess.STARTING_FEN    # the board was restored


def test_respects_node_limit():
    move, _ = run(chess.STARTING_FEN, depth=10, nodes=2000)
    assert move in chess.Board().legal_moves


def test_stop_during_search_returns_legal_move():
    board = chess.Board(KIWIPETE)
    stop = threading.Event()
    reports: list[dict] = []
    threading.Timer(0.5, stop.set).start()
    move = search(board, SearchLimits(infinite=True), stop, lambda **r: reports.append(r))
    assert move in board.legal_moves
    assert reports                                  # at least one depth finished
    assert board.fen() == chess.Board(KIWIPETE).fen()


# ---------------------------------------------------------------- iterative deepening

def test_reports_every_depth_in_order():
    reports: list[dict] = []
    search(chess.Board(), SearchLimits(depth=4), threading.Event(), lambda **r: reports.append(r))
    assert [r["depth"] for r in reports] == [1, 2, 3, 4]
    assert all(r["seldepth"] >= r["depth"] for r in reports)


def test_single_legal_move_is_played_immediately():
    # Black only has Kb8
    reports: list[dict] = []
    start = time.perf_counter()
    move = search(
        chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1"),
        SearchLimits(wtime=60_000, btime=60_000),
        threading.Event(),
        lambda **r: reports.append(r),
    )
    assert move == chess.Move.from_uci("a8b8")
    assert time.perf_counter() - start < 0.05
    assert reports == []


def test_finds_mate_with_time_only_and_stops_early():
    start = time.perf_counter()
    _, info = run("k7/8/2K5/8/8/8/8/1R6 w - - 0 1", movetime=3000)
    assert info["score_mate"] == 2
    assert time.perf_counter() - start < 1.0       # mate proven -> stop searching


# ---------------------------------------------------------------- time management

def test_time_limits():
    board_w = chess.Board()
    board_b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

    # no time limit
    assert time_limits(board_w, SearchLimits(infinite=True, wtime=1000)) == (None, None)
    assert time_limits(board_w, SearchLimits(depth=6)) == (None, None)
    # fixed time per move
    assert time_limits(board_w, SearchLimits(movetime=1000)) == (950, 950)
    # clock: 30 estimated moves left, hard = 3 x soft
    assert time_limits(board_w, SearchLimits(wtime=30_000, btime=1)) == (1000, 3000)
    # Black uses btime/binc
    assert time_limits(board_b, SearchLimits(wtime=1, btime=30_000, binc=400)) == (1300, 3900)
    # movestogo is used
    assert time_limits(board_w, SearchLimits(wtime=30_000, movestogo=10)) == (3000, 9000)
    # little time, large increment: never more than half the remaining time
    assert time_limits(board_w, SearchLimits(wtime=1000, winc=2000)) == (500, 500)


def test_movetime_is_respected():
    start = time.perf_counter()
    move, _ = run(KIWIPETE, movetime=300)
    assert move in chess.Board(KIWIPETE).legal_moves
    assert time.perf_counter() - start < 0.4


def test_clock_time_is_respected():
    # 5 s left: soft ~0.17 s, hard 0.5 s
    start = time.perf_counter()
    move, _ = run(KIWIPETE, wtime=5000, btime=5000)
    assert move in chess.Board(KIWIPETE).legal_moves
    assert time.perf_counter() - start < 0.6


# ---------------------------------------------------------------- score format

def test_score_to_uci():
    assert score_to_uci(37) == {"score_cp": 37}
    assert score_to_uci(MATE_SCORE - 1) == {"score_mate": 1}
    assert score_to_uci(MATE_SCORE - 3) == {"score_mate": 2}
    assert score_to_uci(-MATE_SCORE + 2) == {"score_mate": -1}
    assert score_to_uci(-MATE_SCORE + 4) == {"score_mate": -2}

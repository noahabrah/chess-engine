import threading

import chess

from engine.search import MATE_SCORE, SearchLimits, score_from_tt, score_to_tt, search
from engine.tt import ENTRY_BYTES, EXACT, LOWER, UPPER, TranspositionTable, position_key

E4 = chess.Move.from_uci("e2e4")
D4 = chess.Move.from_uci("d2d4")


def test_store_and_probe():
    tt = TranspositionTable(1)
    tt.store(12345, depth=3, score=50, flag=EXACT, move=E4)
    entry = tt.probe(12345)
    assert entry is not None
    assert (entry.depth, entry.score, entry.flag, entry.move) == (3, 50, EXACT, E4)


def test_probe_other_position_on_same_slot_returns_none():
    tt = TranspositionTable(1)
    tt.store(5, depth=3, score=50, flag=EXACT, move=E4)
    assert tt.probe(5 + tt.size) is None        # same slot, different position


def test_replacement():
    tt = TranspositionTable(1)
    tt.store(7, depth=5, score=10, flag=EXACT, move=E4)
    tt.store(7, depth=2, score=99, flag=LOWER, move=D4)    # same position, shallower -> keep
    assert tt.probe(7).depth == 5
    tt.store(7, depth=6, score=20, flag=UPPER, move=D4)    # deeper -> replace
    assert tt.probe(7).depth == 6
    tt.store(7 + tt.size, depth=1, score=0, flag=EXACT, move=None)  # different position -> replace
    assert tt.probe(7) is None


def test_clear_and_resize():
    tt = TranspositionTable(1)
    tt.store(1, depth=1, score=0, flag=EXACT, move=None)
    tt.clear()
    assert tt.probe(1) is None
    tt.resize(2)
    assert tt.size == 2 * 1024 * 1024 // ENTRY_BYTES


def test_position_key_ignores_move_order():
    a = chess.Board()
    for m in ["g1f3", "g8f6", "d2d4"]:
        a.push_uci(m)
    b = chess.Board()
    for m in ["d2d4", "g8f6", "g1f3"]:
        b.push_uci(m)
    assert position_key(a) == position_key(b)
    assert position_key(a) != position_key(chess.Board())


def test_mate_scores_are_stored_relative_to_position():
    # Mate in 3 plies from a position at ply 4 = mate at ply 7 from the root
    at_root = MATE_SCORE - 7
    stored = score_to_tt(at_root, ply=4)
    assert stored == MATE_SCORE - 3
    assert score_from_tt(stored, ply=2) == MATE_SCORE - 5     # same position, different ply
    assert score_from_tt(score_to_tt(-MATE_SCORE + 6, 2), 2) == -MATE_SCORE + 6
    assert score_to_tt(150, 9) == 150


def test_shared_tt_speeds_up_next_search():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    tt = TranspositionTable(16)
    first: dict = {}
    second: dict = {}
    move1 = search(board, SearchLimits(depth=4), threading.Event(), first.update, tt)
    move2 = search(board, SearchLimits(depth=4), threading.Event(), second.update, tt)
    assert move1 == move2
    assert second["nodes"] < first["nodes"] / 5

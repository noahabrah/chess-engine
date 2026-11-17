import random
import threading

import chess
import numpy as np
import pytest

from engine import nnue
from engine.evaluate import evaluate, evaluate_pesto
from engine.nnue import Network
from engine.search import SearchLimits, search
from engine.uci import UCIEngine

ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
QUEEN_UP_WHITE = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"


def random_network(hidden: int = 32, seed: int = 0) -> Network:
    rng = np.random.default_rng(seed)
    return Network(
        ft_weight=rng.normal(0, 0.1, (768, hidden)),
        ft_bias=rng.normal(0, 0.1, hidden),
        out_weight=rng.normal(0, 0.5, 2 * hidden),
        out_bias=0.1,
        eval_scale=400,
    )


def random_positions(count: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randrange(2, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def test_mirrored_position_gets_same_score():
    net = random_network()
    for board in random_positions(30):
        # The float summation order can differ -> at most 1 cp rounding difference
        assert abs(net.evaluate(board) - net.evaluate(board.mirror())) <= 1


def test_evaluate_uses_network_only_when_enabled(monkeypatch):
    net = random_network()
    board = chess.Board(ITALIAN)
    monkeypatch.setattr(nnue, "loaded", net)
    nnue.set_enabled(True)
    assert evaluate(board) == net.evaluate(board)
    nnue.set_enabled(False)
    assert evaluate(board) == evaluate_pesto(board)


def test_score_is_limited():
    net = random_network()
    net.out_bias = 1000.0           # absurdly large network output
    assert net.evaluate(chess.Board()) == nnue.MAX_EVAL


def test_engine_matches_pytorch_model(tmp_path):
    """Export + NumPy inference must give exactly what the network computes in PyTorch."""
    torch = pytest.importorskip("torch")
    from nnue.export import export
    from nnue.model import EVAL_SCALE, NNUE
    from nnue.prepare_data import MAX_PIECES, encode_pieces

    torch.manual_seed(0)
    model = NNUE(32)
    torch.save({"model": model.state_dict(), "hidden": 32, "epoch": 0, "val_loss": 0.0}, tmp_path / "net.pt")
    export(str(tmp_path / "net.pt"), str(tmp_path / "net.npz"))
    net = Network.from_file(tmp_path / "net.npz")

    boards = random_positions(50)
    pieces = torch.tensor([encode_pieces(b) + [-1] * (MAX_PIECES - len(encode_pieces(b))) for b in boards])
    stm = torch.tensor([b.turn == chess.WHITE for b in boards], dtype=torch.uint8)
    with torch.no_grad():
        expected = (model(pieces, stm) * EVAL_SCALE).tolist()
    for board, want in zip(boards, expected):
        assert abs(net.evaluate(board) - want) <= 1


def test_uci_option_switches_evaluation(monkeypatch, capsys):
    import engine.uci as uci
    monkeypatch.setattr(nnue, "loaded", random_network())
    monkeypatch.setattr(uci, "CHECK_OPTIONS", {"UseNNUE": True})
    engine = UCIEngine()
    engine.handle("uci")
    assert "option name UseNNUE type check default true" in capsys.readouterr().out
    engine.handle("setoption name UseNNUE value true")
    assert nnue.network is nnue.loaded
    engine.handle("setoption name usennue value false")
    assert nnue.network is None


# ---------------------------------------------------------------- with the trained network

needs_trained_net = pytest.mark.skipif(not nnue.DEFAULT_PATH.exists(), reason="engine/nnue.npz is missing")


@pytest.fixture
def trained_net(monkeypatch):
    monkeypatch.setattr(nnue, "network", nnue.loaded)
    return nnue.loaded


@needs_trained_net
def test_trained_net_is_plausible(trained_net):
    assert abs(evaluate(chess.Board())) < 100
    assert evaluate(chess.Board(QUEEN_UP_WHITE)) > 500
    assert evaluate(chess.Board(QUEEN_UP_WHITE).mirror()) > 500     # Black is the queen up, Black to move


@needs_trained_net
def test_search_with_trained_net(trained_net):
    info: dict = {}
    move = search(chess.Board("k7/8/2K5/8/8/8/8/1R6 w - - 0 1"), SearchLimits(depth=3), threading.Event(), info.update)
    assert info["score_mate"] == 2
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")        # black queen is hanging
    move = search(board, SearchLimits(depth=3), threading.Event(), info.update)
    assert move == chess.Move.from_uci("d1d5")

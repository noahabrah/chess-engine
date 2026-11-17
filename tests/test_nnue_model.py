import random

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")    # only installed with the "train" group

from nnue.model import NNUE, flip_features, loss_fn, to_dense  # noqa: E402
from nnue.prepare_data import MAX_PIECES, decode_pieces, encode_pieces  # noqa: E402

ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


def as_tensors(boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
    pieces = [encode_pieces(b) + [-1] * (MAX_PIECES - len(encode_pieces(b))) for b in boards]
    stm = [b.turn == chess.WHITE for b in boards]
    return torch.tensor(pieces, dtype=torch.int32), torch.tensor(stm, dtype=torch.uint8)


def random_positions(count: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randrange(2, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def test_encode_decode_roundtrip():
    board = chess.Board(ITALIAN)
    decoded = decode_pieces(encode_pieces(board), board.turn)
    assert decoded.board_fen() == board.board_fen() and decoded.turn == board.turn


def test_flip_features_equals_mirrored_board():
    board = chess.Board(ITALIAN)
    pieces, _ = as_tensors([board])
    mirrored, _ = as_tensors([board.mirror()])
    assert sorted(flip_features(pieces)[0].tolist()) == sorted(mirrored[0].tolist())
    assert flip_features(torch.tensor([[-1]])).item() == -1     # padding stays padding


def test_to_dense():
    pieces, _ = as_tensors([chess.Board()])
    dense = to_dense(pieces)
    assert dense.shape == (1, 768)
    assert dense.sum().item() == 32


def test_mirrored_position_gets_same_output():
    # Mirrored position (colors swapped, other side to move) = same score for the side to move
    torch.manual_seed(0)
    model = NNUE(32)
    boards = random_positions(20)
    out = model(*as_tensors(boards))
    out_mirrored = model(*as_tensors([b.mirror() for b in boards]))
    assert torch.allclose(out, out_mirrored, atol=1e-6)


def test_short_training_reduces_loss():
    # The network learns a simple target: material balance from the side to move's point of view
    torch.manual_seed(0)
    values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    boards = random_positions(512)
    pieces, stm = as_tensors(boards)
    score = torch.tensor([
        sum(values[p.piece_type] * (1 if p.color == b.turn else -1) for p in b.piece_map().values())
        for b in boards
    ])
    model = NNUE(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = loss_fn(model(pieces, stm), score).item()
    for _ in range(100):
        loss = loss_fn(model(pieces, stm), score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first / 2
    assert np.isfinite(loss.item())

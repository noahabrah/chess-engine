"""Build training data from the Lichess evaluation database.

Source: https://database.lichess.org/#evals (lichess_db_eval.jsonl.zst, about 22 GB).
One line = one position with Stockfish evaluations from the Lichess cloud analysis:
    {"fen": "...", "evals": [{"depth": 36, "knodes": ..., "pvs": [{"cp": 31, "line": "g1f3 ..."}]}]}
cp/mate are always from White's point of view.

The file is streamed (straight from the URL or from a local copy) and reading stops
after --positions usable positions, so nothing has to be downloaded completely.

Stored per position (the format is deliberately independent of the network architecture):
- pieces: up to 32 values  piece * 64 + square  from White's point of view,
          piece = 0-5 white pawn…king, 6-11 black pawn…king; padded with -1
- stm:    side to move (1 = White, 0 = Black)
- score:  evaluation in centipawns from the side to move's point of view (clipped to ±MAX_SCORE)

Output: <out>/train_000.npz, train_001.npz, … and <out>/val.npz (a random VAL_FRACTION).
Running the script again overwrites these files.

Usage:
    uv run --group train python -m nnue.prepare_data --positions 1000000
"""

import argparse
import http.client
import io
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import chess
import numpy as np

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
MIN_DEPTH = 20          # only use evaluations with at least this search depth
MAX_SCORE = 2000        # centipawns; larger scores and mates are clipped to this
MAX_PIECES = 32
SHARD_SIZE = 1_000_000  # positions per file
VAL_FRACTION = 0.01     # share of positions held out for validation (never trained on)
TIMEOUT = 60            # seconds without data from the server -> give up instead of waiting forever


def encode_pieces(board: chess.Board) -> list[int]:
    """All pieces as  piece * 64 + square  (from White's point of view)."""
    pieces = []
    for square, piece in board.piece_map().items():
        index = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        pieces.append(index * 64 + square)
    return pieces


def decode_pieces(pieces, white_to_move: bool) -> chess.Board:
    """Inverse of encode_pieces: piece list (padded with -1) -> board."""
    board = chess.Board(None)
    for feature in pieces:
        if feature >= 0:
            index, square = divmod(int(feature), 64)
            color = chess.WHITE if index < 6 else chess.BLACK
            board.set_piece_at(square, chess.Piece(index % 6 + 1, color))
    board.turn = bool(white_to_move)
    return board


def best_eval(evals: list[dict]) -> dict | None:
    """The evaluation with the highest search depth."""
    return max(evals, key=lambda e: e.get("depth", 0), default=None)


def process_line(line: str) -> tuple[list[int], bool, int] | str:
    """One database line -> (pieces, white_to_move, score), or the reason it was rejected.

    Only quiet positions are kept: not in check and the best move is neither a capture
    nor a promotion. In those positions the static evaluation (what the network learns)
    should match the search result; tactics are left to the quiescence search.
    """
    data = json.loads(line)
    ev = best_eval(data.get("evals", []))
    if ev is None or not ev.get("pvs"):
        return "no evaluation"
    if ev.get("depth", 0) < MIN_DEPTH:
        return "depth too low"

    board = chess.Board(data["fen"])
    # Positions from the Lichess board editor can be impossible (e.g. ten queens)
    if not board.is_valid() or chess.popcount(board.occupied) > MAX_PIECES:
        return "invalid position"
    if board.is_check():
        return "in check"

    pv = ev["pvs"][0]
    line_moves = pv.get("line", "").split()
    if not line_moves:
        return "no principal variation"
    best = chess.Move.from_uci(line_moves[0])
    if board.is_capture(best) or best.promotion:
        return "best move is a capture/promotion"

    if "mate" in pv:
        white_score = MAX_SCORE if pv["mate"] > 0 else -MAX_SCORE
    else:
        white_score = max(-MAX_SCORE, min(MAX_SCORE, pv["cp"]))
    score = white_score if board.turn == chess.WHITE else -white_score
    return encode_pieces(board), board.turn == chess.WHITE, score


def open_stream(source: str) -> io.TextIOWrapper:
    """URL or local .zst file as a text stream (decompressed while reading)."""
    import zstandard  # only needed here (group "train"), so the tests also run without it

    raw = urllib.request.urlopen(source, timeout=TIMEOUT) if source.startswith("http") else open(source, "rb")  # noqa: SIM115
    reader = zstandard.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(reader, encoding="utf-8")


class ShardWriter:
    """Collects positions and writes them to .npz files of shard_size positions each."""

    def __init__(self, path_pattern: str, shard_size: int) -> None:
        self.path_pattern = path_pattern
        self.shard_size = shard_size
        self.shard_index = 0
        self.pieces: list[list[int]] = []
        self.stm: list[bool] = []
        self.score: list[int] = []

    def add(self, pieces: list[int], stm: bool, score: int) -> None:
        self.pieces.append(pieces + [-1] * (MAX_PIECES - len(pieces)))
        self.stm.append(stm)
        self.score.append(score)
        if len(self.score) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.score:
            return
        path = self.path_pattern.format(self.shard_index)
        np.savez(
            path,
            pieces=np.array(self.pieces, dtype=np.int16),
            stm=np.array(self.stm, dtype=np.uint8),
            score=np.array(self.score, dtype=np.int16),
        )
        print(f"  -> {path} ({len(self.score):,} positions)")
        self.shard_index += 1
        self.pieces, self.stm, self.score = [], [], []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, default=1_000_000, help="number of usable positions to collect")
    parser.add_argument("--source", default=URL, help="URL or local .zst file")
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train = ShardWriter(str(out / "train_{:03d}.npz"), SHARD_SIZE)
    val = ShardWriter(str(out / "val.npz"), 10**12)     # everything in a single file

    rejected: Counter[str] = Counter()
    scores: list[int] = []      # sample of the scores for the statistics at the end
    accepted = read = 0
    start = time.perf_counter()

    # On an abort (Ctrl+C, connection hangs or drops) everything collected so far is still saved
    try:
        with open_stream(args.source) as stream:
            for line in stream:
                read += 1
                result = process_line(line)
                if isinstance(result, str):
                    rejected[result] += 1
                    continue
                accepted += 1
                (val if rng.random() < VAL_FRACTION else train).add(*result)
                if len(scores) < 1_000_000:
                    scores.append(result[2])
                if accepted % 100_000 == 0:
                    rate = read / (time.perf_counter() - start)
                    print(f"{accepted:,} / {args.positions:,} positions ({rate:,.0f} lines/s)")
                if accepted >= args.positions:
                    break
    except (KeyboardInterrupt, OSError, http.client.HTTPException) as error:
        print(f"\nAborted ({type(error).__name__}: {error}) - saving what was collected so far")

    train.flush()
    val.flush()

    elapsed = time.perf_counter() - start
    print(f"\nRead {read:,} lines in {elapsed:.0f} s, accepted {accepted:,}")
    for reason, count in rejected.most_common():
        print(f"  rejected ({reason}): {count:,}")
    s = np.abs(np.array(scores))
    print(
        f"Score (side to move): mean {np.mean(scores):+.0f} cp, "
        f"|score| < 100: {np.mean(s < 100):.0%}, < 500: {np.mean(s < 500):.0%}, "
        f"= {MAX_SCORE} (mate/clipped): {np.mean(s == MAX_SCORE):.0%}"
    )


if __name__ == "__main__":
    main()

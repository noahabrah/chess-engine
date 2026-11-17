"""Transposition table (TT): a cache of positions that were already searched.

The same position can arise from different move orders (a transposition),
e.g. 1.Nf3 Nf6 2.d4 and 1.d4 Nf6 2.Nf3. For each position the TT stores:
- how deep it was searched and with which result -> maybe no need to search it again
- the best move -> tried first next time (the best move ordering there is)

Laid out like in real engines: a list of fixed size, indexed by key % size.
If the slot is taken, a simple replacement rule decides which entry stays.
"""

from typing import NamedTuple

import chess

# Kind of stored score (alpha-beta doesn't always return the exact value)
EXACT = 0   # exact value
LOWER = 1   # beta cutoff: the real value is >= score
UPPER = 2   # no move beat alpha: the real value is <= score

# Estimated memory per entry in Python (list slot + tuple + ints)
ENTRY_BYTES = 200


class TTEntry(NamedTuple):
    key: int
    depth: int
    score: int
    flag: int
    move: chess.Move | None


def position_key(board: chess.Board) -> int:
    """Hash of the position: pieces, side to move, castling rights, en passant.

    _transposition_key() is internal to python-chess (used there for is_repetition),
    but much faster than chess.polyglot.zobrist_hash().
    """
    return hash(board._transposition_key())


class TranspositionTable:
    def __init__(self, size_mb: int = 16) -> None:
        self.resize(size_mb)

    def resize(self, size_mb: int) -> None:
        """New size in MB (UCI option Hash). Clears the table."""
        self.size = max(1, size_mb * 1024 * 1024 // ENTRY_BYTES)
        self.clear()

    def clear(self) -> None:
        self.table: list[TTEntry | None] = [None] * self.size

    def probe(self, key: int) -> TTEntry | None:
        entry = self.table[key % self.size]
        if entry is not None and entry.key == key:
            return entry
        return None     # empty, or a different position in the same slot

    def store(self, key: int, depth: int, score: int, flag: int, move: chess.Move | None) -> None:
        index = key % self.size
        old = self.table[index]
        # Replace, unless it's the same position from a deeper search
        if old is None or old.key != key or depth >= old.depth:
            self.table[index] = TTEntry(key, depth, score, flag, move)

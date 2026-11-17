"""Search: iterative deepening + negamax with alpha-beta pruning + quiescence search
+ transposition table + move ordering (TT move, MVV-LVA, killer moves, history)
+ check extension + null move pruning + principal variation search (PVS)
+ late move reductions (LMR).

Conventions:
- All scores are from the side to move's point of view (negamax): a child's score
  is negated, and alpha and beta are swapped and negated.
- Being mated is scored as -MATE_SCORE + ply. So a quick mate is worth more than
  a slow one, and being mated later is better than being mated sooner.
- Aborting (stop, time, node limit) works through the exception SearchAborted.
  search() then restores the board to the root position.

Time management with two limits:
- Soft limit: the time we aim to spend on this move. Once half of it is used up,
  no new depth is started.
- Hard limit: after this the current depth is aborted immediately.
"""

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import chess

from .evaluate import evaluate
from .tt import EXACT, LOWER, UPPER, TranspositionTable, position_key

MATE_SCORE = 100_000
MATE_THRESHOLD = MATE_SCORE - 1_000   # |score| above this = mate in X
INF = MATE_SCORE + 1
MAX_DEPTH = 64                        # upper bound for iterative deepening
MAX_PLY = 128                         # safety limit for the distance from the root (incl. quiescence)
DEFAULT_MOVES_TO_GO = 30              # estimated moves left if the GUI sends no movestogo
CHECK_EVERY = 1024                    # nodes between two stop/time checks (power of 2)
NULL_MOVE_MIN_DEPTH = 3               # null move pruning only from this remaining depth on
LMR_MIN_DEPTH = 3                     # late move reductions only from this remaining depth on
LMR_MIN_MOVE_INDEX = 3                # never reduce the first moves (TT move, good captures)
LMR_MAX_MOVES = 256                   # size of the LMR table (more legal moves never occur)


@dataclass
class SearchLimits:
    """All limits from the UCI command `go`. None = not given."""

    wtime: int | None = None      # White's remaining time in ms
    btime: int | None = None      # Black's remaining time in ms
    winc: int = 0                 # White's increment in ms
    binc: int = 0                 # Black's increment in ms
    movestogo: int | None = None  # moves until the next time control
    movetime: int | None = None   # fixed time for this move in ms
    depth: int | None = None      # maximum search depth
    nodes: int | None = None      # maximum number of nodes
    infinite: bool = False        # search until `stop`
    move_overhead: int = 50       # safety margin in ms (UCI option)


# report(depth=..., nodes=..., pv=[...], score_cp=... or score_mate=...)
ReportFn = Callable[..., None]


class SearchAborted(Exception):
    """Aborts the search (stop, time is up, node limit reached)."""


def time_limits(board: chess.Board, limits: SearchLimits) -> tuple[float | None, float | None]:
    """(soft limit, hard limit) in ms for this move, or (None, None) without a time limit."""
    if limits.infinite:
        return None, None

    if limits.movetime is not None:
        budget = max(limits.movetime - limits.move_overhead, 10)
        return budget, budget

    time_left = limits.wtime if board.turn == chess.WHITE else limits.btime
    if time_left is None:
        return None, None       # e.g. `go depth 6` or `go nodes 10000`
    inc = limits.winc if board.turn == chess.WHITE else limits.binc
    moves_to_go = max(limits.movestogo or DEFAULT_MOVES_TO_GO, 1)

    soft = time_left / moves_to_go + inc * 3 / 4
    hard = min(soft * 3, time_left / 2, time_left - limits.move_overhead)
    hard = max(hard, 10)
    return min(soft, hard), hard


def score_to_uci(score: int) -> dict[str, int]:
    """Internal score -> keyword arguments for report (centipawns or mate in moves)."""
    if score >= MATE_THRESHOLD:
        return {"score_mate": (MATE_SCORE - score + 1) // 2}
    if score <= -MATE_THRESHOLD:
        return {"score_mate": -((MATE_SCORE + score) // 2)}
    return {"score_cp": score}


def score_to_tt(score: int, ply: int) -> int:
    """Store mate scores in the TT relative to the position instead of the root.

    Internally a mate is "MATE_SCORE - distance from the root". But the same position
    can show up at a different ply in another search. So the TT stores the distance
    from this position, and it is converted back when reading.
    """
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    """Inverse of score_to_tt: mate distance from this position -> distance from the root."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def is_proven_mate(score: int, depth: int) -> bool:
    """True if a mate was found that lies completely within `depth` plies.

    Searching deeper can't change it anymore.
    """
    return abs(score) >= MATE_THRESHOLD and MATE_SCORE - abs(score) <= depth


def mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Sort key for captures: most valuable victim first, then least valuable attacker.

    (Most Valuable Victim – Least Valuable Attacker.) Higher = searched earlier.
    """
    victim = board.piece_type_at(move.to_square)
    if victim is None:      # empty target square: en passant or a non-capturing promotion
        victim = chess.PAWN if board.is_en_passant(move) else 0
    attacker = board.piece_type_at(move.from_square)
    score = 10 * victim - attacker
    if move.promotion:
        score += 10 * move.promotion
    return score


# Precomputed reductions for late move reductions: _LMR_TABLE[depth][move_index].
# The deeper the search and the later the move in the ordering, the larger the reduction
# (logarithmic, the usual formula from engines like Ethereal).
_LMR_TABLE = [
    [0 if d == 0 or i == 0 else int(0.75 + math.log(d) * math.log(i) / 2.25) for i in range(LMR_MAX_MOVES)]
    for d in range(MAX_PLY)
]


def lmr_reduction(depth: int, move_index: int) -> int:
    """By how many plies a late quiet move is searched less deeply.

    At least depth 1 is left, so the move doesn't drop straight into quiescence search.
    """
    reduction = _LMR_TABLE[min(depth, MAX_PLY - 1)][min(move_index, LMR_MAX_MOVES - 1)]
    return max(0, min(reduction, depth - 2))


def has_non_pawn_material(board: chess.Board) -> bool:
    """True if the side to move still has a knight, bishop, rook or queen."""
    return bool(board.occupied_co[board.turn] & ~(board.pawns | board.kings))


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    """Quiet move = neither a capture nor a promotion."""
    return not move.promotion and not board.is_capture(move)


def order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    first_move: chess.Move | None = None,
    killers: list[chess.Move | None] | None = None,
    history: list[list[list[int]]] | None = None,
) -> None:
    """Sort moves in place. The earlier the best move comes, the more alpha-beta can prune.

    Order:
    1. first_move (TT move, or the best move of the previous depth)
    2. captures/promotions by MVV-LVA
    3. killer moves (quiet moves that already caused a cutoff at the same ply)
    4. remaining quiet moves by history score
    """
    killers = killers or []
    side_history = history[board.turn] if history is not None else None

    def key(move: chess.Move) -> int:
        if move == first_move:
            return 100_000_000
        if not is_quiet(board, move):
            return 10_000_000 + mvv_lva(board, move)
        if move in killers:
            return 9_000_000 - killers.index(move)      # killer 1 before killer 2
        if side_history is not None:
            return min(side_history[move.from_square][move.to_square], 8_000_000)
        return 0

    moves.sort(key=key, reverse=True)


def noisy_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and queen promotions, sorted by MVV-LVA."""
    moves = list(board.generate_legal_captures())
    pawns = board.pawns & board.occupied_co[board.turn]
    promotion_rank = chess.BB_RANK_8 if board.turn == chess.WHITE else chess.BB_RANK_1
    # Promotions without a capture (capturing promotions are already included above)
    for move in board.generate_legal_moves(pawns, promotion_rank & ~board.occupied):
        if move.promotion == chess.QUEEN:
            moves.append(move)
    moves.sort(key=lambda m: mvv_lva(board, m), reverse=True)
    return moves


class Searcher:
    def __init__(
        self,
        board: chess.Board,
        limits: SearchLimits,
        stop_event: threading.Event,
        tt: TranspositionTable | None = None,
    ) -> None:
        self.board = board
        self.stop_event = stop_event
        self.tt = tt if tt is not None else TranspositionTable()
        self.max_nodes = limits.nodes if limits.nodes is not None else float("inf")
        self.nodes = 0
        self.seldepth = 0       # largest distance from the root reached (incl. quiescence)

        # Ordering of quiet moves, updated on every beta cutoff:
        # - killers[ply]: the last two quiet moves that caused a cutoff at this ply
        # - history[color][from][to]: how often (weighted by depth²) this move caused a cutoff
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history = [[[0] * 64 for _ in range(64)] for _ in chess.COLORS]

        soft_ms, hard_ms = time_limits(board, limits)
        now = time.perf_counter()
        # A new depth takes several times as long as the previous one. If half of the
        # soft limit is already used up, it would hardly finish -> don't even start it.
        self.soft_deadline = None if soft_ms is None else now + soft_ms / 2 / 1000
        self.hard_deadline = None if hard_ms is None else now + hard_ms / 1000

        # Result of the current depth (best root move searched completely so far)
        self.best_move: chess.Move | None = None
        self.best_score = -INF
        self.pv: list[chess.Move] = []

    def check_abort(self) -> None:
        if self.stop_event.is_set():
            raise SearchAborted
        if self.nodes >= self.max_nodes:
            raise SearchAborted
        if self.hard_deadline is not None and time.perf_counter() >= self.hard_deadline:
            raise SearchAborted

    def past_soft_deadline(self) -> bool:
        return self.soft_deadline is not None and time.perf_counter() >= self.soft_deadline

    def search_root(self, depth: int, first_move: chess.Move | None = None) -> None:
        """Search all root moves. The result goes into best_move/best_score/pv.

        first_move (the best move of the previous depth) is searched first,
        then captures by MVV-LVA, then quiet moves.
        Root moves are never reduced (no LMR), only PVS is used.
        """
        board = self.board
        moves = list(board.legal_moves)
        order_moves(board, moves, first_move)

        self.best_move = None
        self.best_score = -INF
        self.pv = []
        self.seldepth = 0
        alpha, beta = -INF, INF

        for i, move in enumerate(moves):
            child_pv: list[chess.Move] = []
            board.push(move)
            if i == 0:
                score = -self.negamax(depth - 1, 1, -beta, -alpha, child_pv)
            else:
                # PVS: only check whether the move beats the best one so far (null window)
                score = -self.negamax(depth - 1, 1, -alpha - 1, -alpha, child_pv)
                if score > alpha:   # it does -> search again with the full window for the exact value
                    child_pv.clear()
                    score = -self.negamax(depth - 1, 1, -beta, -alpha, child_pv)
            board.pop()

            if score > alpha:
                alpha = score
                self.best_move = move
                self.best_score = score
                self.pv = [move, *child_pv]

    def negamax(self, depth: int, ply: int, alpha: int, beta: int, pv: list[chess.Move]) -> int:
        """Score of the current position from the side to move's point of view.

        pv is filled with the principal variation from this position on.
        """
        self.nodes += 1
        if self.nodes & (CHECK_EVERY - 1) == 0 or self.nodes >= self.max_nodes:
            self.check_abort()

        board = self.board

        # Draws. Inside the search a single repetition is enough:
        # if repeating were good for us once, it would be good the second time too.
        if board.is_repetition(2) or board.is_insufficient_material():
            return 0
        # 50-move rule, unless the last move delivered mate (mate takes precedence)
        if board.halfmove_clock >= 100 and not board.is_checkmate():
            return 0
        if ply >= MAX_PLY:
            return evaluate(board)      # safety limit (only reached in extremely long check sequences)

        # Check extension: a side in check is searched one ply deeper. That way the
        # search never stops in the middle of a sequence of checks, and mates or wins
        # after forced moves are found earlier.
        in_check = board.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return self.quiescence(ply, alpha, beta)

        # Transposition table: was this position already searched deeply enough?
        key = position_key(board)
        entry = self.tt.probe(key)
        tt_move = None
        if entry is not None:
            tt_move = entry.move
            if entry.depth >= depth:
                score = score_from_tt(entry.score, ply)
                if (
                    entry.flag == EXACT
                    or (entry.flag == LOWER and score >= beta)
                    or (entry.flag == UPPER and score <= alpha)
                ):
                    pv[:] = [tt_move] if tt_move else []   # the PV ends here (the rest isn't in the TT)
                    return score

        # Null move pruning: we let the opponent move twice in a row (we "pass")
        # and search with reduced depth. If our position is still >= beta even then,
        # it would be even better with a real move. The opponent won't allow it,
        # so cut off without searching all moves.
        # Not allowed:
        # - in check (passing would be illegal)
        # - right after a null move (otherwise both sides just keep passing)
        # - without pieces other than pawns: zugzwang is common in pawn endgames, where
        #   passing would be better than any real move and the assumption breaks
        # - with a mate score as beta: mate searches should stay exact
        if (
            not in_check
            and depth >= NULL_MOVE_MIN_DEPTH
            and board.move_stack and board.peek() != chess.Move.null()
            and abs(beta) < MATE_THRESHOLD
            and has_non_pawn_material(board)
            and evaluate(board) >= beta
        ):
            reduction = 2 + depth // 4
            board.push(chess.Move.null())
            score = -self.negamax(depth - 1 - reduction, ply + 1, -beta, -beta + 1, [])
            board.pop()
            if score >= beta:
                # A mate found after passing isn't proven -> return just beta
                return beta if score >= MATE_THRESHOLD else score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0   # checkmate or stalemate
        order_moves(board, moves, tt_move, self.killers[ply], self.history)

        alpha_start = alpha
        best = -INF
        best_move = None
        killers = self.killers[ply]
        for i, move in enumerate(moves):
            child_pv: list[chess.Move] = []
            quiet = is_quiet(board, move)
            board.push(move)
            if i == 0:
                # First move (usually the best): full window, full depth
                score = -self.negamax(depth - 1, ply + 1, -beta, -alpha, child_pv)
            else:
                # Late move reductions: thanks to good move ordering, late quiet moves
                # are almost never the best -> search them less deeply first.
                # Not reduced: captures, promotions, killers, checks
                # and moves out of check.
                reduction = 0
                if (
                    depth >= LMR_MIN_DEPTH
                    and i >= LMR_MIN_MOVE_INDEX
                    and quiet
                    and not in_check
                    and move not in killers
                    and not board.is_check()
                ):
                    reduction = lmr_reduction(depth, i)

                # PVS: for all other moves only check whether they beat alpha.
                # A null window (alpha, alpha + 1) is enough for that and prunes much more.
                score = -self.negamax(depth - 1 - reduction, ply + 1, -alpha - 1, -alpha, child_pv)
                if score > alpha and reduction:
                    # Looked good at reduced depth -> verify at full depth
                    child_pv.clear()
                    score = -self.negamax(depth - 1, ply + 1, -alpha - 1, -alpha, child_pv)
                if alpha < score < beta:
                    # Really better (only possible in PV nodes, elsewhere beta = alpha + 1):
                    # search with the full window for the exact value and the principal variation
                    child_pv.clear()
                    score = -self.negamax(depth - 1, ply + 1, -beta, -alpha, child_pv)
            board.pop()

            if score > best:
                best = score
                best_move = move
                if score > alpha:
                    alpha = score
                    pv[:] = [move, *child_pv]
                    if alpha >= beta:
                        if quiet:
                            self.update_quiet_stats(move, depth, ply)
                        break   # beta cutoff: the opponent won't allow this position

        if best >= beta:
            flag = LOWER        # cut off: the real value may be even higher
        elif best > alpha_start:
            flag = EXACT
        else:
            flag = UPPER        # no move beat alpha: the real value may be even lower
        self.tt.store(key, depth, score_to_tt(best, ply), flag, best_move)
        return best

    def update_quiet_stats(self, move: chess.Move, depth: int, ply: int) -> None:
        """A quiet move caused a beta cutoff: remember it as a killer and in the history."""
        killers = self.killers[ply]
        if killers[0] != move:
            killers[1] = killers[0]
            killers[0] = move
        # Deep cutoffs count more: they save more work and are more reliable
        self.history[self.board.turn][move.from_square][move.to_square] += depth * depth

    def quiescence(self, ply: int, alpha: int, beta: int) -> int:
        """At the horizon, keep searching only captures until the position is "quiet".

        Prevents the horizon effect: without it, e.g. "queen takes pawn" would count
        as winning a pawn, because the recapture lies beyond the horizon.
        """
        self.nodes += 1
        if self.nodes & (CHECK_EVERY - 1) == 0 or self.nodes >= self.max_nodes:
            self.check_abort()
        self.seldepth = max(self.seldepth, ply)

        board = self.board
        if ply >= MAX_PLY:
            return evaluate(board)

        # Detect checkmate. Searching all evasions from check would be more accurate,
        # but creates very long chains (check, evasion, capture with check, …).
        if board.is_check() and not any(board.generate_legal_moves()):
            return -MATE_SCORE + ply

        # Stand pat: the side to move doesn't have to capture. So the static
        # evaluation is a lower bound for the value of the position.
        best = evaluate(board)
        if best >= beta:
            return best
        alpha = max(alpha, best)

        for move in noisy_moves(board):
            board.push(move)
            score = -self.quiescence(ply + 1, -beta, -alpha)
            board.pop()

            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best


def search(
    board: chess.Board,
    limits: SearchLimits,
    stop_event: threading.Event,
    report: ReportFn,
    tt: TranspositionTable | None = None,
) -> chess.Move | None:
    """Return the best move, or None if there is no legal move.

    Searches depth 1, 2, 3, … until the depth limit, the time, the node limit or `stop`.
    After every completed depth, report(...) is called.
    tt: transposition table that is kept across moves (from uci.py).
    Without one, a new table is created for this search.
    """
    moves = list(board.legal_moves)
    if not moves:
        return None

    # Only one legal move: don't waste thinking time, unless the GUI
    # explicitly asks for an analysis (infinite) or a specific depth.
    if len(moves) == 1 and not limits.infinite and limits.depth is None:
        return moves[0]

    searcher = Searcher(board, limits, stop_event, tt)
    max_depth = min(max(1, limits.depth or MAX_DEPTH), MAX_DEPTH)
    root_ply = board.ply()
    best_move = moves[0]    # fallback in case even depth 1 is aborted

    for depth in range(1, max_depth + 1):
        try:
            searcher.check_abort()      # react to `stop` right away before a new depth
            searcher.search_root(depth, best_move)
        except SearchAborted:
            # The exception skipped the push/pop pairs -> restore the board.
            while board.ply() > root_ply:
                board.pop()
            # The previous best move was searched first. If its search finished,
            # searcher.best_move is at least as good -> use it.
            if searcher.best_move is not None:
                best_move = searcher.best_move
            break

        best_move = searcher.best_move
        report(
            depth=depth,
            seldepth=searcher.seldepth,
            nodes=searcher.nodes,
            pv=searcher.pv,
            **score_to_uci(searcher.best_score),
        )

        if searcher.past_soft_deadline():
            break
        if not limits.infinite and is_proven_mate(searcher.best_score, depth):
            break

    return best_move

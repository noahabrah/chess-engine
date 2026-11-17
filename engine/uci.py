"""UCI protocol: reads commands from stdin, writes responses to stdout.

This file contains no chess logic. It only translates text into function calls
(set up the board, start/stop the search) and back.

Rules:
- Everything on stdout is for the GUI. Debug output goes to stderr only, via log().
- The search runs in its own thread, so `stop`, `isready` and `quit`
  are answered during a search as well.
"""

import sys
import threading
import time

import chess

from . import nnue
from .search import SearchLimits, search
from .tt import TranspositionTable

ENGINE_NAME = "MeineEngine 0.1"
ENGINE_AUTHOR = "Noah"

# UCI spin options: name -> (type, default, minimum, maximum)
OPTIONS: dict[str, tuple[str, int, int, int]] = {
    "Hash": ("spin", 16, 1, 1024),            # MB for the transposition table
    "Move Overhead": ("spin", 50, 0, 5000),   # ms safety margin for network/GUI lag
}
# On/off options: name -> default. UseNNUE only exists if engine/nnue.npz exists.
CHECK_OPTIONS: dict[str, bool] = {"UseNNUE": True} if nnue.loaded is not None else {}

_output_lock = threading.Lock()


def send(line: str) -> None:
    """Send one line to the GUI (thread-safe, flushed immediately)."""
    with _output_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    """Debug output to stderr; the GUI doesn't treat it as a UCI command."""
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


class UCIEngine:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.options = {name: default for name, (_, default, _, _) in OPTIONS.items()}
        self.tt = TranspositionTable(self.options["Hash"])  # kept across all moves
        self.stop_event = threading.Event()
        self.search_thread: threading.Thread | None = None
        self.search_start = 0.0

    def loop(self) -> None:
        while True:
            line = sys.stdin.readline()
            if not line:            # EOF: the GUI closed the connection
                self.cmd_quit()
                return
            line = line.strip()
            if not line:
                continue
            try:
                if not self.handle(line):
                    return
            except Exception as exc:  # the engine must never crash  # noqa: BLE001
                log(f"Error in '{line}': {exc!r}")

    def handle(self, line: str) -> bool:
        """Process one command. Returns False if the engine should quit."""
        tokens = line.split()
        cmd, args = tokens[0], tokens[1:]

        match cmd:
            case "uci":
                self.cmd_uci()
            case "isready":
                send("readyok")
            case "ucinewgame":
                self.stop_search()
                self.board = chess.Board()
                self.tt.clear()     # positions from the old game are of no use anymore
            case "setoption":
                self.cmd_setoption(args)
            case "position":
                self.stop_search()
                self.cmd_position(args)
            case "go":
                self.cmd_go(args)
            case "stop":
                self.stop_search()
            case "ponderhit":
                pass                # pondering is not supported (yet)
            case "quit":
                self.cmd_quit()
                return False
            case "d":               # not a UCI command, only for debugging by hand
                send(str(self.board))
                send(f"FEN: {self.board.fen()}")
            case _:
                log(f"Unknown command: {line}")
        return True

    def cmd_uci(self) -> None:
        send(f"id name {ENGINE_NAME}")
        send(f"id author {ENGINE_AUTHOR}")
        for name, (typ, default, lo, hi) in OPTIONS.items():
            send(f"option name {name} type {typ} default {default} min {lo} max {hi}")
        for name, default in CHECK_OPTIONS.items():
            send(f"option name {name} type check default {str(default).lower()}")
        send("uciok")

    def cmd_setoption(self, args: list[str]) -> None:
        # setoption name <name, may contain spaces> [value <value>]
        if "name" not in args:
            return
        name_start = args.index("name") + 1
        if "value" in args:
            value_idx = args.index("value")
            name = " ".join(args[name_start:value_idx])
            value = " ".join(args[value_idx + 1:])
        else:
            name, value = " ".join(args[name_start:]), ""

        # UCI option names are case-insensitive
        check_key = next((k for k in CHECK_OPTIONS if k.lower() == name.lower()), None)
        if check_key == "UseNNUE":
            self.stop_search()      # don't switch the evaluation in the middle of a search
            nnue.set_enabled(value.lower() == "true")
            self.tt.clear()         # the stored scores come from the other evaluation
            return

        key = next((k for k in OPTIONS if k.lower() == name.lower()), None)
        if key is None:
            log(f"Unknown option: {name}")
            return
        _, _, lo, hi = OPTIONS[key]
        try:
            self.options[key] = max(lo, min(hi, int(value)))
        except ValueError:
            log(f"Invalid value for {key}: {value}")
            return

        if key == "Hash":
            self.stop_search()      # the search must not write into a table that is being replaced
            self.tt.resize(self.options["Hash"])

    def cmd_position(self, args: list[str]) -> None:
        # position startpos [moves ...]
        # position fen <FEN> [moves ...]
        if not args:
            return
        moves_idx = args.index("moves") if "moves" in args else len(args)

        if args[0] == "startpos":
            board = chess.Board()
        elif args[0] == "fen":
            board = chess.Board(" ".join(args[1:moves_idx]))
        else:
            log(f"Invalid position command: {' '.join(args)}")
            return

        for uci_move in args[moves_idx + 1:]:
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                log(f"Illegal move in position: {uci_move}")
                return
            board.push(move)
        self.board = board

    def cmd_go(self, args: list[str]) -> None:
        self.stop_search()          # finish a search that may still be running first
        limits = self.parse_go(args)
        self.stop_event.clear()
        self.search_start = time.perf_counter()
        self.search_thread = threading.Thread(
            target=self.run_search,
            args=(self.board.copy(), limits),   # a copy: search and UCI never share a board
            daemon=True,
        )
        self.search_thread.start()

    def cmd_quit(self) -> None:
        self.stop_search()

    # ------------------------------------------------------------------ search

    def parse_go(self, args: list[str]) -> SearchLimits:
        limits = SearchLimits(move_overhead=self.options["Move Overhead"])
        int_params = {"wtime", "btime", "winc", "binc", "movestogo",
                      "movetime", "depth", "nodes"}
        i = 0
        while i < len(args):
            token = args[i]
            if token in int_params and i + 1 < len(args):
                setattr(limits, token, int(args[i + 1]))
                i += 2
                continue
            if token == "infinite":
                limits.infinite = True
            # "ponder", "searchmoves" and "mate" are ignored (for now)
            i += 1
        return limits

    def run_search(self, board: chess.Board, limits: SearchLimits) -> None:
        """Runs in the search thread and always prints exactly one `bestmove` at the end."""
        try:
            move = search(board, limits, self.stop_event, self.report_info, self.tt)
        except Exception as exc:  # noqa: BLE001
            log(f"Error in search: {exc!r}")
            move = None

        # Fallback: any legal move is better than none
        if move is None or move not in board.legal_moves:
            move = next(iter(board.legal_moves), None)

        # With `go infinite`, bestmove may only be sent after `stop`
        if limits.infinite:
            self.stop_event.wait()

        send(f"bestmove {move.uci() if move else '0000'}")

    def stop_search(self) -> None:
        if self.search_thread and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join()
        self.search_thread = None

    def report_info(
        self,
        depth: int,
        nodes: int,
        pv: list[chess.Move],
        score_cp: int | None = None,
        score_mate: int | None = None,
        seldepth: int | None = None,
    ) -> None:
        """Called by the search after every depth -> UCI `info` line."""
        elapsed_ms = max(1, int((time.perf_counter() - self.search_start) * 1000))
        parts = [f"info depth {depth}"]
        if seldepth is not None:
            parts.append(f"seldepth {seldepth}")
        if score_mate is not None:
            parts.append(f"score mate {score_mate}")
        elif score_cp is not None:
            parts.append(f"score cp {score_cp}")
        parts.append(f"nodes {nodes} nps {nodes * 1000 // elapsed_ms} time {elapsed_ms}")
        if pv:
            parts.append("pv " + " ".join(m.uci() for m in pv))
        send(" ".join(parts))


def uci_loop() -> None:
    UCIEngine().loop()

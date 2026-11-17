#!/usr/bin/env bash
# Test games with fastchess.
#
#   tools/match.sh ref [-g GAMES] [-e STOCKFISH_ELO] [-t TIME_CONTROL] [-m NOTE]
#       Current engine against Stockfish with limited strength.
#       The result is appended as a row to tools/results.md.
#
#   tools/match.sh snapshot
#       Copies the current state (engine/ + main.py) to tools/reference/.
#       Run this before a change that you want to test with `sprt` afterwards.
#
#   tools/match.sh sprt [-g MAX_GAMES] [-t TIME_CONTROL]
#       Current engine against the reference until the SPRT decides:
#       "H1 accepted" = the change is better, "H0 accepted" = it is not.
#       The current state is copied to tools/candidate/ for this, so you
#       can keep working on engine/ while the test is running.
#
#   tools/match.sh accept
#       After "H1 accepted": the tested state (tools/candidate/) becomes
#       the new reference for the next SPRT.
#
# Time control as seconds+increment, e.g. 10+0.1.
# Number of parallel games via the environment variable CONCURRENCY (default 4).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FASTCHESS="$ROOT/tools/bin/fastchess"
BOOK="$ROOT/tools/books/8moves_v3.pgn"
PYTHON="$ROOT/.venv/bin/python"
REF_DIR="$ROOT/tools/reference"
CAND_DIR="$ROOT/tools/candidate"
RESULTS="$ROOT/tools/results.md"
LOG_DIR="$ROOT/tools/logs"
CONCURRENCY="${CONCURRENCY:-4}"

mode="${1:-}"
shift || true

games=200
sf_elo=1800
tc="10+0.1"
note=""
while getopts "g:e:t:m:" opt; do
    case "$opt" in
        g) games="$OPTARG" ;;
        e) sf_elo="$OPTARG" ;;
        t) tc="$OPTARG" ;;
        m) note="$OPTARG" ;;
        *) sed -n '2,23p' "$0"; exit 1 ;;
    esac
done

check_tools() {
    [[ -x "$FASTCHESS" ]] || { echo "fastchess is missing: $FASTCHESS"; exit 1; }
    [[ -f "$BOOK" ]] || { echo "Opening book is missing: $BOOK"; exit 1; }
    [[ -x "$PYTHON" ]] || { echo "Python environment is missing: run 'uv sync' once"; exit 1; }
}

# Runs fastchess, writes everything to the log and shows only the important lines.
run_fastchess() {
    local log="$1"
    shift
    mkdir -p "$LOG_DIR"
    # Start in the log directory: fastchess saves its progress file config.json there
    (cd "$LOG_DIR" && "$FASTCHESS" "$@" \
        -openings file="$BOOK" format=pgn order=random \
        -repeat -concurrency "$CONCURRENCY" -recover \
        -pgnout file="${log%.log}.pgn") \
        2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee "$log" \
        | grep -E "^(Finished game|Results of|Elo:|LOS:|Games:|Ptnml|SPRT|LLR|-----)" || true
}

# Copies engine/ + main.py into directory $1 (emptied first).
copy_engine() {
    rm -rf "$1"
    mkdir -p "$1"
    cp -R "$ROOT/engine" "$ROOT/main.py" "$1/"
    find "$1" -name "__pycache__" -type d -prune -exec rm -rf {} +
    date "+%Y-%m-%d %H:%M:%S" > "$1/SNAPSHOT_DATE"
}

case "$mode" in
    ref)
        check_tools
        command -v stockfish > /dev/null || { echo "Stockfish is missing: brew install stockfish"; exit 1; }
        log="$LOG_DIR/ref_$(date +%Y%m%d_%H%M%S).log"
        run_fastchess "$log" \
            -engine cmd="$PYTHON" args=main.py dir="$ROOT" name=MeineEngine \
            -engine cmd=stockfish name="SF$sf_elo" \
                option.UCI_LimitStrength=true option.UCI_Elo="$sf_elo" \
                option.Threads=1 option.Hash=16 \
            -each tc="$tc" \
            -rounds $(( (games + 1) / 2 ))

        # Read the result from the last summary and add it to results.md
        elo_line=$(grep "^Elo:" "$log" | tail -1)
        games_line=$(grep "^Games:" "$log" | tail -1)
        elo=$(sed -E 's/^Elo: ([^ ]+) \+\/- ([^,]+),.*/\1/' <<< "$elo_line")
        err=$(sed -E 's/^Elo: ([^ ]+) \+\/- ([^,]+),.*/\2/' <<< "$elo_line")
        wdl=$(sed -E 's/^Games: ([0-9]+), Wins: ([0-9]+), Losses: ([0-9]+), Draws: ([0-9]+), Points: ([^ ]+) \(([^ ]+) %\).*/\2-\4-\3 (\6 %)/' <<< "$games_line")
        n=$(sed -E 's/^Games: ([0-9]+),.*/\1/' <<< "$games_line")
        if [[ "$elo" =~ ^-?[0-9.]+$ ]]; then
            estimate=$(printf "%.0f" "$(echo "$sf_elo + $elo" | bc -l)")
            elo_text=$(printf "%+.0f ± %.0f" "$elo" "$err")
        else
            estimate="?"
            elo_text="$elo (too one-sided, pick a different Stockfish strength)"
        fi

        if [[ ! -f "$RESULTS" ]]; then
            {
                echo "# Results against Stockfish with limited strength"
                echo
                echo "Created by tools/match.sh ref. Elo difference = strength relative to Stockfish;"
                echo "estimated Elo = Stockfish Elo + difference (only rough: Stockfish's UCI_Elo is calibrated at 60+0.6)."
                echo
                echo "| Date | Note | Time control | Opponent | Games | W-D-L (score) | Elo difference | Estimated Elo |"
                echo "|---|---|---|---|---|---|---|---|"
            } > "$RESULTS"
        fi
        echo "| $(date +%Y-%m-%d) | ${note:--} | $tc | SF $sf_elo | $n | $wdl | $elo_text | $estimate |" >> "$RESULTS"
        echo
        echo "Added to tools/results.md: $elo_text against SF $sf_elo -> about $estimate Elo"
        ;;

    snapshot)
        copy_engine "$REF_DIR"
        echo "Snapshot saved to tools/reference/ ($(cat "$REF_DIR/SNAPSHOT_DATE"))"
        ;;

    accept)
        [[ -f "$CAND_DIR/main.py" ]] || { echo "No tested state in tools/candidate/"; exit 1; }
        rm -rf "$REF_DIR"
        cp -R "$CAND_DIR" "$REF_DIR"
        echo "New reference: tested state from $(cat "$REF_DIR/SNAPSHOT_DATE")"
        ;;

    sprt)
        check_tools
        [[ -f "$REF_DIR/main.py" ]] || { echo "No reference yet: run 'tools/match.sh snapshot' first (before the change)"; exit 1; }
        [[ "$games" == 200 ]] && games=4000     # an SPRT often needs more games, but stops by itself
        copy_engine "$CAND_DIR"     # frozen copy: engine/ may change while the test is running
        log="$LOG_DIR/sprt_$(date +%Y%m%d_%H%M%S).log"
        run_fastchess "$log" \
            -engine cmd="$PYTHON" args=main.py dir="$CAND_DIR" name=New \
            -engine cmd="$PYTHON" args=main.py dir="$REF_DIR" name=Reference \
            -each tc="$tc" \
            -rounds $(( (games + 1) / 2 )) \
            -sprt elo0=0 elo1=20 alpha=0.05 beta=0.05
        echo
        echo "Log and games: $log"
        ;;

    *)
        sed -n '2,23p' "$0"
        exit 1
        ;;
esac

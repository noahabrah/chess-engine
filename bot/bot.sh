#!/usr/bin/env bash
# Lichess bot for the engine (using the official lichess-bot program).
#
#   bot/bot.sh setup     download lichess-bot and set up its Python environment (once)
#   bot/bot.sh update    copy the current state (engine/ + main.py) to bot/engine/
#   bot/bot.sh upgrade   turn the Lichess account into a BOT account (once, CANNOT be undone),
#                        then start the bot
#   bot/bot.sh start     start the bot; it stays online until Ctrl+C
#
# The bot always plays with the copy in bot/engine/, never with engine/ directly.
# That way you can keep working on the code without the bot using half-finished code.
#
# The API token lives in bot/.env (not in git):  LICHESS_BOT_TOKEN=...

set -euo pipefail

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$BOT_DIR")"
LB_DIR="$BOT_DIR/lichess-bot"
VENV="$BOT_DIR/.venv"
ENGINE_DIR="$BOT_DIR/engine"
LB_REPO="https://github.com/lichess-bot-devs/lichess-bot.git"
TOKEN_PLACEHOLDER="paste-your-token-here"

check_setup() {
    [[ -f "$LB_DIR/lichess-bot.py" && -x "$VENV/bin/python" ]] \
        || { echo "lichess-bot is not set up: run 'bot/bot.sh setup' first"; exit 1; }
    [[ -x "$ROOT/.venv/bin/python" ]] \
        || { echo "The engine's Python environment is missing: run 'uv sync' in the project directory"; exit 1; }
}

load_token() {
    [[ -f "$BOT_DIR/.env" ]] || { echo "bot/.env is missing: run 'bot/bot.sh setup' first"; exit 1; }
    set -a
    source "$BOT_DIR/.env"
    set +a
    if [[ -z "${LICHESS_BOT_TOKEN:-}" || "$LICHESS_BOT_TOKEN" == "$TOKEN_PLACEHOLDER" ]]; then
        echo "No token: fill in the line LICHESS_BOT_TOKEN=... in bot/.env"
        exit 1
    fi
}

update_engine() {
    rm -rf "$ENGINE_DIR"
    mkdir -p "$ENGINE_DIR"
    cp -R "$ROOT/engine" "$ROOT/main.py" "$ENGINE_DIR/"
    find "$ENGINE_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} +
    # Start script for lichess-bot: runs the engine with the project's Python environment
    cat > "$ENGINE_DIR/run.sh" <<'EOF'
#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec ../../.venv/bin/python main.py
EOF
    chmod +x "$ENGINE_DIR/run.sh"
    date "+%Y-%m-%d %H:%M:%S" > "$ENGINE_DIR/SNAPSHOT_DATE"
    echo "Bot engine updated ($(cat "$ENGINE_DIR/SNAPSHOT_DATE"))"
}

# Run lichess-bot from its own directory (its logs end up there too)
run_lichess_bot() {
    [[ -f "$ENGINE_DIR/main.py" ]] || update_engine
    echo "Bot engine from $(cat "$ENGINE_DIR/SNAPSHOT_DATE") - update with 'bot/bot.sh update'"
    cd "$LB_DIR"
    exec "$VENV/bin/python" lichess-bot.py --config "$BOT_DIR/config.yml" "$@"
}

case "${1:-}" in
    setup)
        [[ -d "$LB_DIR" ]] || git clone --depth 1 "$LB_REPO" "$LB_DIR"
        [[ -x "$VENV/bin/python" ]] || uv venv "$VENV" --python 3.13
        uv pip install --python "$VENV/bin/python" -r "$LB_DIR/requirements.txt"
        if [[ ! -f "$BOT_DIR/.env" ]]; then
            echo "LICHESS_BOT_TOKEN=$TOKEN_PLACEHOLDER" > "$BOT_DIR/.env"
            chmod 600 "$BOT_DIR/.env"
        fi
        echo
        echo "Done. Now put your token into bot/.env."
        ;;

    update)
        update_engine
        ;;

    upgrade)
        check_setup
        load_token
        echo "This turns the account into a BOT account. It CANNOT be undone."
        read -r -p "Continue? (yes/no) " answer
        [[ "$answer" == "yes" ]] || { echo "Aborted."; exit 1; }
        run_lichess_bot -u
        ;;

    start)
        check_setup
        load_token
        run_lichess_bot
        ;;

    *)
        sed -n '2,13p' "$0"
        exit 1
        ;;
esac

#!/bin/sh
# Starts the engine for chess GUIs (e.g. En Croissant) that expect a single executable file.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python main.py

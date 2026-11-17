"""Entry point of the engine. It only starts the UCI loop.

Important: never print() anything here, stdout belongs to the GUI.
"""

from engine.uci import uci_loop


def main() -> None:
    uci_loop()


if __name__ == "__main__":
    main()

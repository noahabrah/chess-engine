# Python Chess Engine with NNUE

A UCI chess engine written from scratch in Python. It uses an alpha-beta search
with the standard pruning techniques, plus a small **NNUE** (efficiently updatable
neural network) evaluation that I trained myself on 27 million positions
evaluated by Stockfish, taken from the public Lichess evaluation database.

The engine plays on Lichess as **[noahabrahsbot](https://lichess.org/@/noahabrahsbot)**
and works in any UCI chess GUI (e.g. En Croissant).

After the first working version, every change was measured with self-play
SPRT tests before it was kept (see [Results](#results)).

## Features

**Search** ([`engine/search.py`](engine/search.py))
- Negamax with alpha-beta pruning and iterative deepening
- Quiescence search (captures and queen promotions) against the horizon effect
- Transposition table with mate-score correction ([`engine/tt.py`](engine/tt.py))
- Move ordering: TT move → captures by MVV-LVA → killer moves → history heuristic
- Check extension
- Null move pruning (disabled in check and in pawn endgames because of zugzwang)
- Principal variation search (PVS) with null-window re-searches
- Late move reductions (LMR) with a logarithmic reduction table
- Time management with a soft and a hard limit; `stop` is handled at any time

**Evaluation** ([`engine/evaluate.py`](engine/evaluate.py))
- **NNUE** (default): a 768 → 256×2 → 1 network, inference in pure NumPy ([`engine/nnue.py`](engine/nnue.py))
- PeSTO piece-square tables with tapered eval as a handcrafted fallback (UCI option `UseNNUE false`)

**UCI** ([`engine/uci.py`](engine/uci.py))
- The search runs in its own thread, so `stop`, `isready` and `quit` are answered while searching
- Options: `Hash`, `Move Overhead`, `UseNNUE`

## Results

### SPRT self-play tests

Each change was tested against the previous version with
[fastchess](https://github.com/Disservin/fastchess) (10+0.1, 8-move opening book,
SPRT with elo0 = 0, elo1 = 20, α = β = 0.05).

| Change | Games | Elo vs. previous version | Result |
|---|---|---|---|
| Null move pruning + LMR + PVS | 238 | +74 ± 38 | H1 accepted |
| NNUE instead of PeSTO | 356 | +60 ± 35 | H1 accepted |

The NNUE gain comes *despite* the engine getting slower: one NNUE evaluation takes
about 18 µs instead of 8 µs for PeSTO, which lowers the search speed from about
62k to 41–52k nodes per second.

### Against Stockfish with limited strength

200 games at 10+0.1 against Stockfish with `UCI_LimitStrength` (full history in
[`tools/results.md`](tools/results.md)). These numbers are only a rough estimate of
the absolute strength: Stockfish's Elo setting is calibrated for slower time
controls, and the statistical error at 200 games is about ±50 Elo. All three
versions are within that error of each other, so these runs can't resolve the
improvements; the SPRT tests above, which play directly against the previous
version, can. Self-play also tends to exaggerate gains.

| Version | W-D-L | Elo difference | Estimated Elo |
|---|---|---|---|
| Phase 1 (alpha-beta, quiescence, TT, move ordering, PeSTO) | 101-7-92 | +16 ± 50 vs. SF 1800 | ~1816 |
| + null move pruning, LMR, PVS | 96-5-99 | −5 ± 48 vs. SF 1800 | ~1795 |
| + NNUE | 99-7-94 | +9 ± 50 vs. SF 1800 | ~1809 |

## The NNUE

### Training data ([`nnue/prepare_data.py`](nnue/prepare_data.py))

The [Lichess evaluation database](https://database.lichess.org/#evals) contains
positions from Lichess together with deep Stockfish evaluations from the cloud
analysis. The script streams the ~22 GB file straight from the URL and keeps
positions that are good training targets for a *static* evaluation:

- search depth ≥ 20
- valid position, not in check
- the best move is neither a capture nor a promotion (only quiet positions,
  tactics are left to the quiescence search)
- scores clipped to ±2000 cp, mates mapped to ±2000, converted to the side to move's point of view

About 64 % of the lines pass these filters. Each position is stored compactly as
the list of its pieces (`piece * 64 + square`), the side to move and the score,
in shards of one million positions. 1 % is held out for validation.

### Architecture ([`nnue/model.py`](nnue/model.py))

```
input: 768 features (12 piece types × 64 squares), one-hot
        │
        ├── from White's view ──► feature transformer (768 → 256) ──┐
        └── from Black's view ──► same weights      (768 → 256) ──┤
                                                                     ▼
                       [side to move | opponent] = 512, clipped ReLU (0..1)
                                                                     ▼
                                          output layer (512 → 1) × 400 = centipawns
```

- **Two perspectives:** the position is fed in once as it is and once with colors
  swapped and the board mirrored. The half of the side to move always comes first,
  so the network learns "my pieces vs. your pieces" instead of White vs. Black.
- **Loss:** mean squared error between `sigmoid(output)` and `sigmoid(score / 400)`.
  That compares win probabilities, so ±100 cp matters a lot in a balanced position
  but barely in a position that is already won.

### Training ([`nnue/train.py`](nnue/train.py))

PyTorch on MPS, 27 million positions held in GPU memory,
Adam (lr 1e-3) with cosine annealing, batch size 16384, 20 epochs of about 40 s each.

| Evaluation | Validation loss | Mean error (\|score\| < 1000) |
|---|---|---|
| always 0 | 0.0615 | – |
| PeSTO (best scale factor) | 0.0260 | 128 cp |
| **NNUE** | **0.0105** | **93 cp** |

Training and validation loss stayed almost identical, so the network does not
overfit; with this size it is limited by its capacity rather than by the data.

### Inference in the engine ([`engine/nnue.py`](engine/nnue.py))

[`nnue/export.py`](nnue/export.py) saves the weights as plain NumPy arrays
(`engine/nnue.npz`, 790 KB), so the engine itself doesn't need PyTorch.
Because the input is sparse (at most 32 active features out of 768), the first
layer is computed as the sum of the weight rows of the pieces on the board instead
of a full matrix multiplication. A test checks that the engine's result matches
PyTorch to within 1 cp.

## Usage

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install the engine (python-chess, numpy)
./engine.sh                      # start the engine (UCI on stdin/stdout)
```

**In a chess GUI:** add `engine.sh` as a local UCI engine (e.g. in En Croissant under *Engines*).

**Tests:**
```bash
uv run --group train pytest -q   # 64 tests; without "--group train" the PyTorch tests are skipped
```

**Training a new network:**
```bash
uv run --group train python -m nnue.prepare_data --positions 27000000   # data -> data/
uv run --group train python -m nnue.train --epochs 20                   # -> checkpoints/<time>/
uv run --group train python -m nnue.export checkpoints/<time>/best.pt  # -> engine/nnue.npz
```

**Testing a change** (needs `tools/bin/fastchess`, the opening book
`tools/books/8moves_v3.pgn` and `brew install stockfish`):
```bash
tools/match.sh snapshot          # before the change: save the current version as reference
tools/match.sh sprt              # after the change: new version vs. reference until the SPRT decides
tools/match.sh accept            # if H1 was accepted: the tested version becomes the new reference
tools/match.sh ref -m "note"     # 200 games against Stockfish 1800 -> tools/results.md
```

**Lichess bot** (via [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)):
```bash
bot/bot.sh setup                 # once: download lichess-bot, then put the API token into bot/.env
bot/bot.sh update                # copy the current engine to bot/engine/
bot/bot.sh start                 # play on Lichess until Ctrl+C
```

## Project structure

```
engine/          the engine (only needs python-chess and numpy)
  uci.py         UCI protocol, search thread
  search.py      search
  evaluate.py    evaluation: NNUE or PeSTO
  nnue.py        NNUE inference in NumPy
  nnue.npz       trained network
  pst.py, tt.py  PeSTO tables, transposition table
nnue/            training pipeline (PyTorch): data, model, training, export
tests/           pytest tests for search, evaluation, TT, UCI and the NNUE pipeline
tools/match.sh   SPRT and Stockfish matches with fastchess
bot/             Lichess bot configuration and scripts
main.py          entry point, engine.sh  start script for GUIs
```

## Possible next steps

- **Incremental accumulator:** update the first layer after each move instead of
  recomputing it, which is what makes NNUE fast in real engines
- A bigger network (e.g. 512 hidden neurons) trained on more data
- Pondering, aspiration windows, static exchange evaluation (SEE)

## Credits

- [python-chess](https://github.com/niklasf/python-chess) for move generation
- PeSTO piece-square tables by Ronald Friederich ([Chess Programming Wiki](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function))
- Training data: [Lichess evaluation database](https://database.lichess.org/#evals) (CC0)
- [fastchess](https://github.com/Disservin/fastchess) and [Stockfish](https://stockfishchess.org/)
  for testing

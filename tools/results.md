# Results against Stockfish with limited strength

Created by tools/match.sh ref. Elo difference = strength relative to Stockfish;
estimated Elo = Stockfish Elo + difference (only rough: Stockfish's UCI_Elo is calibrated at 60+0.6).

| Date | Note | Time control | Opponent | Games | W-D-L (score) | Elo difference | Estimated Elo |
|---|---|---|---|---|---|---|---|
| 2026-11-11 | Phase 1 done | 10+0.1 | SF 1800 | 200 | 101-7-92 (52.25 %) | +16 ± 50 | 1816 |
| 2026-11-12 | Null Move + LMR + PVS | 10+0.1 | SF 1800 | 200 | 96-5-99 (49.25 %) | -5 ± 48 | 1795 |
| 2026-11-17 | NNUE | 10+0.1 | SF 1800 | 200 | 99-7-94 (51.25 %) | +9 ± 50 | 1809 |

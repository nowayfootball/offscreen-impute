# Off-Screen Player Imputation for Broadcast-Based Spatial Football Analytics

Training-free, online imputation of off-camera players, benchmarked by what actually
matters downstream: **pitch-control error and team control-share bias**, not just
trajectory fidelity.

Broadcast cameras show only 10–16 of 22 players. Computing spatial metrics on the
visible subset silently hands the hidden half of the pitch to the wrong team:

![control panels](figures/control_panels.png)
*Left: ground-truth pitch control with all 22 players. Middle: visible-only (current
GSR practice) — the hidden defenders vanish and the right half flips. Right: imputed.*

## Results (Metrica open data, simulated 44 m broadcast viewport, 2 matches)

| policy | hidden-zone control MAE | team share bias | median position error |
|---|---|---|---|
| B0 ignore (status quo) | 26.9 / 25.6 %p | 13.4 / 12.5 %p | — |
| B1 last-seen + decay | 22.1 / 20.0 | 10.6 / 9.5 | 19.6 / 17.9 m |
| B2 formation anchor | 15.7 / 14.6 | 6.2 / 5.4 | 13.6 / 12.8 m |
| B3 + velocity extrapolation | 12.8 / 11.8 | 5.7 / 4.6 | 14.6 / 14.0 m |
| **B4 centroid voting** | **13.3 / 12.2** | **4.7 / 4.5** | **11.6 / 10.0 m** |

**B4 (the interesting one):** each visible player votes for the full-team centroid as
*(position − running role offset)*; the voted centroid replaces the viewport-biased
visible mean both for storing offsets and for imputing. No training data, no future
observations, real-time on CPU. In the short-occlusion regime (≤9.6 s) covered by
learned imputers (e.g. Graph Imputer, trained on 105 proprietary matches with
bidirectional context), B4 reaches 3–9 m median error — while 57% of real hidden-player
mass lies beyond that regime and is scored here for the first time.

![ladder and sensitivity](figures/ladder_sensitivity.png)

## Reproduce (CPU-only, ~2 min per run)

```bash
pip install -r requirements.txt
bash scripts/download_data.sh
python src/impute_bench.py --game 1 --width 44 --fps 5 --minutes 45
python src/impute_bench.py --game 2 --width 44 --fps 5 --minutes 45
# snapshot figure:
python src/impute_bench.py --game 1 --minutes 20 --viz 2500
```

Every number in the paper comes from these commands. Total compute cost: $0.

## Paper

Draft: [`paper/offscreen_impute_EN.md`](paper/offscreen_impute_EN.md)
(Korean version: [`paper/offscreen_impute_KO.md`](paper/offscreen_impute_KO.md))

## Data & license

- Code: MIT.
- Tracking data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data),
  fetched by the download script, **not redistributed here** — see their repository for terms.

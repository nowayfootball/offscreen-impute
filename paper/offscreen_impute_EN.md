# Training-Free Off-Screen Player Imputation for Broadcast-Based Spatial Football Analytics

*Summary aligned with the arXiv source (`paper/latex/main.tex`). Code: `src/impute_bench.py`, `src/pitch_control.py` (CPU-only, open data).*

## Abstract

Spatial football metrics such as pitch control assume access to the positions of all 22 players, yet the most widely available source of positional data — the broadcast main camera — shows only 10–16 of them at any moment. We quantify the resulting distortion with an open, reproducible benchmark: a simulated broadcast viewport applied to open full-pitch tracking data (Metrica Sports; three matches, one held out from method development). Ignoring off-screen players — the visible-only baseline implied whenever a video-based game-state-reconstruction (GSR) pipeline adds no imputation layer — inflates hidden-zone pitch-control error to 25.1–26.9 percentage points and produces a mean absolute control-share error of 11.1–13.4 points across the three matches. We then evaluate a ladder of *training-free, online* imputation baselines that use only observations from the match being analysed. The best overall on the decision-relevant metrics, *role-anchored centroid voting*, roughly halves hidden-zone error (to 12.2–13.8 points) and cuts control-share error to 28–48% of the ignore policy at every viewport width from 36 m to 60 m in all three matches (4.5–4.7 points at W = 44 m). In the short-occlusion regime (≤9.6 s) covered by the closest learned prior work, our training-free method reaches binwise median position errors of 3.3–8.9 m; 50–57% of hidden-player observations under the simulated viewport, however, lie beyond that regime, outside the 9.6 s sequence protocol of that prior work. We integrate the method end-to-end into a broadcast-video GSR pipeline and show that imputation changes a downstream possession-quality score (Space-Creation Index) by 15.6 and 17.2 points on two real World Cup broadcast windows, flipping the verdict class (under prespecified operational thresholds) in one of them.

## 1. Introduction

Video-based game state reconstruction turns ordinary broadcast footage into player coordinates, promising tracking-level analytics without stadium hardware. But the broadcast camera pans and zooms with the ball: in our World Cup clips, and in the SoccerNet-GSR literature, only 10–16 of 22 players are visible per frame. Team-level spatial metrics — pitch control, space value, block compactness — are then computed over a biased subset: the team whose defenders sit off-screen silently cedes control of the hidden half of the pitch. This distortion is routinely acknowledged and, to our knowledge, has not been quantified on open data for downstream team-level metrics.

This paper makes three contributions. **(i) A reproducible distortion benchmark.** We simulate a broadcast viewport (ball-tracking pan, width W) over open full-pitch tracking data and score any imputation policy by hidden-zone pitch-control MAE, team control-share error, and hidden-player position error — metrics defined by the downstream analytics rather than by trajectory fidelity alone. **(ii) A ladder of training-free online baselines**, culminating in role-anchored centroid voting (B4), which halves the hidden-zone control error and cuts control-share error to 28–48% of the ignore-policy baseline, without any training data, future observations, or GPU. **(iii) End-to-end integration** into a video→GSR→metrics pipeline, with a case study on real World Cup broadcasts where imputation moves a possession-quality score by a decision-relevant magnitude, changing the verdict class in one of two windows.

Our position is deliberately modest: learned imputers remain preferable when training data and future context are available. What we establish is how large the distortion is under a simulated viewport, a training-free floor that learned methods should beat, and an explicit scoring of the long-occlusion regime (>9.6 s) that lies outside the sequence protocol of the closest prior work.

## 2. Related Work

**Graph Imputer** (Omidshafiei et al., *Scientific Reports* 2022) is the closest prior work: a graph-network VAE trained on 105 proprietary EPL tracking matches, evaluated with a simulated camera view-cone, and applied to pitch control under partial observability. Two protocol differences preclude direct comparison: their model is bidirectional (it sees *future* observations within 9.6-second windows) and occlusions are truncated by the window length, whereas our setting is strictly online with unbounded occlusions — we show 50–57% of hidden-player observations exceed their window. **MIDAS** (ECML-PKDD 2025) and related set-transformer imputers represent the learned state of the art on trajectory imputation with camera-mask experiments. Event-based continuous tracking (R. Soc. Open Science 2025) reconstructs trajectories from sparse event observations, an adjacent observation model. The *ghosting* line (Le et al., SSAC 2017; NFL Ghosts 2024) generates counterfactual league-average positions for coaching evaluation — a different goal from observation recovery. SoccerNet-GSR and GS-HOTA define recovery quality for *visible* players only; our benchmark scores the invisible remainder.

## 3. Benchmark

**Data.** Metrica Sports open tracking, three matches, 22 players + ball at 25 fps (games 1–2 ship as full-match CSVs; game 3, in EPTS-FIFA format, covers one half). We use the first 45 minutes of each and evaluate at 5 fps. Games 1–2 were used for method and hyperparameter development; game 3 — and the ablation variants B3E/B3V/B5 — were specified and frozen before game 3 was first evaluated, so game 3 serves as a held-out check of the method ladder.

**Viewport.** A virtual main camera pans horizontally following an exponentially smoothed ball position (α = 0.06/frame, reproducing broadcast pan lag); the visible region is a width-W window spanning the full pitch height. W = 44 m yields 14.6–15.0 visible players on average (games 1–3), matching our real broadcast clips (10–16). Sensitivity is reported for W ∈ {36, 44, 52, 60}.

**Metrics.** (1) Hidden-player position error (m), over players outside the viewport that have been observed at least once earlier in the half. (2) Pitch-control map MAE (3 m grid, arrival-time model with sigmoid), reported for the full pitch and for the hidden zone only. (3) Team control-share error (percentage points). Pitch control is computed with zero velocities for all conditions to isolate the imputation effect. Hidden-zone MAE and share error carry 95% block-bootstrap confidence intervals over one-minute blocks (temporal sampling uncertainty within a match, not match-to-match variation).

## 4. Imputation Ladder

- **B0 — ignore**: the visible-only baseline implied whenever no imputation layer is added; hidden players are simply absent from the metric.
- **B1 — last-seen with decay**: hold the last observed position, blending toward the visible-team mean with τ = 8 s.
- **B2 — formation anchor**: while visible, store each player's offset relative to the visible-team centroid; when hidden, place the player at the current visible centroid plus the stored offset.
- **B5 — fixed formation template**: as B2, but the offset is the cumulative mean over all of the player's visible frames — the closest online analogue of a static role template.
- **B3 — B2 + EMA offsets + short-gap velocity extrapolation**; its ingredients are isolated as B3E (EMA only) and B3V (velocity only). These are best read as diagnostic reference-mismatch variants: they estimate offsets against the voted centroid but apply them at the visible centroid.
- **B4 — role-anchored centroid voting**: each visible player *i* votes for the full-team centroid as (pᵢ − offᵢ), where offᵢ is its running role offset; the voted centroid replaces the biased visible mean both when storing offsets (self-consistent bootstrap) and when imputing. This cancels the subset bias that B2 inherits from the viewport. When fewer than three voters are available it falls back to B2, and B2 to B1.

All methods are causal, use only observations from the current match, and run in real time on CPU.

## 5. Results

**Ladder (W = 44 m, three matches; game 3 held out).** Hidden-zone control MAE / share error / median position error:

| | game 1 | game 2 | game 3 (held out) |
|---|---|---|---|
| B0 ignore | 26.9 %p · 13.4 %p · — | 25.6 · 12.5 · — | 25.1 · 11.1 · — |
| B1 last-seen | 22.1 · 10.6 · 19.6 m | 20.0 · 9.5 · 17.9 m | 19.5 · 8.2 · 18.4 m |
| B2 anchor | 15.7 · 6.2 · 13.6 m | 14.6 · 5.4 · 12.8 m | 14.3 · 4.4 · 12.5 m |
| B3 EMA+velocity | 12.8 · 5.7 · 14.6 m | 11.8 · 4.6 · 14.0 m | 13.2 · 4.7 · 15.1 m |
| **B4 centroid vote** | **13.3 · 4.7 · 11.6 m** | **12.2 · 4.5 · 10.0 m** | **13.8 · 4.7 · 9.7 m** |

B4 has the lowest position error in all three matches and the lowest or statistically unresolved control-share error; in the held-out game 3, B2/B3V edge it on share error by 0.3 pp (4.4 vs. 4.7), a gap whose paired block-bootstrap interval [−0.4, +1.3] pp includes zero. B3 attains the lowest hidden-zone MAE, but at worse position error than B4.

**Ablation.** Velocity blending alone (B3V) slightly *improves* position error over B2 in all three matches — short-gap extrapolation is mildly helpful, not harmful. The clearest negative result is the fixed template (B5): far worse than any dynamic anchor (position error 20.7–22.7 m), consistent with within-half role drift.

**Sensitivity.** Across W = 36–60 m the ordering is consistent in all three matches; B4 reduces share error to 28–48% of B0 at every width, with the largest absolute gains in the narrowest (hardest) viewports.

**Occlusion-time stratification (fair comparison to learned imputers).** B4 median position error: 3.3–3.7 m for gaps ≤ 2 s, 7.2–8.9 m for 2–9.6 s, 15.6–16.9 m beyond 9.6 s. Within the ≤9.6 s regime scored by Graph Imputer a training-free online method is competitive with learned bidirectional models; 50–57% (frame-weighted) of hidden-player observations fall beyond that regime — long occlusions of far-side defenders during sustained attacks, precisely where spatial metrics need imputation most.

## 6. Application: possession-quality verdicts on real broadcasts

We integrate a B4-inspired ghosting layer into a broadcast GSR pipeline assembled from open-source components (calibration bridge, BoT-SORT tracking, colour-based team assignment; GS-HOTA 35.2 on the publicly downloadable SoccerNet-GSR sequences, for context only against the 22.26 official test-set baseline) and render off-screen players as fading "ghosts". We evaluate the two event-flagged junk-possession windows for a FIFA World Cup 2026 Round-of-32 match between the Netherlands and Morocco (1–1 after extra time; Morocco won 3–2 on penalties) — flagged from event data alone, before any spatial analysis, and the only flagged windows for the match. On window 1 (game clock 11:58–12:32) imputation moves the Space-Creation Index from +15.6 to +32.8; on window 2 (72:48–73:45) from −10.9 ("dead junk") to +4.7 ("weak progression"), changing the verdict class. Absent full-pitch ground truth for broadcast footage we make no claim that the imputed scores are more accurate; the point is a sensitivity result — visible-only spatial verdicts can move by 15.6–17.2 SCI points under imputation, so possession-quality reported from broadcast video without imputation is sensitive to camera framing and the chosen off-screen-player policy.

## 7. Limitations and Future Work

Identity fragmentation in real GSR output (track fragments, jersey OCR coverage ≈ 45%) makes roster-aware ghosting approximate; appearance re-identification is the natural next step. The pitch-control model is velocity-free here; velocity-aware control with imputed velocities is open. The benchmark uses three matches from a single provider; the simulated viewport pans but does not zoom or tilt. Learned imputers (bidirectional, trained) remain preferable when future context and training data are available — our contribution is the strong training-free floor, the harsher online/long-occlusion benchmark, and the downstream-metric framing.

## 8. Reproducibility

All benchmark experiments run on CPU with open data (Metrica) and open code; scripts and exact commands are in the repository. The broadcast case study uses World Cup footage that cannot be redistributed; the benchmark code is publicly available, while the case-study pipeline and footage are not included in the public repository. Total compute cost of the benchmark experiments: $0.

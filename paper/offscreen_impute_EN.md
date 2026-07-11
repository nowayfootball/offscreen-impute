# Training-Free Off-Screen Player Imputation for Broadcast-Based Spatial Football Analytics

*Draft v1.0 — target: arXiv preprint / SSAC research track. Working notes: `../OFFSCREEN_IMPUTE.md`. Code: `analysis/gsr/impute_bench.py`, `impute_ghost.py` (CPU-only, open data).*

## Abstract

Spatial football metrics such as pitch control assume access to the positions of all 22 players, yet the dominant source of positional data — the broadcast main camera — shows only 10–16 of them at any moment. We quantify the resulting distortion with an open, reproducible benchmark: a simulated broadcast viewport applied to open full-pitch tracking data (Metrica Sports). Ignoring off-screen players, the standard practice of video-based game-state-reconstruction (GSR) pipelines, inflates hidden-zone pitch-control error to 26.9 percentage points and biases team control share by 13.4 points. We then evaluate a ladder of *training-free, online* imputation baselines that use only observations from the match being analysed. The strongest, *role-anchored centroid voting* — each visible player votes for the full-team centroid by subtracting its running role offset, cancelling the viewport-induced subset bias — cuts hidden-zone error to 13.3 points and share bias to 4.7 points, consistently across two matches and viewport widths from 36 m to 60 m. In the short-occlusion regime (≤9.6 s) covered by prior learned imputers, our training-free method reaches median position error of 3–9 m, comparable to learned models that additionally exploit future observations; 57% of real hidden-player mass, however, lies beyond that regime, which our benchmark is the first to score. We integrate the method end-to-end into an open broadcast-video GSR pipeline and show that imputation changes a downstream possession-quality verdict (Space-Creation Index) by 15–17 points on real World Cup broadcast clips — a decision-relevant magnitude.

## 1. Introduction

Video-based game state reconstruction turns ordinary broadcast footage into player coordinates, promising tracking-level analytics without stadium hardware. But the broadcast camera pans and zooms with the ball: in our World Cup clips, and in the SoccerNet-GSR literature, only 10–16 of 22 players are visible per frame. Team-level spatial metrics — pitch control, space value, block compactness — are then computed over a biased subset: the team whose defenders sit off-screen silently cedes control of the hidden half of the pitch.

This paper makes three contributions. **(i) A reproducible distortion benchmark.** We simulate a broadcast viewport (ball-tracking pan, width W) over open full-pitch tracking data and score any imputation policy by hidden-zone pitch-control MAE, team control-share bias, and hidden-player position error — metrics defined by the downstream analytics rather than by trajectory fidelity alone. **(ii) A ladder of training-free online baselines**, culminating in role-anchored centroid voting (B4), which halves the control-map error and cuts share bias to a third without any training data, future observations, or GPU. **(iii) End-to-end integration** into an open video→GSR→metrics pipeline, with a case study where imputation flips a real-match possession-quality verdict.

## 2. Related Work

**Graph Imputer** (Omidshafiei et al., *Scientific Reports* 2022) is the closest prior work: a graph-network VAE trained on 105 proprietary EPL tracking matches, evaluated with a simulated camera view-cone, and applied to pitch control under partial observability. Two protocol differences preclude direct comparison: their model is bidirectional (it sees *future* observations within 9.6-second windows) and occlusions are truncated by the window length, whereas our setting is strictly online with unbounded occlusions — we show 57% of hidden-player observations exceed their window. **MIDAS** (ECML-PKDD 2025) and related set-transformer imputers represent the learned state of the art on trajectory imputation with camera-mask experiments. Event-based continuous tracking (R. Soc. Open Science 2025) reconstructs trajectories from sparse event observations, an adjacent observation model. The *ghosting* line (Le et al., SSAC 2017; NFL Ghosts 2024) generates counterfactual league-average positions for coaching evaluation — a different goal from observation recovery. SoccerNet-GSR and GS-HOTA define recovery quality for *visible* players only; our benchmark scores the invisible remainder.

## 3. Benchmark

**Data.** Metrica Sports open tracking, two full matches, 22 players + ball at 25 fps; first halves (45 min each), evaluated at 5 fps.

**Viewport.** A virtual main camera pans horizontally following an exponentially smoothed ball position (α = 0.06/frame, reproducing broadcast pan lag); the visible region is a width-W window. W = 44 m yields 14.6 visible players on average, matching our real broadcast clips (10–16). Sensitivity is reported for W ∈ {36, 44, 52, 60}.

**Metrics.** (1) Hidden-player position error (m). (2) Pitch-control map MAE (3 m grid, arrival-time model with sigmoid), reported for the full pitch and for the hidden zone only. (3) Team control-share bias (percentage points). Pitch control is computed with zero velocities for all conditions to isolate the imputation effect.

## 4. Imputation Ladder

- **B0 — ignore** (current GSR practice): hidden players are simply absent.
- **B1 — last-seen with decay**: hold the last observed position, blending toward the visible-team mean with τ = 8 s.
- **B2 — formation anchor**: while visible, store each player's offset relative to the visible-team centroid; when hidden, place the player at the current visible centroid plus the stored offset.
- **B3 — B2 + EMA offsets + short-gap velocity extrapolation** (negative result, kept for completeness).
- **B4 — role-anchored centroid voting**: each visible player *i* votes for the full-team centroid as (pᵢ − offᵢ), where offᵢ is its running role offset; the voted centroid replaces the biased visible mean both when storing offsets (self-consistent bootstrap) and when imputing. This cancels the subset bias that B2 inherits from the viewport.

All methods are causal, use only observations from the current match, and run in real time on CPU.

## 5. Results

**Ladder (W = 44 m, two matches).** Hidden-zone control MAE / share bias / median position error:

| | game 1 | game 2 |
|---|---|---|
| B0 ignore | 26.9 %p · 13.4 %p · — | 25.6 · 12.5 · — |
| B1 last-seen | 22.1 · 10.6 · 19.6 m | 20.0 · 9.5 · 17.9 m |
| B2 anchor | 15.7 · 6.2 · 13.6 m | 14.6 · 5.4 · 12.8 m |
| B3 +velocity | 12.8 · 5.7 · 14.6 m | 11.8 · 4.6 · 14.0 m |
| **B4 centroid vote** | **13.3 · 4.7 · 11.6 m** | **12.2 · 4.5 · 10.0 m** |

**Negative result.** Velocity extrapolation degrades position error: players typically leave the viewport moving outward but then hold their line, so extrapolation keeps pushing them away. The simple latest-formation snapshot is the better prior.

**Sensitivity.** Across W = 36–60 m the ordering is monotone and consistent in both matches; B4 reduces share bias to roughly one third of B0 at every width, with the largest absolute gains in the narrowest (hardest) viewports.

**Occlusion-time stratification (fair comparison to learned imputers).** B4 median position error: 3.3–3.7 m for gaps ≤ 2 s, 7.2–8.9 m for 2–9.6 s, ~15.8 m beyond 9.6 s. Within the ≤9.6 s regime scored by Graph Imputer, a training-free online method is thus competitive with learned bidirectional models; 57% of hidden-player observations fall outside that regime and are scored here for the first time.

## 6. Application: possession-quality verdicts on real broadcasts

We integrate B4 into an open broadcast GSR pipeline (calibration bridge, BoT-SORT tracking, colour-based team assignment; official SoccerNet GS-HOTA 35.2 vs. 26.6 baseline) and render off-screen players as fading "ghosts". On two event-flagged junk-possession windows from a World Cup match (Morocco–Netherlands), imputation moves the Space-Creation Index from +15.6 to +32.8 in one window (hidden retreating defenders amplify the genuine space-creation signal) and from −10.9 ("dead junk") to +4.7 ("weak progression") in the other — a verdict-changing shift. Absent full-pitch ground truth we make no correctness claim on real clips; the point is that visible-only spatial verdicts can move by 15–17 SCI points, a magnitude that mandates imputation.

## 7. Limitations and Future Work

Identity fragmentation in real GSR output (track fragments, jersey OCR coverage ≈ 45%) makes roster-aware ghosting approximate; appearance re-identification is the natural next step. The pitch-control model is velocity-free here; velocity-aware control with imputed velocities is open. Learned imputers (bidirectional, trained) remain preferable when future context and training data are available — our contribution is the strong training-free floor, the harsher online/long-occlusion benchmark, and the downstream-metric framing.

## 8. Reproducibility

All experiments run on CPU with open data (Metrica) and an open pipeline; scripts and exact commands are in the repository. Total compute cost of every experiment in this paper: $0.

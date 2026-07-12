"""Off-screen player imputation benchmark with a simulated broadcast viewport.

Applies a virtual broadcast camera (ball-tracking pan, width W meters) to open
full-pitch tracking data (Metrica Sports), masks out-of-view players, imputes them
with a ladder of training-free online baselines, and scores each policy by:
  (1) hidden-player position error,
  (2) pitch-control map MAE vs the full-22 ground truth (full pitch and hidden zone),
  (3) team control-share bias.

Baselines:
  B0 ignore           hidden players simply absent (current GSR practice)
  B1 last-seen        hold last observed position, decay toward visible-team mean
  B2 formation anchor visible-centroid-relative offset snapshot
  B3 B2 + EMA offsets + short-gap velocity extrapolation (negative result)
  B4 centroid voting  each visible player votes for the full-team centroid as
                      (position - running role offset), cancelling viewport bias;
                      offsets are stored against the voted centroid (self-consistent)
Ablations / extra baselines:
  B3e EMA offsets only        (B2 with EMA role offsets, no velocity blend)
  B3v velocity blend only     (B2 snapshot anchor + short-gap velocity blend)
  B5  fixed template          cumulative-mean offsets (static formation template)

Usage:
  python impute_bench.py --game 1 --width 44 --fps 5 --minutes 45 [--viz IDX]
Games 1 and 2 use Metrica's CSV format; game 3 (held-out) uses the EPTS-FIFA format.
Data: run scripts/download_data.sh first (fetches Metrica open data into data/).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from pitch_control import pitch_control

PITCH_L, PITCH_W = 105.0, 68.0
DATA = Path(__file__).resolve().parent.parent / "data"


def load_team(path):
    """Metrica CSV -> (frame -> {pid: (x, y) meters}), (frame -> ball (x, y))."""
    rows = list(csv.reader(open(path)))
    header = rows[2]
    pids = []
    for c in range(3, len(header) - 2, 2):
        name = header[c].strip()
        if name and name != "Ball":
            pids.append((c, name))
    ball_c = len(header) - 2
    frames, ball = {}, {}
    for row in rows[3:]:
        if not row or row[0] != "1":              # first half only
            continue
        f = int(row[1])
        cur = {}
        for c, pid in pids:
            try:
                x, y = float(row[c]), float(row[c + 1])
            except (ValueError, IndexError):
                continue
            if np.isnan(x) or np.isnan(y):
                continue
            cur[pid] = (x * PITCH_L, y * PITCH_W)
        frames[f] = cur
        try:
            bx, by = float(row[ball_c]), float(row[ball_c + 1])
            if not (np.isnan(bx) or np.isnan(by)):
                ball[f] = (bx * PITCH_L, by * PITCH_W)
        except (ValueError, IndexError):
            pass
    return frames, ball


def load_game3():
    """Metrica game 3 (EPTS-FIFA format) -> home/away/ball dicts, same structure
    as the game 1/2 loader. Line format:
      frame : player x,y pairs ';'-separated : ballx,bally
    (normalized 0..1, 25 fps, first-half duration)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(DATA / "g3_meta.xml").getroot()

    def tag(e):
        return e.tag.split("}")[-1]

    pid_team = {}
    for el in root.iter():
        if tag(el) == "Player":
            pid_team[el.attrib["id"]] = el.attrib["teamId"]
    chan_pids = []                                  # playerId per channel (x,y pair order)
    ch2pid = {el.attrib["id"]: el.attrib["playerId"]
              for el in root.iter() if tag(el) == "PlayerChannel"}
    for el in root.iter():
        if tag(el) == "PlayerChannelRef" and el.attrib["playerChannelId"].endswith("_x"):
            chan_pids.append(ch2pid[el.attrib["playerChannelId"]])
    team_ids = sorted({t for t in pid_team.values()})
    home, away, ball = {}, {}, {}
    for line in open(DATA / "g3_tracking.txt"):
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue
        f = int(parts[0])
        cur = {team_ids[0]: {}, team_ids[1]: {}}
        for pid, pair in zip(chan_pids, parts[1].split(";")):
            try:
                x, y = map(float, pair.split(","))
            except ValueError:
                continue
            if np.isnan(x) or np.isnan(y):
                continue
            cur[pid_team[pid]][pid] = (x * PITCH_L, y * PITCH_W)
        home[f], away[f] = cur[team_ids[0]], cur[team_ids[1]]
        try:
            bx, by = map(float, parts[2].split(","))
            if not (np.isnan(bx) or np.isnan(by)):
                ball[f] = (bx * PITCH_L, by * PITCH_W)
        except ValueError:
            pass
    return home, away, ball


def viewport_x(ball, frames_idx, alpha=0.06):
    """Virtual camera pan: EMA of ball x (reproduces broadcast pan lag)."""
    cx, prev, last_ball = {}, PITCH_L / 2, PITCH_L / 2
    for f in frames_idx:
        if f in ball:
            last_ball = ball[f][0]
        prev = prev + alpha * (last_ball - prev)
        cx[f] = prev
    return cx


def b1_impute(last_seen, team_mean, gap_s, tau=8.0):
    """Last-seen position blended toward team mean with time decay."""
    w = np.exp(-gap_s / tau)
    return (w * last_seen[0] + (1 - w) * team_mean[0],
            w * last_seen[1] + (1 - w) * team_mean[1])


def b2_impute(offset, vis_centroid):
    """Formation anchor: stored centroid-relative offset applied to visible centroid."""
    return (vis_centroid[0] + offset[0], vis_centroid[1] + offset[1])


def draw_panels(panels, vis, hid, imps, cx, half_w, out_png):
    """Three-panel control maps with players / ghosts / viewport overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, ny = len(np.arange(1, PITCH_L, 3.0)), len(np.arange(1, PITCH_W, 3.0))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.6))
    for ax, (title, ctrl) in zip(axes, panels):
        ax.imshow(ctrl.reshape(ny, nx), extent=[0, PITCH_L, PITCH_W, 0],
                  cmap="RdBu_r", vmin=0, vmax=1, alpha=0.85)
        ax.axvspan(0, max(cx - half_w, 0), color="k", alpha=0.25)
        ax.axvspan(min(cx + half_w, PITCH_L), PITCH_L, color="k", alpha=0.25)
        for t, mk in ((0, "o"), (1, "s")):
            col = "crimson" if t == 0 else "navy"
            if vis[t]:
                p = np.array(list(vis[t].values()))
                ax.scatter(p[:, 0], p[:, 1], c=col, marker=mk, s=60,
                           edgecolors="white", zorder=5)
            if hid[t]:
                p = np.array(list(hid[t].values()))
                ax.scatter(p[:, 0], p[:, 1], c=col, marker="x", s=50, zorder=5)
            key = title.split()[0].lower()
            for pid, est in imps.get(key, {}).get(t, {}).items():
                gt = hid[t].get(pid)
                ax.scatter([est[0]], [est[1]], facecolors="none", edgecolors=col,
                           marker=mk, s=90, linewidths=1.8, zorder=6)
                if gt is not None:
                    ax.plot([est[0], gt[0]], [est[1], gt[1]], c=col, lw=0.8, alpha=0.6)
        ax.set_title(title)
        ax.set_xlim(0, PITCH_L); ax.set_ylim(PITCH_W, 0)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pitch control - solid=visible / x=hidden GT / hollow=imputed / shade=off-camera")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print("->", out_png)


def block_ci(vals, block=300, n_boot=1000, seed=7):
    """Block bootstrap 95% CI of the mean (1-minute blocks = 300 eval steps @5fps)."""
    v = np.asarray(vals, dtype=float)
    if len(v) < block * 2:
        return None
    blocks = [v[i:i + block] for i in range(0, len(v) - block + 1, block)]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([blocks[j] for j in
                             rng.integers(0, len(blocks), len(blocks))]).mean()
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=1)
    ap.add_argument("--width", type=float, default=44.0, help="viewport width (m)")
    ap.add_argument("--fps", type=int, default=5, help="evaluation sampling fps (raw 25)")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--viz", type=int, default=None,
                    help="render 3-panel PNG at this evaluation-frame index")
    args = ap.parse_args()

    if args.game == 3:
        home, away, ball = load_game3()
    else:
        home, ball = load_team(DATA / f"Sample_Game_{args.game}_RawTrackingData_Home_Team.csv")
        away, _ = load_team(DATA / f"Sample_Game_{args.game}_RawTrackingData_Away_Team.csv")
    step = 25 // args.fps
    dt = step / 25.0
    max_f = min(max(home), int(args.minutes * 60 * 25))
    eval_frames = [f for f in range(step * 10, max_f, step) if f in home and f in away]
    cx = viewport_x(ball, sorted(home.keys()))
    half_w = args.width / 2

    gx, gy = np.meshgrid(np.arange(1, PITCH_L, 3.0), np.arange(1, PITCH_W, 3.0))
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    last_seen = {}     # (team,pid) -> (pos, frame)
    offsets = {}       # (team,pid) -> latest visible-centroid-relative offset (B2)
    offs_ema = {}      # (team,pid) -> EMA role offset against voted centroid (B3/B4)
    offs_cum = {}      # (team,pid) -> B5: fixed template (cumulative-mean offset)
    n_cum = {}
    vels = {}          # (team,pid) -> velocity when visible on consecutive eval steps
    prev_vis = {}
    MODES = ("b0", "b1", "b2", "b3e", "b3v", "b3", "b4", "b5")
    stats = dict(n_vis=[], err_b4_gap=[],
                 ctrl_mae={m: [] for m in MODES}, share_bias={m: [] for m in MODES},
                 ctrl_mae_hid={m: [] for m in MODES},
                 b4_src={"vote": 0, "b2": 0, "b1": 0})  # B4 estimate source (fallback share)
    for m in MODES[1:]:
        stats[f"err_{m}"] = []

    def cen_vote(t, vis_t):
        votes = [(p[0] - offs_ema[(t, pid)][0], p[1] - offs_ema[(t, pid)][1])
                 for pid, p in vis_t.items() if (t, pid) in offs_ema]
        return np.mean(votes, axis=0) if len(votes) >= 3 else None

    for f in eval_frames:
        teams = {0: home[f], 1: away[f]}
        vis, hid = {0: {}, 1: {}}, {0: {}, 1: {}}
        for t, players in teams.items():
            for pid, p in players.items():
                if abs(p[0] - cx[f]) <= half_w:
                    vis[t][pid] = p
                    last_seen[(t, pid)] = (p, f)
                else:
                    hid[t][pid] = p
        # update offsets from *visible* players only (matches deployment conditions)
        for t in (0, 1):
            if len(vis[t]) >= 3:
                cen = np.mean(list(vis[t].values()), axis=0)
                for pid, p in vis[t].items():
                    offsets[(t, pid)] = (p[0] - cen[0], p[1] - cen[1])
                cv = cen_vote(t, vis[t])
                ref = cv if cv is not None else cen
                for pid, p in vis[t].items():
                    off = (p[0] - ref[0], p[1] - ref[1])
                    o = offs_ema.get((t, pid), off)
                    offs_ema[(t, pid)] = (0.9 * o[0] + 0.1 * off[0],
                                          0.9 * o[1] + 0.1 * off[1])
                for pid, p in vis[t].items():     # B5: cumulative mean (fixed template)
                    key = (t, pid)
                    off = (p[0] - cen[0], p[1] - cen[1])
                    n = n_cum.get(key, 0) + 1
                    c0 = offs_cum.get(key, off)
                    offs_cum[key] = (c0[0] + (off[0] - c0[0]) / n,
                                     c0[1] + (off[1] - c0[1]) / n)
                    n_cum[key] = n
            for pid, p in vis[t].items():
                key = (t, pid)
                if key in prev_vis and prev_vis[key][1] == f - step:
                    q = prev_vis[key][0]
                    vels[key] = ((p[0] - q[0]) / dt, (p[1] - q[1]) / dt)
                prev_vis[key] = (p, f)

        stats["n_vis"].append(len(vis[0]) + len(vis[1]))

        def players_list(imp_mode, imp_out=None):
            out = []
            for t in (0, 1):
                for pid, p in vis[t].items():      # zero velocity: isolate imputation effect
                    out.append((p[0], p[1], 0.0, 0.0, t))
                if imp_mode == "b0":
                    continue
                cen = (np.mean(list(vis[t].values()), axis=0)
                       if vis[t] else np.array([PITCH_L / 2, PITCH_W / 2]))
                for pid, gt_p in hid[t].items():
                    key = (t, pid)
                    if imp_mode == "b1":
                        if key not in last_seen:
                            continue
                        ls, lf = last_seen[key]
                        est = b1_impute(ls, cen, (f - lf) / 25.0)
                    elif imp_mode == "b2":
                        if key in offsets:
                            est = b2_impute(offsets[key], cen)
                        elif key in last_seen:
                            est = b1_impute(last_seen[key][0], cen,
                                            (f - last_seen[key][1]) / 25.0)
                        else:
                            continue
                    elif imp_mode == "b5":         # fixed template (cumulative-mean offset)
                        if key in offs_cum:
                            est = b2_impute(offs_cum[key], cen)
                        elif key in last_seen:
                            est = b1_impute(last_seen[key][0], cen,
                                            (f - last_seen[key][1]) / 25.0)
                        else:
                            continue
                    elif imp_mode == "b3e":        # ablation: B2 + EMA offsets only
                        if key in offs_ema:
                            est = b2_impute(offs_ema[key], cen)
                        elif key in last_seen:
                            est = b1_impute(last_seen[key][0], cen,
                                            (f - last_seen[key][1]) / 25.0)
                        else:
                            continue
                    elif imp_mode == "b3v":        # ablation: B2 snapshot + velocity blend only
                        if key in offsets:
                            anchor = b2_impute(offsets[key], cen)
                        elif key in last_seen:
                            anchor = b1_impute(last_seen[key][0], cen,
                                               (f - last_seen[key][1]) / 25.0)
                        else:
                            continue
                        est = anchor
                        if key in last_seen and key in vels:
                            gap_s = (f - last_seen[key][1]) / 25.0
                            w = float(np.exp(-gap_s / 1.5))
                            ex = (last_seen[key][0][0] + vels[key][0] * gap_s,
                                  last_seen[key][0][1] + vels[key][1] * gap_s)
                            est = (w * ex[0] + (1 - w) * anchor[0],
                                   w * ex[1] + (1 - w) * anchor[1])
                    elif imp_mode == "b4":
                        cv = cen_vote(t, vis[t])
                        if cv is not None and key in offs_ema:
                            est = (cv[0] + offs_ema[key][0], cv[1] + offs_ema[key][1])
                            stats["b4_src"]["vote"] += 1
                        elif key in offsets:
                            est = b2_impute(offsets[key], cen)
                            stats["b4_src"]["b2"] += 1
                        elif key in last_seen:
                            est = b1_impute(last_seen[key][0], cen,
                                            (f - last_seen[key][1]) / 25.0)
                            stats["b4_src"]["b1"] += 1
                        else:
                            continue
                    else:                          # b3
                        anchor = (b2_impute(offs_ema[key], cen) if key in offs_ema
                                  else (b1_impute(last_seen[key][0], cen,
                                                  (f - last_seen[key][1]) / 25.0)
                                        if key in last_seen else None))
                        if anchor is None:
                            continue
                        est = anchor
                        if key in last_seen and key in vels:
                            gap_s = (f - last_seen[key][1]) / 25.0
                            w = float(np.exp(-gap_s / 1.5))
                            ex = (last_seen[key][0][0] + vels[key][0] * gap_s,
                                  last_seen[key][0][1] + vels[key][1] * gap_s)
                            est = (w * ex[0] + (1 - w) * anchor[0],
                                   w * ex[1] + (1 - w) * anchor[1])
                    e = float(np.hypot(est[0] - gt_p[0], est[1] - gt_p[1]))
                    stats[f"err_{imp_mode}"].append(e)
                    if imp_mode == "b4" and key in last_seen:
                        stats["err_b4_gap"].append(((f - last_seen[key][1]) / 25.0, e))
                    out.append((est[0], est[1], 0.0, 0.0, t))
                    if imp_out is not None:
                        imp_out.setdefault(t, {})[pid] = est
            return out

        full = [(p[0], p[1], 0.0, 0.0, t) for t in (0, 1) for p in teams[t].values()]
        ctrl_gt = pitch_control(full, grid)
        if ctrl_gt is None:
            continue
        ctrls, imps = {}, {}
        for mode in MODES:
            imps[mode] = {}
            pl = players_list(mode, imps[mode])
            if len({p[4] for p in pl}) < 2:
                continue
            c = pitch_control(pl, grid)
            ctrls[mode] = c
            stats["ctrl_mae"][mode].append(float(np.mean(np.abs(c - ctrl_gt))) * 100)
            stats["share_bias"][mode].append((float(np.mean(c)) - float(np.mean(ctrl_gt))) * 100)
            hidmask = np.abs(grid[:, 0] - cx[f]) > half_w
            if hidmask.any():
                stats["ctrl_mae_hid"][mode].append(
                    float(np.mean(np.abs(c[hidmask] - ctrl_gt[hidmask]))) * 100)
        if (args.viz is not None and eval_frames.index(f) == args.viz
                and "b0" in ctrls and "b4" in ctrls):
            draw_panels([("GT (full 22)", ctrl_gt), ("B0 visible-only", ctrls["b0"]),
                         ("B4 imputed", ctrls["b4"])],
                        vis, hid, imps, cx[f], half_w,
                        str(DATA / f"impute_viz_g{args.game}_f{f}.png"))

    print(f"game {args.game} · first {args.minutes} min · viewport {args.width} m "
          f"· {len(eval_frames)} eval frames")
    print(f"visible players: mean {np.mean(stats['n_vis']):.1f}/22 "
          f"(median {np.median(stats['n_vis']):.0f})")
    for m in MODES[1:]:
        if stats[f"err_{m}"]:
            e = stats[f"err_{m}"]
            print(f"{m.upper()} position error: median {np.median(e):.2f} m "
                  f"· mean {np.mean(e):.2f} m")
    if stats["err_b4_gap"]:
        for lo, hi, tag in ((0, 2, "<=2s"), (2, 9.6, "2-9.6s"), (9.6, 1e9, ">9.6s")):
            es = [e for g, e in stats["err_b4_gap"] if lo < g <= hi]
            if es:
                print(f"  B4 occlusion {tag:7s}: median {np.median(es):.2f} m (n={len(es)})")
    for mode in MODES:
        m = stats["ctrl_mae"][mode]
        h = stats["ctrl_mae_hid"][mode]
        s = stats["share_bias"][mode]
        if m:
            print(f"{mode.upper()}: control MAE {np.mean(m):.2f}%p "
                  f"(hidden zone {np.mean(h):.2f}%p) "
                  f"· team share bias |{np.mean(np.abs(s)):.2f}|%p")
    src = stats["b4_src"]
    tot = sum(src.values())
    if tot:
        print("B4 estimate source: "
              + " · ".join(f"{k} {v/tot*100:.1f}%" for k, v in src.items()))
    for m in ("b0", "b4"):                # 1-minute block bootstrap 95% CI
        ci_h = block_ci(stats["ctrl_mae_hid"][m], block=60 * args.fps)
        ci_s = block_ci(np.abs(stats["share_bias"][m]), block=60 * args.fps)
        if ci_h and ci_s:
            print(f"{m.upper()} 95% CI: hidden MAE [{ci_h[0]:.1f}, {ci_h[1]:.1f}]%p "
                  f"· share err [{ci_s[0]:.1f}, {ci_s[1]:.1f}]%p")
    for m in ("b2", "b3v"):  # paired difference CI: B4 - m (share error)
        a, b = stats["share_bias"]["b4"], stats["share_bias"][m]
        if len(a) == len(b):
            diff = np.abs(a) - np.abs(b)
            ci = block_ci(diff, block=60 * args.fps)
            if ci:
                print(f"paired share err B4-{m.upper()} 95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}]%p")


if __name__ == "__main__":
    main()

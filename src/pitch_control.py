"""Minimal arrival-time pitch control model.

Each player starts from an inertia-adjusted position (current position advanced by
T_REACT seconds of current velocity), then moves at V_MAX. Team control probability
at each grid cell is a sigmoid of the difference between the two teams' minimum
arrival times.
"""
import numpy as np

V_MAX = 7.8      # m/s
T_REACT = 0.7    # s
SIG_SCALE = 0.45


def pitch_control(players, grid_xy):
    """players: [(x, y, vx, vy, team)] with team in {0,1}; grid (N,2) in meters.
    Returns team-0 control probability per grid cell (N,), or None if a team is empty."""
    t_min = {0: None, 1: None}
    for x, y, vx, vy, team in players:
        ox, oy = x + vx * T_REACT, y + vy * T_REACT
        t = T_REACT + np.hypot(grid_xy[:, 0] - ox, grid_xy[:, 1] - oy) / V_MAX
        if t_min[team] is None:
            t_min[team] = t
        else:
            np.minimum(t_min[team], t, out=t_min[team])
    if t_min[0] is None or t_min[1] is None:
        return None
    return 1.0 / (1.0 + np.exp((t_min[0] - t_min[1]) / SIG_SCALE))

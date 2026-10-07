"""Classify a polar-wander trajectory: onset, episodes, final state.

Replaces the ad hoc metrics of the 2026-10-05 run scripts, which (Fermi
review) measured misalignment against the NEAREST principal axis and called a
pole "settled" when it merely stopped moving, so a pole parked on a saddle
(the mid axis) counted as settled. Here:
  * misalignment is always measured against the MAX-moment axis;
  * "settled" requires misalignment from the max axis < settle_misalign_deg
    AND a quiet final window; a quiet pole off the max axis is "parked".

Axes are undirected: angles between a pole and an axis use |cos|.
"""
import numpy as np


def axis_angle_deg(u, v):
    """Angle between two undirected axes [deg] (rows or single vectors)."""
    c = np.abs(np.sum(np.asarray(u) * np.asarray(v), axis=-1))
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


def summarize(t_yr, spin, v_max, onset_deg=5.0, settle_misalign_deg=1.0,
              quiet_rate_deg_per_myr=1.0, min_episode_deg=5.0, merge_gap_myr=1.0,
              final_window_frac=0.05, final_window_min_myr=2.0):
    """Summarize a pole track.

    Parameters
    ----------
    t_yr : (N,) times [yr], increasing
    spin : (N, 3) unit spin vectors (body frame)
    v_max : (N, 3) max-moment axis at each record (any sign)

    Returns dict with
      onset_Myr       first time the pole is > onset_deg from its t=0 position
      travel_max_deg  largest angle from the t=0 pole
      arc_deg         total path length
      misalign_end_deg / misalign_start_deg   angle to the MAX axis
      final_state     'settled' | 'parked' (quiet, off the max axis) | 'moving'
      settle_Myr      (settled only) last time the pole was > 1 deg from its
                      final position
      episodes        [(t_start_Myr, t_end_Myr, arc_deg)], runs of rate >
                      quiet_rate separated by > merge_gap_myr, arc >= min_episode_deg
      history         'none' | 'single' | 'episodic' (>= 2 episodes) |
                      'continuous' (one episode, still running at the last record)
      final_colat_deg, final_lon_deg   final pole, body frame (lon east of body x)
    """
    t = np.asarray(t_yr, dtype=float) / 1e6
    s = np.asarray(spin, dtype=float)
    vm = np.asarray(v_max, dtype=float)
    if len(t) < 2:
        raise ValueError('summarize needs at least two records')
    from0 = axis_angle_deg(s, s[0])
    step = axis_angle_deg(s[1:], s[:-1])
    dt = np.diff(t)
    rate = np.where(dt > 0, step / np.where(dt > 0, dt, 1.0), 0.0)
    moving = rate > quiet_rate_deg_per_myr
    # episodes: runs of moving intervals, merged across short quiet gaps
    runs, i = [], 0
    while i < len(moving):
        if moving[i]:
            j = i
            while j + 1 < len(moving) and moving[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and t[r[0]] - t[merged[-1][1] + 1] <= merge_gap_myr:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    episodes = []
    for a, b in merged:
        arc = float(step[a:b + 1].sum())
        if arc >= min_episode_deg:
            episodes.append((float(t[a]), float(t[b + 1]), arc))
    win = max(final_window_frac * (t[-1] - t[0]), final_window_min_myr)
    fin = t >= t[-1] - win
    fin_iv = fin[1:] & fin[:-1]
    quiet_end = not np.any(moving[fin_iv]) if fin_iv.any() else not moving[-1]
    mis_end = float(axis_angle_deg(s[-1], vm[-1]))
    if quiet_end and mis_end < settle_misalign_deg:
        final_state = 'settled'
    elif quiet_end:
        final_state = 'parked'
    else:
        final_state = 'moving'
    settle = None
    if final_state == 'settled':
        far = axis_angle_deg(s, s[-1]) > 1.0
        settle = float(t[np.where(far)[0][-1] + 1]) if far.any() else float(t[0])
    if not episodes:
        history = 'none'
    elif len(episodes) == 1:
        # 'continuous' only if the one episode is still running at the end;
        # a slow sub-episode drift after it ended stays 'single' (final_state
        # then reports 'moving')
        history = 'continuous' if episodes[0][1] >= t[-1] else 'single'
    else:
        history = 'episodic'
    onset = float(t[np.argmax(from0 > onset_deg)]) if (from0 > onset_deg).any() else None
    return {
        'onset_Myr': onset,
        'travel_max_deg': float(from0.max()),
        'arc_deg': float(step.sum()),
        'misalign_start_deg': float(axis_angle_deg(s[0], vm[0])),
        'misalign_end_deg': mis_end,
        'final_state': final_state,
        'settle_Myr': settle,
        'episodes': episodes,
        'n_episodes': len(episodes),
        'history': history,
        'max_rate_deg_per_myr': float(rate.max()) if len(rate) else 0.0,
        'final_colat_deg': float(np.degrees(np.arccos(np.clip(s[-1, 2], -1, 1)))),
        'final_lon_deg': float(np.degrees(np.arctan2(s[-1, 1], s[-1, 0]))),
    }

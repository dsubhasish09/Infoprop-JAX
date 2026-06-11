"""Wheelbot racing tracks scaled up to humanoid proportions.

Loads the same 200 pre-generated tracks as the Wheelbot env and scales all xy
coordinates (and the track width) by the ratio of the two robots' nominal
body-centre heights, so the humanoid races tracks of equivalent relative size.
"""
import jax.numpy as jp

from infoprop_jax.envs.wheelbot.assets.track.generator import load_track_by_seed
from infoprop_jax.envs.wheelbot.trajectory import (
    Trajectory,
    pad_line_segments_to_size,
    points_to_line_segemnt,
)

# Nominal free-joint resting heights (humanoid torso / wheelbot body, from the MJCF models).
HUMANOID_NOMINAL_Z = 1.282
WHEELBOT_NOMINAL_Z = 0.0645
TRACK_SCALE = HUMANOID_NOMINAL_Z / WHEELBOT_NOMINAL_Z

NUM_TRACKS = 200

# Create scaled trajectories statically to be used in reset/observation methods.
tracks = [load_track_by_seed(i) for i in range(NUM_TRACKS)]
track_width = float(tracks[0]['width']) * TRACK_SCALE
# Scale before segment/padding construction so the padding sentinels stay untouched.
_centerlines = [track['centerline'] * TRACK_SCALE for track in tracks]
_trajectories = [points_to_line_segemnt(centerline) for centerline in _centerlines]
trajectory_lengths = jp.array([t.shape[0] for t in _trajectories])
_max_length = max(t.shape[0] for t in _trajectories)
trajectories_flattened = jp.array(
    [pad_line_segments_to_size(t, _max_length) for t in _trajectories]
)


def get_trajectory_by_seed(track_seed: int, lookahead: int = 10) -> Trajectory:
    """Load the scaled pre-generated track trajectory for the given integer seed."""
    traj_flattened = trajectories_flattened[track_seed]
    size = trajectory_lengths[track_seed]
    return Trajectory(traj_flattened, size, lookahead)


def scaled_cones(track_seed: int):
    """Return the scaled (inner, outer) boundary cones; concrete seed, XML injection only."""
    track = tracks[track_seed]
    return track['inner_cones'] * TRACK_SCALE, track['outer_cones'] * TRACK_SCALE

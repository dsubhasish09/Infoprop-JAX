"""
Track trajectory representation and query utilities.

A Trajectory stores the racing track centreline as a sequence of line segments
and provides JAX-compatible methods to compute the task-relevant state features:
cross-track error, cross-angle error, and lookahead waypoint coordinates.

These features form the task state s^task used by the policy and dynamics model.
"""
import jax.numpy as jp
from jax import vmap, tree_util, jit
import jax
from dataclasses import dataclass


def distance_to_segment(pos_xy: jp.ndarray, a: jp.ndarray, b: jp.ndarray):
    # Compute the projection of pos onto the segment a_b
    a_b = b - a
    a_pos = pos_xy - a
    t = jp.dot(a_pos, a_b) / jp.dot(a_b, a_b)
    t = jp.clip(t, 0, 1)  # Clamp to segment
    closest = a + t * a_b
    return jp.linalg.norm(pos_xy - closest)


def angle_of_segment(a: jp.ndarray, b: jp.ndarray):
    # computes angle of line segment a->b to x axis. returns angle in radians in [-π, π]
    d = b - a
    return jp.arctan2(d[1], d[0])


def angle_difference(angle1, angle2):
    # computes the angle to add to angle2 so that it aligns with angle1. Result lies within (-pi, pi)
    diff = angle1 - angle2
    return (diff + jp.pi) % (2 * jp.pi) - jp.pi


def points_to_line_segemnt(points: jp.ndarray) -> jp.ndarray:
    """
    Convert array of points to array of line segments i.e. pairs of points
    input cannot include any padding!
    """
    next_points = jp.roll(points, shift=-1, axis=0)
    line_segment = jp.stack([points, next_points], axis=1)
    return line_segment


def pad_line_segments_to_size(points: jp.ndarray, size: int) -> jp.ndarray:
    """
    Pads a (N, 2, 2) array of point pairs with [[1e10, 1e10], [1e10, 1e10]] rows until it
    reaches the given size.
    """
    n = points.shape[0]
    if n > size:
        raise Exception(
            f"Cannot pad points of size {n} to size {size}. "
            "Points already have more elements than size."
        )
    pad_count = size - n
    padding_pair = jp.array([[1e9, 1e9], [1e10, 1e10]])
    pad = jp.tile(padding_pair[None, :, :], (pad_count, 1, 1))
    return jp.concatenate([points, pad], axis=0)


@dataclass(frozen=True)
class Trajectory:
    """Immutable segment-based representation of a racing track centreline.

    Attributes:
        line_segments: (N, 2, 2) array of consecutive segment endpoint pairs.
        centerline: (N, 2) array of centreline waypoints.
        lookahead: Number of upcoming waypoints to include in the observation.
        track_width: Width of the track (used for reward normalisation).
    """

    line_segments: jp.ndarray
    size: int
    lookahead: int = 10

    def get_next_points(self, index: int) -> jp.ndarray:
        """Return the next `lookahead` centreline waypoints starting from the closest segment."""
        size = self.size
        indices = (jp.arange(0, self.lookahead) + index) % size
        next_line_segments = self.line_segments[indices]
        next_points = next_line_segments[:, 0]
        return next_points

    def find_closest_segment_index(self, pos_xy):
        dists = vmap(lambda pair: distance_to_segment(pos_xy, *pair))(self.line_segments)
        return jp.argmin(dists)

    def cross_track_error(self, pos_xy):
        """Signed minimum distance from robot_pos to the nearest track segment."""
        dists = vmap(lambda pair: distance_to_segment(pos_xy, *pair))(self.line_segments)
        return jp.min(dists)

    def cross_angle_error(self, pos_xy, yaw_angle):
        """Difference between the robot heading and the nearest track segment direction."""
        min_i = self.find_closest_segment_index(pos_xy)
        min_line_segment = self.line_segments[min_i]
        min_line_segment_angle = angle_of_segment(*min_line_segment)
        return angle_difference(min_line_segment_angle, yaw_angle)

    def get_next_distances(self, pos_xy):
        # compute closest line segment
        min_i = self.find_closest_segment_index(pos_xy)
        # index of the endpoint of the closest line segment
        next_i = min_i + 1
        next_points = self.get_next_points(next_i)
        distances = vmap(lambda point: jp.linalg.norm(point - pos_xy))(next_points)
        return distances

    def get_next_angles(self, pos_xy, yaw_angle):
        # compute closest line segment
        min_i = self.find_closest_segment_index(pos_xy)
        # index of the endpoint of the closest line segment
        next_i = min_i + 1
        next_points = self.get_next_points(next_i)
        # compute angles of direction for the next points
        direction_angles = vmap(lambda point: angle_of_segment(pos_xy, point))(next_points)
        # compute relative angle i.e. how must the yaw change to align with direction angle
        relative_angles = vmap(lambda angle: angle_difference(angle, yaw_angle))(direction_angles)
        return relative_angles

    def get_sin_cos_next_angles(self, pos_xy, yaw_angle):
        angles = self.get_next_angles(pos_xy, yaw_angle)
        sin_cos = jp.stack([jp.sin(angles), jp.cos(angles)], axis=1)
        return sin_cos.flatten()

    def get_state(self, pos_xy, yaw_angle, sin_cos_encoding: bool = False):
        """Return the full trajectory feature vector for the current robot pose.

        Concatenates: [cross_track_error, cross_angle_error (or sin/cos),
                       lookahead_distances, lookahead_angles (or sin/cos)].

        Args:
            robot_pos: (2,) global XY position.
            robot_yaw: Scalar heading angle.
            sin_cos_encoding: If True, encode angles as (sin, cos) pairs.

        Returns:
            Feature vector of shape (22,) or (32,) depending on sin_cos_encoding.
        """
        e_ct = self.cross_track_error(pos_xy)
        e_ca = self.cross_angle_error(pos_xy, yaw_angle)
        distances = self.get_next_distances(pos_xy)
        angles = (
            self.get_next_angles(pos_xy, yaw_angle)
            if not sin_cos_encoding
            else self.get_sin_cos_next_angles(pos_xy, yaw_angle)
        )
        if sin_cos_encoding:
            return jp.array([e_ct, jp.sin(e_ca), jp.cos(e_ca), *distances, *angles])
        else:
            return jp.array([e_ct, e_ca, *distances, *angles])

    def get_init_pos(self, index: int):
        """Return the robot position and yaw corresponding to segment index `idx`."""
        index = index % self.size
        init_line_segment = self.line_segments[index]
        init_xy = init_line_segment[0]
        init_angle = angle_of_segment(*init_line_segment)
        return init_xy, init_angle

    def get_rand_init_pos(self, rng: jp.ndarray):
        """Sample a random initial position and yaw from the centreline."""
        index = jax.random.randint(rng, (), 0, self.size)
        init_xy, init_angle = self.get_init_pos(index)
        return init_xy, init_angle


def check_equal(a, b):
    assert jp.allclose(a, b, atol=1e-6), f'{a} =! {b}'


def main():
    points = jp.array([[2, -2], [2, -5], [5, -5], [5, -2]])
    line_segments = points_to_line_segemnt(points)
    padded_line_segments = pad_line_segments_to_size(line_segments, 6)

    traj = Trajectory(padded_line_segments, 4, 4)

    pos_xy = jp.array([4, -1])
    yaw = -jp.pi
    init_xy_0, init_angle_0 = traj.get_init_pos(0)
    init_xy_1, init_angle_1 = traj.get_init_pos(1)
    init_xy_2, init_angle_2 = traj.get_init_pos(2)
    init_xy_3, init_angle_3 = traj.get_init_pos(3)

    rng = jax.random.PRNGKey(3)
    init_xy_4, init_angle_4 = traj.get_rand_init_pos(rng)

    check_equal(init_xy_0, jp.array([2, -2]))
    check_equal(init_angle_0, -jp.pi / 2)
    check_equal(init_xy_1, jp.array([2, -5]))
    check_equal(init_angle_1, 0)
    check_equal(init_xy_2, jp.array([5, -5]))
    check_equal(init_angle_2, jp.pi / 2)
    check_equal(init_xy_3, jp.array([5, -2]))
    check_equal(init_angle_3, jp.pi)
    check_equal(traj.cross_track_error(pos_xy), 1)
    check_equal(traj.cross_angle_error(pos_xy, yaw), 0)
    check_equal(
        traj.get_next_distances(pos_xy),
        jp.array([jp.sqrt(5), jp.sqrt(20), jp.sqrt(17), jp.sqrt(2)]),
    )
    check_equal(traj.get_next_angles(pos_xy, yaw)[3], 3 * jp.pi / 4)

    points = jp.array([[3, 2], [3, 3], [2, 3], [1, 3], [1, 2], [1, 1], [2, 1], [3, 1]]) * 0.001
    line_segments = points_to_line_segemnt(points)
    padded_line_segments = pad_line_segments_to_size(line_segments, 20)
    traj = Trajectory(padded_line_segments, 8, 8)
    pos_xy = jp.array([2, 2]) * 0.001
    yaw = 0

    check_equal(traj.cross_track_error(pos_xy), 0.001)
    check_equal(traj.cross_angle_error(pos_xy, yaw), jp.pi / 2)
    check_equal(
        traj.get_next_distances(pos_xy),
        jp.array([jp.sqrt(2), 1, jp.sqrt(2), 1, jp.sqrt(2), 1, jp.sqrt(2), 1]) * 0.001,
    )
    check_equal(
        traj.get_next_angles(pos_xy, yaw),
        jp.array([
            jp.pi / 4, jp.pi / 2, 3 * jp.pi / 4, -jp.pi,
            -3 * jp.pi / 4, -jp.pi / 2, -jp.pi / 4, 0,
        ]),
    )

    points = jp.array([[0, 0], [2, 2], [4, 0], [2, -2]])
    line_segments = points_to_line_segemnt(points)
    padded_line_segments = pad_line_segments_to_size(line_segments, 20)
    traj = Trajectory(padded_line_segments, 4, 4)
    init_xy_0, init_angle_0 = traj.get_init_pos(0)
    init_xy_1, init_angle_1 = traj.get_init_pos(1)
    init_xy_2, init_angle_2 = traj.get_init_pos(2)
    init_xy_3, init_angle_3 = traj.get_init_pos(3)

    check_equal(init_xy_0, jp.array([0, 0]))
    check_equal(init_angle_0, jp.pi / 4)
    check_equal(init_xy_1, jp.array([2, 2]))
    check_equal(init_angle_1, -jp.pi / 4)
    check_equal(init_xy_2, jp.array([4, 0]))
    check_equal(init_angle_2, -3 * jp.pi / 4)
    check_equal(init_xy_3, jp.array([2, -2]))
    check_equal(init_angle_3, 3 * jp.pi / 4)

    print('All tests passed')


if __name__ == "__main__":
    main()

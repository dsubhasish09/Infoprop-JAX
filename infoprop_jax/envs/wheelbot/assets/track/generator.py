"""
Procedural racing track generation using Catmull-Rom splines.

Generates closed-loop tracks by:
  1. Sampling random waypoints in a bounded area.
  2. Computing the convex hull and injecting non-convexity.
  3. Interpolating with Catmull-Rom splines for smooth centrelines.
  4. Validating geometry (angle constraints, no self-intersections).
  5. Scaling and discretising boundaries to cone positions.

Pre-generated tracks are bundled in this package and loaded at training time via
load_track_by_seed(). New tracks can be generated with create_track().
"""

import numpy as np
from shapely.geometry import MultiPoint, LinearRing, LineString, Point, Polygon
import matplotlib.pyplot as plt
from time import time
import jax.numpy as jp
import time
from pathlib import Path


TRACK_ASSET_DIR = Path(__file__).resolve().parent / "saved_tracks"

class BadTrackException(Exception):
    pass


def push_apart(pts, dist=.05, reps=3):
    for _ in range(reps):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                h = pts[j] - pts[i]
                h_dist = np.linalg.norm(h)
                if h_dist < dist:
                    h = h / h_dist * (dist - h_dist)
                    pts[j] += h
                    pts[i] -= h


def make_non_convex(pts, offset=.2, dist=.05, rng=None):
    # If given a closed path, remove the end-point, we already handle it, and it will cause issues
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    if rng is None:
        rng = np.random.default_rng()

    # positive rolls towards right, i.e. roll([1, 2, 3], 1) -> [3, 1, 2]
    # negative rolls towards left, i.e. roll([1, 2, 3], -1) -> [2, 3, 1]
    lerps = rng.random((len(pts), 1)) * .5 + .25
    lerp_points = (pts * lerps + np.roll(pts, -1, axis=0) * (1 - lerps))
    dists = pts - np.roll(pts, -1, axis=0)
    orthos = np.stack([dists[:, 1], -dists[:, 0]], axis=-1)
    norms = np.linalg.norm(orthos, axis=-1)
    orthos /= norms[:, None]
    new_pts = lerp_points + orthos * (rng.random(size=(len(pts), 1)) * 2 * offset - offset)

    result = np.empty((len(pts) * 2, 2), dtype=pts.dtype)
    result[::2] = pts
    result[1::2] = new_pts

    push_apart(result, dist=dist)

    return result


def catmull_rom_closed(points, alpha=.5, num_points=20):
    if np.allclose(points[0], points[-1]):
        raise ValueError("Unexpected repetition of start point")

    p0 = np.roll(points, 2, axis=0)
    a = np.roll(points, 1, axis=0)
    b = points
    p3 = np.roll(points, -1, axis=0)

    def tj(ti, pi, pj):
        delta = pi - pj
        lengths = np.linalg.norm(delta, axis=-1)
        return ti + lengths ** alpha

    t0 = np.zeros(len(points))
    t1 = tj(t0, p0, a)
    t2 = tj(t1, a, b)
    t3 = tj(t2, b, p3)

    t0 = t0[:, None, None]
    t1 = t1[:, None, None]
    t2 = t2[:, None, None]
    t3 = t3[:, None, None]

    t = np.linspace(0, 1, num_points)[:, None] * (t2 - t1) + t1
    # t shape is (num_segments, num_points, 1) with last reserved for num_dimensions=2

    a1 = (t1 - t) / (t1 - t0) * p0[:, None, :] + (t - t0) / (t1 - t0) * a[:, None, :]
    a2 = (t2 - t) / (t2 - t1) * a[:, None, :] + (t - t1) / (t2 - t1) * b[:, None, :]
    a3 = (t3 - t) / (t3 - t2) * b[:, None, :] + (t - t2) / (t3 - t2) * p3[:, None, :]
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    points = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    return points.reshape(-1, 2)


def make_track_centerline(extends, min_dist=15., offset=20., rng=None):
    """Generate a smooth closed-loop centreline from random waypoints via Catmull-Rom spline."""
    if rng is None:
        rng = np.random.default_rng()

    points = rng.uniform(0, 1, size=(rng.integers(10, 20, endpoint=True), 2)) * extends
    push_apart(points, dist=min_dist)

    hull = np.array(MultiPoint(points).convex_hull.exterior.coords.xy).T
    track_pts = make_non_convex(hull, offset=offset, dist=15, rng=rng)
    track_cm = catmull_rom_closed(track_pts)

    return track_cm


def calculate_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2

    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    cosine_angle = dot_product / magnitude_product
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle numerical precision issues

    return np.degrees(angle)


def check_angles(pts):
    """Validate that all track angles are within acceptable bounds (no hairpin turns)."""
    check = True

    for i in range(len(pts)):
        pt = pts[i]
        pred = pts[(i-7) % len(pts)]
        suc = pts[(i+7) % len(pts)]

        angle = calculate_angle(pred, pt, suc)

        if not 115 < angle < 245:
            check = False
            break

    return check


def check_valid_track(centerline, width):
    """Check centreline for valid angles, simplicity, and single interior region."""
    lr = LinearRing(centerline)

    if not check_angles(centerline):
        raise BadTrackException("Too sharp angles")

    if not lr.is_simple:
        from shapely.validation import explain_validity
        raise BadTrackException(f"Not simple: {explain_validity(lr)}")

    polygon = lr.buffer(width / 2)

    if type(polygon) != Polygon:
        # Can also be MultiPolygon, which is certainly false
        raise BadTrackException("Centerline is MultiPolygon")

    if len(polygon.interiors) != 1:
        raise BadTrackException("Centerline has more than one interiors")

    return True


def make_cones_and_start_pose(centerline, width):
    from .utils import discretize_contour

    poly = make_polygon_from_centerline(centerline, width)

    if type(poly) is not Polygon:
        raise BadTrackException("Extended centerline is not Polygon")

    outer = discretize_contour(np.asarray(poly.exterior.coords)[:, :2], 5.)
    inner = discretize_contour(np.asarray(poly.interiors[0].coords)[:, :2], 5.)

    cone_pos = np.concatenate([outer, inner])
    cone_type = np.concatenate([np.full(len(outer), 1, dtype=int), np.full(len(inner), 2, dtype=int)])

    # Find start pos
    forward = centerline[1] - centerline[0]
    forward = forward / np.linalg.norm(forward)
    full_left = centerline[0] + np.array([-forward[1], forward[0]]) * width / 2

    theta = np.arctan2(-forward[1], -forward[0])

    xy = centerline[0]

    return cone_pos, cone_type, xy, theta


def adjust_start_pos(centerline):

    for i in range(0, len(centerline), 3):
        angle1 = calculate_angle(centerline[0], centerline[5], centerline[10])
        angle2 = calculate_angle(centerline[0], centerline[8], centerline[16])
        angle3 = calculate_angle(centerline[len(centerline) - 5], centerline[0], centerline[5])
        angle4 = calculate_angle(centerline[len(centerline) - 8], centerline[0], centerline[8])
        if not 150 < angle1 < 210 or not 150 < angle2 < 210 or not 150 < angle3 < 210 or not 150 < angle4 < 210:
            centerline = np.roll(centerline, +5, axis=0)
        else:
            return centerline

    raise BadTrackException("Could not find a suita_ble starting position.")


def make_full_environment(extends=(200, 200), width=5., cone_width=5., rng=None):
    """Main generation entry point: centreline → validation → cone positions → start pose."""
    centerline = None

    if rng is None:
        rng = np.random.default_rng()

    while True:
        try:
            centerline = make_track_centerline(extends, rng=rng)
            check_valid_track(centerline, width)
            centerline = adjust_start_pos(centerline)
            cone_pos, cone_type, xy, theta = make_cones_and_start_pose(centerline, cone_width)
            break
        except BadTrackException:
            pass

    return {
        'centerline': centerline,
        'length': LinearRing(centerline).length,
        'width': width,
        'start_xy': xy,
        'start_theta': theta,
        'cone_pos': cone_pos,
        'cone_type': cone_type
    }


def make_polygon_from_centerline(centerline, width=5.):
    poly = LinearRing(centerline).buffer(width / 2)

    return poly


def write_tum_track(track, path, total_width=7.):
    with open(path, 'w') as fp:
        print("# x_m,y_m,w_tr_right_m,w_tr_left_m", file=fp)

        for pos, y in track:
            print("{},{},{},{}".format(pos, y, total_width / 2, total_width / 2), file=fp)


def plot_cross_line(plt, a, b):
    # --- Plot starting line ---
    # Direction vector from a to b
    direction = b - a
    # Orthogonal vector (rotate by 90 degrees)
    ortho = np.array([-direction[1], direction[0]])
    ortho = ortho / np.linalg.norm(ortho)  # Normalize
    half_length = 0.36 / 2
    start_pt1 = a + ortho * half_length
    start_pt2 = a - ortho * half_length
    plt.plot([start_pt1[0], start_pt2[0]], [start_pt1[1], start_pt2[1]], 'b-', linewidth=3, label='waypoint line')
    # --- End starting line ---

#my part
def plot_track(centerline, inner_cones, outer_cones, start_xy, start_theta, name = 'track_plot.png'):
    plt.figure(figsize=(8, 8))
    
    # Plot centerline
    plt.plot(centerline[:, 0], centerline[:, 1], 'k-', label='Centerline')

    # Plot cones
    plt.plot(outer_cones[:, 0], outer_cones[:, 1], 'r-', alpha=0.7, label='track')
    plt.plot(inner_cones[:, 0], inner_cones[:, 1], 'r-', alpha=0.7)

    # Plot start direction arrow if provided
    if start_xy is not None and start_theta is not None:
        arrow_length = 0.3  # Adjust as needed for visibility
        dx = arrow_length * np.cos(start_theta)
        dy = arrow_length * np.sin(start_theta)
        plt.arrow(start_xy[0], start_xy[1], dx, dy, 
                  head_width=0.1, head_length=0.2, fc='green', ec='green', label='Start Direction')
        plt.scatter([start_xy[0]], [start_xy[1]], c='green', s=50, marker='*', label='Start Position')

    plt.axis('equal')
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title('Race Track Top-Down View')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('track_plots/' + name)
    plt.close()


def valid_extend(cone_pos) -> bool:
    for pos in cone_pos:
        if not (-8 <= pos[0] <= 8 and -8 <= pos[1] <= 8):
            return False
    return True


def remove_duplicates(points: jp.ndarray) -> jp.ndarray:
    # Detect duplicates by comparing adjacent rows
    diffs = jp.any(jp.diff(points, axis=0) != 0, axis=1)
    mask = jp.concatenate([jp.array([True]), diffs])  # Keep first point

    return points[mask]


def compute_cone_width(inner_cones, outer_cones):
    """
    Underestimates track width by finding minimal distance between cones
    """
    min_dists = []
    for inner in inner_cones:
        # Compute distances from this inner cone to all outer cones
        dists = jp.linalg.norm(outer_cones - inner, axis=1)
        min_dists.append(jp.min(dists))
    return jp.min(jp.array(min_dists))


def create_track(seed, zoom_factor=0.09):
    """Generate, scale, and return a complete track dict (centreline, cones, width)."""
    rng = np.random.default_rng(seed)
    track = make_full_environment(extends=(1, 1), width=4., cone_width=4., rng=rng)
    centerline = jp.array(track['centerline'])
    start_xy = jp.array(track['start_xy'])
    start_theta = track['start_theta']
    cone_pos = jp.array(track['cone_pos'])
    cone_type = jp.array(track['cone_type'])

    #make_full_environment generates the starting angle in the opposite direction of the track
    start_theta = (start_theta + jp.pi) % (2 * jp.pi)

    # Apply zoom factor
    centerline_scaled = centerline * zoom_factor
    cone_pos_scaled = cone_pos * zoom_factor
    start_xy_scaled = start_xy * zoom_factor

    #remove duplicates
    center_line_consecutively_unique = remove_duplicates(centerline_scaled)
    if jp.array_equal(center_line_consecutively_unique[0], center_line_consecutively_unique[-1]):
        centerline_unique = center_line_consecutively_unique[:-1]
    else:
        centerline_unique = center_line_consecutively_unique

    #Shift all positions so that start_xy is 0 0
    cone_pos_shifted = cone_pos_scaled - start_xy_scaled
    centerline_shifted = centerline_unique - start_xy_scaled
    start_xy_shifted = jp.array([0, 0])

    #if not valid_extend(cone_pos_shifted):
        #raise ValueError("Track is not valid, cones are out of bounds. Consider scaling the track down further.")

    #Split cone array into 2 arrays one for the inner and one for the outer cones
    outer_mask = cone_type == 1
    inner_mask = cone_type == 2

    outer_cones = cone_pos_shifted[outer_mask]
    inner_cones = cone_pos_shifted[inner_mask]

    #add the first point again so that in the mujoco sim the track is closed.
    first_outer_cone_point = outer_cones[0:1]
    first_inner_cone_point = inner_cones[0:1]
    outer_cones_closed = jp.concatenate([outer_cones, first_outer_cone_point], axis=0)
    inner_cones_closed = jp.concatenate([inner_cones, first_inner_cone_point], axis=0)

    track_width = compute_cone_width(inner_cones_closed, outer_cones_closed)

    plot_track(centerline_shifted, inner_cones_closed, outer_cones_closed, start_xy_shifted, start_theta, f"track_plot_{seed}")
    return {
        'centerline': centerline_shifted,
        'start_angle': start_theta,
        'inner_cones': inner_cones_closed,
        'outer_cones': outer_cones_closed,
        'width': track_width
    }

def save_track(track, filename):
    """Save a track dict to a .npz file."""
    # Save the track dictionary arrays using savez
    jp.savez(filename,
             centerline=track['centerline'],
             start_angle=track['start_angle'],
             inner_cones=track['inner_cones'],
             outer_cones=track['outer_cones'],
             width=track['width']
            )


def load_track(filename):
    """Load a track from a .npz file at the given path."""
    loaded = jp.load(filename)
    track = {
        'centerline': loaded['centerline'],
        'start_angle': loaded['start_angle'],
        'inner_cones': loaded['inner_cones'],
        'outer_cones': loaded['outer_cones'],
        'width': loaded['width']
    }
    return track


def load_track_by_seed(seed: int):
    """Load the pre-generated track with the given integer seed from bundled assets."""
    return load_track(TRACK_ASSET_DIR / f'track_{seed}.npz')


def main():
    start = time.time()
    for seed in range(100):
        track = create_track(seed=seed, zoom_factor=0.09)
        save_track(track, TRACK_ASSET_DIR / f'track_{seed}.npz')
        track_loaded = load_track_by_seed(seed)
        assert jp.all(track['centerline'] == track_loaded['centerline'])
        assert jp.all(track['inner_cones'] == track_loaded['inner_cones'])
        assert jp.all(track['outer_cones'] == track_loaded['outer_cones'])
        assert track['start_angle'] == track_loaded['start_angle']
        assert track['width'] == track_loaded['width']
        
    end = time.time()
    elapsed = end - start
    print(f"All tests passed. {elapsed:.3f} seconds.")


if __name__ == "__main__":
    main()

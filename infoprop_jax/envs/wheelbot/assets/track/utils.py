"""Utility functions for track processing: contour discretisation and coordinate transforms."""

import numpy as np


def make_orthogonals(track):
    raise NotImplementedError("Deprecated")


def discretize_contour(track, step=10.):
    """Discretise a polygon contour into evenly spaced cone positions."""
    from shapely.geometry import LineString, Point, MultiPoint

    lines = [LineString([track[i], track[i+1]]) for i in range(len(track) - 1)]

    # Add final loop-closing line
    lines.append(LineString([track[-1], track[0]]))

    current_xy = track[0]
    result = [current_xy]
    curr_i = 0

    while True:
        circ = Point(current_xy).buffer(step).exterior

        for i in range(curr_i, len(lines)):
            intersection = lines[i].intersection(circ)
            if intersection:
                if isinstance(intersection, Point):
                    ps = np.array(intersection.xy).T
                elif isinstance(intersection, MultiPoint):
                    ps = np.concatenate([np.array(p.xy).T for p in intersection.geoms])
                else:
                    raise TypeError(str(type(intersection)))

                ld = np.array(lines[i].xy).T

                if i == curr_i:
                    max_advance = np.linalg.norm(current_xy - ld[0])
                else:
                    max_advance = -1
                best_p = None

                for p in ps:
                    assert p.shape == (2,)
                    advance = np.linalg.norm(p - ld[0])
                    if advance <= max_advance:
                        continue
                    max_advance = advance
                    best_p = p

                if best_p is not None:
                    current_xy = best_p
                    result.append(best_p)
                    curr_i = i
                    break
        else:
            return np.array(result)


def annotate_scale(h, ax=None, text=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    if ax is None:
        ax = plt.gca()

    if text is None:
        text = "h = {:.3}".format(h)

    artist_scale = AnchoredSizeBar(ax.transData, h, text,
                                   'upper right', label_top=True, borderpad=1, frameon=False)
    ax.add_artist(artist_scale)
    return artist_scale

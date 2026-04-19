import json
import math
import random
import os

import numpy as np
from scipy.special import binom
from shapely.geometry import MultiPoint, Point, LineString
from shapely.ops import nearest_points

from gym_line_follower.line_interpolation import interpolate_points

root_dir = os.path.dirname(__file__)


def bernstein(n, k, t):
    return binom(n, k) * t ** k * (1. - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 200)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    per_vertex_r = kw.pop("per_vertex_r", None)
    segments = []
    for i in range(len(points) - 1):
        seg_kw = dict(kw)
        if per_vertex_r is not None:
            seg_kw["r"] = float(per_vertex_r[i % len(per_vertex_r)])
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **seg_kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0, sharp_corner_prob=0.0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest.
    *sharp_corner_prob* is the probability [0,1] for each vertex to
           become a sharp corner (small control-point radius)."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    n_verts = len(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)

    per_vertex_r = None
    if sharp_corner_prob > 0.0:
        per_vertex_r = np.where(
            np.random.random(n_verts) < sharp_corner_prob,
            0.03,   # sharp / tight corner
            rad,    # smooth corner
        )

    s, c = get_curve(a, r=rad, method="var", per_vertex_r=per_vertex_r)
    x, y = c.T
    return x, y, a


def generate_polygon(ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts):
    """
    Start with the centre of the geometry at ctrX, ctrY,
    then creates the geometry by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the geometry
    aveRadius - in px, the average radius of this geometry, this roughly controls how large the geometry is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order."""

    irregularity = np.clip(irregularity, 0, 1) * 2 * math.pi / numVerts
    spikeyness = np.clip(spikeyness, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / numVerts) - irregularity
    upper = (2 * math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(numVerts):
        r_i = np.clip(random.gauss(aveRadius, spikeyness), 0, 2 * aveRadius)
        x = ctrX + r_i * math.cos(angle)
        y = ctrY + r_i * math.sin(angle)
        points.append((int(x), int(y)))

        angle = angle + angleSteps[i]

    return points


class Track:
    """
    Line follower follows a Track instance. This class contains methods for randomly generating, rendering and
    calculating relative follower distance, speed and direction.
    """

    def __init__(self, pts, nb_checkpoints=100, render_params=None):
        l = LineString(pts).length
        n = int(l / 3e-3)  # Get number of points for 3 mm spacing

        self.pts = interpolate_points(np.array(pts), n)  # interpolate points to get the right spacing
        self.x = self.pts[:, 0]
        self.y = self.pts[:, 1]

        self.render_params = render_params

        self.mpt = MultiPoint(self.pts)
        self.string = LineString(self.pts)

        # Find starting point and angle
        self.start_xy = self.x[0], self.y[0]
        self.start_angle = self.angle_at_index(0)

        # Get length
        self.length = self.string.length

        # Progress tracking setup
        self.progress = 0.
        self.progress_idx = 0
        self.nb_checkpoints = nb_checkpoints
        self.checkpoints = [i * (self.length / self.nb_checkpoints) for i in range(1, self.nb_checkpoints + 1)]
        self.next_checkpoint_idx = 0
        self.done = False

        self._setup_features()

    @classmethod
    def generate(cls, approx_width=1., hw_ratio=0.5, seed=None, irregularity=0.2,
                 spikeyness=0.2, num_verts=10, *args, **kwargs):
        """
        Generate random track.
        Adapted from: https://stackoverflow.com/a/45618741/9908077
        :param approx_width: approx. width of generated track
        :param hw_ratio: ratio height / width
        :param seed: seed for random generator
        :return: Track instance
        """
        # Extract generation params from render_params when provided
        render_params = kwargs.get("render_params", None)
        sharp_corner_prob = 0.0
        if render_params is not None:
            nv = render_params.get("num_verts", None)
            if nv is not None:
                num_verts = int(round(float(nv)))
            scp = render_params.get("sharp_corner_prob", None)
            if scp is not None:
                sharp_corner_prob = float(scp)

        # Generate random points
        random.seed(seed)
        upscale = 1000.  # upscale so curve gen fun works
        r = upscale * approx_width / 2.
        pts = generate_polygon(0, 0, r, irregularity=irregularity, spikeyness=spikeyness, numVerts=num_verts)
        pts = np.array(pts)

        # Generate curve with points
        x, y, _ = get_bezier_curve(pts, rad=0.2, edgy=0, sharp_corner_prob=sharp_corner_prob)
        # Remove duplicated point
        x = x[:-1]
        y = y[:-1]

        # Scale y
        y = y * hw_ratio

        # Scale units
        unit_scale = 1000
        x, y = x / unit_scale, y / unit_scale
        pts = np.stack((x, y), axis=-1)

        # Check width / height:
        if max(abs(min(x)), max(x)) * 2 > 1.5 * approx_width or max(abs(min(y)), max(y)) * 2 > 1.5 * approx_width * hw_ratio:
            return cls.generate(approx_width, hw_ratio, seed, irregularity, spikeyness, num_verts, *args, **kwargs)

        # Randomly flip track direction
        np.random.seed(seed)
        if np.random.choice([True, False]):
            pts = np.flip(pts, axis=0)
        return cls(pts, *args, **kwargs)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        with open(path, "r") as f:
            d = json.load(f)
        points = d["points"]
        points.append(points[0])  # Close the loop
        points = interpolate_points(points, 1000)
        return cls(points, *args, **kwargs)

    # ------------------------------------------------------------------
    # Feature geometry helpers (dead ends, crossings, gaps, width)
    # ------------------------------------------------------------------

    def _setup_features(self):
        """Compute and cache all per-episode track features from render_params."""
        self.dead_ends = []
        self.crossings = []
        self.gap_mask = np.zeros(max(0, len(self.pts) - 1), dtype=bool)
        self.width_schedule = []

        rp = self.render_params
        if rp is None:
            return

        rng = np.random.default_rng()

        if rp.get("dead_ends_enabled", False):
            n = int(round(float(rp.get("num_dead_ends", 3))))
            length = float(rp.get("dead_end_length", 0.08))
            self.dead_ends = self._generate_dead_ends(n, length, rng)

        if rp.get("crossings_enabled", False):
            n = int(round(float(rp.get("num_crossings", 2))))
            length = float(rp.get("crossing_length", 0.10))
            self.crossings = self._generate_crossings(n, length, rng)

        if rp.get("gaps_enabled", False):
            n = int(round(float(rp.get("num_gaps", 2))))
            gap_length = float(rp.get("gap_length", 0.04))
            self.gap_mask = self._generate_gap_mask(n, gap_length, rng)

        if rp.get("variable_line_width", False):
            n_segs = int(round(float(rp.get("num_width_segments", 8))))
            base_thickness = float(rp.get("line_thickness", 0.030))
            self.width_schedule = self._generate_width_schedule(n_segs, base_thickness, rng)

    def _generate_dead_ends(self, n, length, rng):
        """Short stubs branching off the main track at random positions."""
        result = []
        n_pts = len(self.pts)
        margin = max(1, n_pts // 10)
        avail = n_pts - 2 * margin
        if avail <= 0:
            return result
        indices = rng.choice(np.arange(margin, n_pts - margin), size=min(n, avail), replace=False)
        for idx in indices:
            tangent = self.vector_at_index(int(idx))
            perp = np.array([-tangent[1], tangent[0]])
            side = rng.choice([-1.0, 1.0])
            stub_len = length * rng.uniform(0.5, 1.0)
            p1 = self.pts[idx].copy()
            p2 = p1 + side * perp * stub_len
            result.append((p1, p2))
        return result

    def _generate_crossings(self, n, length, rng):
        """Lines that cross perpendicularly through the track (T/X junctions)."""
        result = []
        n_pts = len(self.pts)
        margin = max(1, n_pts // 10)
        avail = n_pts - 2 * margin
        if avail <= 0:
            return result
        indices = rng.choice(np.arange(margin, n_pts - margin), size=min(n, avail), replace=False)
        for idx in indices:
            tangent = self.vector_at_index(int(idx))
            perp = np.array([-tangent[1], tangent[0]])
            half = length * rng.uniform(0.4, 0.6)
            center = self.pts[idx].copy()
            p1 = center - perp * half
            p2 = center + perp * (length - half)
            result.append((p1, p2))
        return result

    def _generate_gap_mask(self, n, gap_length_m, rng):
        """Boolean mask over segments: True = skip drawing (gap in line)."""
        n_pts = len(self.pts)
        mask = np.zeros(max(0, n_pts - 1), dtype=bool)
        if n_pts < 3:
            return mask
        margin = max(1, n_pts // 10)
        gap_pts = max(1, int(gap_length_m / 3e-3))
        upper = n_pts - margin - gap_pts
        if upper <= margin:
            return mask
        candidates = np.arange(margin, upper)
        size = min(n, len(candidates))
        starts = rng.choice(candidates, size=size, replace=False)
        for s in starts:
            end = min(int(s) + gap_pts, n_pts - 1)
            mask[int(s):end] = True
        return mask

    def _generate_width_schedule(self, n_segments, base_thickness, rng):
        """List of (start_idx, width_meters) giving abrupt width changes along the track."""
        n_pts = len(self.pts)
        boundaries = np.linspace(0, n_pts, n_segments + 1, dtype=int)
        result = []
        for i in range(n_segments):
            w = rng.uniform(0.020, 0.060)
            result.append((int(boundaries[i]), w))
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, w=3., h=2., ppm=1500, line_thickness=0.015, save=None, line_color="black",
                background="white", line_opacity=0.8, dashed=False,
                variable_line_width=False,
                line_noise_enabled=False, line_noise_intensity=0.3,
                line_opacity_variation=0.0,
                **kwargs):
        """
        Render track using open-cv
        :param w: canvas width in meters
        :param h: canvas height in meters
        :param ppm: pixel per meter
        :param line_thickness: line thickness in meters (base value; overridden per-segment when variable_line_width)
        :param save: path to save
        :param line_color: string or BGR tuple
                           options: [black, red, green, blue]
        :param background: string or BGR tuple
                           options: [wood, wood_2, concrete, brick, checkerboard, white, gray]
        :param line_opacity: opacity of line in range 0, 1 where 0 is fully transparent
        :param variable_line_width: use per-section width variation (precomputed in self.width_schedule)
        :param line_noise_enabled: apply noise / dead-pixel effects to the rendered line
        :param line_noise_intensity: noise strength in [0, 1]
        :param line_opacity_variation: smooth per-area opacity jitter in [0, 1]
        :return: rendered track image array
        """
        import cv2
        w_res = int(round(w * ppm))
        h_res = int(round(h * ppm))
        t_res = int(round(line_thickness * ppm))

        background_bgr = None
        if isinstance(background, str):
            background = background.lower()
            if background == "wood":
                bg = cv2.imread(os.path.join(root_dir, "track_textures", "wood.jpg"))
            elif background == "wood_2":
                bg = cv2.imread(os.path.join(root_dir, "track_textures", "wood_2.jpg"))
            elif background == "concrete":
                bg = cv2.imread(os.path.join(root_dir, "track_textures", "concrete.jpg"))
            elif background == "brick":
                bg = cv2.imread(os.path.join(root_dir, "track_textures", "brick.jpg"))
            elif background == "checkerboard":
                bg = cv2.imread(os.path.join(root_dir, "track_textures", "checkerboard.jpg"))
            elif background == "white":
                background_bgr = (255, 255, 255)
            elif background == "gray":
                background_bgr = (150, 150, 150)
            else:
                raise ValueError("Invalid background string.")

            if background_bgr:
                bg = np.ones((h_res, w_res, 3), dtype=np.uint8)
                bg[:, :, 0] *= background_bgr[0]
                bg[:, :, 1] *= background_bgr[1]
                bg[:, :, 2] *= background_bgr[2]
            else:
                bg = cv2.resize(bg, (w_res, h_res), interpolation=cv2.INTER_LINEAR)

        elif isinstance(background, tuple):
            bg = np.ones((h_res, w_res, 3), dtype=np.uint8)
            bg[:, :, 0] *= background[0]
            bg[:, :, 1] *= background[1]
            bg[:, :, 2] *= background[2]
        else:
            raise ValueError("Invalid background.")

        if isinstance(line_color, str):
            line_color = line_color.lower()
            if line_color == "black":
                line_bgr = (0, 0, 0)
            elif line_color == "red":
                line_bgr = (0, 0, 255)
            elif line_color == "green":
                line_bgr = (0, 128, 0)
            elif line_color == "blue":
                line_bgr = (255, 0, 0)
            else:
                raise ValueError("Invalid color string.")
        elif isinstance(line_color, tuple):
            line_bgr = line_color
        else:
            raise ValueError("Invalid line_color.")

        line = bg.copy()

        # Build per-segment thickness (pixels) array for variable-width mode
        gap_mask = getattr(self, "gap_mask", np.zeros(max(0, len(self.pts) - 1), dtype=bool))
        width_schedule = getattr(self, "width_schedule", [])

        if variable_line_width and width_schedule:
            n_segs = max(0, len(self.pts) - 1)
            seg_thickness = np.full(n_segs, t_res, dtype=int)
            for j, (start_idx, width_m) in enumerate(width_schedule):
                end_idx = width_schedule[j + 1][0] if j + 1 < len(width_schedule) else n_segs
                seg_thickness[start_idx:end_idx] = max(1, int(round(width_m * ppm)))
        else:
            seg_thickness = None

        def _world_to_img(x, y):
            xi = int(round((x + w / 2) * ppm))
            yi = int(round(h_res - (y + h / 2) * ppm))
            return xi, yi

        if dashed:
            pts = interpolate_points(self.pts, 1000)
            n = self.length / dashed
            chunks = np.array_split(pts, n)[::2]
            for c in chunks:
                for i in range(len(c) - 1):
                    x1, y1 = c[i]
                    x2, y2 = c[i + 1]
                    cv2.line(line, _world_to_img(x1, y1), _world_to_img(x2, y2),
                             color=line_bgr, thickness=t_res, lineType=cv2.LINE_AA)
        else:
            for i in range(len(self.pts) - 1):
                if gap_mask[i]:
                    continue
                x1, y1 = self.pts[i]
                x2, y2 = self.pts[i + 1]
                thickness = int(seg_thickness[i]) if seg_thickness is not None else t_res
                cv2.line(line, _world_to_img(x1, y1), _world_to_img(x2, y2),
                         color=line_bgr, thickness=thickness, lineType=cv2.LINE_AA)

        # Draw dead ends (one-sided stubs)
        for p1, p2 in getattr(self, "dead_ends", []):
            cv2.line(line, _world_to_img(p1[0], p1[1]), _world_to_img(p2[0], p2[1]),
                     color=line_bgr, thickness=t_res, lineType=cv2.LINE_AA)

        # Draw crossings (perpendicular lines through the track)
        for p1, p2 in getattr(self, "crossings", []):
            cv2.line(line, _world_to_img(p1[0], p1[1]), _world_to_img(p2[0], p2[1]),
                     color=line_bgr, thickness=t_res, lineType=cv2.LINE_AA)

        # Composite line onto background — with optional smooth opacity variation
        alpha = line_opacity
        if line_opacity_variation > 0:
            small_h = max(1, h_res // 50)
            small_w = max(1, w_res // 50)
            small_noise = np.random.uniform(-line_opacity_variation, line_opacity_variation,
                                            (small_h, small_w)).astype(np.float32)
            alpha_map = np.clip(
                alpha + cv2.resize(small_noise, (w_res, h_res), interpolation=cv2.INTER_LINEAR),
                0.2, 1.0
            )
            alpha3 = alpha_map[:, :, np.newaxis]
            out = np.clip(
                line.astype(np.float32) * alpha3 + bg.astype(np.float32) * (1.0 - alpha3),
                0, 255
            ).astype(np.uint8)
        else:
            out = cv2.addWeighted(line, alpha, bg, 1 - alpha, 0)

        # Line noise / texture post-processing
        if line_noise_enabled and line_noise_intensity > 0:
            intensity = float(line_noise_intensity)

            # Gaussian noise across the whole image
            gaussian = np.random.normal(0, intensity * 18.0, out.shape)
            out = np.clip(out.astype(np.int16) + gaussian.astype(np.int16), 0, 255).astype(np.uint8)

            # Dead-pixel / blotch effect: blank small patches within the line area
            line_diff = np.max(np.abs(line.astype(np.int16) - bg.astype(np.int16)), axis=2)
            line_pixel_mask = line_diff > 15
            ys, xs = np.where(line_pixel_mask)
            if len(ys) > 0:
                n_patches = max(1, int(intensity * 40))
                patch_indices = np.random.randint(0, len(ys), size=n_patches)
                for pi in patch_indices:
                    cy, cx = int(ys[pi]), int(xs[pi])
                    r_px = np.random.randint(1, max(2, int(intensity * 8) + 1))
                    bg_color = tuple(int(bg[cy, cx, c]) for c in range(3))
                    cv2.circle(out, (cx, cy), r_px, bg_color, -1)

        if save is not None:
            cv2.imwrite(save, out)
        return out

    def render(self, *args, **kwargs):
        if self.render_params:
            return self._render(*args, **kwargs, **self.render_params)
        else:
            return self._render(*args, **kwargs)

    def distance_from_point(self, pt):
        """
        Calculate minimal distance of a position from track.
        :param pt: position. [x, y] or shapely.geometry.Point instance
        :return: minimal absolute distance to track, float
        """
        if not isinstance(pt, Point):
            pt = Point(pt)
        return pt.distance(self.mpt)

    def vector_at_index(self, idx):
        """
        Return normalized track direction vector at desired index.
        :param idx: index of track point
        :return: unit direction vector
        """
        x, y = self.x, self.y

        # Handle indexing last track point
        if idx < len(self.pts) - 2:
            vect = np.array([x[idx + 1] - x[idx], y[idx + 1] - y[idx]])
        else:
            vect = np.array([x[0] - x[idx], y[0] - y[idx]])

        # Find track angle
        norm = np.linalg.norm(vect)
        vect = (vect / norm) if norm > 0.0 else np.array([1., 0])  # normalize vector to unit length
        return vect

    def angle_at_index(self, idx):
        """
        Calculate track angle at desired index. Angle is calculated from x-axis, CCW is positive. Angle is returned in
        radians in range [0, 2pi]
        :param idx: index of track point
        :return: angle in radians, range [0, 2pi]
        """
        vect = self.vector_at_index(idx)
        x_vect = np.array([1, 0])
        dot = np.dot(vect, x_vect)
        det = np.linalg.det([x_vect, vect])
        track_ang = np.arctan2(det, dot)
        if track_ang < 0.:
            track_ang += 2 * np.pi
        return track_ang

    def nearest_point(self, pt):
        """
        Determine point on track that is nearest to provided point.
        :param pt: point to search nearest track point for, Point instance or coordinate array [x, y]
        :return: nearest track point coordinates [x, y]
        """
        if not isinstance(pt, Point):
            pt = Point(pt)
        nearest = nearest_points(pt, self.mpt)[1]
        return nearest.x, nearest.y

    def nearest_angle(self, pt):
        """
        Calculate track angle at the point on track nearest to provided point-
        :param pt: point to search nearest track point for, Point instance or coordinate array [x, y]
        :return: angle, float
        """
        near_x, near_y = self.nearest_point(pt)
        near_idx = np.where(self.x == near_x)[0][0]
        return self.angle_at_index(near_idx)

    def nearest_vector(self, pt):
        """
        Calculate track angle at the point on track nearest to provided point.
        :param pt: point to search nearest track point for, Point instance or coordinate array [x, y]
        :return: unit track direction vector
        """
        near_x, near_y = self.nearest_point(pt)
        near_idx = np.where(self.x == near_x)[0][0]
        return self.vector_at_index(near_idx)

    def length_between_idx(self, idx1, idx2, shortest=True):
        """
        Calculate length of track segment between two point indexes. Direction is determined based on index order.
        :param idx1: first index
        :param idx2: second index
        :param shortest: True to return shortest path, False to return longest
        :return: segment length, float, positive or negative based on direction
        """
        if idx1 == idx2:
            return 0.
        if idx1 < idx2:
            first = idx1
            second = idx2
        else:
            first = idx2
            second = idx1
        string_1 = LineString(self.pts[first:second+1])
        string_2 = LineString(np.concatenate((self.pts[0:first+1], self.pts[second:])))
        len_1 = string_1.length
        len_2 = string_2.length

        if len_1 < len_2:
            if idx1 < idx2:
                if shortest:
                    return len_1
                else:
                    return -len_2
            else:
                if shortest:
                    return -len_1
                else:
                    return len_2
        else:
            if idx1 < idx2:
                if shortest:
                    return -len_2
                else:
                    return len_1
            else:
                if shortest:
                    return len_2
                else:
                    return -len_1

    def length_along_track(self, pt1, pt2):
        """
        Calculate length along track between two points near to track. Returns the shortest possible path.
        Order of argument points is arbitrary.
        :param pt1: first point
        :param pt2: second point
        :return: length, float, positive if in direction of track, negative otherwise
        """
        near_1 = self.nearest_point(pt1)
        near_2 = self.nearest_point(pt2)

        idx_1 = np.where(self.x == near_1[0])[0][0]
        idx_2 = np.where(self.x == near_2[0])[0][0]
        return self.length_between_idx(idx_1, idx_2, shortest=True)

    def position_along(self, pt):
        """
        Calculate position along track from start of track.
        :param pt:
        :return: position in range [0, track length]
        """
        near = self.nearest_point(pt)
        idx = np.where(self.x == near[0])[0][0]
        return (idx / len(self.pts)) * self.length

    def update_progress(self, position):
        """
        Update track progress and return passed checkpoints.
        :param position: position along track in meters from starting point
        :return: number of checkpoints passed
        """
        if self.done:
            return 0
        if position > self.progress:
            self.progress = position
            self.progress_idx = int(round((self.progress / self.length) * len(self.pts)))
        ret = 0
        while self.progress >= self.checkpoints[self.next_checkpoint_idx]:
            self.next_checkpoint_idx += 1
            ret += 1
            if self.next_checkpoint_idx >= self.nb_checkpoints-1:
                self.done = True
                break
        return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # t = Track.generate(2.0, hw_ratio=0.7, seed=4125,
    #                    spikeyness=0.2, nb_checkpoints=500)

    # img = t.render()
    # plt.imshow(img)
    # plt.show()

    for i in range(9):
        t = Track.generate(2.0, hw_ratio=0.7, seed=None,
                           spikeyness=0.2, nb_checkpoints=500)
        img = t.render(ppm=1000)
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.axis("off")
    # plt.tight_layout()
    plt.savefig("track_generator.png", dpi=300)
    plt.show()

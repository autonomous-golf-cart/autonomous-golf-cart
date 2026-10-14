"""
Path planning on BEV segmentation maps.
Two approaches:
  1. Frenet Frame Optimal Trajectory - samples lateral offsets along GPS reference path
  2. MPC with Bicycle Model - optimizes trajectory within driveable corridor

Both take a BEV driveable mask + GPS route and output a trajectory in BEV coordinates.
"""
import numpy as np
import math
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


# ── GPS / coordinate helpers ─────────────────────────────────────────

def gps_to_local(lat, lon, ref_lat, ref_lon):
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    x = dlon * 6371000 * math.cos(math.radians(ref_lat))
    y = dlat * 6371000
    return x, y


def load_gps_route(gps_path):
    import json
    with open(gps_path) as f:
        data = json.load(f)
    ref_lat, ref_lon = data["_reference_origin_latlon"]
    points = []
    for s in data["samples"]:
        x, y = gps_to_local(s["lat"], s["lon"], ref_lat, ref_lon)
        points.append((s["rel_t"], x, y))
    return np.array(points), ref_lat, ref_lon


def load_map_route(map_path, ref_lat, ref_lon):
    """Load the pre-annotated map route (A→B) and convert lat/lon to local coords."""
    import json
    with open(map_path) as f:
        data = json.load(f)
    route_latlon = data["route_latlon"]
    points = []
    for lat, lon in route_latlon:
        x, y = gps_to_local(lat, lon, ref_lat, ref_lon)
        points.append((x, y))
    return np.array(points)


def localize_on_map_route(map_route, ego_x, ego_y):
    """Find the nearest point on the map route to the current GPS position.
    Returns the index and the fractional position along the route."""
    dx = map_route[:, 0] - ego_x
    dy = map_route[:, 1] - ego_y
    dists = np.sqrt(dx ** 2 + dy ** 2)
    nearest_idx = np.argmin(dists)
    return nearest_idx, dists[nearest_idx]


def get_map_route_ahead(map_route, nearest_idx, lookahead_pts=None):
    """Extract the portion of the map route ahead of the current position."""
    if lookahead_pts is None:
        return map_route[nearest_idx:]
    end = min(nearest_idx + lookahead_pts, len(map_route))
    return map_route[nearest_idx:end]


def build_reference_path_from_map(map_route_ahead, ego_x, ego_y, ego_yaw, lookahead_m=20.0):
    """Build a cubic spline reference path from the map route ahead, in ego-local coords."""
    if len(map_route_ahead) < 4:
        return None

    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    dx = map_route_ahead[:, 0] - ego_x
    dy = map_route_ahead[:, 1] - ego_y
    local_x = dx * c - dy * s
    local_y = dx * s + dy * c

    arc_len = np.zeros(len(local_x))
    for i in range(1, len(local_x)):
        arc_len[i] = arc_len[i - 1] + np.sqrt(
            (local_x[i] - local_x[i - 1]) ** 2 + (local_y[i] - local_y[i - 1]) ** 2
        )

    if arc_len[-1] < 0.5:
        return None

    trim = np.searchsorted(arc_len, lookahead_m)
    if trim < 4:
        trim = min(len(arc_len), max(4, trim))
    arc_len = arc_len[:trim]
    local_x = local_x[:trim]
    local_y = local_y[:trim]

    unique_mask = np.diff(arc_len, prepend=-1) > 0.01
    arc_len = arc_len[unique_mask]
    local_x = local_x[unique_mask]
    local_y = local_y[unique_mask]

    if len(arc_len) < 4:
        return None

    cs_x = CubicSpline(arc_len, local_x)
    cs_y = CubicSpline(arc_len, local_y)

    return cs_x, cs_y, arc_len[-1]


def load_ego_data(ego_path):
    import json
    egos = []
    with open(ego_path) as f:
        for line in f:
            line = line.strip()
            if not line or '"_schema"' in line:
                continue
            egos.append(json.loads(line))
    return egos


def get_ego_at_time(egos, t):
    for i in range(len(egos) - 1):
        if egos[i]["rel_t"] <= t <= egos[i + 1]["rel_t"]:
            alpha = (t - egos[i]["rel_t"]) / (egos[i + 1]["rel_t"] - egos[i]["rel_t"] + 1e-9)
            e0, e1 = egos[i], egos[i + 1]
            speed = e0["speed_mps"] * (1 - alpha) + e1["speed_mps"] * alpha
            return speed
    return egos[-1]["speed_mps"]


def get_ego_from_gps(gps_route, t):
    """Get ego position, heading, and speed from GPS data directly."""
    idx = np.searchsorted(gps_route[:, 0], t) - 1
    idx = np.clip(idx, 0, len(gps_route) - 2)

    alpha = (t - gps_route[idx, 0]) / (gps_route[idx + 1, 0] - gps_route[idx, 0] + 1e-9)
    x = gps_route[idx, 1] * (1 - alpha) + gps_route[idx + 1, 1] * alpha
    y = gps_route[idx, 2] * (1 - alpha) + gps_route[idx + 1, 2] * alpha

    # Heading from nearby GPS points
    lookahead = min(idx + 5, len(gps_route) - 1)
    lookbehind = max(idx - 2, 0)
    dx = gps_route[lookahead, 1] - gps_route[lookbehind, 1]
    dy = gps_route[lookahead, 2] - gps_route[lookbehind, 2]
    yaw = math.atan2(dy, dx)

    # Speed from GPS
    dt = gps_route[min(idx + 1, len(gps_route) - 1), 0] - gps_route[idx, 0]
    ddx = gps_route[min(idx + 1, len(gps_route) - 1), 1] - gps_route[idx, 1]
    ddy = gps_route[min(idx + 1, len(gps_route) - 1), 2] - gps_route[idx, 2]
    speed = math.sqrt(ddx ** 2 + ddy ** 2) / max(dt, 0.01)

    return x, y, yaw, speed


# ── BEV coordinate transforms ───────────────────────────────────────

def world_to_bev(wx, wy, ego_x, ego_y, ego_yaw, bev_size, range_fwd, range_side):
    """Convert world coords to BEV pixel coords relative to ego."""
    dx = wx - ego_x
    dy = wy - ego_y
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
    local_fwd = dx * c - dy * s
    local_left = dx * s + dy * c
    bx = int((local_left / range_side * 0.5 + 0.5) * bev_size)
    by = int((1 - local_fwd / range_fwd) * bev_size)
    return bx, by


def world_to_bev_batch(wx, wy, ego_x, ego_y, ego_yaw, bev_size, range_fwd, range_side):
    dx = wx - ego_x
    dy = wy - ego_y
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    local_fwd = dx * c - dy * s
    local_left = dx * s + dy * c
    bx = ((local_left / range_side * 0.5 + 0.5) * bev_size).astype(int)
    by = ((1 - local_fwd / range_fwd) * bev_size).astype(int)
    return bx, by


def bev_to_local(bx, by, bev_size, range_fwd, range_side):
    """BEV pixel to ego-local coords (fwd, left)."""
    local_left = (bx / bev_size - 0.5) * 2 * range_side
    local_fwd = (1 - by / bev_size) * range_fwd
    return local_fwd, local_left


def map_route_to_bev(map_route, ego_x, ego_y, ego_yaw, bev_size, range_fwd, range_side):
    """Convert map route (Nx2 world xy) to BEV pixel coordinates relative to ego."""
    dx = map_route[:, 0] - ego_x
    dy = map_route[:, 1] - ego_y
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    local_fwd = dx * c - dy * s
    local_left = dx * s + dy * c
    bx = (local_left / range_side * 0.5 + 0.5) * bev_size
    by = (1 - local_fwd / range_fwd) * bev_size
    valid = (bx >= 0) & (bx < bev_size) & (by >= 0) & (by < bev_size) & (local_fwd > 0)
    return bx[valid], by[valid]


def gps_to_bev(gps_route, ego_x, ego_y, ego_yaw, bev_size, range_fwd, range_side):
    """Convert GPS route to BEV pixel coordinates relative to ego."""
    dx = gps_route[:, 1] - ego_x
    dy = gps_route[:, 2] - ego_y
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    local_fwd = dx * c - dy * s
    local_left = dx * s + dy * c
    bx = (local_left / range_side * 0.5 + 0.5) * bev_size
    by = (1 - local_fwd / range_fwd) * bev_size
    valid = (bx >= 0) & (bx < bev_size) & (by >= 0) & (by < bev_size) & (local_fwd > 0)
    return bx[valid], by[valid]


# ── Driveable area extraction ────────────────────────────────────────

def get_driveable_mask(seg_map, calib, bev_size=500):
    """Project segmentation to BEV and return boolean mask of driveable area."""
    from .bev import create_bev
    from .cityscapes import CITYSCAPES_COLORS
    bev_img = create_bev(seg_map, calib, bev_size)
    road_color = np.array(CITYSCAPES_COLORS[0])
    sidewalk_color = np.array(CITYSCAPES_COLORS[1])
    terrain_color = np.array(CITYSCAPES_COLORS[9])
    mask = (
        np.all(bev_img == road_color, axis=-1) |
        np.all(bev_img == sidewalk_color, axis=-1) |
        np.all(bev_img == terrain_color, axis=-1)
    )
    return mask


def get_driveable_mask_direct(seg_map, calib, bev_size=500):
    """Build driveable BEV mask directly from seg_map projection (faster, no color matching)."""
    f = calib["intrinsics"]["focal_length"]
    cx_p = calib["intrinsics"]["cx"]
    cy_p = calib["intrinsics"]["cy"]
    k1 = calib["intrinsics"]["k1"]
    k2 = calib["intrinsics"]["k2"]
    h = calib["extrinsics"]["height_m"]
    img_h, img_w = seg_map.shape[:2]

    FT_TO_M = 0.3048
    range_fwd = 50 * FT_TO_M
    range_side = 25 * FT_TO_M

    pitch = math.radians(calib["extrinsics"]["pitch_deg"])
    roll = math.radians(calib["extrinsics"]["roll_deg"])
    yaw = math.radians(calib["extrinsics"]["yaw_deg"])
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    cyw, syw = math.cos(yaw), math.sin(yaw)

    Ryaw = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]])
    Rbase = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Rpitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    R = Rroll @ Rpitch @ Rbase @ Ryaw

    by_arr, bx_arr = np.mgrid[0:bev_size, 0:bev_size]
    by_flat = by_arr.ravel()
    bx_flat = bx_arr.ravel()

    wx = (bx_flat / bev_size - 0.5) * 2 * range_side
    wy = (1 - by_flat / bev_size) * range_fwd

    m = wy > 0.15
    wx, wy = wx[m], wy[m]
    by_f, bx_f = by_flat[m], bx_flat[m]

    pts = np.stack([wx, wy, np.full_like(wx, -h)])
    cam = R @ pts

    m2 = cam[2] > 0.01
    cam = cam[:, m2]
    by_f, bx_f = by_f[m2], bx_f[m2]

    r3d = np.sqrt(cam[0] ** 2 + cam[1] ** 2)
    theta = np.arctan2(r3d, cam[2])
    m3 = theta < math.pi * 0.47
    cam, r3d, theta = cam[:, m3], r3d[m3], theta[m3]
    by_f, bx_f = by_f[m3], bx_f[m3]

    t2 = theta ** 2
    td = theta * (1 + k1 * t2 + k2 * t2 * t2)
    rp = f * td

    safe = r3d > 1e-8
    u = np.where(safe, cx_p + rp * cam[0] / r3d, cx_p)
    v = np.where(safe, cy_p + rp * cam[1] / r3d, cy_p)

    iu = np.floor(u).astype(np.int32)
    iv = np.floor(v).astype(np.int32)
    m4 = (iu >= 0) & (iu < img_w) & (iv >= 0) & (iv < img_h)
    iu, iv = iu[m4], iv[m4]
    by_f, bx_f = by_f[m4], bx_f[m4]

    cls_ids = seg_map[iv, iu]
    driveable_classes = {0, 1, 9}  # road, sidewalk, terrain
    is_driveable = np.isin(cls_ids, list(driveable_classes))

    mask = np.zeros((bev_size, bev_size), dtype=bool)
    mask[by_f[is_driveable], bx_f[is_driveable]] = True

    road_only = cls_ids == 0
    road_mask = np.zeros((bev_size, bev_size), dtype=bool)
    road_mask[by_f[road_only], bx_f[road_only]] = True

    return mask, range_fwd, range_side, road_mask


# ── Vision-only centerline path planner ─────────────────────────────

def _snap_to_road(bx_arr, by_arr, mask, bev_size):
    """Push any off-road points to the nearest road pixel in that row.
    Falls back to searching neighboring rows (±10) if no road in exact row."""
    out_bx = bx_arr.copy()
    for i in range(len(bx_arr)):
        bxi = int(np.clip(bx_arr[i], 0, bev_size - 1))
        byi = int(np.clip(by_arr[i], 0, bev_size - 1))
        if mask[byi, bxi]:
            continue
        # Search this row first, then expand to neighboring rows
        found = False
        for dy in range(0, 15):
            for y_try in ([byi] if dy == 0 else [byi - dy, byi + dy]):
                if y_try < 0 or y_try >= bev_size:
                    continue
                row = mask[y_try, :]
                if np.any(row):
                    cols = np.where(row)[0]
                    nearest = cols[np.argmin(np.abs(cols - bxi))]
                    out_bx[i] = float(nearest)
                    found = True
                    break
            if found:
                break
    return out_bx


def driveable_centerline_path(driveable_mask, bev_size=500, range_fwd=15.24, range_side=7.62,
                               gps_bx=None, gps_by=None, road_mask=None):
    """
    Extract a smooth path along the road centerline visible in BEV.
    Scans road_mask row-by-row from near to far, finds road center,
    fits a smooth polynomial, and post-validates against the driveable mask.
    Uses map route BEV points for region selection at forks.
    """
    gps_sorted_by = None
    gps_sorted_bx = None
    if gps_bx is not None and gps_by is not None and len(gps_bx) > 1:
        order = np.argsort(gps_by)
        gps_sorted_by = gps_by[order]
        gps_sorted_bx = gps_bx[order]

    # Clean road mask: keep only the connected region nearest to ego (bottom-center)
    scan_mask = road_mask if road_mask is not None else driveable_mask
    from scipy.ndimage import label as ndlabel
    labeled, n_features = ndlabel(scan_mask)
    if n_features > 0:
        ego_region = labeled[bev_size - 5, bev_size // 2]
        if ego_region == 0:
            for dy in range(1, 200):
                for dx in range(-50, 51, 5):
                    yy = max(0, bev_size - 5 - dy)
                    xx = np.clip(bev_size // 2 + dx, 0, bev_size - 1)
                    if labeled[yy, xx] > 0:
                        ego_region = labeled[yy, xx]
                        break
                if ego_region > 0:
                    break
        if ego_region > 0:
            # Use the label from dilated to select region, but scan original mask
            region_mask = (labeled == ego_region)
            scan_mask = scan_mask & region_mask

    centers = []
    y_coords = []
    prev_center = bev_size / 2.0
    gap_count = 0
    step = 2

    for y in range(bev_size - 8, -1, -step):
        row = scan_mask[y, :]
        if not np.any(row):
            gap_count += 1
            if gap_count > 20 and len(centers) > 5:
                break
            continue
        gap_count = 0

        cols = np.where(row)[0]
        runs = []
        run_start = cols[0]
        for j in range(1, len(cols)):
            if cols[j] - cols[j - 1] > 5:
                runs.append((run_start, cols[j - 1]))
                run_start = cols[j]
        runs.append((run_start, cols[-1]))

        desired = bev_size / 2.0
        desired_i = int(np.clip(desired, 0, bev_size - 1))

        if scan_mask[y, desired_i]:
            center = desired
        else:
            cols = np.where(row)[0]
            dists = np.abs(cols - desired)
            nearby = cols[dists <= 30]
            if len(nearby) > 0:
                center = float(nearby[np.argmin(np.abs(nearby - desired))])
            else:
                gap_count += 1
                continue

        prev_center = center
        centers.append(center)
        y_coords.append(float(y))

    if len(centers) < 10:
        return None, None

    centers = np.array(centers)
    y_coords = np.array(y_coords)

    # Gaussian smoothing to eliminate row-by-row noise while preserving curves
    sigma = max(10, len(centers) // 8)
    centers = gaussian_filter1d(centers, sigma=sigma, mode='nearest')

    # Subsample smoothed centers directly (no polynomial — avoids overshoot)
    n_out = 40
    indices = np.linspace(0, len(centers) - 1, n_out).astype(int)
    bx = centers[indices].astype(float)
    by = y_coords[indices].astype(float)
    bx = np.clip(bx, 0, bev_size - 1)
    by = np.clip(by, 0, bev_size - 1)

    # Snap → smooth → snap using the ego's connected road region
    bx = _snap_to_road(bx, by, scan_mask, bev_size)
    bx = gaussian_filter1d(bx, sigma=3, mode='nearest')
    bx = np.clip(bx, 0, bev_size - 1)
    bx = _snap_to_road(bx, by, scan_mask, bev_size)

    local_left = (bx / bev_size - 0.5) * 2 * range_side
    local_fwd = (1 - by / bev_size) * range_fwd

    bev_traj = np.stack([bx.astype(int), by.astype(int)], axis=1)
    local_traj = np.stack([local_fwd, local_left], axis=1)

    return bev_traj, local_traj


# ── Lane-aware centerline path planner ───────────────────────────────

def _find_ego_connected_road(scan_mask, bev_size):
    """Keep only the road region connected to ego (bottom-center of BEV)."""
    from scipy.ndimage import label as ndlabel
    labeled, n_features = ndlabel(scan_mask)
    if n_features == 0:
        return scan_mask
    ego_region = labeled[bev_size - 5, bev_size // 2]
    if ego_region == 0:
        for dy in range(1, 200):
            for dx in range(-50, 51, 5):
                yy = max(0, bev_size - 5 - dy)
                xx = np.clip(bev_size // 2 + dx, 0, bev_size - 1)
                if labeled[yy, xx] > 0:
                    ego_region = labeled[yy, xx]
                    break
            if ego_region > 0:
                break
    if ego_region > 0:
        return scan_mask & (labeled == ego_region)
    return scan_mask


def _find_runs(row_mask):
    """Find contiguous True runs in a 1D boolean array. Returns list of (start, end) inclusive."""
    cols = np.where(row_mask)[0]
    if len(cols) == 0:
        return []
    runs = []
    run_start = cols[0]
    for j in range(1, len(cols)):
        if cols[j] - cols[j - 1] > 5:
            runs.append((run_start, cols[j - 1]))
            run_start = cols[j]
    runs.append((run_start, cols[-1]))
    return runs


def _build_map_route_lookup(gps_bx, gps_by, bev_size):
    """Build an interpolator from BEV row → expected BEV column from map route."""
    if gps_bx is None or gps_by is None or len(gps_bx) < 2:
        return None
    from scipy.interpolate import interp1d
    order = np.argsort(gps_by)
    sorted_by = np.array(gps_by)[order]
    sorted_bx = np.array(gps_bx)[order]
    unique_mask = np.diff(sorted_by, prepend=-999) > 0.5
    sorted_by = sorted_by[unique_mask]
    sorted_bx = sorted_bx[unique_mask]
    if len(sorted_by) < 2:
        return None
    return interp1d(sorted_by, sorted_bx, bounds_error=False, fill_value="extrapolate")


def _snap_point_to_road(bx, by, scan_mask, bev_size):
    """Snap a single (bx, by) to the nearest road pixel. Searches the exact
    row first, then expands to neighboring rows."""
    bxi = int(np.clip(round(bx), 0, bev_size - 1))
    byi = int(np.clip(round(by), 0, bev_size - 1))
    if scan_mask[byi, bxi]:
        return float(bxi), float(byi)
    for dy in range(0, 30):
        for y_try in ([byi] if dy == 0 else [byi - dy, byi + dy]):
            if y_try < 0 or y_try >= bev_size:
                continue
            row = scan_mask[y_try, :]
            if np.any(row):
                cols = np.where(row)[0]
                nearest = cols[np.argmin(np.abs(cols - bxi))]
                return float(nearest), float(y_try)
    return float(bxi), float(byi)


def _constrained_smooth(centers, y_coords, scan_mask, bev_size, sigma=5, iterations=5):
    """Smooth the path while forcing every point to remain on road.
    Each iteration: Gaussian smooth → snap every point back to the nearest
    road pixel in its row. Converges to a smooth on-road path."""
    c = centers.copy()
    for _ in range(iterations):
        c = gaussian_filter1d(c, sigma=sigma, mode='nearest')
        for i in range(len(c)):
            y = int(np.clip(round(y_coords[i]), 0, bev_size - 1))
            bx = int(np.clip(round(c[i]), 0, bev_size - 1))
            if not scan_mask[y, bx]:
                row = scan_mask[y, :]
                if np.any(row):
                    cols = np.where(row)[0]
                    c[i] = float(cols[np.argmin(np.abs(cols - bx))])
    return c


def lane_aware_centerline_path(driveable_mask, bev_size=500, range_fwd=15.24, range_side=7.62,
                                road_mask=None, gps_bx=None, gps_by=None,
                                road_width_ft=20.0):
    """
    Lane-aware centerline planner that is strictly constrained to the road mask.

    Every point is computed from road pixels via the distance transform, and
    all smoothing is followed by a snap-back to the nearest road pixel. The
    output path is guaranteed to lie entirely within the driveable area.

    Args:
        road_width_ft: expected full road width in feet (default from calibration).
            Used to set the single/two-lane threshold at 70% of this value.
    """
    FT_TO_M = 0.3048
    px_per_ft = bev_size / (2 * range_side / FT_TO_M)
    two_lane_threshold_px = road_width_ft * 0.7 * px_per_ft

    scan_mask = road_mask if road_mask is not None else driveable_mask
    scan_mask = _find_ego_connected_road(scan_mask, bev_size)

    # Erode the mask so the path stays a few pixels inside the road edge,
    # then re-run connected-component from ego. Erosion can break thin
    # bridges; without re-checking connectivity the path would jump gaps.
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(scan_mask, iterations=4)
    eroded_connected = _find_ego_connected_road(eroded, bev_size)
    if np.any(eroded_connected):
        scan_mask = eroded_connected
    else:
        eroded2 = binary_erosion(scan_mask, iterations=2)
        eroded2_connected = _find_ego_connected_road(eroded2, bev_size)
        if np.any(eroded2_connected):
            scan_mask = eroded2_connected

    map_bx_lookup = _build_map_route_lookup(gps_bx, gps_by, bev_size)

    from scipy.ndimage import distance_transform_edt
    dist_xform = distance_transform_edt(scan_mask)

    # Find the ego start: lowest road pixel at center column
    ego_start_y = bev_size - 8
    for ey in range(bev_size - 1, -1, -1):
        if scan_mask[ey, bev_size // 2]:
            ego_start_y = ey
            break

    centers = [float(bev_size // 2)]
    y_coords = [float(ego_start_y)]
    step = 2
    gap_count = 0
    prev_center = bev_size / 2.0

    for y in range(ego_start_y - step, -1, -step):
        row = scan_mask[y, :]
        if not np.any(row):
            gap_count += 1
            if gap_count > 20 and len(centers) > 5:
                break
            continue
        gap_count = 0

        runs = _find_runs(row)
        if not runs:
            continue

        best_run = min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - prev_center))
        run_left, run_right = best_run
        run_width = run_right - run_left + 1
        run_mid = (run_left + run_right) / 2.0

        if run_width > two_lane_threshold_px:
            row_dist = dist_xform[y, run_left:run_right + 1]
            if len(row_dist) > 10:
                ridge_bx = run_left + np.argmax(row_dist)
            else:
                ridge_bx = int(run_mid)

            use_right_lane = True
            if map_bx_lookup is not None:
                map_bx = float(map_bx_lookup(y))
                use_right_lane = map_bx < ridge_bx
            else:
                use_right_lane = prev_center < run_mid

            if use_right_lane:
                lane_left, lane_right = run_left, ridge_bx
            else:
                lane_left, lane_right = ridge_bx, run_right

            lane_dist = dist_xform[y, lane_left:lane_right + 1]
            if len(lane_dist) > 0 and lane_dist.max() > 0:
                center = lane_left + int(np.argmax(lane_dist))
            else:
                center = (lane_left + lane_right) // 2
        else:
            row_dist = dist_xform[y, run_left:run_right + 1]
            if len(row_dist) > 0 and row_dist.max() > 0:
                center = run_left + int(np.argmax(row_dist))
            else:
                center = int(run_mid)

        # Hard verify — only accept points that are actually on road
        if not scan_mask[y, int(np.clip(center, 0, bev_size - 1))]:
            continue

        # Continuity check: if the center jumps too far laterally, verify
        # that the road actually connects. Walk pixel-by-pixel from the
        # previous point to this one; if any pixel is off the mask, reject.
        if len(centers) > 0:
            px0 = int(np.clip(prev_center, 0, bev_size - 1))
            py0 = int(y_coords[-1])
            px1 = int(np.clip(center, 0, bev_size - 1))
            py1 = y
            seg_len = max(abs(px1 - px0), abs(py1 - py0), 1)
            n_check = seg_len + 1
            ts = np.linspace(0, 1, n_check)
            cx = np.clip((px0 + ts * (px1 - px0)).astype(int), 0, bev_size - 1)
            cy = np.clip((py0 + ts * (py1 - py0)).astype(int), 0, bev_size - 1)
            if not np.all(scan_mask[cy, cx]):
                gap_count += 1
                if gap_count > 20 and len(centers) > 5:
                    break
                continue

        prev_center = center
        centers.append(float(center))
        y_coords.append(float(y))

    if len(centers) < 10:
        return None, None

    centers = np.array(centers)
    y_coords = np.array(y_coords)

    # Constrained smoothing: smooth then snap back to road, repeat
    centers = _constrained_smooth(centers, y_coords, scan_mask, bev_size,
                                   sigma=max(4, len(centers) // 20), iterations=6)

    # Keep dense points (every raw sample) so line segments can't jump off-road
    bx = centers.astype(float)
    by = y_coords.copy()

    # Hard verify + snap every point
    for i in range(len(bx)):
        bx[i], by[i] = _snap_point_to_road(bx[i], by[i], scan_mask, bev_size)

    # Final sweep: truncate the path at the first segment that leaves the road.
    # Every drawn line between consecutive points must be entirely on-road.
    last_good = 0
    for i in range(len(bx) - 1):
        x0, y0, x1, y1 = bx[i], by[i], bx[i + 1], by[i + 1]
        seg_len = max(abs(x1 - x0), abs(y1 - y0), 1)
        n_px = int(seg_len) + 1
        ts = np.linspace(0, 1, n_px)
        sx = np.clip((x0 + ts * (x1 - x0)).astype(int), 0, bev_size - 1)
        sy = np.clip((y0 + ts * (y1 - y0)).astype(int), 0, bev_size - 1)
        if np.all(scan_mask[sy, sx]):
            last_good = i + 1
        else:
            break
    bx = bx[:last_good + 1]
    by = by[:last_good + 1]

    if len(bx) < 3:
        return None, None

    local_left = (bx / bev_size - 0.5) * 2 * range_side
    local_fwd = (1 - by / bev_size) * range_fwd

    bev_traj = np.stack([bx.astype(int), by.astype(int)], axis=1)
    local_traj = np.stack([local_fwd, local_left], axis=1)

    return bev_traj, local_traj


# ── Cubic spline reference path ──────────────────────────────────────

def build_reference_path(gps_route, ego_x, ego_y, ego_yaw, lookahead_m=20.0, behind_m=2.0):
    """Extract GPS route segment near ego, transform to ego-local coords, fit spline."""
    dists = np.sqrt((gps_route[:, 1] - ego_x) ** 2 + (gps_route[:, 2] - ego_y) ** 2)
    nearest_idx = np.argmin(dists)

    cum_dist = 0
    start_idx = nearest_idx
    for i in range(nearest_idx, 0, -1):
        d = np.sqrt((gps_route[i, 1] - gps_route[i - 1, 1]) ** 2 +
                     (gps_route[i, 2] - gps_route[i - 1, 2]) ** 2)
        cum_dist += d
        if cum_dist >= behind_m:
            start_idx = i
            break

    cum_dist = 0
    end_idx = nearest_idx
    for i in range(nearest_idx, len(gps_route) - 1):
        d = np.sqrt((gps_route[i + 1, 1] - gps_route[i, 1]) ** 2 +
                     (gps_route[i + 1, 2] - gps_route[i, 2]) ** 2)
        cum_dist += d
        if cum_dist >= lookahead_m:
            end_idx = i + 1
            break
    else:
        end_idx = len(gps_route) - 1

    segment = gps_route[start_idx:end_idx + 1]
    if len(segment) < 4:
        return None

    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    dx = segment[:, 1] - ego_x
    dy = segment[:, 2] - ego_y
    local_x = dx * c - dy * s  # forward
    local_y = dx * s + dy * c  # left

    arc_len = np.zeros(len(local_x))
    for i in range(1, len(local_x)):
        arc_len[i] = arc_len[i - 1] + np.sqrt(
            (local_x[i] - local_x[i - 1]) ** 2 + (local_y[i] - local_y[i - 1]) ** 2
        )

    if arc_len[-1] < 0.5:
        return None

    unique_mask = np.diff(arc_len, prepend=-1) > 0.01
    arc_len = arc_len[unique_mask]
    local_x = local_x[unique_mask]
    local_y = local_y[unique_mask]

    if len(arc_len) < 4:
        return None

    cs_x = CubicSpline(arc_len, local_x)
    cs_y = CubicSpline(arc_len, local_y)

    return cs_x, cs_y, arc_len[-1]


# ── Approach 1: Frenet Frame Optimal Trajectory ──────────────────────

def quintic_poly(t, a0, a1, a2, a3, a4, a5):
    return a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5


def quintic_poly_coeffs(x0, v0, a0, xf, vf, af, T):
    """Solve quintic polynomial boundary conditions."""
    A = np.array([
        [T ** 3, T ** 4, T ** 5],
        [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [6 * T, 12 * T ** 2, 20 * T ** 3],
    ])
    b = np.array([
        xf - x0 - v0 * T - 0.5 * a0 * T ** 2,
        vf - v0 - a0 * T,
        af - a0,
    ])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return [x0, v0, 0.5 * a0, c[0], c[1], c[2]]


def frenet_optimal_trajectory(driveable_mask, gps_route, ego_x, ego_y, ego_yaw,
                               bev_size=500, range_fwd=15.24, range_side=7.62,
                               n_lateral=11, lookahead_m=15.0, map_route_ahead=None):
    """
    Frenet frame planner: sample lateral offsets from reference path,
    pick the smoothest trajectory that stays in driveable area.
    Uses map route when available, falls back to GPS recording.

    Returns trajectory as array of (bev_x, bev_y) pixel coordinates, or None.
    Also returns trajectory in ego-local (fwd, left) meters.
    """
    ref = None
    if map_route_ahead is not None and len(map_route_ahead) >= 4:
        ref = build_reference_path_from_map(map_route_ahead, ego_x, ego_y, ego_yaw,
                                             lookahead_m=lookahead_m)
    if ref is None:
        ref = build_reference_path(gps_route, ego_x, ego_y, ego_yaw,
                                    lookahead_m=lookahead_m, behind_m=2.0)
    if ref is None:
        return None, None

    cs_x, cs_y, total_s = ref

    n_points = 40
    s_vals = np.linspace(0, min(total_s, lookahead_m), n_points)
    ref_x = cs_x(s_vals)
    ref_y = cs_y(s_vals)

    ref_dx = cs_x(s_vals, 1)
    ref_dy = cs_y(s_vals, 1)
    ref_heading = np.arctan2(ref_dy, ref_dx)
    normal_x = -np.sin(ref_heading)
    normal_y = np.cos(ref_heading)

    max_lateral = 3.0
    d_offsets = np.linspace(-max_lateral, max_lateral, n_lateral)

    best_traj = None
    best_cost = float("inf")
    best_local = None

    for d_final in d_offsets:
        d_vals = np.linspace(0, d_final, n_points)

        traj_fwd = ref_x + d_vals * normal_x
        traj_left = ref_y + d_vals * normal_y

        bx = ((traj_left / range_side * 0.5 + 0.5) * bev_size).astype(int)
        by = ((1 - traj_fwd / range_fwd) * bev_size).astype(int)

        in_bev = (bx >= 0) & (bx < bev_size) & (by >= 0) & (by < bev_size)
        if not np.all(in_bev):
            fwd_in = traj_fwd[in_bev]
            if len(fwd_in) == 0 or fwd_in.max() < 3.0:
                continue

        valid_bx = np.clip(bx, 0, bev_size - 1)
        valid_by = np.clip(by, 0, bev_size - 1)

        in_fov_count = in_bev.sum()
        if in_fov_count < 3:
            continue

        on_road = driveable_mask[valid_by[in_bev], valid_bx[in_bev]]
        collision_frac = 1 - on_road.sum() / in_fov_count
        if collision_frac > 0.15:
            continue

        deviation_cost = np.sum(d_vals ** 2) * 0.1
        smoothness_cost = 0
        if len(traj_fwd) > 2:
            ddx = np.diff(traj_fwd, 2)
            ddy = np.diff(traj_left, 2)
            smoothness_cost = np.sum(ddx ** 2 + ddy ** 2)
        collision_cost = collision_frac * 2000

        cost = deviation_cost + smoothness_cost + collision_cost

        if cost < best_cost:
            best_cost = cost
            best_traj = np.stack([bx, by], axis=1)
            best_local = np.stack([traj_fwd, traj_left], axis=1)

    return best_traj, best_local


# ── Approach 2: MPC with Bicycle Model ───────────────────────────────

GOLF_CART_L = 2.5  # wheelbase meters
MAX_STEER = math.radians(35)  # max steering angle
MAX_SPEED = 5.0  # m/s


def extract_corridor(driveable_mask, ref_fwd, ref_left, ref_heading,
                      bev_size, range_fwd, range_side, max_width=5.0):
    """For each point along reference path, find left/right driveable boundaries."""
    n = len(ref_fwd)
    d_left = np.full(n, max_width)
    d_right = np.full(n, max_width)

    for i in range(n):
        nx = -math.sin(ref_heading[i])
        ny = math.cos(ref_heading[i])

        for sign, arr in [(1, d_left), (-1, d_right)]:
            for d in np.arange(0.1, max_width, 0.2):
                probe_fwd = ref_fwd[i] + sign * d * nx
                probe_left = ref_left[i] + sign * d * ny
                bx = int((probe_left / range_side * 0.5 + 0.5) * bev_size)
                by = int((1 - probe_fwd / range_fwd) * bev_size)
                if bx < 0 or bx >= bev_size or by < 0 or by >= bev_size:
                    arr[i] = d - 0.2
                    break
                if not driveable_mask[by, bx]:
                    arr[i] = d - 0.2
                    break

    return d_left, d_right


def mpc_bicycle_trajectory(driveable_mask, gps_route, ego_x, ego_y, ego_yaw, ego_speed,
                            bev_size=500, range_fwd=15.24, range_side=7.62,
                            horizon=20, dt=0.2, lookahead_m=15.0, map_route_ahead=None,
                            road_mask=None):
    """
    MPC planner: optimize steering/acceleration over a horizon using bicycle model,
    constrained to stay within driveable corridor extracted from BEV seg.
    Uses map route when available, falls back to GPS recording.

    Returns trajectory as (bev_x, bev_y) pixel coords and ego-local (fwd, left).
    """
    ref = None
    if map_route_ahead is not None and len(map_route_ahead) >= 4:
        ref = build_reference_path_from_map(map_route_ahead, ego_x, ego_y, ego_yaw,
                                             lookahead_m=lookahead_m)
    if ref is None:
        ref = build_reference_path(gps_route, ego_x, ego_y, ego_yaw,
                                    lookahead_m=lookahead_m, behind_m=2.0)
    if ref is None:
        return None, None

    cs_x, cs_y, total_s = ref

    n_ref = 50
    s_vals = np.linspace(0, min(total_s, lookahead_m), n_ref)
    ref_x = cs_x(s_vals)
    ref_y = cs_y(s_vals)
    ref_dx = cs_x(s_vals, 1)
    ref_dy = cs_y(s_vals, 1)
    ref_heading = np.arctan2(ref_dy, ref_dx)

    d_left, d_right = extract_corridor(
        driveable_mask, ref_x, ref_y, ref_heading,
        bev_size, range_fwd, range_side
    )

    speed = max(ego_speed, 1.5)
    if speed > MAX_SPEED:
        speed = MAX_SPEED

    def objective(u):
        steers = u[:horizon]
        x, y, theta = 0.0, 0.0, 0.0
        traj_x, traj_y = [x], [y]
        cost = 0.0

        for k in range(horizon):
            delta = np.clip(steers[k], -MAX_STEER, MAX_STEER)
            x += speed * math.cos(theta) * dt
            y += speed * math.sin(theta) * dt
            theta += speed / GOLF_CART_L * math.tan(delta) * dt
            traj_x.append(x)
            traj_y.append(y)

            s_approx = math.sqrt(x ** 2 + y ** 2)
            s_approx = np.clip(s_approx, 0, total_s - 0.01)
            rx = float(cs_x(s_approx))
            ry = float(cs_y(s_approx))

            cost += 0.5 * ((x - rx) ** 2 + (y - ry) ** 2)

            if k > 0:
                cost += 5.0 * (steers[k] - steers[k - 1]) ** 2
            cost += 0.5 * steers[k] ** 2

            bx = int((y / range_side * 0.5 + 0.5) * bev_size)
            by = int((1 - x / range_fwd) * bev_size)
            if 0 <= bx < bev_size and 0 <= by < bev_size:
                if not driveable_mask[by, bx]:
                    cost += 5000.0
            else:
                if x > 0:
                    cost += 5000.0

        return cost

    u0 = np.zeros(horizon)
    bounds = [(-MAX_STEER, MAX_STEER)] * horizon

    result = minimize(objective, u0, method="SLSQP", bounds=bounds,
                      options={"maxiter": 100, "ftol": 1e-4})

    steers = result.x
    x, y, theta = 0.0, 0.0, 0.0
    traj_fwd, traj_left = [x], [y]
    for k in range(horizon):
        delta = np.clip(steers[k], -MAX_STEER, MAX_STEER)
        x += speed * math.cos(theta) * dt
        y += speed * math.sin(theta) * dt
        theta += speed / GOLF_CART_L * math.tan(delta) * dt
        traj_fwd.append(x)
        traj_left.append(y)

    traj_fwd = np.array(traj_fwd)
    traj_left = np.array(traj_left)

    bx = (traj_left / range_side * 0.5 + 0.5) * bev_size
    by = (1 - traj_fwd / range_fwd) * bev_size
    bx = np.clip(bx, 0, bev_size - 1)
    by = np.clip(by, 0, bev_size - 1)

    bev_traj = np.stack([bx.astype(int), by.astype(int)], axis=1)
    local_traj = np.stack([traj_fwd, traj_left], axis=1)

    return bev_traj, local_traj

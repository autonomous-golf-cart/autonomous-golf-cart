"""
test_geo.py — cartlib.geo distance, bearing, projection, and path-snapping math.

Uses a campus-scale origin; tolerances are loose enough for the flat-earth
approximation over tens of metres.

Run: python3 -m pytest tests/
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartlib import geo

ORIGIN = (37.4275, -122.1697)
METRES_PER_DEG_LAT = geo.EARTH_R * math.pi / 180.0


def north(m, origin=ORIGIN):
    return geo.from_local_xy(origin, 0.0, m)


def east(m, origin=ORIGIN):
    return geo.from_local_xy(origin, m, 0.0)


# --- haversine -------------------------------------------------------------
def test_haversine_zero():
    assert geo.haversine_m(ORIGIN, ORIGIN) == 0.0


def test_haversine_one_degree_latitude():
    a, b = (0.0, 0.0), (1.0, 0.0)
    assert geo.haversine_m(a, b) == pytest.approx(METRES_PER_DEG_LAT, rel=1e-9)


def test_haversine_symmetric():
    b = (37.43, -122.17)
    assert geo.haversine_m(ORIGIN, b) == pytest.approx(geo.haversine_m(b, ORIGIN))


def test_haversine_matches_local_offset():
    assert geo.haversine_m(ORIGIN, north(25.0)) == pytest.approx(25.0, rel=1e-4)
    assert geo.haversine_m(ORIGIN, east(25.0)) == pytest.approx(25.0, rel=1e-4)


# --- bearing ---------------------------------------------------------------
@pytest.mark.parametrize("target, expected", [
    ((1.0, 0.0), 0.0),
    ((0.0, 1.0), 90.0),
    ((-1.0, 0.0), 180.0),
    ((0.0, -1.0), 270.0),
])
def test_bearing_cardinals(target, expected):
    assert geo.bearing_deg((0.0, 0.0), target) == pytest.approx(expected, abs=1e-9)


def test_bearing_range():
    b = geo.bearing_deg(ORIGIN, (37.42, -122.18))
    assert 0.0 <= b < 360.0


# --- angle_diff ------------------------------------------------------------
@pytest.mark.parametrize("target, source, expected", [
    (10.0, 0.0, 10.0),
    (0.0, 10.0, -10.0),
    (350.0, 10.0, -20.0),
    (10.0, 350.0, 20.0),
    (180.0, 0.0, 180.0),
    (0.0, 180.0, 180.0),
    (270.0, 90.0, 180.0),
])
def test_angle_diff(target, source, expected):
    assert geo.angle_diff_deg(target, source) == pytest.approx(expected)


def test_angle_diff_range():
    for t in range(0, 720, 7):
        for s in range(0, 360, 11):
            d = geo.angle_diff_deg(float(t), float(s))
            assert -180.0 < d <= 180.0


# --- local projection ------------------------------------------------------
def test_local_xy_origin_is_zero():
    assert geo.local_xy(ORIGIN, ORIGIN) == (0.0, 0.0)


def test_local_xy_axes():
    x, y = geo.local_xy(ORIGIN, north(10.0))
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(10.0)
    x, y = geo.local_xy(ORIGIN, east(-7.5))
    assert x == pytest.approx(-7.5)
    assert y == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("x, y", [(0.0, 0.0), (12.3, -4.5), (-30.0, 60.0)])
def test_local_xy_round_trip(x, y):
    p = geo.from_local_xy(ORIGIN, x, y)
    rx, ry = geo.local_xy(ORIGIN, p)
    assert rx == pytest.approx(x, abs=1e-6)
    assert ry == pytest.approx(y, abs=1e-6)


# --- path helpers ----------------------------------------------------------
PATH = [north(0.0), north(10.0), north(20.0)]


def test_nearest_index():
    assert geo.nearest_index(PATH, north(1.0)) == 0
    assert geo.nearest_index(PATH, north(11.0)) == 1
    assert geo.nearest_index(PATH, north(100.0)) == 2


def test_nearest_point_on_path_left_is_positive():
    snap = geo.nearest_point_on_path(PATH, geo.from_local_xy(ORIGIN, -3.0, 5.0))
    assert snap.segment_index == 0
    assert snap.fraction == pytest.approx(0.5, abs=1e-3)
    assert snap.distance_m == pytest.approx(3.0, rel=1e-3)
    assert snap.signed_distance_m == pytest.approx(3.0, rel=1e-3)
    assert snap.along_m == pytest.approx(5.0, rel=1e-3)


def test_nearest_point_on_path_right_is_negative():
    snap = geo.nearest_point_on_path(PATH, geo.from_local_xy(ORIGIN, 2.0, 15.0))
    assert snap.segment_index == 1
    assert snap.signed_distance_m == pytest.approx(-2.0, rel=1e-3)
    assert snap.along_m == pytest.approx(15.0, rel=1e-3)


def test_nearest_point_on_path_clamps_before_start():
    snap = geo.nearest_point_on_path(PATH, north(-4.0))
    assert snap.segment_index == 0
    assert snap.fraction == 0.0
    assert snap.distance_m == pytest.approx(4.0, rel=1e-3)


def test_nearest_point_single_vertex():
    snap = geo.nearest_point_on_path([ORIGIN], north(5.0))
    assert snap.point == ORIGIN
    assert snap.distance_m == pytest.approx(5.0, rel=1e-4)


def test_point_at_distance():
    p, idx = geo.point_at_distance(PATH, 15.0)
    assert idx == 2
    assert geo.haversine_m(p, north(15.0)) < 0.01


def test_point_at_distance_past_end_returns_last():
    p, idx = geo.point_at_distance(PATH, 1000.0)
    assert p == PATH[-1]
    assert idx == len(PATH) - 1


def test_lookahead_point():
    pos = geo.from_local_xy(ORIGIN, 1.0, 4.0)  # beside the path at 4 m along
    p, idx = geo.lookahead_point(PATH, pos, 3.0)
    assert idx == 1
    assert geo.haversine_m(p, north(7.0)) < 0.01

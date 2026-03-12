import math


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def center_point(p1, p2):
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


def angle_degrees(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

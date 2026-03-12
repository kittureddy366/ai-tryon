import numpy as np


def lerp(current, target, alpha):
    return current + (target - current) * alpha


def smooth_vector(current, target, alpha):
    return current + (target - current) * alpha


def bounded_spring_step(value, velocity, target, stiffness, damping, dt, min_v, max_v):
    velocity += (target - value) * stiffness * dt
    velocity *= damping
    value += velocity * dt * 60.0
    value = float(np.clip(value, min_v, max_v))
    return value, velocity


def exponential_smooth(current, previous, alpha):
    if previous is None:
        return current
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * current + (1.0 - alpha) * previous

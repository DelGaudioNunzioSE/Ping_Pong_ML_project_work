import math


def trajectory(state, target_z=0.1):
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Gravitational acceleration

    # Solve the quadratic equation to find the time t when z(t) = target_z
    a = -0.5 * g
    b = vz
    c = bz - target_z

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # The ball never reaches z = target_z
        return None, None

    # Calculate the possible times
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    # Take the positive time
    t = max(t1, t2)

    if t < 0:
        # Both times are negative, the ball never reaches z = target_z
        return None, None

    # Calculate the x and y positions where the ball reaches z = target_z
    x = bx + vx * t
    y = by + vy * t

    return x, y

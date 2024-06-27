import math


def max_height_point(state):
    """
    Calculates the maximum height point of a projectile trajectory.

    Args:
        state (list or tuple): State vector containing ball position (bx, by, bz) and velocity (vx, vy, vz).

    Returns:
        tuple: x, y, z coordinates of the maximum height point.
               Returns (None, None, None) if the ball is already falling (negative time to max height).
    """
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Acceleration due to gravity

    # Time at which the ball reaches the apex of the trajectory
    t_max = vz / g

    if t_max < 0:
        # If time is negative, the ball is already falling
        return None, None, None

    # Calculate x, y, z positions at the maximum height
    x_max = bx + vx * t_max
    y_max = by + vy * t_max
    z_max = bz + vz * t_max - 0.5 * g * t_max ** 2

    return x_max, y_max, z_max


def trajectory(state, target_z=0.1):
    """
    Calculates the x, y positions where a projectile reaches a specified z-height.

    Args:
        state (list or tuple): State vector containing ball position (bx, by, bz) and velocity (vx, vy, vz).
        target_z (float, optional): Target z-height to reach. Defaults to 0.1.

    Returns:
        tuple: x, y coordinates where the projectile reaches the target_z height.
               Returns (None, None) if the projectile does not reach the target_z height.
    """
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Acceleration due to gravity

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

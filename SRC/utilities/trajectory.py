import math


def max_height_point(state):
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Accelerazione gravitazionale

    # Il tempo in cui la pallina raggiunge l'apice della traiettoria è dato da:
    t_max = vz / g

    if t_max < 0:
        # Se il tempo è negativo, significa che la pallina sta già cadendo e non c'è un punto di massimo
        return None, None, None

    # Calcolare le posizioni x e y quando la pallina raggiunge l'apice
    x_max = bx + vx * t_max
    y_max = by + vy * t_max
    z_max = bz + vz * t_max - 0.5 * g * t_max ** 2

    return x_max, y_max, z_max

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

# def bounce_trajectory(state, target_y):
#     bx, by, bz = state[17:20]
#     vx, vy, vz = state[20:23]
#     g = 9.81  # Gravitational acceleration
#     bounce_z = 0.05  # Z value at which the ball bounces
#
#     # Step 1: Calculate time and position at first bounce (z = 0.05)
#     a = -0.5 * g
#     b = vz
#     c = bz - bounce_z
#
#     # Calculate the discriminant
#     discriminant = b**2 - 4 * a * c
#
#     if discriminant < 0:
#         # The ball never reaches z = bounce_z
#         return None, None
#
#     # Calculate the possible times
#     t1 = (-b + math.sqrt(discriminant)) / (2 * a)
#     t2 = (-b - math.sqrt(discriminant)) / (2 * a)
#
#     # Take the positive time
#     t_bounce = max(t1, t2)
#
#     if t_bounce < 0:
#         # Both times are negative, the ball never reaches z = bounce_z
#         return None, None
#
#     # Position at the bounce
#     x_bounce = bx + vx * t_bounce
#     y_bounce = by + vy * t_bounce
#     z_bounce = bounce_z
#
#     # Velocity after the bounce (vertical component inverted)
#     vz_after_bounce = -vz + g * t_bounce
#
#     # Step 2: Calculate new trajectory after the bounce to reach target_y
#
#     # Time to reach target_y after bounce
#     if vy != 0:
#         t_to_target_y = (target_y - y_bounce) / vy
#     else:
#         # Ball doesn't move in y direction
#         t_to_target_y = float('inf')
#
#     if t_to_target_y < 0:
#         return None, None
#
#     # Position at target_y
#     x_at_target_y = x_bounce + vx * t_to_target_y
#     z_at_target_y = z_bounce + vz_after_bounce * t_to_target_y - 0.5 * g * t_to_target_y**2
#
#     return x_at_target_y, z_at_target_y

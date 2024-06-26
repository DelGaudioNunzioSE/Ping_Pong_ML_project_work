import math


def touch_the_net(state, target_y=1.2):
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Accelerazione gravitazionale

    # Calcolare il tempo t quando y(t) = target_y
    if vy == 0:
        # Se la velocità in y è zero, la palla non raggiungerà mai target_y
        return False

    t = (target_y - by) / vy

    if t < 0:
        # Se il tempo è negativo, significa che la pallina non raggiungerà target_y in futuro
        return False

    # Calcolare la posizione z in questo tempo
    z = bz + vz * t - 0.5 * g * t ** 2

    if z <= 0.13:
        return True
    else:
        return False


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